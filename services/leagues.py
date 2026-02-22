from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from typing import Any

from sqlalchemy import and_, asc, case, desc, func, or_, select
from sqlalchemy.orm import Session, aliased

from db.models import ChlGame, ChlPredictionRecord, ChlTeam
from services.data_backend import CHL_LEAGUES, DataBackendError, require_supported_league_code
from services.league_time import league_today as _shared_league_today

SUPPORTED_LEAGUES = [
    {
        "code": code,
        "name": str(cfg["name"]),
        "sport": str(cfg["sport"]),
        "active": bool(cfg.get("active", True)),
        "timezone": str(cfg.get("timezone") or "UTC"),
    }
    for code, cfg in sorted(CHL_LEAGUES.items(), key=lambda item: int(item[1]["id"]))
]

FIFTY_BY_FIFTY_LOGO_PROVIDER_IDS = {201, 204, 208, 213, 222, 277}
VALID_SORTS = {"prediction_date_desc", "prediction_date_asc", "game_date_desc", "game_date_asc"}
VALID_RESULTS = {"all", "correct", "incorrect", "pending"}
VALID_RESOLVED_RESULTS = {"all", "correct", "incorrect"}


class LeagueServiceError(RuntimeError):
    pass


class UnknownLeagueError(LeagueServiceError):
    pass


class DailyReportNotFoundError(LeagueServiceError):
    pass


class GameNotFoundError(LeagueServiceError):
    pass


class TeamNotFoundError(LeagueServiceError):
    pass


def _ensure_league(league_code: str) -> str:
    try:
        return require_supported_league_code(league_code)
    except DataBackendError as exc:
        raise UnknownLeagueError(str(exc)) from exc


def _league_id_for_code(league_code: str) -> int:
    cfg = CHL_LEAGUES.get(league_code)
    if not cfg:
        raise UnknownLeagueError(f"Unsupported league: {league_code}")
    return int(cfg["id"])


def default_logo_url_for_provider(provider_team_id: int, league_code: str = "whl") -> str:
    normalized_league = _ensure_league(league_code)
    if normalized_league == "whl" and provider_team_id in FIFTY_BY_FIFTY_LOGO_PROVIDER_IDS:
        return f"https://assets.leaguestat.com/{normalized_league}/logos/50x50/{provider_team_id}.png"
    return f"https://assets.leaguestat.com/{normalized_league}/logos/{provider_team_id}.png"


def _serialize_team(
    team: ChlTeam | None,
    league_code: str,
    fallback_provider_id: int | None = None,
    fallback_name: str | None = None,
) -> dict[str, Any]:
    provider_id = team.hockeytech_id if team is not None else int(fallback_provider_id or 0)
    logo_url = team.logo_url if (team and team.logo_url) else (default_logo_url_for_provider(provider_id, league_code=league_code) if provider_id else None)

    return {
        "db_team_id": int(team.id) if team is not None else 0,
        "provider_team_id": provider_id,
        "name": team.name if team is not None else (fallback_name or "Unknown Team"),
        "city": team.city if team is not None else None,
        "conference": team.conference if team is not None else None,
        "division": team.division if team is not None else None,
        "logo_url": logo_url,
        "active": bool(team.active) if team is not None else True,
    }


def _to_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _prediction_datetime_bounds(
    prediction_date_from: dt.date | None,
    prediction_date_to: dt.date | None,
) -> tuple[dt.datetime | None, dt.datetime | None]:
    start = dt.datetime.combine(prediction_date_from, dt.time.min) if prediction_date_from else None
    end = dt.datetime.combine(prediction_date_to, dt.time.max) if prediction_date_to else None
    return start, end


def _league_today(league_code: str, now_utc: dt.datetime | None = None) -> dt.date:
    normalized = _ensure_league(league_code)
    return _shared_league_today(normalized, now_utc=now_utc)


def _game_date_value(item: dict[str, Any]) -> dt.date | None:
    game_date = item.get("game_date")
    if isinstance(game_date, dt.date):
        return game_date

    scheduled_time = item.get("scheduled_time_utc")
    if isinstance(scheduled_time, dt.datetime):
        return scheduled_time.date()

    return None


def _scheduled_time_value(item: dict[str, Any]) -> dt.datetime:
    scheduled_time = item.get("scheduled_time_utc")
    if isinstance(scheduled_time, dt.datetime):
        return scheduled_time
    return dt.datetime.max


def _is_resolved(item: dict[str, Any]) -> bool:
    return item.get("consensus_correct") is not None


def _is_upcoming(item: dict[str, Any], as_of_date: dt.date) -> bool:
    if _is_resolved(item):
        return False
    game_date = _game_date_value(item)
    return game_date is not None and game_date >= as_of_date


def _sort_for_upcoming(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda row: (
            _game_date_value(row) or dt.date.max,
            _scheduled_time_value(row),
            int(row["game_id"]),
        ),
    )


def _paginate_items(items: list[dict[str, Any]], *, limit: int, offset: int) -> tuple[list[dict[str, Any]], int, bool]:
    total = len(items)
    paged_items = items[offset : offset + limit]
    has_more = (offset + limit) < total
    return paged_items, total, has_more


def _build_prediction_page(items: list[dict[str, Any]], *, limit: int, offset: int) -> dict[str, Any]:
    paged_items, total, has_more = _paginate_items(items, limit=limit, offset=offset)
    return {
        "items": paged_items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": has_more,
    }


def list_supported_leagues() -> list[dict[str, Any]]:
    return [dict(league) for league in SUPPORTED_LEAGUES]


def list_league_teams(
    db: Session,
    league_code: str,
    active_only: bool = True,
) -> list[dict[str, Any]]:
    normalized_league = _ensure_league(league_code)
    league_id = _league_id_for_code(normalized_league)

    stmt = select(ChlTeam).where(ChlTeam.league_id == league_id).order_by(ChlTeam.name.asc())
    if active_only:
        stmt = stmt.where(ChlTeam.active.is_(True))

    teams = db.scalars(stmt).all()
    return [_serialize_team(team, league_code=normalized_league) for team in teams]


def _empty_team_performance_stats() -> dict[str, Any]:
    return {
        "games_with_predictions": 0,
        "resolved_games": 0,
        "pending_games": 0,
        "correct_predictions": 0,
        "accuracy_pct": None,
        "actual_wins": 0,
        "actual_losses": 0,
        "predicted_wins": 0,
        "predicted_losses": 0,
        "win_delta": 0,
    }


def _finalize_team_performance_stats(stats: dict[str, Any]) -> dict[str, Any]:
    resolved_games = int(stats["resolved_games"])
    games_with_predictions = int(stats["games_with_predictions"])
    correct_predictions = int(stats["correct_predictions"])
    actual_wins = int(stats["actual_wins"])
    predicted_wins = int(stats["predicted_wins"])

    stats["pending_games"] = max(0, games_with_predictions - resolved_games)
    stats["accuracy_pct"] = round((correct_predictions / resolved_games) * 100.0, 2) if resolved_games > 0 else None
    stats["win_delta"] = predicted_wins - actual_wins
    return stats


def _query_consensus_prediction_rows(
    db: Session,
    *,
    league_id: int,
    game_date_from: dt.date | None,
    game_date_to: dt.date | None,
    team_db_id: int | None = None,
) -> list[dict[str, Any]]:
    priority_expr = case(
        (ChlPredictionRecord.k_value == 15, 0),
        else_=1,
    )
    row_rank = func.row_number().over(
        partition_by=(ChlPredictionRecord.league_id, ChlPredictionRecord.game_id),
        order_by=(
            priority_expr.asc(),
            ChlPredictionRecord.k_value.desc(),
            ChlPredictionRecord.prediction_date.desc().nullslast(),
            ChlPredictionRecord.id.desc(),
        ),
    )

    ranked_stmt = (
        select(
            ChlPredictionRecord.game_id.label("game_id"),
            ChlPredictionRecord.home_team_id.label("home_team_id"),
            ChlPredictionRecord.away_team_id.label("away_team_id"),
            ChlPredictionRecord.predicted_winner_id.label("predicted_winner_id"),
            ChlPredictionRecord.actual_winner_id.label("actual_winner_id"),
            ChlPredictionRecord.correct.label("correct"),
            row_rank.label("row_rank"),
        )
        .join(
            ChlGame,
            and_(
                ChlGame.league_id == ChlPredictionRecord.league_id,
                ChlGame.game_id == ChlPredictionRecord.game_id,
            ),
        )
        .where(ChlPredictionRecord.league_id == league_id)
        .where(ChlGame.game_date.is_not(None))
    )

    if game_date_from is not None:
        ranked_stmt = ranked_stmt.where(ChlGame.game_date >= game_date_from)
    if game_date_to is not None:
        ranked_stmt = ranked_stmt.where(ChlGame.game_date <= game_date_to)
    if team_db_id is not None:
        ranked_stmt = ranked_stmt.where(
            or_(
                ChlPredictionRecord.home_team_id == team_db_id,
                ChlPredictionRecord.away_team_id == team_db_id,
            )
        )

    ranked_subquery = ranked_stmt.subquery("ranked_predictions")
    consensus_stmt = select(
        ranked_subquery.c.game_id,
        ranked_subquery.c.home_team_id,
        ranked_subquery.c.away_team_id,
        ranked_subquery.c.predicted_winner_id,
        ranked_subquery.c.actual_winner_id,
        ranked_subquery.c.correct,
    ).where(ranked_subquery.c.row_rank == 1)

    rows = db.execute(consensus_stmt).mappings().all()
    return [dict(row) for row in rows]


def _build_team_performance_items(
    *,
    teams: list[ChlTeam],
    league_code: str,
    consensus_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    team_by_id = {int(team.id): team for team in teams}
    stats_by_team_id = {int(team.id): _empty_team_performance_stats() for team in teams}

    for row in consensus_rows:
        home_team_id = int(row["home_team_id"]) if row["home_team_id"] is not None else None
        away_team_id = int(row["away_team_id"]) if row["away_team_id"] is not None else None
        correct = row.get("correct")
        predicted_winner_id = int(row["predicted_winner_id"]) if row["predicted_winner_id"] is not None else None
        actual_winner_id = int(row["actual_winner_id"]) if row["actual_winner_id"] is not None else None

        for team_id in (home_team_id, away_team_id):
            if team_id is None:
                continue
            stats = stats_by_team_id.get(team_id)
            if stats is None:
                continue

            stats["games_with_predictions"] += 1

            if correct is None:
                continue

            stats["resolved_games"] += 1
            if correct is True:
                stats["correct_predictions"] += 1

            if actual_winner_id is not None:
                if actual_winner_id == team_id:
                    stats["actual_wins"] += 1
                else:
                    stats["actual_losses"] += 1

            if predicted_winner_id is not None:
                if predicted_winner_id == team_id:
                    stats["predicted_wins"] += 1
                else:
                    stats["predicted_losses"] += 1

    items: list[dict[str, Any]] = []
    for team in teams:
        team_id = int(team.id)
        finalized_stats = _finalize_team_performance_stats(stats_by_team_id.get(team_id, _empty_team_performance_stats()))
        items.append(
            {
                "team": _serialize_team(team_by_id[team_id], league_code=league_code),
                "stats": finalized_stats,
            }
        )

    return items


def list_team_performance(
    db: Session,
    league_code: str,
    *,
    date_from: dt.date | None,
    date_to: dt.date | None,
    active_only: bool = True,
) -> dict[str, Any]:
    normalized_league = _ensure_league(league_code)
    league_id = _league_id_for_code(normalized_league)

    if date_from is not None and date_to is not None and date_from > date_to:
        raise ValueError("date_from cannot be after date_to.")

    teams_stmt = select(ChlTeam).where(ChlTeam.league_id == league_id).order_by(ChlTeam.name.asc())
    if active_only:
        teams_stmt = teams_stmt.where(ChlTeam.active.is_(True))
    teams = db.scalars(teams_stmt).all()

    consensus_rows = _query_consensus_prediction_rows(
        db,
        league_id=league_id,
        game_date_from=date_from,
        game_date_to=date_to,
        team_db_id=None,
    )
    items = _build_team_performance_items(
        teams=teams,
        league_code=normalized_league,
        consensus_rows=consensus_rows,
    )

    return {
        "league_code": normalized_league,
        "date_from": date_from,
        "date_to": date_to,
        "items": items,
    }


def get_team_performance(
    db: Session,
    league_code: str,
    *,
    team_provider_id: int,
    date_from: dt.date | None,
    date_to: dt.date | None,
) -> dict[str, Any]:
    normalized_league = _ensure_league(league_code)
    league_id = _league_id_for_code(normalized_league)

    if date_from is not None and date_to is not None and date_from > date_to:
        raise ValueError("date_from cannot be after date_to.")

    team = db.scalar(
        select(ChlTeam).where(
            ChlTeam.league_id == league_id,
            ChlTeam.hockeytech_id == team_provider_id,
        )
    )
    if team is None:
        raise TeamNotFoundError(f"Team not found for provider_team_id={team_provider_id}")

    consensus_rows = _query_consensus_prediction_rows(
        db,
        league_id=league_id,
        game_date_from=date_from,
        game_date_to=date_to,
        team_db_id=int(team.id),
    )
    items = _build_team_performance_items(
        teams=[team],
        league_code=normalized_league,
        consensus_rows=consensus_rows,
    )

    return {
        "league_code": normalized_league,
        "date_from": date_from,
        "date_to": date_to,
        "item": items[0],
    }


def _query_prediction_rows(
    db: Session,
    league_code: str,
    *,
    prediction_date_from: dt.date | None,
    prediction_date_to: dt.date | None,
    game_date_from: dt.date | None,
    game_date_to: dt.date | None,
    team_provider_id: int | None,
) -> list[tuple[ChlPredictionRecord, ChlGame | None, ChlTeam | None, ChlTeam | None, ChlTeam | None, ChlTeam | None]]:
    normalized_league = _ensure_league(league_code)
    league_id = _league_id_for_code(normalized_league)

    home_team = aliased(ChlTeam)
    away_team = aliased(ChlTeam)
    predicted_team = aliased(ChlTeam)
    actual_team = aliased(ChlTeam)

    stmt = (
        select(
            ChlPredictionRecord,
            ChlGame,
            home_team,
            away_team,
            predicted_team,
            actual_team,
        )
        .join(
            ChlGame,
            and_(
                ChlGame.game_id == ChlPredictionRecord.game_id,
                ChlGame.league_id == ChlPredictionRecord.league_id,
            ),
            isouter=True,
        )
        .join(home_team, home_team.id == ChlPredictionRecord.home_team_id)
        .join(away_team, away_team.id == ChlPredictionRecord.away_team_id)
        .join(predicted_team, predicted_team.id == ChlPredictionRecord.predicted_winner_id, isouter=True)
        .join(actual_team, actual_team.id == ChlPredictionRecord.actual_winner_id, isouter=True)
        .where(ChlPredictionRecord.league_id == league_id)
    )

    prediction_start, prediction_end = _prediction_datetime_bounds(prediction_date_from, prediction_date_to)
    if prediction_start:
        stmt = stmt.where(ChlPredictionRecord.prediction_date >= prediction_start)
    if prediction_end:
        stmt = stmt.where(ChlPredictionRecord.prediction_date <= prediction_end)

    if game_date_from:
        stmt = stmt.where(ChlGame.game_date >= game_date_from)
    if game_date_to:
        stmt = stmt.where(ChlGame.game_date <= game_date_to)

    if team_provider_id is not None:
        stmt = stmt.where(
            or_(
                home_team.hockeytech_id == team_provider_id,
                away_team.hockeytech_id == team_provider_id,
            )
        )

    stmt = stmt.order_by(desc(ChlPredictionRecord.prediction_date), desc(ChlPredictionRecord.id))
    return db.execute(stmt).all()


def _choose_consensus_prediction(predictions_by_k: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not predictions_by_k:
        return None

    for prediction in predictions_by_k:
        if prediction["k_value"] == 15:
            return prediction

    return max(predictions_by_k, key=lambda row: row["k_value"])


def _aggregate_predicted_games(
    rows: list[tuple[ChlPredictionRecord, ChlGame | None, ChlTeam | None, ChlTeam | None, ChlTeam | None, ChlTeam | None]],
    league_code: str,
) -> list[dict[str, Any]]:
    aggregated: dict[int, dict[str, Any]] = {}

    for record, game, home_team, away_team, predicted_team, actual_team in rows:
        key = int(record.game_id)
        item = aggregated.get(key)
        if item is None:
            home_score = game.home_goal_count if game else None
            away_score = game.away_goal_count if game else None
            item = {
                "league_code": league_code,
                "game_id": key,
                "game_date": game.game_date if game else None,
                "prediction_timestamp": record.prediction_date,
                "prediction_date": record.prediction_date.date() if record.prediction_date else None,
                "status": game.status if game else None,
                "period": game.period if game else None,
                "venue": game.venue if game else None,
                "scheduled_time_utc": game.scheduled_time_utc if game else None,
                "final_score": {"home": home_score, "away": away_score},
                "home_team": _serialize_team(home_team, league_code=league_code),
                "away_team": _serialize_team(away_team, league_code=league_code),
                "predictions_by_k": [],
                "consensus_k_value": None,
                "consensus_predicted_winner_db_team_id": None,
                "consensus_predicted_winner_provider_team_id": None,
                "consensus_predicted_winner_name": None,
                "consensus_correct": None,
                "resolved_count": 0,
                "correct_count": 0,
            }
            aggregated[key] = item

        item["predictions_by_k"].append(
            {
                "k_value": int(record.k_value),
                "home_team_probability": _to_float(record.home_team_probability),
                "away_team_probability": _to_float(record.away_team_probability),
                "predicted_winner_db_team_id": int(predicted_team.id) if predicted_team else None,
                "predicted_winner_provider_team_id": int(predicted_team.hockeytech_id) if predicted_team else None,
                "predicted_winner_name": predicted_team.name if predicted_team else None,
                "actual_winner_db_team_id": int(actual_team.id) if actual_team else None,
                "actual_winner_provider_team_id": int(actual_team.hockeytech_id) if actual_team else None,
                "actual_winner_name": actual_team.name if actual_team else None,
                "correct": record.correct,
                "model_version": record.model_version,
                "model_family": record.model_family,
                "raw_model_outputs": record.raw_model_outputs,
            }
        )

        if record.correct is not None:
            item["resolved_count"] += 1
            if record.correct:
                item["correct_count"] += 1

        if item["prediction_timestamp"] is None and record.prediction_date is not None:
            item["prediction_timestamp"] = record.prediction_date
            item["prediction_date"] = record.prediction_date.date()

    items = list(aggregated.values())

    for item in items:
        item["predictions_by_k"].sort(key=lambda row: row["k_value"])
        consensus = _choose_consensus_prediction(item["predictions_by_k"])
        if consensus:
            item["consensus_k_value"] = consensus["k_value"]
            item["consensus_predicted_winner_db_team_id"] = consensus["predicted_winner_db_team_id"]
            item["consensus_predicted_winner_provider_team_id"] = consensus["predicted_winner_provider_team_id"]
            item["consensus_predicted_winner_name"] = consensus["predicted_winner_name"]
            item["consensus_correct"] = consensus["correct"]

    return items


def _apply_result_filter(items: list[dict[str, Any]], result: str) -> list[dict[str, Any]]:
    if result not in VALID_RESULTS:
        raise ValueError(f"Unsupported result filter: {result}")

    if result == "all":
        return items

    if result == "correct":
        return [item for item in items if item["consensus_correct"] is True]

    if result == "incorrect":
        return [item for item in items if item["consensus_correct"] is False]

    return [item for item in items if item["consensus_correct"] is None]


def _sort_predicted_games(items: list[dict[str, Any]], sort: str) -> list[dict[str, Any]]:
    if sort not in VALID_SORTS:
        raise ValueError(f"Unsupported sort: {sort}")

    if sort == "prediction_date_asc":
        return sorted(
            items,
            key=lambda row: (row["prediction_timestamp"] or dt.datetime.min, row["game_id"]),
        )

    if sort == "prediction_date_desc":
        return sorted(
            items,
            key=lambda row: (row["prediction_timestamp"] or dt.datetime.min, row["game_id"]),
            reverse=True,
        )

    if sort == "game_date_asc":
        return sorted(
            items,
            key=lambda row: (row["game_date"] or dt.date.min, row["game_id"]),
        )

    return sorted(
        items,
        key=lambda row: (row["game_date"] or dt.date.min, row["game_id"]),
        reverse=True,
    )


def list_predicted_games(
    db: Session,
    league_code: str,
    *,
    prediction_date_from: dt.date | None,
    prediction_date_to: dt.date | None,
    game_date_from: dt.date | None,
    game_date_to: dt.date | None,
    team_provider_id: int | None,
    result: str,
    limit: int,
    offset: int,
    sort: str,
) -> list[dict[str, Any]]:
    normalized_league = _ensure_league(league_code)

    rows = _query_prediction_rows(
        db,
        normalized_league,
        prediction_date_from=prediction_date_from,
        prediction_date_to=prediction_date_to,
        game_date_from=game_date_from,
        game_date_to=game_date_to,
        team_provider_id=team_provider_id,
    )

    items = _aggregate_predicted_games(rows, league_code=normalized_league)
    items = _apply_result_filter(items, result=result)
    items = _sort_predicted_games(items, sort=sort)

    return items[offset : offset + limit]


def get_next_predicted_slate(
    db: Session,
    league_code: str,
    *,
    as_of_date: dt.date | None,
    team_provider_id: int | None,
) -> dict[str, Any]:
    normalized_league = _ensure_league(league_code)
    effective_as_of = as_of_date or _league_today(normalized_league)

    rows = _query_prediction_rows(
        db,
        normalized_league,
        prediction_date_from=None,
        prediction_date_to=None,
        game_date_from=effective_as_of,
        game_date_to=None,
        team_provider_id=team_provider_id,
    )
    items = _aggregate_predicted_games(rows, league_code=normalized_league)
    upcoming_items = _sort_for_upcoming([item for item in items if _is_upcoming(item, effective_as_of)])

    if not upcoming_items:
        return {
            "league_code": normalized_league,
            "as_of_date": effective_as_of,
            "target_game_date": None,
            "games_count": 0,
            "predictions_count": 0,
            "games": [],
        }

    target_game_date = _game_date_value(upcoming_items[0])
    target_games = [item for item in upcoming_items if _game_date_value(item) == target_game_date]
    predictions_count = sum(len(item.get("predictions_by_k") or []) for item in target_games)

    return {
        "league_code": normalized_league,
        "as_of_date": effective_as_of,
        "target_game_date": target_game_date,
        "games_count": len(target_games),
        "predictions_count": predictions_count,
        "games": target_games,
    }


def list_upcoming_predictions(
    db: Session,
    league_code: str,
    *,
    as_of_date: dt.date | None,
    game_date_from: dt.date | None,
    game_date_to: dt.date | None,
    team_provider_id: int | None,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    normalized_league = _ensure_league(league_code)
    effective_as_of = as_of_date or _league_today(normalized_league)
    effective_game_date_from = game_date_from or effective_as_of

    rows = _query_prediction_rows(
        db,
        normalized_league,
        prediction_date_from=None,
        prediction_date_to=None,
        game_date_from=effective_game_date_from,
        game_date_to=game_date_to,
        team_provider_id=team_provider_id,
    )
    items = _aggregate_predicted_games(rows, league_code=normalized_league)
    filtered = _sort_for_upcoming([item for item in items if _is_upcoming(item, effective_as_of)])
    return _build_prediction_page(filtered, limit=limit, offset=offset)


def list_prediction_results(
    db: Session,
    league_code: str,
    *,
    game_date_from: dt.date | None,
    game_date_to: dt.date | None,
    team_provider_id: int | None,
    result: str,
    sort: str,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    normalized_league = _ensure_league(league_code)

    if result not in VALID_RESOLVED_RESULTS:
        raise ValueError(f"Unsupported result filter: {result}")

    rows = _query_prediction_rows(
        db,
        normalized_league,
        prediction_date_from=None,
        prediction_date_to=None,
        game_date_from=game_date_from,
        game_date_to=game_date_to,
        team_provider_id=team_provider_id,
    )
    items = _aggregate_predicted_games(rows, league_code=normalized_league)
    resolved_items = [item for item in items if _is_resolved(item)]

    if result == "correct":
        resolved_items = [item for item in resolved_items if item.get("consensus_correct") is True]
    elif result == "incorrect":
        resolved_items = [item for item in resolved_items if item.get("consensus_correct") is False]

    resolved_items = _sort_predicted_games(resolved_items, sort=sort)
    return _build_prediction_page(resolved_items, limit=limit, offset=offset)


def _resolve_team_db_id_for_league(
    db: Session,
    *,
    league_id: int,
    team_provider_id: int | None,
) -> int | None:
    if team_provider_id is None:
        return None

    team = db.scalar(
        select(ChlTeam.id).where(
            ChlTeam.league_id == league_id,
            ChlTeam.hockeytech_id == team_provider_id,
        )
    )
    return int(team) if team is not None else None


def _consensus_results_base_stmt(
    *,
    league_id: int,
    team_db_id: int | None,
    result: str,
    game_date_from: dt.date | None,
    game_date_to: dt.date | None,
):
    stmt = (
        select(ChlPredictionRecord.game_id, ChlGame.game_date)
        .join(
            ChlGame,
            and_(
                ChlGame.game_id == ChlPredictionRecord.game_id,
                ChlGame.league_id == ChlPredictionRecord.league_id,
            ),
        )
        .where(ChlPredictionRecord.league_id == league_id)
        .where(ChlPredictionRecord.k_value == 15)
        .where(ChlPredictionRecord.correct.is_not(None))
        .where(ChlGame.game_date.is_not(None))
    )

    if game_date_from:
        stmt = stmt.where(ChlGame.game_date >= game_date_from)
    if game_date_to:
        stmt = stmt.where(ChlGame.game_date <= game_date_to)

    if team_db_id is not None:
        stmt = stmt.where(
            or_(
                ChlPredictionRecord.home_team_id == team_db_id,
                ChlPredictionRecord.away_team_id == team_db_id,
            )
        )

    if result == "correct":
        stmt = stmt.where(ChlPredictionRecord.correct.is_(True))
    elif result == "incorrect":
        stmt = stmt.where(ChlPredictionRecord.correct.is_(False))

    return stmt


def _count_distinct_result_days(db: Session, base_stmt) -> int:
    day_subquery = base_stmt.with_only_columns(ChlGame.game_date).distinct().subquery()
    return int(db.scalar(select(func.count()).select_from(day_subquery)) or 0)


def _fetch_result_day_keys(
    db: Session,
    base_stmt,
    *,
    days_per_page: int,
    offset: int,
) -> list[dt.date]:
    rows = (
        db.execute(
            base_stmt.with_only_columns(ChlGame.game_date)
            .distinct()
            .order_by(ChlGame.game_date.desc())
            .offset(offset)
            .limit(days_per_page)
        )
        .scalars()
        .all()
    )
    return [row for row in rows if isinstance(row, dt.date)]


def _fetch_consensus_game_ids_for_days(
    db: Session,
    base_stmt,
    *,
    day_keys: list[dt.date],
) -> list[int]:
    if not day_keys:
        return []

    rows = (
        db.execute(
            base_stmt.where(ChlGame.game_date.in_(day_keys))
            .with_only_columns(ChlPredictionRecord.game_id)
            .distinct()
        )
        .scalars()
        .all()
    )
    return [int(row) for row in rows]


def _query_prediction_rows_for_game_ids(
    db: Session,
    *,
    league_id: int,
    game_ids: list[int],
) -> list[tuple[ChlPredictionRecord, ChlGame | None, ChlTeam | None, ChlTeam | None, ChlTeam | None, ChlTeam | None]]:
    if not game_ids:
        return []

    home_team = aliased(ChlTeam)
    away_team = aliased(ChlTeam)
    predicted_team = aliased(ChlTeam)
    actual_team = aliased(ChlTeam)

    stmt = (
        select(
            ChlPredictionRecord,
            ChlGame,
            home_team,
            away_team,
            predicted_team,
            actual_team,
        )
        .join(
            ChlGame,
            and_(
                ChlGame.game_id == ChlPredictionRecord.game_id,
                ChlGame.league_id == ChlPredictionRecord.league_id,
            ),
            isouter=True,
        )
        .join(home_team, home_team.id == ChlPredictionRecord.home_team_id)
        .join(away_team, away_team.id == ChlPredictionRecord.away_team_id)
        .join(predicted_team, predicted_team.id == ChlPredictionRecord.predicted_winner_id, isouter=True)
        .join(actual_team, actual_team.id == ChlPredictionRecord.actual_winner_id, isouter=True)
        .where(ChlPredictionRecord.league_id == league_id)
        .where(ChlPredictionRecord.game_id.in_(game_ids))
        .order_by(desc(ChlPredictionRecord.prediction_date), desc(ChlPredictionRecord.id))
    )

    return db.execute(stmt).all()


def list_prediction_results_by_day(
    db: Session,
    league_code: str,
    *,
    game_date_from: dt.date | None,
    game_date_to: dt.date | None,
    team_provider_id: int | None,
    result: str,
    page: int,
    days_per_page: int,
) -> dict[str, Any]:
    normalized_league = _ensure_league(league_code)
    league_id = _league_id_for_code(normalized_league)

    if result not in VALID_RESOLVED_RESULTS:
        raise ValueError(f"Unsupported result filter: {result}")

    if page < 1:
        raise ValueError("page must be >= 1")
    if days_per_page < 1:
        raise ValueError("days_per_page must be >= 1")

    team_db_id = _resolve_team_db_id_for_league(
        db,
        league_id=league_id,
        team_provider_id=team_provider_id,
    )
    if team_provider_id is not None and team_db_id is None:
        return {
            "league_code": normalized_league,
            "items": [],
            "page": page,
            "days_per_page": days_per_page,
            "total_days": 0,
            "total_pages": 0,
            "has_more": False,
            "has_prev": False,
        }

    base_stmt = _consensus_results_base_stmt(
        league_id=league_id,
        team_db_id=team_db_id,
        result=result,
        game_date_from=game_date_from,
        game_date_to=game_date_to,
    )

    total_days = _count_distinct_result_days(db, base_stmt)
    if total_days == 0:
        return {
            "league_code": normalized_league,
            "items": [],
            "page": page,
            "days_per_page": days_per_page,
            "total_days": 0,
            "total_pages": 0,
            "has_more": False,
            "has_prev": False,
        }

    total_pages = math.ceil(total_days / days_per_page)
    offset = (page - 1) * days_per_page
    day_keys = _fetch_result_day_keys(
        db,
        base_stmt,
        days_per_page=days_per_page,
        offset=offset,
    )

    if not day_keys:
        return {
            "league_code": normalized_league,
            "items": [],
            "page": page,
            "days_per_page": days_per_page,
            "total_days": total_days,
            "total_pages": total_pages,
            "has_more": False,
            "has_prev": page > 1,
        }

    game_ids = _fetch_consensus_game_ids_for_days(db, base_stmt, day_keys=day_keys)
    rows = _query_prediction_rows_for_game_ids(
        db,
        league_id=league_id,
        game_ids=game_ids,
    )
    games = _aggregate_predicted_games(rows, league_code=normalized_league)

    games_by_day: dict[dt.date, list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        game_date = _game_date_value(game)
        if game_date is None:
            continue
        games_by_day[game_date].append(game)

    items: list[dict[str, Any]] = []
    for game_date in day_keys:
        day_games = sorted(
            games_by_day.get(game_date, []),
            key=lambda row: (_scheduled_time_value(row), int(row["game_id"])),
        )
        resolved_games = [game for game in day_games if game.get("consensus_correct") is not None]
        resolved_count = len(resolved_games)
        correct_count = sum(1 for game in resolved_games if game.get("consensus_correct") is True)
        accuracy_pct = round((correct_count / resolved_count) * 100.0, 2) if resolved_count > 0 else None

        items.append(
            {
                "game_date": game_date,
                "games_count": len(day_games),
                "resolved_count": resolved_count,
                "correct_count": correct_count,
                "accuracy_pct": accuracy_pct,
                "games": day_games,
            }
        )

    return {
        "league_code": normalized_league,
        "items": items,
        "page": page,
        "days_per_page": days_per_page,
        "total_days": total_days,
        "total_pages": total_pages,
        "has_more": page < total_pages,
        "has_prev": page > 1,
    }


def list_recent_result_days(
    db: Session,
    league_code: str,
    *,
    days: int,
    team_provider_id: int | None,
    as_of_date: dt.date | None,
) -> dict[str, Any]:
    normalized_league = _ensure_league(league_code)
    effective_as_of = as_of_date or _league_today(normalized_league)

    rows = _query_prediction_rows(
        db,
        normalized_league,
        prediction_date_from=None,
        prediction_date_to=None,
        game_date_from=None,
        game_date_to=effective_as_of,
        team_provider_id=team_provider_id,
    )
    items = _aggregate_predicted_games(rows, league_code=normalized_league)

    grouped: dict[dt.date, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        game_date = _game_date_value(item)
        if game_date is None or game_date > effective_as_of:
            continue
        if not _is_resolved(item):
            continue
        grouped[game_date].append(item)

    day_rows: list[dict[str, Any]] = []
    for game_date in sorted(grouped.keys(), reverse=True)[:days]:
        games = sorted(grouped[game_date], key=lambda row: (_scheduled_time_value(row), int(row["game_id"])))
        resolved_games = [game for game in games if game.get("consensus_correct") is not None]
        resolved_count = len(resolved_games)
        correct_count = sum(1 for game in resolved_games if game.get("consensus_correct") is True)
        accuracy_pct = round((correct_count / resolved_count) * 100.0, 2) if resolved_count > 0 else None
        day_rows.append(
            {
                "game_date": game_date,
                "games_count": len(games),
                "resolved_count": resolved_count,
                "correct_count": correct_count,
                "accuracy_pct": accuracy_pct,
                "games": games,
            }
        )

    return {
        "league_code": normalized_league,
        "as_of_date": effective_as_of,
        "days": day_rows,
    }


def _build_daily_summary(prediction_date: dt.date, games: list[dict[str, Any]]) -> dict[str, Any]:
    predictions_count = sum(len(game["predictions_by_k"]) for game in games)
    resolved_count = sum(game["resolved_count"] for game in games)
    correct_count = sum(game["correct_count"] for game in games)

    accuracy_pct = None
    if resolved_count > 0:
        accuracy_pct = round((correct_count / resolved_count) * 100.0, 2)

    return {
        "prediction_date": prediction_date,
        "games_count": len(games),
        "predictions_count": predictions_count,
        "resolved_count": resolved_count,
        "correct_count": correct_count,
        "accuracy_pct": accuracy_pct,
    }


def list_daily_reports(
    db: Session,
    league_code: str,
    *,
    date_from: dt.date | None,
    date_to: dt.date | None,
    team_provider_id: int | None,
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    normalized_league = _ensure_league(league_code)

    rows = _query_prediction_rows(
        db,
        normalized_league,
        prediction_date_from=date_from,
        prediction_date_to=date_to,
        game_date_from=None,
        game_date_to=None,
        team_provider_id=team_provider_id,
    )

    predicted_games = _aggregate_predicted_games(rows, league_code=normalized_league)

    grouped: dict[dt.date, list[dict[str, Any]]] = defaultdict(list)
    for game in predicted_games:
        prediction_date = game["prediction_date"]
        if prediction_date is None:
            continue
        grouped[prediction_date].append(game)

    summaries = [
        _build_daily_summary(prediction_date=prediction_date, games=games)
        for prediction_date, games in grouped.items()
    ]
    summaries.sort(key=lambda item: item["prediction_date"], reverse=True)
    return summaries[offset : offset + limit]


def get_daily_report_detail(
    db: Session,
    league_code: str,
    prediction_date: dt.date,
    *,
    team_provider_id: int | None,
) -> dict[str, Any]:
    normalized_league = _ensure_league(league_code)

    games = list_predicted_games(
        db,
        normalized_league,
        prediction_date_from=prediction_date,
        prediction_date_to=prediction_date,
        game_date_from=None,
        game_date_to=None,
        team_provider_id=team_provider_id,
        result="all",
        limit=10_000,
        offset=0,
        sort="game_date_desc",
    )

    if not games:
        raise DailyReportNotFoundError(f"No prediction report found for {prediction_date.isoformat()}")

    summary = _build_daily_summary(prediction_date=prediction_date, games=games)
    return {
        "league_code": normalized_league,
        "prediction_date": prediction_date,
        "summary": summary,
        "games": games,
    }


def _scorebar_links(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    data = snapshot or {}
    return {
        "ticket_url": data.get("TicketUrl"),
        "game_summary_url": data.get("GameSummaryUrl"),
        "home_audio_url": data.get("HomeAudioUrl"),
        "away_audio_url": data.get("VisitorAudioUrl"),
        "home_video_url": data.get("HomeVideoUrl"),
        "away_video_url": data.get("VisitorVideoUrl"),
        "home_webcast_url": data.get("HomeWebcastUrl"),
        "away_webcast_url": data.get("VisitorWebcastUrl"),
    }


def get_game_detail(db: Session, league_code: str, game_id: int) -> dict[str, Any]:
    normalized_league = _ensure_league(league_code)
    league_id = _league_id_for_code(normalized_league)

    game = db.scalar(select(ChlGame).where(ChlGame.league_id == league_id, ChlGame.game_id == game_id))
    if game is None:
        raise GameNotFoundError(f"Game not found: {game_id}")

    home_team = None
    away_team = None
    if game.home_team_id is not None:
        home_team = db.scalar(
            select(ChlTeam).where(ChlTeam.league_id == league_id, ChlTeam.hockeytech_id == game.home_team_id)
        )
    if game.away_team_id is not None:
        away_team = db.scalar(
            select(ChlTeam).where(ChlTeam.league_id == league_id, ChlTeam.hockeytech_id == game.away_team_id)
        )

    predicted_team = aliased(ChlTeam)
    actual_team = aliased(ChlTeam)

    prediction_rows = db.execute(
        select(ChlPredictionRecord, predicted_team, actual_team)
        .join(predicted_team, predicted_team.id == ChlPredictionRecord.predicted_winner_id, isouter=True)
        .join(actual_team, actual_team.id == ChlPredictionRecord.actual_winner_id, isouter=True)
        .where(ChlPredictionRecord.league_id == league_id, ChlPredictionRecord.game_id == game_id)
        .order_by(asc(ChlPredictionRecord.k_value))
    ).all()

    predictions_by_k: list[dict[str, Any]] = []
    for record, predicted, actual in prediction_rows:
        predictions_by_k.append(
            {
                "k_value": int(record.k_value),
                "home_team_probability": _to_float(record.home_team_probability),
                "away_team_probability": _to_float(record.away_team_probability),
                "predicted_winner_db_team_id": int(predicted.id) if predicted else None,
                "predicted_winner_provider_team_id": int(predicted.hockeytech_id) if predicted else None,
                "predicted_winner_name": predicted.name if predicted else None,
                "actual_winner_db_team_id": int(actual.id) if actual else None,
                "actual_winner_provider_team_id": int(actual.hockeytech_id) if actual else None,
                "actual_winner_name": actual.name if actual else None,
                "correct": record.correct,
                "model_version": record.model_version,
                "model_family": record.model_family,
                "raw_model_outputs": record.raw_model_outputs,
            }
        )

    consensus = _choose_consensus_prediction(predictions_by_k)

    return {
        "league_code": normalized_league,
        "game_id": int(game.game_id),
        "game_date": game.game_date,
        "status": game.status,
        "venue": game.venue,
        "scheduled_time_utc": game.scheduled_time_utc,
        "period": game.period,
        "final_score": {
            "home": game.home_goal_count,
            "away": game.away_goal_count,
        },
        "home_team": _serialize_team(
            home_team,
            league_code=normalized_league,
            fallback_provider_id=game.home_team_id,
            fallback_name=game.home_team,
        ),
        "away_team": _serialize_team(
            away_team,
            league_code=normalized_league,
            fallback_provider_id=game.away_team_id,
            fallback_name=game.away_team,
        ),
        "scoring_breakdown": game.scoring_breakdown,
        "shots_on_goal": game.shots_on_goal,
        "power_play": game.power_play,
        "fow": game.fow,
        "links": _scorebar_links(game.scorebar_snapshot),
        "predictions_by_k": predictions_by_k,
        "consensus_k_value": consensus["k_value"] if consensus else None,
        "consensus_predicted_winner_db_team_id": consensus["predicted_winner_db_team_id"] if consensus else None,
        "consensus_predicted_winner_provider_team_id": consensus["predicted_winner_provider_team_id"] if consensus else None,
        "consensus_predicted_winner_name": consensus["predicted_winner_name"] if consensus else None,
        "consensus_correct": consensus["correct"] if consensus else None,
    }
