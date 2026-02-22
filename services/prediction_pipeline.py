from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from sqlalchemy import and_, desc, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from services.data_backend import (
    DataBackendError,
    HockeyStoreModels,
    apply_league_scope,
    build_store_chain,
    game_conflict_columns,
    hockeytech_api_key_for_league,
    hockeytech_client_code_for_league,
    prediction_conflict_columns,
    primary_store,
    require_supported_league_code,
    resolve_league_id_for_store,
    rolling_conflict_columns,
)
from services.feature_builder import (
    InsufficientHistoryError,
    TeamNotFoundError,
    build_features_by_k,
)
from services.hockeytech_client import HockeyTechClient
from services.predictor import predict_from_payload

K_VALUES = [5, 10, 15]
FINAL_STATUS_TOKENS = {"4", "final", "completed", "game over", "end"}


@dataclass
class UpcomingRunStats:
    target_date: dt.date
    predictions_written: int
    skipped_games: int


def prediction_date_bounds(
    date_from: dt.date | None,
    date_to: dt.date | None,
) -> tuple[dt.datetime | None, dt.datetime | None]:
    start = dt.datetime.combine(date_from, dt.time.min) if date_from else None
    end = dt.datetime.combine(date_to, dt.time.max) if date_to else None
    return start, end


def _parse_iso_datetime(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        return None


def _parse_iso_date(value: str | None) -> dt.date | None:
    parsed = _parse_iso_datetime(value)
    return parsed.date() if parsed else None


def _first_present(payload: dict[str, Any], keys: list[str]) -> Any | None:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalized_status_token(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _status_is_final(value: Any) -> bool:
    token = _normalized_status_token(value)
    if token in FINAL_STATUS_TOKENS:
        return True
    return "final" in token or "completed" in token


def _is_zero_like(value: Any) -> bool:
    if value in (None, ""):
        return True
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return float(value) == 0.0
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return True
        try:
            return float(stripped) == 0.0
        except ValueError:
            return False
    if isinstance(value, dict):
        if not value:
            return True
        return all(_is_zero_like(v) for v in value.values())
    if isinstance(value, list):
        if not value:
            return True
        return all(_is_zero_like(v) for v in value)
    return False


def _is_non_final_placeholder_bundle(values: dict[str, Any]) -> bool:
    if _status_is_final(values.get("status")):
        return False

    goals_zero_or_missing = (values.get("home_goal_count") in (None, 0)) and (values.get("away_goal_count") in (None, 0))
    if not goals_zero_or_missing:
        return False

    return all(
        [
            _is_zero_like(values.get("scoring_breakdown")),
            _is_zero_like(values.get("shots_on_goal")),
            _is_zero_like(values.get("power_play")),
            _is_zero_like(values.get("fow")),
            _is_zero_like(values.get("home_shots_on_goal_total")),
            _is_zero_like(values.get("away_shots_on_goal_total")),
        ]
    )


def _clear_placeholder_stat_fields(values: dict[str, Any]) -> None:
    for key in [
        "home_goal_count",
        "away_goal_count",
        "scoring_breakdown",
        "shots_on_goal",
        "power_play",
        "fow",
        "home_power_play_percentage",
        "away_power_play_percentage",
        "home_faceoff_win_percentage",
        "away_faceoff_win_percentage",
        "home_shots_on_goal_total",
        "away_shots_on_goal_total",
    ]:
        values[key] = None


def _extract_schedule_game_id(game_data: dict[str, Any]) -> int | None:
    raw = _first_present(game_data, ["ID", "id", "game_id"])
    return _to_int(raw)


def build_schedule_game_upsert_values(
    game_data: dict[str, Any],
    now: dt.datetime,
    season_name_fallback: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    game_id = _extract_schedule_game_id(game_data)
    if game_id is None:
        raise DataBackendError(f"Missing game id in schedule payload: keys={sorted(game_data.keys())}")

    season_id = _first_present(game_data, ["SeasonID", "season_id"])
    season_name = _first_present(game_data, ["SeasonName", "season_name"])
    if season_name in (None, "") and season_name_fallback not in (None, ""):
        season_name = season_name_fallback

    game_iso = _first_present(
        game_data,
        ["GameDateISO8601", "game_date_iso_8601", "date_time_played", "scheduled_time"],
    )
    date_played = _first_present(game_data, ["date_played", "GameDate"])
    game_date = _parse_iso_date(str(game_iso)) if game_iso is not None else None
    if game_date is None and date_played is not None:
        try:
            game_date = dt.date.fromisoformat(str(date_played))
        except ValueError:
            game_date = None

    home_team_name = _first_present(game_data, ["HomeLongName", "home_team_name", "home_team"])
    away_team_name = _first_present(game_data, ["VisitorLongName", "visiting_team_name", "visiting_team"])

    values = {
        "game_id": game_id,
        "season_id": str(season_id) if season_id is not None else None,
        "season_name": str(season_name) if season_name is not None else None,
        "game_date": game_date,
        "venue": _first_present(game_data, ["venue_name", "venue", "location"]),
        "status": _first_present(game_data, ["game_status", "GameStatusString", "progress", "status"]),
        "home_goal_count": _to_int(_first_present(game_data, ["home_goal_count", "HomeGoals"])),
        "away_goal_count": _to_int(_first_present(game_data, ["visiting_goal_count", "VisitorGoals", "away_goal_count"])),
        "game_number": _to_int(_first_present(game_data, ["game_number", "GameNumber"])),
        "period": _first_present(game_data, ["period", "period_trans", "Period"]),
        "home_team": str(home_team_name) if home_team_name is not None else None,
        "away_team": str(away_team_name) if away_team_name is not None else None,
        "home_team_id": _to_int(_first_present(game_data, ["HomeID", "home_team", "home_team_id"])),
        "away_team_id": _to_int(_first_present(game_data, ["VisitorID", "visiting_team", "away_team_id"])),
        "scorebar_snapshot": game_data,
        "scheduled_time_utc": _parse_iso_datetime(str(game_iso)) if game_iso is not None else None,
        "created_at": now,
        "updated_at": now,
    }

    # Prevent scheduled-game placeholder zero scores from being persisted as real outcomes.
    if not _status_is_final(values.get("status")) and values.get("home_goal_count") in (None, 0) and values.get("away_goal_count") in (None, 0):
        values["home_goal_count"] = None
        values["away_goal_count"] = None

    update_values = {**values}
    update_values.pop("created_at", None)
    return values, update_values


def _clock_game_upsert_values(
    game_data: dict[str, Any],
    clock: dict[str, Any],
    now: dt.datetime,
) -> tuple[dict[str, Any], dict[str, Any]]:
    schedule_values, _ = build_schedule_game_upsert_values(game_data=game_data, now=now)

    power_play = clock.get("power_play") or {}
    power_play_total = power_play.get("total") or {}
    power_play_goals = power_play.get("goals") or {}

    home_pp_attempts = _to_float(power_play_total.get("home"), default=0.0)
    away_pp_attempts = _to_float(power_play_total.get("visiting"), default=0.0)
    home_pp_goals = _to_float(power_play_goals.get("home"), default=0.0)
    away_pp_goals = _to_float(power_play_goals.get("visiting"), default=0.0)

    home_pp = (home_pp_goals / home_pp_attempts) if home_pp_attempts > 0 else 0.0
    away_pp = (away_pp_goals / away_pp_attempts) if away_pp_attempts > 0 else 0.0

    fow = clock.get("fow") or {}
    home_fow = _to_float(fow.get("home"), default=0.0)
    away_fow = _to_float(fow.get("visiting"), default=0.0)
    fow_total = home_fow + away_fow
    home_fowp = (home_fow / fow_total) if fow_total > 0 else 0.5
    away_fowp = (away_fow / fow_total) if fow_total > 0 else 0.5

    shots_on_goal = clock.get("shots_on_goal") or {}
    home_sog = sum(_to_int(v) or 0 for v in (shots_on_goal.get("home") or {}).values())
    away_sog = sum(_to_int(v) or 0 for v in (shots_on_goal.get("visiting") or {}).values())

    home_team = clock.get("home_team") or {}
    away_team = clock.get("visiting_team") or {}

    values = {
        "game_id": schedule_values["game_id"],
        "season_id": str(_first_present(clock, ["season_id"]) or schedule_values["season_id"]) if (
            _first_present(clock, ["season_id"]) or schedule_values["season_id"]
        ) is not None else None,
        "season_name": str(_first_present(clock, ["season_name"]) or schedule_values["season_name"]) if (
            _first_present(clock, ["season_name"]) or schedule_values["season_name"]
        ) is not None else None,
        "game_date": _parse_iso_date(str(_first_present(clock, ["game_date_iso_8601"])))
        or schedule_values["game_date"],
        "venue": _first_present(clock, ["venue"]) or schedule_values["venue"],
        "status": _first_present(clock, ["progress", "status"]) or schedule_values["status"],
        "home_team_id": _to_int(_first_present(home_team, ["team_id"])) or schedule_values["home_team_id"],
        "away_team_id": _to_int(_first_present(away_team, ["team_id"])) or schedule_values["away_team_id"],
        "home_team": _first_present(home_team, ["name"]) or schedule_values["home_team"],
        "away_team": _first_present(away_team, ["name"]) or schedule_values["away_team"],
        "home_goal_count": _to_int(_first_present(clock, ["home_goal_count"])),
        "away_goal_count": _to_int(_first_present(clock, ["visiting_goal_count"])),
        "game_number": _to_int(_first_present(clock, ["game_number"])),
        "period": _first_present(clock, ["period"]),
        "scoring_breakdown": _first_present(clock, ["scoring"]),
        "shots_on_goal": _first_present(clock, ["shots_on_goal"]),
        "power_play": power_play or None,
        "fow": fow or None,
        "home_power_play_percentage": home_pp,
        "away_power_play_percentage": away_pp,
        "home_faceoff_win_percentage": home_fowp,
        "away_faceoff_win_percentage": away_fowp,
        "home_shots_on_goal_total": home_sog,
        "away_shots_on_goal_total": away_sog,
        "scorebar_snapshot": game_data,
        "scheduled_time_utc": schedule_values["scheduled_time_utc"],
        "created_at": now,
        "updated_at": now,
    }

    if _is_non_final_placeholder_bundle(values):
        _clear_placeholder_stat_fields(values)

    update_values = {**values}
    update_values.pop("created_at", None)
    return values, update_values


def build_clock_game_upsert_values(
    game_data: dict[str, Any],
    clock: dict[str, Any],
    now: dt.datetime,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _clock_game_upsert_values(game_data=game_data, clock=clock, now=now)


def _resolve_write_stores(db: Session, league_code: str | None) -> tuple[str, list[tuple[HockeyStoreModels, int | None]]]:
    normalized = require_supported_league_code(league_code)

    stores: list[tuple[HockeyStoreModels, int | None]] = []
    for store in build_store_chain():
        league_id = resolve_league_id_for_store(db, store, normalized)
        stores.append((store, league_id))

    if not stores:
        raise DataBackendError(f"No writable stores configured for league_code={normalized}")

    return normalized, stores


def _resolve_read_store(db: Session, league_code: str | None) -> tuple[str, HockeyStoreModels, int | None]:
    normalized = require_supported_league_code(league_code)
    store = primary_store()
    league_id = resolve_league_id_for_store(db, store, normalized)
    return normalized, store, league_id


def _scoped_stmt(stmt, store: HockeyStoreModels, league_id: int | None):
    return apply_league_scope(stmt, store.game_model if hasattr(store, "game_model") else store.team_model, league_id)


def _upsert_game_row(
    db: Session,
    *,
    store: HockeyStoreModels,
    league_id: int | None,
    values: dict[str, Any],
    update_values: dict[str, Any],
) -> None:
    if store.league_scoped:
        values = {**values, "league_id": league_id}
        update_values = {**update_values, "league_id": league_id}

    db.execute(
        insert(store.game_model)
        .values(**values)
        .on_conflict_do_update(
            index_elements=game_conflict_columns(store),
            set_=update_values,
        )
    )


def upsert_upcoming_schedule(
    db: Session,
    target_date: dt.date,
    league_code: str | None = None,
) -> int:
    normalized_league, stores = _resolve_write_stores(db, league_code)
    client = HockeyTechClient(
        client_code=hockeytech_client_code_for_league(normalized_league),
        api_key=hockeytech_api_key_for_league(normalized_league),
    )
    games = client.get_schedule_for_date(target_date)

    season_name_by_id: dict[str, str] = {}
    try:
        season_rows = client.get_seasons()
        season_name_by_id = {
            str(row.get("season_id")): str(row.get("season_name"))
            for row in season_rows
            if row.get("season_id") not in (None, "") and row.get("season_name") not in (None, "")
        }
    except Exception:  # noqa: BLE001
        season_name_by_id = {}

    rows_written = 0
    now = dt.datetime.now(dt.UTC)
    for game_data in games:
        season_id_value = _first_present(game_data, ["SeasonID", "season_id"])
        season_name_fallback = season_name_by_id.get(str(season_id_value)) if season_id_value not in (None, "") else None
        values, update_values = build_schedule_game_upsert_values(
            game_data=game_data,
            now=now,
            season_name_fallback=season_name_fallback,
        )

        for store, league_id in stores:
            _upsert_game_row(db, store=store, league_id=league_id, values=values, update_values=update_values)

        rows_written += 1

    db.commit()
    return rows_written


def update_completed_games(
    db: Session,
    target_date: dt.date,
    league_code: str | None = None,
) -> int:
    normalized_league, stores = _resolve_write_stores(db, league_code)
    client = HockeyTechClient(
        client_code=hockeytech_client_code_for_league(normalized_league),
        api_key=hockeytech_api_key_for_league(normalized_league),
    )
    games = client.get_scorebar(number_of_days_ahead=0, number_of_days_back=1)

    updated = 0
    now = dt.datetime.now(dt.UTC)
    for game_data in games:
        game_date = _parse_iso_date(game_data.get("GameDateISO8601"))
        if game_date != target_date:
            continue

        game_id_raw = game_data.get("ID")
        if not game_id_raw:
            continue

        clock = client.get_clock(int(game_id_raw))
        values, update_values = build_clock_game_upsert_values(game_data=game_data, clock=clock, now=now)

        for store, league_id in stores:
            _upsert_game_row(db, store=store, league_id=league_id, values=values, update_values=update_values)
        updated += 1

    db.commit()
    _update_prediction_record_correctness(db, target_date, league_code=normalized_league)
    return updated


def _update_prediction_record_correctness(
    db: Session,
    target_date: dt.date,
    league_code: str | None = None,
) -> None:
    normalized, stores = _resolve_write_stores(db, league_code)
    del normalized

    for store, league_id in stores:
        game_model = store.game_model
        team_model = store.team_model
        prediction_model = store.prediction_model

        games_stmt = apply_league_scope(select(game_model).where(game_model.game_date == target_date), game_model, league_id)
        games = db.scalars(games_stmt).all()

        for game in games:
            if game.home_goal_count is None or game.away_goal_count is None:
                continue

            home_team_stmt = select(team_model).where(team_model.hockeytech_id == game.home_team_id)
            away_team_stmt = select(team_model).where(team_model.hockeytech_id == game.away_team_id)
            home_team = db.scalar(apply_league_scope(home_team_stmt, team_model, league_id))
            away_team = db.scalar(apply_league_scope(away_team_stmt, team_model, league_id))
            if not home_team or not away_team:
                continue

            actual_winner_id: int | None
            if game.home_goal_count > game.away_goal_count:
                actual_winner_id = home_team.id
            elif game.home_goal_count < game.away_goal_count:
                actual_winner_id = away_team.id
            else:
                actual_winner_id = None

            records_stmt = apply_league_scope(
                select(prediction_model).where(prediction_model.game_id == game.game_id),
                prediction_model,
                league_id,
            )
            records = db.scalars(records_stmt).all()
            for record in records:
                predicted_winner_id = record.home_team_id if (record.home_team_probability or 0) >= (record.away_team_probability or 0) else record.away_team_id
                record.predicted_winner_id = predicted_winner_id
                record.actual_winner_id = actual_winner_id
                record.correct = (predicted_winner_id == actual_winner_id) if actual_winner_id is not None else None

    db.commit()


def recompute_rolling_averages(
    db: Session,
    k_values: list[int] | None = None,
    league_code: str | None = None,
) -> int:
    normalized_league, stores = _resolve_write_stores(db, league_code)
    del normalized_league

    k_targets = k_values or K_VALUES
    rows_written = 0
    now = dt.datetime.now(dt.UTC)

    for store, league_id in stores:
        team_model = store.team_model
        game_model = store.game_model
        rolling_model = store.rolling_model

        teams_stmt = apply_league_scope(select(team_model), team_model, league_id)
        teams = db.scalars(teams_stmt).all()

        for team in teams:
            games_stmt = (
                select(game_model)
                .where(or_(game_model.home_team_id == team.hockeytech_id, game_model.away_team_id == team.hockeytech_id))
                .where(game_model.game_date.is_not(None))
                .order_by(game_model.game_date, game_model.game_id)
            )
            games_stmt = apply_league_scope(games_stmt, game_model, league_id)
            games = db.scalars(games_stmt).all()

            for idx, game in enumerate(games):
                for k in k_targets:
                    if idx < k:
                        continue

                    window = games[idx - k : idx]

                    target_goals: list[float] = []
                    target_ppp: list[float] = []
                    target_sog: list[float] = []
                    target_fowp: list[float] = []

                    opp_goals: list[float] = []
                    opp_ppp: list[float] = []
                    opp_sog: list[float] = []
                    opp_fowp: list[float] = []

                    for g in window:
                        is_home = g.home_team_id == team.hockeytech_id
                        target_goals.append(_to_float(g.home_goal_count if is_home else g.away_goal_count, default=0.0))
                        opp_goals.append(_to_float(g.away_goal_count if is_home else g.home_goal_count, default=0.0))

                        target_ppp.append(_to_float(g.home_power_play_percentage if is_home else g.away_power_play_percentage, default=0.0))
                        opp_ppp.append(_to_float(g.away_power_play_percentage if is_home else g.home_power_play_percentage, default=0.0))

                        target_sog.append(_to_float(g.home_shots_on_goal_total if is_home else g.away_shots_on_goal_total, default=0.0))
                        opp_sog.append(_to_float(g.away_shots_on_goal_total if is_home else g.home_shots_on_goal_total, default=0.0))

                        target_fowp.append(_to_float(g.home_faceoff_win_percentage if is_home else g.away_faceoff_win_percentage, default=0.0))
                        opp_fowp.append(_to_float(g.away_faceoff_win_percentage if is_home else g.home_faceoff_win_percentage, default=0.0))

                    def avg(values: list[float]) -> float:
                        return sum(values) / len(values) if values else 0.0

                    is_home = game.home_team_id == team.hockeytech_id
                    team_goals = game.home_goal_count if is_home else game.away_goal_count
                    opp_goals_now = game.away_goal_count if is_home else game.home_goal_count

                    target_win = None
                    if team_goals is not None and opp_goals_now is not None:
                        target_win = 1 if team_goals > opp_goals_now else 0

                    row_data = {
                        "game_id": game.game_id,
                        store.rolling_team_id_field: team.id,
                        "k_value": k,
                        "goals_for_avg": avg(target_goals),
                        "goals_against_avg": avg(opp_goals),
                        "shots_for_avg": avg(target_sog),
                        "shots_against_avg": avg(opp_sog),
                        "power_play_percentage_avg": avg(target_ppp),
                        "power_play_percentage_against_avg": avg(opp_ppp),
                        "faceoff_win_percentage_avg": avg(target_fowp),
                        "faceoff_win_percentage_against_avg": avg(opp_fowp),
                        "home_away": 1 if is_home else 0,
                        "goals_diff": avg(target_goals) - avg(opp_goals),
                        "ppp_diff": avg(target_ppp) - avg(opp_ppp),
                        "sog_diff": avg(target_sog) - avg(opp_sog),
                        "fowp_diff": avg(target_fowp) - avg(opp_fowp),
                        "target_win": target_win,
                        "created_at": now,
                        "updated_at": now,
                    }
                    if store.league_scoped:
                        row_data["league_id"] = league_id

                    db.execute(
                        insert(rolling_model)
                        .values(**row_data)
                        .on_conflict_do_update(
                            index_elements=rolling_conflict_columns(store),
                            set_={
                                **{k2: row_data[k2] for k2 in row_data if k2 != "created_at"},
                                "updated_at": now,
                            },
                        )
                    )
                    rows_written += 1

    db.commit()
    return rows_written


def _resolve_write_entities_for_store(
    db: Session,
    *,
    store: HockeyStoreModels,
    league_id: int | None,
    game: Any,
    home_team: Any,
    away_team: Any,
) -> tuple[Any | None, Any | None, Any | None]:
    if store.name == primary_store().name:
        return game, home_team, away_team

    game_stmt = apply_league_scope(
        select(store.game_model).where(store.game_model.game_id == game.game_id),
        store.game_model,
        league_id,
    )
    home_stmt = apply_league_scope(
        select(store.team_model).where(store.team_model.hockeytech_id == home_team.hockeytech_id),
        store.team_model,
        league_id,
    )
    away_stmt = apply_league_scope(
        select(store.team_model).where(store.team_model.hockeytech_id == away_team.hockeytech_id),
        store.team_model,
        league_id,
    )

    return db.scalar(game_stmt), db.scalar(home_stmt), db.scalar(away_stmt)


def _persist_prediction_rows(
    db: Session,
    game: Any,
    home_team: Any,
    away_team: Any,
    result: dict[str, Any],
    extra_raw_model_outputs_by_k: dict[int, dict[str, Any]] | None = None,
    commit: bool = True,
    league_code: str | None = None,
) -> int:
    normalized_league, stores = _resolve_write_stores(db, league_code)
    del normalized_league

    ensemble_home = float(result["home_team_prob"])
    ensemble_away = float(result["away_team_prob"])
    predicted_ht_id = int(result["predicted_winner_id"]) if result.get("predicted_winner_id") is not None else None

    now = dt.datetime.now(dt.UTC)
    primary_written = 0

    for store, league_id in stores:
        prediction_model = store.prediction_model
        game_write, home_team_write, away_team_write = _resolve_write_entities_for_store(
            db,
            store=store,
            league_id=league_id,
            game=game,
            home_team=home_team,
            away_team=away_team,
        )
        if game_write is None or home_team_write is None or away_team_write is None:
            continue

        predicted_winner_db_id = None
        if predicted_ht_id == home_team_write.hockeytech_id:
            predicted_winner_db_id = home_team_write.id
        elif predicted_ht_id == away_team_write.hockeytech_id:
            predicted_winner_db_id = away_team_write.id

        actual_winner_db_id = None
        if game_write.home_goal_count is not None and game_write.away_goal_count is not None:
            if game_write.home_goal_count > game_write.away_goal_count:
                actual_winner_db_id = home_team_write.id
            elif game_write.home_goal_count < game_write.away_goal_count:
                actual_winner_db_id = away_team_write.id

        for k in K_VALUES:
            comp = (result.get("k_components") or {}).get(str(k), {})
            home_prob = float(comp.get("home_team_prob", ensemble_home))
            away_prob = float(comp.get("away_team_prob", ensemble_away))

            winner_id = predicted_winner_db_id
            if winner_id is None:
                winner_id = home_team_write.id if home_prob >= away_prob else away_team_write.id

            raw_outputs = {
                "ensemble": {
                    "home_team_prob": ensemble_home,
                    "away_team_prob": ensemble_away,
                },
                "k_components": result.get("k_components") or {},
            }
            extra_raw = (extra_raw_model_outputs_by_k or {}).get(k)
            if extra_raw:
                raw_outputs.update(extra_raw)

            record_data = {
                "game_id": game_write.game_id,
                "k_value": k,
                "home_team_id": home_team_write.id,
                "away_team_id": away_team_write.id,
                "home_team_probability": home_prob,
                "away_team_probability": away_prob,
                "predicted_winner_id": winner_id,
                "actual_winner_id": actual_winner_db_id,
                "correct": (winner_id == actual_winner_db_id) if actual_winner_db_id is not None else None,
                "prediction_date": now,
                "model_version": result.get("model_version"),
                "model_family": result.get("model_family"),
                "raw_model_outputs": raw_outputs,
                "created_at": now,
                "updated_at": now,
            }
            if store.league_scoped:
                record_data["league_id"] = league_id

            db.execute(
                insert(prediction_model)
                .values(**record_data)
                .on_conflict_do_update(
                    index_elements=prediction_conflict_columns(store),
                    set_={**record_data, "updated_at": now},
                )
            )

            if store.name == primary_store().name:
                primary_written += 1

    if commit:
        db.commit()
    return primary_written


def get_upcoming_games(db: Session, target_date: dt.date, league_code: str | None = None) -> list[Any]:
    _, store, league_id = _resolve_read_store(db, league_code)
    stmt = select(store.game_model).where(store.game_model.game_date == target_date).order_by(store.game_model.game_id)
    stmt = apply_league_scope(stmt, store.game_model, league_id)
    return db.scalars(stmt).all()


def run_upcoming_predictions(
    db: Session,
    target_date: dt.date,
    league_code: str | None = None,
) -> UpcomingRunStats:
    normalized = require_supported_league_code(league_code)
    upsert_upcoming_schedule(db, target_date, league_code=normalized)
    games = get_upcoming_games(db, target_date, league_code=normalized)

    written = 0
    skipped = 0

    for game in games:
        if game.home_team_id is None or game.away_team_id is None or game.game_date is None:
            skipped += 1
            continue

        try:
            built = build_features_by_k(
                db=db,
                home_team_hockeytech_id=game.home_team_id,
                away_team_hockeytech_id=game.away_team_id,
                game_date=game.game_date,
                league_code=normalized,
            )
        except (InsufficientHistoryError, TeamNotFoundError):
            skipped += 1
            continue

        payload = {
            "game_id": game.game_id,
            "game_date": game.game_date.isoformat(),
            "home_team_id": game.home_team_id,
            "away_team_id": game.away_team_id,
            "features_by_k": built["features_by_k"],
            "context_features": built.get("context_features") or {},
        }

        result = predict_from_payload(payload, league_code=normalized)
        written += _persist_prediction_rows(
            db,
            game,
            built["home_team"],
            built["away_team"],
            result,
            league_code=normalized,
        )

    return UpcomingRunStats(target_date=target_date, predictions_written=written, skipped_games=skipped)


def run_custom_prediction(
    db: Session,
    home_team_hockeytech_id: int,
    away_team_hockeytech_id: int,
    game_date: dt.date,
    store_result: bool,
    league_code: str | None = None,
) -> tuple[dict[str, Any], Any | None]:
    normalized = require_supported_league_code(league_code)
    built = build_features_by_k(
        db=db,
        home_team_hockeytech_id=home_team_hockeytech_id,
        away_team_hockeytech_id=away_team_hockeytech_id,
        game_date=game_date,
        league_code=normalized,
    )

    payload = {
        "game_id": None,
        "game_date": game_date.isoformat(),
        "home_team_id": home_team_hockeytech_id,
        "away_team_id": away_team_hockeytech_id,
        "features_by_k": built["features_by_k"],
        "context_features": built.get("context_features") or {},
    }

    result = predict_from_payload(payload, league_code=normalized)

    stored_primary_record: Any | None = None
    if store_result:
        _, stores = _resolve_write_stores(db, normalized)
        predicted_ht_id = int(result["predicted_winner_id"]) if result.get("predicted_winner_id") is not None else None

        for store, league_id in stores:
            home_stmt = apply_league_scope(
                select(store.team_model).where(store.team_model.hockeytech_id == built["home_team"].hockeytech_id),
                store.team_model,
                league_id,
            )
            away_stmt = apply_league_scope(
                select(store.team_model).where(store.team_model.hockeytech_id == built["away_team"].hockeytech_id),
                store.team_model,
                league_id,
            )
            home_team = db.scalar(home_stmt)
            away_team = db.scalar(away_stmt)
            if not home_team or not away_team:
                continue

            predicted_winner_db_id = None
            if predicted_ht_id == home_team.hockeytech_id:
                predicted_winner_db_id = home_team.id
            elif predicted_ht_id == away_team.hockeytech_id:
                predicted_winner_db_id = away_team.id

            kwargs = {
                "home_team_id": home_team.id,
                "away_team_id": away_team.id,
                "game_date": game_date,
                "home_team_probability": result["home_team_prob"],
                "away_team_probability": result["away_team_prob"],
                "predicted_winner_id": predicted_winner_db_id,
                "model_version": result.get("model_version"),
                "model_family": result.get("model_family"),
                "k_components": result.get("k_components") or {},
            }
            if store.league_scoped:
                kwargs["league_id"] = league_id

            record = store.custom_prediction_model(**kwargs)
            db.add(record)
            db.flush()

            if store.name == primary_store().name:
                stored_primary_record = record

        db.commit()
        if stored_primary_record is not None:
            db.refresh(stored_primary_record)

    return result, stored_primary_record


def list_prediction_history(
    db: Session,
    date_from: dt.date | None,
    date_to: dt.date | None,
    team_hockeytech_id: int | None,
    k_value: int | None,
    limit: int = 500,
    league_code: str | None = None,
) -> list[Any]:
    normalized, store, league_id = _resolve_read_store(db, league_code)
    del normalized

    prediction_model = store.prediction_model
    team_model = store.team_model

    stmt = select(prediction_model).order_by(desc(prediction_model.prediction_date), desc(prediction_model.id))
    stmt = apply_league_scope(stmt, prediction_model, league_id)

    start_dt, end_dt = prediction_date_bounds(date_from=date_from, date_to=date_to)
    if start_dt:
        stmt = stmt.where(prediction_model.prediction_date >= start_dt)
    if end_dt:
        stmt = stmt.where(prediction_model.prediction_date <= end_dt)
    if k_value:
        stmt = stmt.where(prediction_model.k_value == k_value)

    if team_hockeytech_id is not None:
        team_stmt = apply_league_scope(
            select(team_model).where(team_model.hockeytech_id == team_hockeytech_id),
            team_model,
            league_id,
        )
        team = db.scalar(team_stmt)
        if team is None:
            return []
        stmt = stmt.where(or_(prediction_model.home_team_id == team.id, prediction_model.away_team_id == team.id))

    stmt = stmt.limit(limit)
    return db.scalars(stmt).all()


def run_daily_pipeline(
    db: Session,
    run_date: dt.date | None = None,
    league_code: str | None = None,
) -> dict[str, Any]:
    normalized = require_supported_league_code(league_code)
    today = run_date or dt.date.today()
    tomorrow = today + dt.timedelta(days=1)
    yesterday = today - dt.timedelta(days=1)

    schedule_rows = upsert_upcoming_schedule(db, tomorrow, league_code=normalized)
    completed_rows = update_completed_games(db, yesterday, league_code=normalized)
    rolling_rows = recompute_rolling_averages(db, league_code=normalized)

    from services.training import train_and_maybe_promote

    train_result = train_and_maybe_promote(promote=True, league_code=normalized)
    upcoming_stats = run_upcoming_predictions(db, tomorrow, league_code=normalized)

    return {
        "league_code": normalized,
        "run_date": today.isoformat(),
        "target_upcoming_date": tomorrow.isoformat(),
        "target_completed_date": yesterday.isoformat(),
        "schedule_rows": schedule_rows,
        "completed_rows": completed_rows,
        "rolling_rows": rolling_rows,
        "training": train_result,
        "upcoming_predictions": {
            "predictions_written": upcoming_stats.predictions_written,
            "skipped_games": upcoming_stats.skipped_games,
        },
    }
