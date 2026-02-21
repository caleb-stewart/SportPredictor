from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from sqlalchemy import and_, desc, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from db.models import (
    WhlCustomPredictionRecord,
    WhlGame,
    WhlPredictionRecord,
    WhlRollingAverage,
    WhlTeam,
)
from services.feature_builder import (
    InsufficientHistoryError,
    TeamNotFoundError,
    build_features_by_k,
)
from services.hockeytech_client import HockeyTechClient
from services.predictor import predict_from_payload

K_VALUES = [5, 10, 15]


@dataclass
class UpcomingRunStats:
    target_date: dt.date
    predictions_written: int
    skipped_games: int


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


def upsert_upcoming_schedule(db: Session, target_date: dt.date) -> int:
    client = HockeyTechClient()
    games = client.get_schedule_for_date(target_date)

    rows_written = 0
    now = dt.datetime.now(dt.UTC)
    for game_data in games:
        game_id = int(game_data["ID"])

        stmt = (
            insert(WhlGame)
            .values(
                game_id=game_id,
                season_id=game_data.get("SeasonID"),
                season_name=game_data.get("SeasonName"),
                game_date=_parse_iso_date(game_data.get("GameDateISO8601")),
                venue=game_data.get("venue_name"),
                status=game_data.get("GameStatusString"),
                home_team=game_data.get("HomeLongName"),
                away_team=game_data.get("VisitorLongName"),
                home_team_id=int(game_data.get("HomeID")) if game_data.get("HomeID") else None,
                away_team_id=int(game_data.get("VisitorID")) if game_data.get("VisitorID") else None,
                scorebar_snapshot=game_data,
                scheduled_time_utc=_parse_iso_datetime(game_data.get("GameDateISO8601")),
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_update(
                index_elements=[WhlGame.game_id],
                set_={
                    "season_id": game_data.get("SeasonID"),
                    "season_name": game_data.get("SeasonName"),
                    "game_date": _parse_iso_date(game_data.get("GameDateISO8601")),
                    "venue": game_data.get("venue_name"),
                    "status": game_data.get("GameStatusString"),
                    "home_team": game_data.get("HomeLongName"),
                    "away_team": game_data.get("VisitorLongName"),
                    "home_team_id": int(game_data.get("HomeID")) if game_data.get("HomeID") else None,
                    "away_team_id": int(game_data.get("VisitorID")) if game_data.get("VisitorID") else None,
                    "scorebar_snapshot": game_data,
                    "scheduled_time_utc": _parse_iso_datetime(game_data.get("GameDateISO8601")),
                    "updated_at": now,
                },
            )
        )
        db.execute(stmt)
        rows_written += 1

    db.commit()
    return rows_written


def update_completed_games(db: Session, target_date: dt.date) -> int:
    client = HockeyTechClient()
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

        game_id = int(game_id_raw)
        clock = client.get_clock(game_id)

        home_pp_attempts = float((clock.get("power_play") or {}).get("total", {}).get("home") or 0.0)
        away_pp_attempts = float((clock.get("power_play") or {}).get("total", {}).get("visiting") or 0.0)
        home_pp_goals = float((clock.get("power_play") or {}).get("goals", {}).get("home") or 0.0)
        away_pp_goals = float((clock.get("power_play") or {}).get("goals", {}).get("visiting") or 0.0)

        home_pp = (home_pp_goals / home_pp_attempts) if home_pp_attempts > 0 else 0.0
        away_pp = (away_pp_goals / away_pp_attempts) if away_pp_attempts > 0 else 0.0

        home_fow = float((clock.get("fow") or {}).get("home") or 0.0)
        away_fow = float((clock.get("fow") or {}).get("visiting") or 0.0)
        fow_total = home_fow + away_fow
        home_fowp = (home_fow / fow_total) if fow_total > 0 else 0.5
        away_fowp = (away_fow / fow_total) if fow_total > 0 else 0.5

        home_sog = sum(int(v) for v in ((clock.get("shots_on_goal") or {}).get("home") or {}).values())
        away_sog = sum(int(v) for v in ((clock.get("shots_on_goal") or {}).get("visiting") or {}).values())

        stmt = (
            insert(WhlGame)
            .values(
                game_id=game_id,
                season_id=clock.get("season_id") or game_data.get("SeasonID"),
                season_name=clock.get("season_name") or game_data.get("SeasonName"),
                game_date=_parse_iso_date(clock.get("game_date_iso_8601") or game_data.get("GameDateISO8601")),
                venue=clock.get("venue") or game_data.get("venue_name"),
                status=clock.get("progress") or game_data.get("GameStatusString"),
                home_team_id=int((clock.get("home_team") or {}).get("team_id") or game_data.get("HomeID")),
                away_team_id=int((clock.get("visiting_team") or {}).get("team_id") or game_data.get("VisitorID")),
                home_team=(clock.get("home_team") or {}).get("name") or game_data.get("HomeLongName"),
                away_team=(clock.get("visiting_team") or {}).get("name") or game_data.get("VisitorLongName"),
                home_goal_count=int(clock.get("home_goal_count") or 0),
                away_goal_count=int(clock.get("visiting_goal_count") or 0),
                game_number=int(clock.get("game_number") or 0),
                period=clock.get("period"),
                scoring_breakdown=clock.get("scoring"),
                shots_on_goal=clock.get("shots_on_goal"),
                power_play=clock.get("power_play"),
                fow=clock.get("fow"),
                home_power_play_percentage=home_pp,
                away_power_play_percentage=away_pp,
                home_faceoff_win_percentage=home_fowp,
                away_faceoff_win_percentage=away_fowp,
                home_shots_on_goal_total=home_sog,
                away_shots_on_goal_total=away_sog,
                scorebar_snapshot=game_data,
                scheduled_time_utc=_parse_iso_datetime(game_data.get("GameDateISO8601")),
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_update(
                index_elements=[WhlGame.game_id],
                set_={
                    "season_id": clock.get("season_id") or game_data.get("SeasonID"),
                    "season_name": clock.get("season_name") or game_data.get("SeasonName"),
                    "game_date": _parse_iso_date(clock.get("game_date_iso_8601") or game_data.get("GameDateISO8601")),
                    "venue": clock.get("venue") or game_data.get("venue_name"),
                    "status": clock.get("progress") or game_data.get("GameStatusString"),
                    "home_team_id": int((clock.get("home_team") or {}).get("team_id") or game_data.get("HomeID")),
                    "away_team_id": int((clock.get("visiting_team") or {}).get("team_id") or game_data.get("VisitorID")),
                    "home_team": (clock.get("home_team") or {}).get("name") or game_data.get("HomeLongName"),
                    "away_team": (clock.get("visiting_team") or {}).get("name") or game_data.get("VisitorLongName"),
                    "home_goal_count": int(clock.get("home_goal_count") or 0),
                    "away_goal_count": int(clock.get("visiting_goal_count") or 0),
                    "game_number": int(clock.get("game_number") or 0),
                    "period": clock.get("period"),
                    "scoring_breakdown": clock.get("scoring"),
                    "shots_on_goal": clock.get("shots_on_goal"),
                    "power_play": clock.get("power_play"),
                    "fow": clock.get("fow"),
                    "home_power_play_percentage": home_pp,
                    "away_power_play_percentage": away_pp,
                    "home_faceoff_win_percentage": home_fowp,
                    "away_faceoff_win_percentage": away_fowp,
                    "home_shots_on_goal_total": home_sog,
                    "away_shots_on_goal_total": away_sog,
                    "scorebar_snapshot": game_data,
                    "scheduled_time_utc": _parse_iso_datetime(game_data.get("GameDateISO8601")),
                    "updated_at": now,
                },
            )
        )
        db.execute(stmt)
        updated += 1

    db.commit()
    _update_prediction_record_correctness(db, target_date)
    return updated


def _update_prediction_record_correctness(db: Session, target_date: dt.date) -> None:
    games_stmt = select(WhlGame).where(WhlGame.game_date == target_date)
    games = db.scalars(games_stmt).all()

    for game in games:
        if game.home_goal_count is None or game.away_goal_count is None:
            continue

        home_team = db.scalar(select(WhlTeam).where(WhlTeam.hockeytech_id == game.home_team_id))
        away_team = db.scalar(select(WhlTeam).where(WhlTeam.hockeytech_id == game.away_team_id))
        if not home_team or not away_team:
            continue

        actual_winner_id: int | None
        if game.home_goal_count > game.away_goal_count:
            actual_winner_id = home_team.id
        elif game.home_goal_count < game.away_goal_count:
            actual_winner_id = away_team.id
        else:
            actual_winner_id = None

        records = db.scalars(select(WhlPredictionRecord).where(WhlPredictionRecord.game_id == game.game_id)).all()
        for record in records:
            predicted_winner_id = record.home_team_id if (record.home_team_probability or 0) >= (record.away_team_probability or 0) else record.away_team_id
            record.predicted_winner_id = predicted_winner_id
            record.actual_winner_id = actual_winner_id
            record.correct = (predicted_winner_id == actual_winner_id) if actual_winner_id is not None else None

    db.commit()


def recompute_rolling_averages(db: Session, k_values: list[int] | None = None) -> int:
    k_targets = k_values or K_VALUES
    teams = db.scalars(select(WhlTeam)).all()
    rows_written = 0

    now = dt.datetime.now(dt.UTC)

    for team in teams:
        games = db.scalars(
            select(WhlGame)
            .where(or_(WhlGame.home_team_id == team.hockeytech_id, WhlGame.away_team_id == team.hockeytech_id))
            .where(WhlGame.game_date.is_not(None))
            .order_by(WhlGame.game_date, WhlGame.game_id)
        ).all()

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
                    target_goals.append(float(g.home_goal_count if is_home else g.away_goal_count or 0.0))
                    opp_goals.append(float(g.away_goal_count if is_home else g.home_goal_count or 0.0))

                    target_ppp.append(float(g.home_power_play_percentage if is_home else g.away_power_play_percentage or 0.0))
                    opp_ppp.append(float(g.away_power_play_percentage if is_home else g.home_power_play_percentage or 0.0))

                    target_sog.append(float(g.home_shots_on_goal_total if is_home else g.away_shots_on_goal_total or 0.0))
                    opp_sog.append(float(g.away_shots_on_goal_total if is_home else g.home_shots_on_goal_total or 0.0))

                    target_fowp.append(float(g.home_faceoff_win_percentage if is_home else g.away_faceoff_win_percentage or 0.0))
                    opp_fowp.append(float(g.away_faceoff_win_percentage if is_home else g.home_faceoff_win_percentage or 0.0))

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
                    "whl_team_id": team.id,
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

                stmt = (
                    insert(WhlRollingAverage)
                    .values(**row_data)
                    .on_conflict_do_update(
                        index_elements=["game_id", "k_value", "whl_team_id"],
                        set_={
                            **{k2: row_data[k2] for k2 in row_data.keys() if k2 not in {"created_at"}},
                            "updated_at": now,
                        },
                    )
                )
                db.execute(stmt)
                rows_written += 1

    db.commit()
    return rows_written


def _persist_prediction_rows(
    db: Session,
    game: WhlGame,
    home_team: WhlTeam,
    away_team: WhlTeam,
    result: dict[str, Any],
    extra_raw_model_outputs_by_k: dict[int, dict[str, Any]] | None = None,
    commit: bool = True,
) -> int:
    ensemble_home = float(result["home_team_prob"])
    ensemble_away = float(result["away_team_prob"])
    predicted_ht_id = int(result["predicted_winner_id"]) if result.get("predicted_winner_id") is not None else None

    predicted_winner_db_id = None
    if predicted_ht_id == home_team.hockeytech_id:
        predicted_winner_db_id = home_team.id
    elif predicted_ht_id == away_team.hockeytech_id:
        predicted_winner_db_id = away_team.id

    actual_winner_db_id = None
    if game.home_goal_count is not None and game.away_goal_count is not None:
        if game.home_goal_count > game.away_goal_count:
            actual_winner_db_id = home_team.id
        elif game.home_goal_count < game.away_goal_count:
            actual_winner_db_id = away_team.id

    now = dt.datetime.now(dt.UTC)
    written = 0

    for k in K_VALUES:
        comp = (result.get("k_components") or {}).get(str(k), {})
        home_prob = float(comp.get("home_team_prob", ensemble_home))
        away_prob = float(comp.get("away_team_prob", ensemble_away))

        winner_id = predicted_winner_db_id
        if winner_id is None:
            winner_id = home_team.id if home_prob >= away_prob else away_team.id

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
            "game_id": game.game_id,
            "k_value": k,
            "home_team_id": home_team.id,
            "away_team_id": away_team.id,
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

        stmt = (
            insert(WhlPredictionRecord)
            .values(**record_data)
            .on_conflict_do_update(
                index_elements=["game_id", "k_value"],
                set_={**record_data, "updated_at": now},
            )
        )
        db.execute(stmt)
        written += 1

    if commit:
        db.commit()
    return written


def get_upcoming_games(db: Session, target_date: dt.date) -> list[WhlGame]:
    return db.scalars(
        select(WhlGame)
        .where(WhlGame.game_date == target_date)
        .order_by(WhlGame.game_id)
    ).all()


def run_upcoming_predictions(db: Session, target_date: dt.date) -> UpcomingRunStats:
    upsert_upcoming_schedule(db, target_date)
    games = get_upcoming_games(db, target_date)

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
        }

        result = predict_from_payload(payload)
        written += _persist_prediction_rows(db, game, built["home_team"], built["away_team"], result)

    return UpcomingRunStats(target_date=target_date, predictions_written=written, skipped_games=skipped)


def run_custom_prediction(
    db: Session,
    home_team_hockeytech_id: int,
    away_team_hockeytech_id: int,
    game_date: dt.date,
    store_result: bool,
) -> tuple[dict[str, Any], WhlCustomPredictionRecord | None]:
    built = build_features_by_k(
        db=db,
        home_team_hockeytech_id=home_team_hockeytech_id,
        away_team_hockeytech_id=away_team_hockeytech_id,
        game_date=game_date,
    )

    payload = {
        "game_id": None,
        "game_date": game_date.isoformat(),
        "home_team_id": home_team_hockeytech_id,
        "away_team_id": away_team_hockeytech_id,
        "features_by_k": built["features_by_k"],
    }

    result = predict_from_payload(payload)

    record: WhlCustomPredictionRecord | None = None
    if store_result:
        predicted_ht_id = int(result["predicted_winner_id"]) if result.get("predicted_winner_id") is not None else None
        predicted_winner_db_id = None
        if predicted_ht_id == built["home_team"].hockeytech_id:
            predicted_winner_db_id = built["home_team"].id
        elif predicted_ht_id == built["away_team"].hockeytech_id:
            predicted_winner_db_id = built["away_team"].id

        record = WhlCustomPredictionRecord(
            home_team_id=built["home_team"].id,
            away_team_id=built["away_team"].id,
            game_date=game_date,
            home_team_probability=result["home_team_prob"],
            away_team_probability=result["away_team_prob"],
            predicted_winner_id=predicted_winner_db_id,
            model_version=result.get("model_version"),
            model_family=result.get("model_family"),
            k_components=result.get("k_components") or {},
        )
        db.add(record)
        db.commit()
        db.refresh(record)

    return result, record


def list_prediction_history(
    db: Session,
    date_from: dt.date | None,
    date_to: dt.date | None,
    team_hockeytech_id: int | None,
    k_value: int | None,
    limit: int = 500,
) -> list[WhlPredictionRecord]:
    stmt = select(WhlPredictionRecord).order_by(desc(WhlPredictionRecord.prediction_date), desc(WhlPredictionRecord.id))

    if date_from:
        stmt = stmt.where(WhlPredictionRecord.prediction_date >= dt.datetime.combine(date_from, dt.time.min, tzinfo=dt.UTC))
    if date_to:
        stmt = stmt.where(WhlPredictionRecord.prediction_date <= dt.datetime.combine(date_to, dt.time.max, tzinfo=dt.UTC))
    if k_value:
        stmt = stmt.where(WhlPredictionRecord.k_value == k_value)

    if team_hockeytech_id is not None:
        team = db.scalar(select(WhlTeam).where(WhlTeam.hockeytech_id == team_hockeytech_id))
        if team is None:
            return []
        stmt = stmt.where(or_(WhlPredictionRecord.home_team_id == team.id, WhlPredictionRecord.away_team_id == team.id))

    stmt = stmt.limit(limit)
    return db.scalars(stmt).all()


def run_daily_pipeline(db: Session, run_date: dt.date | None = None) -> dict[str, Any]:
    today = run_date or dt.date.today()
    tomorrow = today + dt.timedelta(days=1)
    yesterday = today - dt.timedelta(days=1)

    schedule_rows = upsert_upcoming_schedule(db, tomorrow)
    completed_rows = update_completed_games(db, yesterday)
    rolling_rows = recompute_rolling_averages(db)

    from services.training import train_and_maybe_promote

    train_result = train_and_maybe_promote(promote=True)
    upcoming_stats = run_upcoming_predictions(db, tomorrow)

    return {
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
