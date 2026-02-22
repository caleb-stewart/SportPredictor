from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from sqlalchemy import desc, or_, select
from sqlalchemy.orm import Session

from services.context_features import compute_matchup_context_features
from services.data_backend import (
    CHL_LEAGUES,
    DataBackendError,
    apply_league_scope,
    primary_store,
    require_supported_league_code,
    resolve_league_id_for_store,
)

K_VALUES = [5, 10, 15]


class FeatureBuilderError(RuntimeError):
    pass


class InsufficientHistoryError(FeatureBuilderError):
    pass


class TeamNotFoundError(FeatureBuilderError):
    pass


@dataclass
class MatchupContext:
    home_team: Any
    away_team: Any
    game_date: dt.date


def _resolve_store(db: Session | None, league_code: str | None):
    normalized = require_supported_league_code(league_code)
    store = primary_store()
    if store.league_scoped and db is None:
        raise DataBackendError("db session is required for CHL-scoped feature queries.")
    league_id = resolve_league_id_for_store(db, store, normalized) if db is not None else None
    return normalized, store, league_id


def resolve_matchup_context(
    db: Session,
    home_team_hockeytech_id: int,
    away_team_hockeytech_id: int,
    game_date: dt.date,
    league_code: str | None = None,
) -> MatchupContext:
    _, store, league_id = _resolve_store(db, league_code)

    team_model = store.team_model
    home_stmt = apply_league_scope(
        select(team_model).where(team_model.hockeytech_id == home_team_hockeytech_id),
        team_model,
        league_id,
    )
    away_stmt = apply_league_scope(
        select(team_model).where(team_model.hockeytech_id == away_team_hockeytech_id),
        team_model,
        league_id,
    )

    home_team = db.scalar(home_stmt)
    away_team = db.scalar(away_stmt)

    if not home_team or not away_team:
        raise TeamNotFoundError("Unknown home_team_id or away_team_id.")

    return MatchupContext(home_team=home_team, away_team=away_team, game_date=game_date)


def _feature_map_from_avg(avg: Any) -> dict[str, float]:
    return {
        "goals_for_avg": float(avg.goals_for_avg or 0.0),
        "goals_against_avg": float(avg.goals_against_avg or 0.0),
        "shots_for_avg": float(avg.shots_for_avg or 0.0),
        "shots_against_avg": float(avg.shots_against_avg or 0.0),
        "power_play_percentage_avg": float(avg.power_play_percentage_avg or 0.0),
        "power_play_percentage_against_avg": float(avg.power_play_percentage_against_avg or 0.0),
        "faceoff_win_percentage_avg": float(avg.faceoff_win_percentage_avg or 0.0),
        "faceoff_win_percentage_against_avg": float(avg.faceoff_win_percentage_against_avg or 0.0),
        "goals_diff": float(avg.goals_diff or 0.0),
        "ppp_diff": float(avg.ppp_diff or 0.0),
        "sog_diff": float(avg.sog_diff or 0.0),
        "fowp_diff": float(avg.fowp_diff or 0.0),
    }


def _latest_rolling_average(
    db: Session,
    team_db_id: int,
    k_value: int,
    game_date: dt.date,
    league_code: str | None,
) -> Any | None:
    stmt = _latest_rolling_average_stmt(
        db=db,
        team_db_id=team_db_id,
        k_value=k_value,
        game_date=game_date,
        league_code=league_code,
    )
    return db.scalar(stmt)


def _latest_rolling_average_stmt(
    team_db_id: int,
    k_value: int,
    game_date: dt.date,
    league_code: str | None = None,
    db: Session | None = None,
):
    if db is None:
        normalized = require_supported_league_code(league_code)
        store = primary_store()
        league_id = int(CHL_LEAGUES[normalized]["id"]) if store.league_scoped else None
    else:
        _, store, league_id = _resolve_store(db, league_code)
    rolling_model = store.rolling_model
    game_model = store.game_model

    stmt = (
        select(rolling_model)
        .join(game_model, rolling_model.game_id == game_model.game_id)
        .where(
            getattr(rolling_model, store.rolling_team_id_field) == team_db_id,
            rolling_model.k_value == k_value,
            game_model.game_date < game_date,
        )
        .order_by(desc(game_model.game_date), desc(game_model.game_id))
        .limit(1)
    )
    stmt = apply_league_scope(stmt, rolling_model, league_id)
    stmt = apply_league_scope(stmt, game_model, league_id)
    return stmt


def build_features_by_k(
    db: Session,
    home_team_hockeytech_id: int,
    away_team_hockeytech_id: int,
    game_date: dt.date,
    k_values: list[int] | None = None,
    league_code: str | None = None,
) -> dict[str, Any]:
    k_targets = k_values or K_VALUES
    _, store, league_id = _resolve_store(db, league_code)

    context = resolve_matchup_context(
        db=db,
        home_team_hockeytech_id=home_team_hockeytech_id,
        away_team_hockeytech_id=away_team_hockeytech_id,
        game_date=game_date,
        league_code=league_code,
    )

    features_by_k: dict[str, dict[str, dict[str, float]]] = {}

    for k in k_targets:
        home_avg = _latest_rolling_average(db, context.home_team.id, k, game_date, league_code=league_code)
        away_avg = _latest_rolling_average(db, context.away_team.id, k, game_date, league_code=league_code)

        if home_avg is None or away_avg is None:
            raise InsufficientHistoryError(
                f"Insufficient rolling history for k={k} before {game_date.isoformat()}."
            )

        features_by_k[str(k)] = {
            "home": _feature_map_from_avg(home_avg),
            "away": _feature_map_from_avg(away_avg),
        }

    game_model = store.game_model
    history_stmt = (
        select(
            game_model.game_id,
            game_model.game_date,
            game_model.home_team_id,
            game_model.away_team_id,
            game_model.home_goal_count,
            game_model.away_goal_count,
        )
        .where(
            game_model.game_date.is_not(None),
            game_model.home_goal_count.is_not(None),
            game_model.away_goal_count.is_not(None),
            game_model.game_date < game_date,
            or_(
                game_model.home_team_id.in_([home_team_hockeytech_id, away_team_hockeytech_id]),
                game_model.away_team_id.in_([home_team_hockeytech_id, away_team_hockeytech_id]),
            ),
        )
        .order_by(game_model.game_date.asc(), game_model.game_id.asc())
    )
    history_stmt = apply_league_scope(history_stmt, game_model, league_id)
    history_rows = db.execute(history_stmt).all()

    context_features = compute_matchup_context_features(
        historical_games=[
            {
                "game_id": row.game_id,
                "game_date": row.game_date,
                "home_team_id": row.home_team_id,
                "away_team_id": row.away_team_id,
                "home_goal_count": row.home_goal_count,
                "away_goal_count": row.away_goal_count,
            }
            for row in history_rows
        ],
        home_team_id=home_team_hockeytech_id,
        away_team_id=away_team_hockeytech_id,
        game_date=game_date,
    )

    k15 = features_by_k.get("15")
    if k15:
        goals_diff_diff = float(k15["home"]["goals_diff"] - k15["away"]["goals_diff"])
        sog_diff_diff = float(k15["home"]["sog_diff"] - k15["away"]["sog_diff"])
        elo_diff = float(context_features.get("elo_diff_pre", 0.0))
        context_features["strength_adjusted_goals_diff"] = goals_diff_diff - (elo_diff / 400.0)
        context_features["strength_adjusted_sog_diff"] = sog_diff_diff - (elo_diff / 400.0)

    return {
        "home_team": context.home_team,
        "away_team": context.away_team,
        "features_by_k": features_by_k,
        "context_features": context_features,
    }
