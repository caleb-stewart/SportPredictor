from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from db.models import WhlGame, WhlRollingAverage, WhlTeam

K_VALUES = [5, 10, 15]


class FeatureBuilderError(RuntimeError):
    pass


class InsufficientHistoryError(FeatureBuilderError):
    pass


class TeamNotFoundError(FeatureBuilderError):
    pass


@dataclass
class MatchupContext:
    home_team: WhlTeam
    away_team: WhlTeam
    game_date: dt.date


def resolve_matchup_context(
    db: Session,
    home_team_hockeytech_id: int,
    away_team_hockeytech_id: int,
    game_date: dt.date,
) -> MatchupContext:
    home_team = db.scalar(select(WhlTeam).where(WhlTeam.hockeytech_id == home_team_hockeytech_id))
    away_team = db.scalar(select(WhlTeam).where(WhlTeam.hockeytech_id == away_team_hockeytech_id))

    if not home_team or not away_team:
        raise TeamNotFoundError("Unknown home_team_id or away_team_id.")

    return MatchupContext(home_team=home_team, away_team=away_team, game_date=game_date)


def _feature_map_from_avg(avg: WhlRollingAverage) -> dict[str, float]:
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
) -> WhlRollingAverage | None:
    stmt = _latest_rolling_average_stmt(team_db_id=team_db_id, k_value=k_value, game_date=game_date)
    return db.scalar(stmt)


def _latest_rolling_average_stmt(
    team_db_id: int,
    k_value: int,
    game_date: dt.date,
):
    return (
        select(WhlRollingAverage)
        .join(WhlGame, WhlRollingAverage.game_id == WhlGame.game_id)
        .where(
            WhlRollingAverage.whl_team_id == team_db_id,
            WhlRollingAverage.k_value == k_value,
            WhlGame.game_date < game_date,
        )
        .order_by(desc(WhlGame.game_date), desc(WhlGame.game_id))
        .limit(1)
    )


def build_features_by_k(
    db: Session,
    home_team_hockeytech_id: int,
    away_team_hockeytech_id: int,
    game_date: dt.date,
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    k_targets = k_values or K_VALUES
    context = resolve_matchup_context(
        db=db,
        home_team_hockeytech_id=home_team_hockeytech_id,
        away_team_hockeytech_id=away_team_hockeytech_id,
        game_date=game_date,
    )

    features_by_k: dict[str, dict[str, dict[str, float]]] = {}

    for k in k_targets:
        home_avg = _latest_rolling_average(db, context.home_team.id, k, game_date)
        away_avg = _latest_rolling_average(db, context.away_team.id, k, game_date)

        if home_avg is None or away_avg is None:
            raise InsufficientHistoryError(
                f"Insufficient rolling history for k={k} before {game_date.isoformat()}."
            )

        features_by_k[str(k)] = {
            "home": _feature_map_from_avg(home_avg),
            "away": _feature_map_from_avg(away_avg),
        }

    return {
        "home_team": context.home_team,
        "away_team": context.away_team,
        "features_by_k": features_by_k,
    }
