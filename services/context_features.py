from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

DEFAULT_ELO = 1500.0
DEFAULT_REST_DAYS = 7.0
ELO_K = 20.0
HOME_ICE_ADVANTAGE = 35.0
TREND_WINDOW = 5

CONTEXT_FEATURE_COLUMNS: list[str] = [
    "elo_home_pre",
    "elo_away_pre",
    "elo_diff_pre",
    "elo_sum_pre",
    "elo_expected_home_prob",
    "home_games_played_pre",
    "away_games_played_pre",
    "games_played_diff",
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    "home_back_to_back",
    "away_back_to_back",
    "elo_home_trend_5",
    "elo_away_trend_5",
    "elo_trend_diff_5",
    "elo_uncertainty_home",
    "elo_uncertainty_away",
    "elo_uncertainty_diff",
]


@dataclass
class TeamState:
    elo: float = DEFAULT_ELO
    games_played: int = 0
    last_game_date: dt.date | None = None
    pregame_elo_history: deque[float] | None = None

    def __post_init__(self) -> None:
        if self.pregame_elo_history is None:
            self.pregame_elo_history = deque(maxlen=TREND_WINDOW + 1)


def _expected_home(elo_home: float, elo_away: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(elo_home + HOME_ICE_ADVANTAGE - elo_away) / 400.0))


def _rest_days(last_game_date: dt.date | None, current_date: dt.date) -> float:
    if last_game_date is None:
        return DEFAULT_REST_DAYS
    return float((current_date - last_game_date).days)


def _trend_from_history(history: deque[float]) -> float:
    if len(history) <= TREND_WINDOW:
        return 0.0
    oldest = history[0]
    newest = history[-1]
    return float(newest - oldest)


def _uncertainty(games_played: int) -> float:
    return 1.0 / math.sqrt(max(games_played, 0) + 1.0)


def _pregame_context(
    home_state: TeamState,
    away_state: TeamState,
    game_date: dt.date,
) -> dict[str, float]:
    home_rest = _rest_days(home_state.last_game_date, game_date)
    away_rest = _rest_days(away_state.last_game_date, game_date)
    elo_expected_home = _expected_home(home_state.elo, away_state.elo)

    home_trend = _trend_from_history(home_state.pregame_elo_history or deque())
    away_trend = _trend_from_history(away_state.pregame_elo_history or deque())

    home_uncertainty = _uncertainty(home_state.games_played)
    away_uncertainty = _uncertainty(away_state.games_played)

    return {
        "elo_home_pre": float(home_state.elo),
        "elo_away_pre": float(away_state.elo),
        "elo_diff_pre": float(home_state.elo - away_state.elo),
        "elo_sum_pre": float((home_state.elo + away_state.elo) / 2.0),
        "elo_expected_home_prob": float(elo_expected_home),
        "home_games_played_pre": float(home_state.games_played),
        "away_games_played_pre": float(away_state.games_played),
        "games_played_diff": float(home_state.games_played - away_state.games_played),
        "home_rest_days": float(home_rest),
        "away_rest_days": float(away_rest),
        "rest_days_diff": float(home_rest - away_rest),
        "home_back_to_back": float(1.0 if home_rest <= 1.0 else 0.0),
        "away_back_to_back": float(1.0 if away_rest <= 1.0 else 0.0),
        "elo_home_trend_5": float(home_trend),
        "elo_away_trend_5": float(away_trend),
        "elo_trend_diff_5": float(home_trend - away_trend),
        "elo_uncertainty_home": float(home_uncertainty),
        "elo_uncertainty_away": float(away_uncertainty),
        "elo_uncertainty_diff": float(home_uncertainty - away_uncertainty),
    }


def _update_states_after_game(
    home_state: TeamState,
    away_state: TeamState,
    game_date: dt.date,
    home_win: int | None,
) -> None:
    expected_home = _expected_home(home_state.elo, away_state.elo)
    if home_win is not None:
        delta = ELO_K * (float(home_win) - expected_home)
        home_state.elo += delta
        away_state.elo -= delta

    home_state.games_played += 1
    away_state.games_played += 1
    home_state.last_game_date = game_date
    away_state.last_game_date = game_date


def build_context_feature_map(games_frame: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Build leakage-safe pregame context features by replaying games chronologically."""
    if games_frame.empty:
        return {}

    required_cols = {"game_id", "game_date", "home_team_id", "away_team_id", "home_goal_count", "away_goal_count"}
    missing = required_cols - set(games_frame.columns)
    if missing:
        raise ValueError(f"Missing required columns for context features: {sorted(missing)}")

    ordered = games_frame.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    states: dict[int, TeamState] = defaultdict(TeamState)
    context_by_game_id: dict[int, dict[str, float]] = {}

    for _, row in ordered.iterrows():
        if pd.isna(row["game_date"]) or pd.isna(row["home_team_id"]) or pd.isna(row["away_team_id"]):
            continue

        game_date = pd.Timestamp(row["game_date"]).date()
        home_team_id = int(row["home_team_id"])
        away_team_id = int(row["away_team_id"])
        game_id = int(row["game_id"])

        home_state = states[home_team_id]
        away_state = states[away_team_id]

        if home_state.pregame_elo_history is not None:
            home_state.pregame_elo_history.append(float(home_state.elo))
        if away_state.pregame_elo_history is not None:
            away_state.pregame_elo_history.append(float(away_state.elo))

        context_by_game_id[game_id] = _pregame_context(
            home_state=home_state,
            away_state=away_state,
            game_date=game_date,
        )

        home_goals = row["home_goal_count"]
        away_goals = row["away_goal_count"]
        home_win: int | None = None
        if not pd.isna(home_goals) and not pd.isna(away_goals):
            home_win = 1 if int(home_goals) > int(away_goals) else 0

        _update_states_after_game(
            home_state=home_state,
            away_state=away_state,
            game_date=game_date,
            home_win=home_win,
        )

    return context_by_game_id


def compute_matchup_context_features(
    historical_games: Iterable[dict[str, object]],
    home_team_id: int,
    away_team_id: int,
    game_date: dt.date,
) -> dict[str, float]:
    rows = list(historical_games)
    if not rows:
        home_state = TeamState()
        away_state = TeamState()
        return _pregame_context(home_state=home_state, away_state=away_state, game_date=game_date)

    frame = pd.DataFrame(rows)
    if frame.empty:
        home_state = TeamState()
        away_state = TeamState()
        return _pregame_context(home_state=home_state, away_state=away_state, game_date=game_date)

    states: dict[int, TeamState] = defaultdict(TeamState)
    ordered = frame.sort_values(["game_date", "game_id"])

    for _, row in ordered.iterrows():
        if pd.isna(row.get("game_date")) or pd.isna(row.get("home_team_id")) or pd.isna(row.get("away_team_id")):
            continue

        row_date = pd.Timestamp(row["game_date"]).date()
        if row_date >= game_date:
            continue

        h_id = int(row["home_team_id"])
        a_id = int(row["away_team_id"])
        home_state = states[h_id]
        away_state = states[a_id]

        if home_state.pregame_elo_history is not None:
            home_state.pregame_elo_history.append(float(home_state.elo))
        if away_state.pregame_elo_history is not None:
            away_state.pregame_elo_history.append(float(away_state.elo))

        home_goals = row.get("home_goal_count")
        away_goals = row.get("away_goal_count")
        home_win: int | None = None
        if home_goals is not None and away_goals is not None and not pd.isna(home_goals) and not pd.isna(away_goals):
            home_win = 1 if int(home_goals) > int(away_goals) else 0

        _update_states_after_game(
            home_state=home_state,
            away_state=away_state,
            game_date=row_date,
            home_win=home_win,
        )

    home_state = states[home_team_id]
    away_state = states[away_team_id]
    if home_state.pregame_elo_history is not None:
        home_state.pregame_elo_history.append(float(home_state.elo))
    if away_state.pregame_elo_history is not None:
        away_state.pregame_elo_history.append(float(away_state.elo))

    return _pregame_context(home_state=home_state, away_state=away_state, game_date=game_date)
