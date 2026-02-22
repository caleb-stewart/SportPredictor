from __future__ import annotations

import datetime as dt

import pandas as pd

from services.context_features import build_context_feature_map, compute_matchup_context_features


def test_build_context_feature_map_uses_pregame_state():
    games = pd.DataFrame(
        [
            {
                "game_id": 1,
                "game_date": dt.date(2025, 1, 1),
                "home_team_id": 10,
                "away_team_id": 20,
                "home_goal_count": 4,
                "away_goal_count": 2,
            },
            {
                "game_id": 2,
                "game_date": dt.date(2025, 1, 2),
                "home_team_id": 10,
                "away_team_id": 20,
                "home_goal_count": 1,
                "away_goal_count": 3,
            },
        ]
    )

    ctx = build_context_feature_map(games)
    assert 1 in ctx and 2 in ctx
    # Pregame state for game 1 should be neutral.
    assert round(ctx[1]["elo_diff_pre"], 6) == 0.0
    # Game 2 should include updated ratings from game 1 result.
    assert ctx[2]["elo_diff_pre"] != 0.0


def test_compute_matchup_context_features_no_future_leakage():
    history = [
        {
            "game_id": 1,
            "game_date": dt.date(2025, 1, 1),
            "home_team_id": 10,
            "away_team_id": 20,
            "home_goal_count": 3,
            "away_goal_count": 1,
        },
        {
            "game_id": 2,
            "game_date": dt.date(2025, 1, 10),
            "home_team_id": 20,
            "away_team_id": 10,
            "home_goal_count": 5,
            "away_goal_count": 1,
        },
    ]

    before_second = compute_matchup_context_features(
        historical_games=history,
        home_team_id=10,
        away_team_id=20,
        game_date=dt.date(2025, 1, 5),
    )
    after_second = compute_matchup_context_features(
        historical_games=history,
        home_team_id=10,
        away_team_id=20,
        game_date=dt.date(2025, 1, 15),
    )

    assert before_second["home_games_played_pre"] < after_second["home_games_played_pre"]
