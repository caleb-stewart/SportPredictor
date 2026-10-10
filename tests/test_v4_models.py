from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from services.predictor import predict_from_payload
from services.v4_models import (
    V4_FEATURE_COLUMNS,
    compute_v4_context_features,
    prepare_v4_games,
    replay_v4_features,
)
from services.v4_training import walk_forward_v4
from services.v4_training import save_v4_artifacts


def synthetic_games() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    game_id = 1
    for day in range(1, 13):
        rows.extend(
            [
                {
                    "game_id": game_id,
                    "game_date": dt.date(2025, 1, day),
                    "status": "Final",
                    "home_team_id": 1,
                    "away_team_id": 2,
                    "home_goal_count": 4 if day % 3 else 3,
                    "away_goal_count": 1,
                },
                {
                    "game_id": game_id + 1,
                    "game_date": dt.date(2025, 1, day),
                    "status": "Final",
                    "home_team_id": 3,
                    "away_team_id": 2,
                    "home_goal_count": 1,
                    "away_goal_count": 2,
                },
            ]
        )
        game_id += 2
    return pd.DataFrame(rows)


def test_prepare_v4_games_rejects_non_final_zero_zero_rows() -> None:
    frame = pd.concat(
        [
            synthetic_games(),
            pd.DataFrame(
                [
                    {
                        "game_id": 999,
                        "game_date": dt.date(2025, 2, 1),
                        "status": "Postponed",
                        "home_team_id": 1,
                        "away_team_id": 2,
                        "home_goal_count": 0,
                        "away_goal_count": 0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    prepared = prepare_v4_games(frame, league_code="ohl")
    assert 999 not in set(prepared["game_id"])


def test_goal_branch_uses_regulation_periods_when_available() -> None:
    frame = pd.DataFrame(
        [
            {
                "game_id": 1000,
                "game_date": dt.date(2025, 2, 2),
                "status": "Final SO",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_goal_count": 4,
                "away_goal_count": 3,
                "scoring_breakdown": {
                    "home": {"1": "1", "2": "1", "3": "1"},
                    "visiting": {"1": "1", "2": "1", "3": "1"},
                },
            }
        ]
    )
    prepared = prepare_v4_games(frame)
    assert prepared.iloc[0]["home_goal_count"] == 4
    assert prepared.iloc[0]["goal_model_home_count"] == 3
    assert prepared.iloc[0]["goal_model_away_count"] == 3


def test_replay_applies_results_after_whole_calendar_date() -> None:
    features = replay_v4_features(synthetic_games(), league_code="whl")
    first_day = features[features["game_date"] == dt.date(2025, 1, 1)]
    assert len(first_day) == 2
    # Team 2's loss must not influence the other forecast from the same date.
    assert first_day.iloc[0]["v4_rating_away_mean"] == 0.0
    assert first_day.iloc[1]["v4_rating_away_mean"] == 0.0
    next_day = features[features["game_date"] == dt.date(2025, 1, 2)]
    assert float(next_day["v4_rating_home_mean"].iloc[0]) != 0.0
    assert set(V4_FEATURE_COLUMNS).issubset(features.columns)


def test_v4_context_matches_replay_cutoff() -> None:
    games = synthetic_games()
    replay = replay_v4_features(games)
    history = games[games["game_date"] < dt.date(2025, 1, 5)].to_dict(orient="records")
    live = compute_v4_context_features(
        history,
        home_team_id=1,
        away_team_id=2,
        game_date=dt.date(2025, 1, 5),
    )
    expected = replay[replay["game_date"] == dt.date(2025, 1, 5)].iloc[0]
    assert np.isclose(live["v4_rating_home_prob"], expected["v4_rating_home_prob"])
    assert np.isclose(live["v4_goal_lambda_home"], expected["v4_goal_lambda_home"])


def test_walk_forward_v4_returns_holdout_metrics() -> None:
    metrics, predictions, combiner = walk_forward_v4(synthetic_games())
    assert metrics["holdout"]["games"] > 0
    assert np.isfinite(metrics["holdout"]["log_loss"])
    assert len(predictions) == len(synthetic_games())
    assert combiner is not None


def test_predictor_accepts_v4_bundle_without_legacy_k_features() -> None:
    bundle = {
        "model_version": "v4-test",
        "model_family": "whl_v4_dynamic_hierarchical",
        "v4_meta_model": None,
    }
    payload = {
        "home_team_id": 1,
        "away_team_id": 2,
        "context_features": {
            "v4_rating_home_prob": 0.75,
            "v4_goal_home_prob": 0.65,
        },
    }
    result = predict_from_payload(payload, bundle=bundle)
    assert result["model_version"] == "v4-test"
    assert result["predicted_winner_id"] == 1
    assert 0.69 < result["home_team_prob"] < 0.71


def test_v4_artifact_does_not_promote_by_default(tmp_path) -> None:
    _, _, combiner = walk_forward_v4(synthetic_games())
    output = save_v4_artifacts(
        tmp_path,
        "test-v4",
        combiner,
        {"holdout": {"log_loss": 0.5}},
    )
    assert (output / "model_bundle.joblib").exists()
    assert not (tmp_path / "active_model.json").exists()
