from __future__ import annotations

import numpy as np
import pytest

from services import predictor


class ConstantModel:
    def __init__(self, p: float) -> None:
        self.p = float(p)

    def predict_proba(self, x):
        rows = len(x)
        return np.column_stack([np.full(rows, 1 - self.p), np.full(rows, self.p)])


def base_stats(goals_for: float, goals_against: float, shots_for: float) -> dict[str, float]:
    return {
        "goals_for_avg": goals_for,
        "goals_against_avg": goals_against,
        "shots_for_avg": shots_for,
        "shots_against_avg": 25.0,
        "power_play_percentage_avg": 0.2,
        "power_play_percentage_against_avg": 0.19,
        "faceoff_win_percentage_avg": 0.51,
        "faceoff_win_percentage_against_avg": 0.49,
        "goals_diff": goals_for - goals_against,
        "ppp_diff": 0.01,
        "sog_diff": 2.0,
        "fowp_diff": 0.02,
    }


def test_predictor_returns_ensemble_and_components(monkeypatch):
    bundle = {
        "model_version": "test-v2",
        "model_family": "whl_v2_hybrid_logistic_stacker",
        "base_models": {
            "5": ConstantModel(0.61),
            "10": ConstantModel(0.63),
            "15": ConstantModel(0.65),
        },
        "meta_model": ConstantModel(0.64),
    }

    monkeypatch.setattr(predictor, "load_active_model", lambda: bundle)

    payload = {
        "home_team_id": 215,
        "away_team_id": 206,
        "features_by_k": {
            "5": {"home": base_stats(3.9, 2.1, 32.0), "away": base_stats(3.0, 2.9, 28.0)},
            "10": {"home": base_stats(3.7, 2.3, 31.0), "away": base_stats(3.1, 2.8, 29.0)},
            "15": {"home": base_stats(3.5, 2.5, 30.0), "away": base_stats(2.9, 2.9, 28.0)},
        },
    }

    out = predictor.predict_from_payload(payload)

    assert out["model_version"] == "test-v2"
    assert out["predicted_winner_id"] == 215
    assert pytest.approx(out["home_team_prob"], rel=1e-6) == 0.64
    assert set(out["k_components"].keys()) == {"5", "10", "15"}


def test_predictor_missing_k_raises(monkeypatch):
    bundle = {
        "model_version": "test-v2",
        "model_family": "whl_v2_hybrid_logistic_stacker",
        "base_models": {"5": ConstantModel(0.61), "10": ConstantModel(0.63), "15": ConstantModel(0.65)},
        "meta_model": ConstantModel(0.64),
    }
    monkeypatch.setattr(predictor, "load_active_model", lambda: bundle)

    payload = {
        "home_team_id": 215,
        "away_team_id": 206,
        "features_by_k": {
            "5": {"home": base_stats(3.9, 2.1, 32.0), "away": base_stats(3.0, 2.9, 28.0)},
            "10": {"home": base_stats(3.7, 2.3, 31.0), "away": base_stats(3.1, 2.8, 29.0)},
        },
    }

    with pytest.raises(predictor.PayloadContractError):
        predictor.predict_from_payload(payload)
