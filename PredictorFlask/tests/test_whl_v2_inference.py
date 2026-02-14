import unittest

import numpy as np

from whl_v2_inference import predict_from_payload


class ConstantModel:
    def __init__(self, p):
        self.p = float(p)

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.p), np.full(n, self.p)])


class InferenceContractTests(unittest.TestCase):
    def test_predict_from_payload_returns_ensemble_and_components(self):
        payload = {
            "home_team_id": 215,
            "away_team_id": 206,
            "features_by_k": {
                "5": {
                    "home": base_stats(4.0, 2.0, 33.0),
                    "away": base_stats(3.0, 3.0, 29.0),
                },
                "10": {
                    "home": base_stats(3.7, 2.4, 32.0),
                    "away": base_stats(3.1, 2.9, 30.0),
                },
                "15": {
                    "home": base_stats(3.5, 2.5, 31.0),
                    "away": base_stats(3.0, 2.8, 29.0),
                },
            },
        }

        bundle = {
            "model_version": "test-v2",
            "model_family": "whl_v2_hybrid_logistic_stacker",
            "feature_columns": [f"f_{i}" for i in range(24)],
            "base_models": {
                "5": ConstantModel(0.61),
                "10": ConstantModel(0.63),
                "15": ConstantModel(0.65),
            },
            "meta_model": ConstantModel(0.64),
        }

        # Replace engineered vector ordering dependency for this unit test.
        bundle["feature_columns"] = [
            "goals_for_avg_diff",
            "goals_against_avg_diff",
            "shots_for_avg_diff",
            "shots_against_avg_diff",
            "power_play_percentage_avg_diff",
            "power_play_percentage_against_avg_diff",
            "faceoff_win_percentage_avg_diff",
            "faceoff_win_percentage_against_avg_diff",
            "goals_diff_diff",
            "ppp_diff_diff",
            "sog_diff_diff",
            "fowp_diff_diff",
            "goals_for_avg_sum",
            "goals_against_avg_sum",
            "shots_for_avg_sum",
            "shots_against_avg_sum",
            "power_play_percentage_avg_sum",
            "power_play_percentage_against_avg_sum",
            "faceoff_win_percentage_avg_sum",
            "faceoff_win_percentage_against_avg_sum",
            "goals_diff_sum",
            "ppp_diff_sum",
            "sog_diff_sum",
            "fowp_diff_sum",
        ]

        out = predict_from_payload(payload, bundle)

        self.assertAlmostEqual(out["home_team_prob"], 0.64)
        self.assertAlmostEqual(out["away_team_prob"], 0.36)
        self.assertEqual(out["predicted_winner_id"], 215)
        self.assertIn("5", out["k_components"])
        self.assertIn("10", out["k_components"])
        self.assertIn("15", out["k_components"])


def base_stats(goals_for, goals_against, shots_for):
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


if __name__ == "__main__":
    unittest.main()
