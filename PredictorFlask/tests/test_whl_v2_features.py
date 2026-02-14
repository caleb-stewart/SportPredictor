import unittest

from whl_v2_features import build_pairwise_features, feature_vector_from_engineered


class FeatureBuilderTests(unittest.TestCase):
    def test_pairwise_diff_and_sum(self):
        home = {
            "goals_for_avg": 4,
            "goals_against_avg": 2,
            "shots_for_avg": 30,
            "shots_against_avg": 20,
            "power_play_percentage_avg": 0.22,
            "power_play_percentage_against_avg": 0.18,
            "faceoff_win_percentage_avg": 0.54,
            "faceoff_win_percentage_against_avg": 0.49,
            "goals_diff": 2,
            "ppp_diff": 0.04,
            "sog_diff": 10,
            "fowp_diff": 0.05,
        }
        away = {
            "goals_for_avg": 3,
            "goals_against_avg": 3,
            "shots_for_avg": 28,
            "shots_against_avg": 25,
            "power_play_percentage_avg": 0.19,
            "power_play_percentage_against_avg": 0.2,
            "faceoff_win_percentage_avg": 0.5,
            "faceoff_win_percentage_against_avg": 0.52,
            "goals_diff": 0,
            "ppp_diff": -0.01,
            "sog_diff": 3,
            "fowp_diff": -0.02,
        }

        engineered = build_pairwise_features(home, away)

        self.assertAlmostEqual(engineered["goals_for_avg_diff"], 1.0)
        self.assertAlmostEqual(engineered["goals_for_avg_sum"], 3.5)
        self.assertAlmostEqual(engineered["power_play_percentage_avg_diff"], 0.03)
        self.assertAlmostEqual(engineered["power_play_percentage_avg_sum"], 0.205)

    def test_feature_vector_is_stable_order(self):
        home = {
            "goals_for_avg": 1,
            "goals_against_avg": 2,
            "shots_for_avg": 3,
            "shots_against_avg": 4,
            "power_play_percentage_avg": 0.1,
            "power_play_percentage_against_avg": 0.2,
            "faceoff_win_percentage_avg": 0.3,
            "faceoff_win_percentage_against_avg": 0.4,
            "goals_diff": -1,
            "ppp_diff": -0.1,
            "sog_diff": -1,
            "fowp_diff": -0.1,
        }
        away = {
            "goals_for_avg": 0,
            "goals_against_avg": 1,
            "shots_for_avg": 2,
            "shots_against_avg": 3,
            "power_play_percentage_avg": 0.05,
            "power_play_percentage_against_avg": 0.1,
            "faceoff_win_percentage_avg": 0.2,
            "faceoff_win_percentage_against_avg": 0.25,
            "goals_diff": -1,
            "ppp_diff": -0.05,
            "sog_diff": -1,
            "fowp_diff": -0.05,
        }

        engineered = build_pairwise_features(home, away)
        vector = feature_vector_from_engineered(engineered)

        self.assertEqual(len(vector), 24)
        self.assertAlmostEqual(vector[0], 1.0)


if __name__ == "__main__":
    unittest.main()
