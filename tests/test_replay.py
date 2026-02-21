from __future__ import annotations

import datetime as dt

from services.replay import build_proof_summary, completed_games_stmt


def test_completed_games_stmt_excludes_unfinished_games():
    stmt = completed_games_stmt(date_from=dt.date(2024, 1, 1), date_to=dt.date(2024, 12, 31))
    sql = str(stmt)

    assert "whl_games.home_goal_count IS NOT NULL" in sql
    assert "whl_games.away_goal_count IS NOT NULL" in sql
    assert "whl_games.game_date IS NOT NULL" in sql


def test_build_proof_summary_shows_statistical_improvement():
    paired_rows = []
    for i in range(600):
        y_true = 1 if i % 2 == 0 else 0
        k_value = [5, 10, 15][i % 3]
        old_prob = 0.35 if y_true == 1 else 0.65
        new_prob = 0.75 if y_true == 1 else 0.25
        paired_rows.append(
            {
                "game_id": 100000 + i,
                "k_value": k_value,
                "actual_home_win": y_true,
                "old_home_prob": old_prob,
                "new_home_prob": new_prob,
            }
        )

    proof = build_proof_summary(paired_rows=paired_rows, bootstrap_samples=400, bootstrap_seed=123)

    assert proof["overall"]["new"]["accuracy"] > proof["overall"]["old"]["accuracy"]
    assert proof["overall"]["new"]["log_loss"] < proof["overall"]["old"]["log_loss"]
    assert proof["stat_tests"]["mcnemar"]["p_value"] < 0.05
    assert proof["stat_tests"]["bootstrap_accuracy_delta"]["ci95_low"] > 0.0
    assert proof["proved_better"] is True
