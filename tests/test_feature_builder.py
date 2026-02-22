from __future__ import annotations

import datetime as dt

from services.feature_builder import _latest_rolling_average_stmt


def test_latest_rolling_average_stmt_uses_strict_less_than_for_game_date():
    stmt = _latest_rolling_average_stmt(team_db_id=1, k_value=5, game_date=dt.date(2026, 2, 15))
    sql = str(stmt)

    assert "chl_games.game_date <" in sql
