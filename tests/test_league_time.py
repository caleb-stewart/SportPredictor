from __future__ import annotations

import datetime as dt

from services.league_time import league_today


def test_league_today_uses_configured_league_timezone():
    now_utc = dt.datetime(2026, 2, 22, 7, 30, tzinfo=dt.timezone.utc)

    assert league_today("whl", now_utc=now_utc) == dt.date(2026, 2, 21)
    assert league_today("ohl", now_utc=now_utc) == dt.date(2026, 2, 22)


def test_league_today_falls_back_to_utc_for_unknown_league():
    now_utc = dt.datetime(2026, 2, 22, 7, 30, tzinfo=dt.timezone.utc)
    assert league_today("unknown", now_utc=now_utc) == dt.date(2026, 2, 22)
