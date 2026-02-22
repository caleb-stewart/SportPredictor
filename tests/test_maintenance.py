from __future__ import annotations

import datetime as dt

from services.maintenance import compute_catchup_window


def test_compute_catchup_window_without_checkpoint_runs_today_only():
    start, end, days = compute_catchup_window(
        today=dt.date(2026, 2, 22),
        last_successful_run_date=None,
        max_catchup_days=14,
    )

    assert start == dt.date(2026, 2, 22)
    assert end == dt.date(2026, 2, 22)
    assert days == 1


def test_compute_catchup_window_uses_checkpoint_plus_one_day():
    start, end, days = compute_catchup_window(
        today=dt.date(2026, 2, 22),
        last_successful_run_date=dt.date(2026, 2, 20),
        max_catchup_days=14,
    )

    assert start == dt.date(2026, 2, 21)
    assert end == dt.date(2026, 2, 22)
    assert days == 2


def test_compute_catchup_window_caps_large_gap_to_max_days():
    start, end, days = compute_catchup_window(
        today=dt.date(2026, 2, 22),
        last_successful_run_date=dt.date(2026, 1, 20),
        max_catchup_days=14,
    )

    assert start == dt.date(2026, 2, 9)
    assert end == dt.date(2026, 2, 22)
    assert days == 14


def test_compute_catchup_window_returns_zero_when_already_current():
    start, end, days = compute_catchup_window(
        today=dt.date(2026, 2, 22),
        last_successful_run_date=dt.date(2026, 2, 22),
        max_catchup_days=14,
    )

    assert start is None
    assert end is None
    assert days == 0
