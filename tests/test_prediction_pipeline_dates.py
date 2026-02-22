from __future__ import annotations

import datetime as dt

from services.prediction_pipeline import prediction_date_bounds


def test_prediction_date_bounds_are_naive_datetimes():
    start, end = prediction_date_bounds(date_from=dt.date(2026, 2, 16), date_to=dt.date(2026, 2, 17))

    assert start == dt.datetime(2026, 2, 16, 0, 0, 0)
    assert end == dt.datetime(2026, 2, 17, 23, 59, 59, 999999)
    assert start.tzinfo is None
    assert end.tzinfo is None


def test_prediction_date_bounds_handle_open_ended_ranges():
    start, end = prediction_date_bounds(date_from=None, date_to=dt.date(2026, 2, 17))
    assert start is None
    assert end == dt.datetime(2026, 2, 17, 23, 59, 59, 999999)
