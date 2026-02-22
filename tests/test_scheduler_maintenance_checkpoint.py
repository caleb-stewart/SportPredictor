from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

from jobs import scheduler


def test_job_predict_upcoming_records_maintenance_checkpoint(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        scheduler,
        "run_upcoming_predictions",
        lambda _db, _target, league_code: SimpleNamespace(
            predictions_written=7,
            skipped_games=2,
            league_code=league_code,
        ),
    )

    def _record_checkpoint(db, *, league_code, run_date, totals):
        captured["db"] = db
        captured["league_code"] = league_code
        captured["run_date"] = run_date
        captured["totals"] = totals
        return {"run_id": "demo-run", "to_date": run_date}

    monkeypatch.setattr(scheduler, "record_scheduler_maintenance_success", _record_checkpoint)

    db = object()
    scheduler._job_predict_upcoming(db, "ohl")

    assert captured["db"] is db
    assert captured["league_code"] == "ohl"
    assert isinstance(captured["run_date"], dt.date)
    assert captured["totals"] == {"predictions_written": 7, "skipped_games": 2}
