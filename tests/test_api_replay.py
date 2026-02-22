from __future__ import annotations

import datetime as dt

from fastapi.testclient import TestClient

from db.session import get_db
from main import app


class _FakeSession:
    pass


def _fake_get_db():
    yield _FakeSession()


def test_replay_run_endpoint_returns_payload(monkeypatch):
    def fake_run(**_kwargs):
        return {
            "run_id": "11111111-1111-1111-1111-111111111111",
            "status": "completed",
            "active_model_version": "20260217T025309Z",
            "games_scanned": 10,
            "games_predicted": 8,
            "games_skipped": 2,
            "rows_upserted": 24,
            "skip_reasons": {"insufficient_history": 2},
            "proof_summary": {"proved_better": True},
        }

    from api.routes import predictions as predictions_route

    monkeypatch.setattr(predictions_route, "run_frozen_model_replay", fake_run)
    app.dependency_overrides[get_db] = _fake_get_db
    client = TestClient(app)
    try:
        response = client.post(
            "/predictions/replay/run",
            json={
                "date_from": "2024-01-01",
                "date_to": "2024-12-31",
                "dry_run": True,
                "overwrite": False,
                "archive_label": "unit",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "11111111-1111-1111-1111-111111111111"
    assert body["status"] == "completed"


def test_replay_report_endpoint_returns_payload(monkeypatch):
    def fake_report(**_kwargs):
        return {
            "run_id": "11111111-1111-1111-1111-111111111111",
            "status": "completed",
            "date_from": dt.date(2024, 1, 1),
            "date_to": dt.date(2024, 12, 31),
            "started_at": None,
            "completed_at": None,
            "active_model_version": "20260217T025309Z",
            "games_scanned": 10,
            "games_predicted": 8,
            "games_skipped": 2,
            "rows_upserted": 24,
            "skip_reasons": {"insufficient_history": 2},
            "proof_summary": {"proved_better": True},
            "error_text": None,
        }

    from api.routes import predictions as predictions_route

    monkeypatch.setattr(predictions_route, "get_replay_report", fake_report)
    app.dependency_overrides[get_db] = _fake_get_db
    client = TestClient(app)
    try:
        response = client.get("/predictions/replay/report/11111111-1111-1111-1111-111111111111")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "11111111-1111-1111-1111-111111111111"
    assert body["proof_summary"]["proved_better"] is True


def test_replay_run_endpoint_accepts_last_n_selection(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {
            "run_id": "22222222-2222-2222-2222-222222222222",
            "status": "completed",
            "selection_mode": "last_n_completed_games",
            "last_n_games": 1500,
            "active_model_version": "20260222T002228Z",
            "games_scanned": 1500,
            "games_predicted": 1497,
            "games_skipped": 3,
            "rows_upserted": 4491,
            "skip_reasons": {"insufficient_history": 3},
            "proof_summary": {"proved_better": False},
        }

    from api.routes import predictions as predictions_route

    monkeypatch.setattr(predictions_route, "run_frozen_model_replay", fake_run)
    app.dependency_overrides[get_db] = _fake_get_db
    client = TestClient(app)
    try:
        response = client.post(
            "/predictions/replay/run",
            json={
                "league_code": "ohl",
                "selection_mode": "last_n_completed_games",
                "last_n_games": 1500,
                "dry_run": False,
                "overwrite": True,
                "rollback_on_proof_failure": False,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["selection_mode"] == "last_n_completed_games"
    assert body["last_n_games"] == 1500
    assert captured["selection_mode"] == "last_n_completed_games"
    assert captured["last_n_games"] == 1500
    assert captured["rollback_on_proof_failure"] is False
