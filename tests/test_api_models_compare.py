from __future__ import annotations

from fastapi.testclient import TestClient

from db.session import get_db
from main import app


class _FakeSession:
    pass


def _fake_get_db():
    yield _FakeSession()


def test_models_compare_run_endpoint(monkeypatch):
    from api.routes import models as models_route

    monkeypatch.setattr(
        models_route,
        "run_model_compare",
        lambda **_kwargs: {
            "run_id": "11111111-1111-1111-1111-111111111111",
            "status": "completed",
            "mode": "frozen_replay",
            "baseline_model_version": "base-v1",
            "candidate_model_version": "cand-v2",
            "games_scanned": 100,
            "games_compared": 95,
            "proof_summary": {"proved_better": True},
        },
    )
    app.dependency_overrides[get_db] = _fake_get_db
    client = TestClient(app)
    try:
        response = client.post(
            "/models/compare/run",
            json={
                "candidate_model_version": "cand-v2",
                "baseline_model_version": "base-v1",
                "mode": "frozen_replay",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["proof_summary"]["proved_better"] is True


def test_models_compare_report_endpoint(monkeypatch):
    from api.routes import models as models_route

    monkeypatch.setattr(
        models_route,
        "get_model_compare_report",
        lambda **_kwargs: {
            "run_id": "33333333-3333-3333-3333-333333333333",
            "status": "completed",
            "mode": "frozen_replay",
            "baseline_model_version": "base-v1",
            "candidate_model_version": "cand-v2",
            "date_from": "2024-01-01",
            "date_to": "2024-12-31",
            "started_at": None,
            "completed_at": None,
            "games_scanned": 100,
            "games_compared": 95,
            "proof_summary": {"proved_better": True},
            "error_text": None,
        },
    )
    app.dependency_overrides[get_db] = _fake_get_db
    client = TestClient(app)
    try:
        response = client.get("/models/compare/report/33333333-3333-3333-3333-333333333333")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "33333333-3333-3333-3333-333333333333"
    assert body["proof_summary"]["proved_better"] is True


def test_experiments_feature_proposal_endpoint(monkeypatch):
    from api.routes import experiments as experiments_route

    monkeypatch.setattr(
        experiments_route,
        "run_feature_proposal_experiment",
        lambda **_kwargs: {
            "experiment_id": "22222222-2222-2222-2222-222222222222",
            "status": "completed",
            "experiment_type": "feature_proposal",
            "result": {"verdict": {"accepted": True}},
        },
    )
    app.dependency_overrides[get_db] = _fake_get_db
    client = TestClient(app)
    try:
        response = client.post(
            "/experiments/feature-proposal/run",
            json={
                "proposal_name": "test",
                "seed": 42,
                "proposals": [
                    {
                        "name": "x_new",
                        "left_feature": "goals_diff_diff",
                        "op": "add",
                        "right_feature": "sog_diff_diff",
                    }
                ],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["experiment_type"] == "feature_proposal"
    assert body["result"]["verdict"]["accepted"] is True
