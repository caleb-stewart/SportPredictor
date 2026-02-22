from __future__ import annotations

import datetime as dt

from fastapi.testclient import TestClient

from db.session import get_db
from main import app


class _FakeSession:
    pass


def _fake_get_db():
    yield _FakeSession()


def _sample_predicted_game(consensus_correct: bool | None = True) -> dict:
    return {
        "league_code": "whl",
        "game_id": 1022738,
        "game_date": dt.date(2026, 2, 15),
        "prediction_timestamp": dt.datetime(2026, 2, 16, 19, 35, 38),
        "prediction_date": dt.date(2026, 2, 16),
        "status": "Final",
        "period": "3",
        "venue": "Town Toyota Center",
        "scheduled_time_utc": dt.datetime(2026, 2, 15, 16, 0, 0),
        "final_score": {"home": 2, "away": 4},
        "home_team": {
            "db_team_id": 23,
            "provider_team_id": 222,
            "name": "Wenatchee Wild",
            "city": "Wenatchee",
            "conference": "Western",
            "division": "U.S. Division",
            "logo_url": "https://assets.leaguestat.com/whl/logos/50x50/222.png",
            "active": True,
        },
        "away_team": {
            "db_team_id": 1,
            "provider_team_id": 215,
            "name": "Spokane Chiefs",
            "city": "Spokane",
            "conference": "Western",
            "division": "U.S. Division",
            "logo_url": "https://assets.leaguestat.com/whl/logos/215.png",
            "active": True,
        },
        "predictions_by_k": [
            {
                "k_value": 5,
                "home_team_probability": 0.35,
                "away_team_probability": 0.65,
                "predicted_winner_db_team_id": 1,
                "predicted_winner_provider_team_id": 215,
                "predicted_winner_name": "Spokane Chiefs",
                "actual_winner_db_team_id": 1,
                "actual_winner_provider_team_id": 215,
                "actual_winner_name": "Spokane Chiefs",
                "correct": consensus_correct,
                "model_version": "20260217T025309Z",
                "model_family": "whl_v2_hybrid_logistic_stacker",
                "raw_model_outputs": {"ensemble": {"home_team_prob": 0.35, "away_team_prob": 0.65}},
            }
        ],
        "consensus_k_value": 5,
        "consensus_predicted_winner_db_team_id": 1,
        "consensus_predicted_winner_provider_team_id": 215,
        "consensus_predicted_winner_name": "Spokane Chiefs",
        "consensus_correct": consensus_correct,
        "resolved_count": 0 if consensus_correct is None else 1,
        "correct_count": 1 if consensus_correct is True else 0,
    }


def test_leagues_endpoint_returns_catalog():
    client = TestClient(app)
    response = client.get("/leagues")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert body[0]["code"] == "whl"


def test_league_teams_endpoint_returns_normalized_team_payload(monkeypatch):
    from api.routes import leagues as leagues_route

    monkeypatch.setattr(
        leagues_route,
        "list_league_teams",
        lambda **_kwargs: [
            {
                "db_team_id": 1,
                "provider_team_id": 215,
                "name": "Spokane Chiefs",
                "city": "Spokane",
                "conference": "Western",
                "division": "U.S. Division",
                "logo_url": "https://assets.leaguestat.com/whl/logos/215.png",
                "active": True,
            }
        ],
    )

    app.dependency_overrides[get_db] = _fake_get_db
    client = TestClient(app)
    try:
        response = client.get("/leagues/whl/teams")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    team = response.json()[0]
    assert {"db_team_id", "provider_team_id", "logo_url", "name"} <= set(team.keys())


def test_predicted_games_endpoint_returns_nested_k_predictions(monkeypatch):
    from api.routes import leagues as leagues_route

    monkeypatch.setattr(
        leagues_route,
        "list_predicted_games",
        lambda **_kwargs: [_sample_predicted_game(consensus_correct=True)],
    )

    app.dependency_overrides[get_db] = _fake_get_db
    client = TestClient(app)
    try:
        response = client.get("/leagues/whl/predicted-games")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["predictions_by_k"][0]["k_value"] == 5
    assert body[0]["period"] == "3"


def test_next_slate_endpoint_returns_payload(monkeypatch):
    from api.routes import leagues as leagues_route

    monkeypatch.setattr(
        leagues_route,
        "get_next_predicted_slate",
        lambda **_kwargs: {
            "league_code": "whl",
            "as_of_date": dt.date(2026, 2, 20),
            "target_game_date": dt.date(2026, 2, 21),
            "games_count": 1,
            "predictions_count": 3,
            "games": [_sample_predicted_game(consensus_correct=None)],
        },
    )

    app.dependency_overrides[get_db] = _fake_get_db
    client = TestClient(app)
    try:
        response = client.get("/leagues/whl/predictions/next-slate")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["target_game_date"] == "2026-02-21"
    assert body["games_count"] == 1
    assert body["games"][0]["consensus_correct"] is None


def test_upcoming_predictions_endpoint_returns_paged_items(monkeypatch):
    from api.routes import leagues as leagues_route

    monkeypatch.setattr(
        leagues_route,
        "list_upcoming_predictions",
        lambda **_kwargs: {
            "items": [_sample_predicted_game(consensus_correct=None)],
            "total": 1,
            "limit": 20,
            "offset": 0,
            "has_more": False,
        },
    )

    app.dependency_overrides[get_db] = _fake_get_db
    client = TestClient(app)
    try:
        response = client.get("/leagues/whl/predictions/upcoming")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert len(body["items"]) == 1
    assert body["items"][0]["consensus_correct"] is None


def test_prediction_results_endpoint_returns_paged_items(monkeypatch):
    from api.routes import leagues as leagues_route

    monkeypatch.setattr(
        leagues_route,
        "list_prediction_results",
        lambda **_kwargs: {
            "items": [_sample_predicted_game(consensus_correct=False)],
            "total": 1,
            "limit": 20,
            "offset": 0,
            "has_more": False,
        },
    )

    app.dependency_overrides[get_db] = _fake_get_db
    client = TestClient(app)
    try:
        response = client.get("/leagues/whl/predictions/results")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["items"][0]["consensus_correct"] is False


def test_recent_result_days_endpoint_returns_grouped_days(monkeypatch):
    from api.routes import leagues as leagues_route

    monkeypatch.setattr(
        leagues_route,
        "list_recent_result_days",
        lambda **_kwargs: {
            "league_code": "whl",
            "as_of_date": dt.date(2026, 2, 22),
            "days": [
                {
                    "game_date": dt.date(2026, 2, 21),
                    "games_count": 1,
                    "resolved_count": 3,
                    "correct_count": 2,
                    "accuracy_pct": 66.67,
                    "games": [_sample_predicted_game(consensus_correct=True)],
                }
            ],
        },
    )

    app.dependency_overrides[get_db] = _fake_get_db
    client = TestClient(app)
    try:
        response = client.get("/leagues/whl/predictions/results/recent-days?days=3")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["days"][0]["game_date"] == "2026-02-21"
    assert body["days"][0]["accuracy_pct"] == 66.67


def test_daily_report_detail_endpoint_returns_summary_and_games(monkeypatch):
    from api.routes import leagues as leagues_route

    monkeypatch.setattr(
        leagues_route,
        "get_daily_report_detail",
        lambda **_kwargs: {
            "league_code": "whl",
            "prediction_date": dt.date(2026, 2, 16),
            "summary": {
                "prediction_date": dt.date(2026, 2, 16),
                "games_count": 2,
                "predictions_count": 6,
                "resolved_count": 6,
                "correct_count": 4,
                "accuracy_pct": 66.67,
            },
            "games": [],
        },
    )

    app.dependency_overrides[get_db] = _fake_get_db
    client = TestClient(app)
    try:
        response = client.get("/leagues/whl/reports/daily/2026-02-16")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["accuracy_pct"] == 66.67
    assert body["prediction_date"] == "2026-02-16"


def test_game_detail_endpoint_returns_prediction_breakdown(monkeypatch):
    from api.routes import leagues as leagues_route

    monkeypatch.setattr(
        leagues_route,
        "get_game_detail",
        lambda **_kwargs: {
            "league_code": "whl",
            "game_id": 1022738,
            "game_date": dt.date(2026, 2, 15),
            "status": "Final",
            "venue": "Town Toyota Center",
            "scheduled_time_utc": dt.datetime(2026, 2, 15, 16, 0, 0),
            "period": "3",
            "final_score": {"home": 2, "away": 4},
            "home_team": {
                "db_team_id": 23,
                "provider_team_id": 222,
                "name": "Wenatchee Wild",
                "city": "Wenatchee",
                "conference": "Western",
                "division": "U.S. Division",
                "logo_url": "https://assets.leaguestat.com/whl/logos/50x50/222.png",
                "active": True,
            },
            "away_team": {
                "db_team_id": 1,
                "provider_team_id": 215,
                "name": "Spokane Chiefs",
                "city": "Spokane",
                "conference": "Western",
                "division": "U.S. Division",
                "logo_url": "https://assets.leaguestat.com/whl/logos/215.png",
                "active": True,
            },
            "scoring_breakdown": {"home": {"1": "0"}, "visiting": {"1": "1"}},
            "shots_on_goal": {"home": {"1": 9}, "visiting": {"1": 11}},
            "power_play": {"total": {"home": "3", "visiting": "4"}},
            "fow": {"home": 30, "visiting": 28},
            "links": {
                "ticket_url": "https://tickets.example.com",
                "game_summary_url": "1022738",
                "home_audio_url": None,
                "away_audio_url": None,
                "home_video_url": None,
                "away_video_url": None,
                "home_webcast_url": None,
                "away_webcast_url": None,
            },
            "predictions_by_k": [],
            "consensus_k_value": None,
            "consensus_predicted_winner_db_team_id": None,
            "consensus_predicted_winner_provider_team_id": None,
            "consensus_predicted_winner_name": None,
            "consensus_correct": None,
        },
    )

    app.dependency_overrides[get_db] = _fake_get_db
    client = TestClient(app)
    try:
        response = client.get("/leagues/whl/games/1022738")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["game_id"] == 1022738
    assert "scoring_breakdown" in body
