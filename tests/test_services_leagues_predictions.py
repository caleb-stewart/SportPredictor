from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

from services import leagues as leagues_service


def _game_item(
    *,
    game_id: int,
    game_date: dt.date,
    scheduled_time: dt.datetime,
    consensus_correct: bool | None,
    resolved_count: int,
    correct_count: int,
    prediction_rows: int = 3,
) -> dict:
    return {
        "league_code": "whl",
        "game_id": game_id,
        "game_date": game_date,
        "prediction_timestamp": dt.datetime(2026, 2, 16, 12, 0, 0),
        "prediction_date": dt.date(2026, 2, 16),
        "status": "4" if consensus_correct is not None else "1",
        "period": "3" if consensus_correct is not None else "1",
        "venue": "Test Arena",
        "scheduled_time_utc": scheduled_time,
        "final_score": {"home": 2, "away": 1},
        "home_team": {
            "db_team_id": 1,
            "provider_team_id": 101,
            "name": "Home",
            "city": None,
            "conference": None,
            "division": None,
            "logo_url": None,
            "active": True,
        },
        "away_team": {
            "db_team_id": 2,
            "provider_team_id": 202,
            "name": "Away",
            "city": None,
            "conference": None,
            "division": None,
            "logo_url": None,
            "active": True,
        },
        "predictions_by_k": [
            {
                "k_value": 5,
                "home_team_probability": 0.55,
                "away_team_probability": 0.45,
                "predicted_winner_db_team_id": 1,
                "predicted_winner_provider_team_id": 101,
                "predicted_winner_name": "Home",
                "actual_winner_db_team_id": 1,
                "actual_winner_provider_team_id": 101,
                "actual_winner_name": "Home",
                "correct": consensus_correct,
                "model_version": "v1",
                "model_family": "test",
                "raw_model_outputs": {},
            }
        ]
        * prediction_rows,
        "consensus_k_value": 15,
        "consensus_predicted_winner_db_team_id": 1,
        "consensus_predicted_winner_provider_team_id": 101,
        "consensus_predicted_winner_name": "Home",
        "consensus_correct": consensus_correct,
        "resolved_count": resolved_count,
        "correct_count": correct_count,
    }


def test_league_today_respects_league_timezones():
    now_utc = dt.datetime(2026, 2, 22, 7, 30, tzinfo=dt.timezone.utc)

    # 07:30 UTC is still previous day in Pacific time.
    assert leagues_service._league_today("whl", now_utc=now_utc) == dt.date(2026, 2, 21)
    assert leagues_service._league_today("ohl", now_utc=now_utc) == dt.date(2026, 2, 22)


def test_get_next_predicted_slate_selects_earliest_future_game_date(monkeypatch):
    items = [
        _game_item(
            game_id=20,
            game_date=dt.date(2026, 2, 23),
            scheduled_time=dt.datetime(2026, 2, 23, 4, 0, 0),
            consensus_correct=None,
            resolved_count=0,
            correct_count=0,
        ),
        _game_item(
            game_id=11,
            game_date=dt.date(2026, 2, 22),
            scheduled_time=dt.datetime(2026, 2, 22, 4, 0, 0),
            consensus_correct=None,
            resolved_count=0,
            correct_count=0,
        ),
        _game_item(
            game_id=10,
            game_date=dt.date(2026, 2, 22),
            scheduled_time=dt.datetime(2026, 2, 22, 2, 0, 0),
            consensus_correct=None,
            resolved_count=0,
            correct_count=0,
        ),
        _game_item(
            game_id=9,
            game_date=dt.date(2026, 2, 21),
            scheduled_time=dt.datetime(2026, 2, 21, 2, 0, 0),
            consensus_correct=True,
            resolved_count=3,
            correct_count=2,
        ),
    ]

    monkeypatch.setattr(leagues_service, "_query_prediction_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(leagues_service, "_aggregate_predicted_games", lambda rows, league_code: items)

    payload = leagues_service.get_next_predicted_slate(
        db=object(),
        league_code="whl",
        as_of_date=dt.date(2026, 2, 22),
        team_provider_id=None,
    )

    assert payload["target_game_date"] == dt.date(2026, 2, 22)
    assert payload["games_count"] == 2
    assert payload["games"][0]["game_id"] == 10
    assert payload["games"][1]["game_id"] == 11


def test_list_recent_result_days_groups_latest_distinct_game_dates(monkeypatch):
    items = [
        _game_item(
            game_id=101,
            game_date=dt.date(2026, 2, 21),
            scheduled_time=dt.datetime(2026, 2, 21, 3, 0, 0),
            consensus_correct=True,
            resolved_count=3,
            correct_count=2,
        ),
        _game_item(
            game_id=102,
            game_date=dt.date(2026, 2, 20),
            scheduled_time=dt.datetime(2026, 2, 20, 3, 0, 0),
            consensus_correct=False,
            resolved_count=3,
            correct_count=1,
        ),
        _game_item(
            game_id=103,
            game_date=dt.date(2026, 2, 19),
            scheduled_time=dt.datetime(2026, 2, 19, 3, 0, 0),
            consensus_correct=True,
            resolved_count=3,
            correct_count=3,
        ),
        _game_item(
            game_id=104,
            game_date=dt.date(2026, 2, 19),
            scheduled_time=dt.datetime(2026, 2, 19, 4, 0, 0),
            consensus_correct=None,
            resolved_count=0,
            correct_count=0,
        ),
        _game_item(
            game_id=105,
            game_date=dt.date(2026, 2, 22),
            scheduled_time=dt.datetime(2026, 2, 22, 3, 0, 0),
            consensus_correct=True,
            resolved_count=3,
            correct_count=3,
        ),
    ]

    monkeypatch.setattr(leagues_service, "_query_prediction_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(leagues_service, "_aggregate_predicted_games", lambda rows, league_code: items)

    payload = leagues_service.list_recent_result_days(
        db=object(),
        league_code="whl",
        days=3,
        team_provider_id=None,
        as_of_date=dt.date(2026, 2, 21),
    )

    assert payload["as_of_date"] == dt.date(2026, 2, 21)
    assert len(payload["days"]) == 3
    assert [row["game_date"] for row in payload["days"]] == [
        dt.date(2026, 2, 21),
        dt.date(2026, 2, 20),
        dt.date(2026, 2, 19),
    ]
    assert payload["days"][0]["resolved_count"] == 1
    assert payload["days"][0]["correct_count"] == 1
    assert payload["days"][0]["accuracy_pct"] == 100.0
    assert payload["days"][1]["resolved_count"] == 1
    assert payload["days"][1]["correct_count"] == 0
    assert payload["days"][1]["accuracy_pct"] == 0.0
    assert payload["days"][2]["games_count"] == 1


def test_list_prediction_results_by_day_paginates_three_day_groups(monkeypatch):
    day_keys = [
        dt.date(2026, 2, 20),
        dt.date(2026, 2, 19),
        dt.date(2026, 2, 18),
    ]

    monkeypatch.setattr(leagues_service, "_count_distinct_result_days", lambda db, base_stmt: 7)
    monkeypatch.setattr(
        leagues_service,
        "_fetch_result_day_keys",
        lambda db, base_stmt, days_per_page, offset: day_keys,
    )
    monkeypatch.setattr(
        leagues_service,
        "_fetch_consensus_game_ids_for_days",
        lambda db, base_stmt, day_keys: [501, 502, 503, 504],
    )
    monkeypatch.setattr(
        leagues_service,
        "_query_prediction_rows_for_game_ids",
        lambda db, league_id, game_ids: [],
    )
    monkeypatch.setattr(
        leagues_service,
        "_aggregate_predicted_games",
        lambda rows, league_code: [
            _game_item(
                game_id=502,
                game_date=dt.date(2026, 2, 20),
                scheduled_time=dt.datetime(2026, 2, 20, 5, 0, 0),
                consensus_correct=False,
                resolved_count=3,
                correct_count=0,
            ),
            _game_item(
                game_id=501,
                game_date=dt.date(2026, 2, 20),
                scheduled_time=dt.datetime(2026, 2, 20, 2, 0, 0),
                consensus_correct=True,
                resolved_count=3,
                correct_count=3,
            ),
            _game_item(
                game_id=503,
                game_date=dt.date(2026, 2, 19),
                scheduled_time=dt.datetime(2026, 2, 19, 3, 0, 0),
                consensus_correct=True,
                resolved_count=3,
                correct_count=2,
            ),
            _game_item(
                game_id=504,
                game_date=dt.date(2026, 2, 18),
                scheduled_time=dt.datetime(2026, 2, 18, 1, 0, 0),
                consensus_correct=False,
                resolved_count=3,
                correct_count=1,
            ),
        ],
    )

    payload = leagues_service.list_prediction_results_by_day(
        db=object(),
        league_code="whl",
        game_date_from=None,
        game_date_to=None,
        team_provider_id=None,
        result="all",
        page=2,
        days_per_page=3,
    )

    assert payload["league_code"] == "whl"
    assert payload["total_days"] == 7
    assert payload["total_pages"] == 3
    assert payload["has_prev"] is True
    assert payload["has_more"] is True
    assert [day["game_date"] for day in payload["items"]] == day_keys
    assert [game["game_id"] for game in payload["items"][0]["games"]] == [501, 502]
    assert payload["items"][0]["resolved_count"] == 2
    assert payload["items"][0]["correct_count"] == 1
    assert payload["items"][0]["accuracy_pct"] == 50.0


def test_list_prediction_results_by_day_team_not_found_returns_empty(monkeypatch):
    monkeypatch.setattr(
        leagues_service,
        "_resolve_team_db_id_for_league",
        lambda db, league_id, team_provider_id: None,
    )

    payload = leagues_service.list_prediction_results_by_day(
        db=object(),
        league_code="whl",
        game_date_from=None,
        game_date_to=None,
        team_provider_id=9999,
        result="all",
        page=1,
        days_per_page=3,
    )

    assert payload["items"] == []
    assert payload["total_days"] == 0
    assert payload["total_pages"] == 0


def test_list_prediction_results_by_day_rejects_invalid_result():
    try:
        leagues_service.list_prediction_results_by_day(
            db=object(),
            league_code="whl",
            game_date_from=None,
            game_date_to=None,
            team_provider_id=None,
            result="pending",
            page=1,
            days_per_page=3,
        )
    except ValueError as exc:
        assert "Unsupported result filter" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid result")


def _team_row(team_id: int, provider_team_id: int, name: str):
    return SimpleNamespace(
        id=team_id,
        hockeytech_id=provider_team_id,
        name=name,
        city=None,
        conference=None,
        division=None,
        logo_url=None,
        active=True,
    )


def test_choose_consensus_prediction_prefers_k15_then_highest_k():
    rows_no_k15 = [
        {"k_value": 5, "predicted_winner_name": "Away"},
        {"k_value": 10, "predicted_winner_name": "Home"},
    ]
    rows_with_k15 = [
        {"k_value": 5, "predicted_winner_name": "Away"},
        {"k_value": 10, "predicted_winner_name": "Away"},
        {"k_value": 15, "predicted_winner_name": "Home"},
    ]

    selected_without_k15 = leagues_service._choose_consensus_prediction(rows_no_k15)
    selected_with_k15 = leagues_service._choose_consensus_prediction(rows_with_k15)

    assert selected_without_k15 is not None
    assert selected_without_k15["k_value"] == 10
    assert selected_with_k15 is not None
    assert selected_with_k15["k_value"] == 15


def test_build_team_performance_items_computes_metrics():
    teams = [
        _team_row(1, 215, "Spokane Chiefs"),
        _team_row(2, 222, "Wenatchee Wild"),
    ]
    consensus_rows = [
        {
            "game_id": 1001,
            "home_team_id": 1,
            "away_team_id": 2,
            "predicted_winner_id": 1,
            "actual_winner_id": 1,
            "correct": True,
        },
        {
            "game_id": 1002,
            "home_team_id": 1,
            "away_team_id": 2,
            "predicted_winner_id": 2,
            "actual_winner_id": 1,
            "correct": False,
        },
        {
            "game_id": 1003,
            "home_team_id": 1,
            "away_team_id": 2,
            "predicted_winner_id": 1,
            "actual_winner_id": None,
            "correct": None,
        },
    ]

    items = leagues_service._build_team_performance_items(
        teams=teams,
        league_code="whl",
        consensus_rows=consensus_rows,
    )
    by_provider_id = {item["team"]["provider_team_id"]: item for item in items}

    chiefs = by_provider_id[215]["stats"]
    wild = by_provider_id[222]["stats"]

    assert chiefs["games_with_predictions"] == 3
    assert chiefs["resolved_games"] == 2
    assert chiefs["pending_games"] == 1
    assert chiefs["correct_predictions"] == 1
    assert chiefs["accuracy_pct"] == 50.0
    assert chiefs["actual_wins"] == 2
    assert chiefs["actual_losses"] == 0
    assert chiefs["predicted_wins"] == 1
    assert chiefs["predicted_losses"] == 1
    assert chiefs["win_delta"] == -1

    assert wild["games_with_predictions"] == 3
    assert wild["resolved_games"] == 2
    assert wild["pending_games"] == 1
    assert wild["correct_predictions"] == 1
    assert wild["accuracy_pct"] == 50.0
    assert wild["actual_wins"] == 0
    assert wild["actual_losses"] == 2
    assert wild["predicted_wins"] == 1
    assert wild["predicted_losses"] == 1
    assert wild["win_delta"] == 1


def test_build_team_performance_items_accuracy_is_null_when_no_resolved_games():
    teams = [_team_row(1, 215, "Spokane Chiefs")]
    consensus_rows = [
        {
            "game_id": 2001,
            "home_team_id": 1,
            "away_team_id": 2,
            "predicted_winner_id": 1,
            "actual_winner_id": None,
            "correct": None,
        }
    ]

    items = leagues_service._build_team_performance_items(
        teams=teams,
        league_code="whl",
        consensus_rows=consensus_rows,
    )
    stats = items[0]["stats"]
    assert stats["resolved_games"] == 0
    assert stats["pending_games"] == 1
    assert stats["accuracy_pct"] is None


def test_team_performance_rejects_invalid_date_range():
    try:
        leagues_service.list_team_performance(
            db=object(),
            league_code="whl",
            date_from=dt.date(2026, 2, 20),
            date_to=dt.date(2025, 9, 1),
            active_only=True,
        )
    except ValueError as exc:
        assert "date_from cannot be after date_to" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid date range")


def test_get_team_performance_raises_for_missing_team():
    class _MissingTeamDb:
        def scalar(self, _stmt):
            return None

    try:
        leagues_service.get_team_performance(
            db=_MissingTeamDb(),
            league_code="whl",
            team_provider_id=9999,
            date_from=None,
            date_to=None,
        )
    except leagues_service.TeamNotFoundError as exc:
        assert "provider_team_id=9999" in str(exc)
    else:
        raise AssertionError("Expected TeamNotFoundError for unknown team")
