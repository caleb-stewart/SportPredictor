from __future__ import annotations

import datetime as dt

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
    assert payload["days"][2]["games_count"] == 1
