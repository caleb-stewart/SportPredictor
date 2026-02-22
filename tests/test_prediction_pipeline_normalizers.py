from __future__ import annotations

import datetime as dt

from services.prediction_pipeline import build_clock_game_upsert_values, build_schedule_game_upsert_values


def test_build_schedule_game_upsert_values_handles_modulekit_schedule_shape():
    now = dt.datetime(2026, 2, 21, tzinfo=dt.UTC)
    game_data = {
        "id": "1022126",
        "season_id": "289",
        "season_name": "2025 - 26 Regular Season",
        "GameDateISO8601": "2025-09-19T19:00:00-05:00",
        "venue_name": "Arena",
        "status": "4",
        "game_status": "Final",
        "home_goal_count": "4",
        "visiting_goal_count": "6",
        "game_number": "123",
        "period": "3",
        "home_team": "201",
        "visiting_team": "207",
        "home_team_name": "Brandon Wheat Kings",
        "visiting_team_name": "Moose Jaw Warriors",
    }

    values, update_values = build_schedule_game_upsert_values(game_data=game_data, now=now)

    assert values["game_id"] == 1022126
    assert values["season_id"] == "289"
    assert values["home_team_id"] == 201
    assert values["away_team_id"] == 207
    assert values["game_date"] == dt.date(2025, 9, 19)
    assert values["status"] == "Final"
    assert values["home_goal_count"] == 4
    assert values["away_goal_count"] == 6
    assert values["game_number"] == 123
    assert values["period"] == "3"
    assert "created_at" in values
    assert "created_at" not in update_values


def test_build_clock_game_upsert_values_hydrates_boxscore_fields():
    now = dt.datetime(2026, 2, 21, tzinfo=dt.UTC)
    game_data = {
        "ID": "1022126",
        "SeasonID": "289",
        "SeasonName": "2025 - 26 Regular Season",
        "GameDateISO8601": "2025-09-19T19:00:00-05:00",
        "HomeID": "201",
        "VisitorID": "207",
        "HomeLongName": "Brandon Wheat Kings",
        "VisitorLongName": "Moose Jaw Warriors",
        "venue_name": "Arena",
    }
    clock = {
        "season_id": "289",
        "season_name": "2025 - 26 Regular Season",
        "game_date_iso_8601": "2025-09-19T19:00:00-05:00",
        "home_team": {"team_id": "201", "name": "Brandon Wheat Kings"},
        "visiting_team": {"team_id": "207", "name": "Moose Jaw Warriors"},
        "home_goal_count": "4",
        "visiting_goal_count": "6",
        "game_number": "1",
        "period": "3",
        "progress": "Final",
        "scoring": {"home": {"1": "1"}, "visiting": {"1": "2"}},
        "shots_on_goal": {"home": {"1": 12}, "visiting": {"1": 15}},
        "power_play": {"total": {"home": "4", "visiting": "3"}, "goals": {"home": "1", "visiting": "0"}},
        "fow": {"home": 31, "visiting": 29},
    }

    values, _ = build_clock_game_upsert_values(game_data=game_data, clock=clock, now=now)

    assert values["home_goal_count"] == 4
    assert values["away_goal_count"] == 6
    assert values["home_shots_on_goal_total"] == 12
    assert values["away_shots_on_goal_total"] == 15
    assert values["home_power_play_percentage"] == 0.25
    assert values["away_power_play_percentage"] == 0.0


def test_build_schedule_game_upsert_values_uses_season_name_fallback():
    now = dt.datetime(2026, 2, 21, tzinfo=dt.UTC)
    game_data = {
        "id": "1023001",
        "season_id": "83",
        "GameDateISO8601": "2026-03-22T19:00:00-05:00",
        "status": "2:07 pm ",
        "home_team": "1",
        "visiting_team": "6",
    }

    values, _ = build_schedule_game_upsert_values(
        game_data=game_data,
        now=now,
        season_name_fallback="2025-26 Regular Season",
    )

    assert values["season_name"] == "2025-26 Regular Season"


def test_build_clock_game_upsert_values_clears_nonfinal_placeholder_stats():
    now = dt.datetime(2026, 2, 21, tzinfo=dt.UTC)
    game_data = {
        "id": "1023001",
        "season_id": "83",
        "GameDateISO8601": "2026-03-22T19:00:00-05:00",
        "status": "2:07 pm ",
        "home_team": "1",
        "visiting_team": "6",
    }
    clock = {
        "season_id": "83",
        "season_name": "2025-26 Regular Season",
        "status": "1",
        "progress": " 2:07 PM",
        "home_team": {"team_id": "1", "name": "Hamilton Bulldogs"},
        "visiting_team": {"team_id": "6", "name": "Peterborough Petes"},
        "home_goal_count": "0",
        "visiting_goal_count": "0",
        "game_number": "518",
        "period": "1",
        "scoring": {"home": {"1": "0"}, "visiting": {"1": "0"}},
        "shots_on_goal": {"home": {"1": 0}, "visiting": {"1": 0}},
        "power_play": {"total": {"home": "0", "visiting": "0"}, "goals": {"home": "0", "visiting": "0"}},
        "fow": {"home": 0, "visiting": 0},
    }

    values, _ = build_clock_game_upsert_values(game_data=game_data, clock=clock, now=now)

    assert values["home_goal_count"] is None
    assert values["away_goal_count"] is None
    assert values["scoring_breakdown"] is None
    assert values["shots_on_goal"] is None
    assert values["power_play"] is None
    assert values["fow"] is None
    assert values["home_shots_on_goal_total"] is None
    assert values["away_shots_on_goal_total"] is None
