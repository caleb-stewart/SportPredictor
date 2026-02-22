from __future__ import annotations

from services.chl_history_ingest import (
    _clock_has_core_fields,
    _merge_clock_with_gamesummary,
    _season_kind,
    _select_target_seasons,
)


def test_season_kind_classification():
    assert _season_kind("2025 - 26 Regular Season") == "regular"
    assert _season_kind("2025 WHL Playoffs") == "playoff"
    assert _season_kind("2025 - 26 Pre-Season") == "preseason"
    assert _season_kind("Prospects Game") is None


def test_select_target_seasons_filters_and_sorts_desc():
    seasons = [
        {"season_id": "10", "season_name": "2024 - 25 Regular Season"},
        {"season_id": "11", "season_name": "2025 WHL Playoffs"},
        {"season_id": "12", "season_name": "2025 - 26 Pre-Season"},
        {"season_id": "13", "season_name": "Prospects Game"},
    ]

    selected = _select_target_seasons(
        seasons,
        include_regular=True,
        include_playoffs=False,
        include_preseason=True,
        all_available=True,
    )

    assert [row["season_id"] for row in selected] == ["12", "10"]


def test_merge_clock_with_gamesummary_backfills_missing_core_fields():
    clock = {
        "status": "4",
        "progress": "Final",
        "home_goal_count": "7",
        "visiting_goal_count": "6",
    }
    gamesummary = {
        "meta": {"period": "3", "game_number": "517", "season_id": "38"},
        "goalsByPeriod": {
            "home": {"1": 4, "2": 2, "3": 1},
            "visitor": {"1": 1, "2": 4, "3": 1},
        },
        "shotsByPeriod": {
            "home": {"1": 22, "2": 17, "3": 13},
            "visitor": {"1": 10, "2": 12, "3": 12},
        },
        "powerPlayCount": {"home": 4, "visitor": 4},
        "powerPlayGoals": {"home": 1, "visitor": 2},
        "totalFaceoffs": {"home": {"won": 0}, "visitor": {"won": 0}},
        "home": {"name": "Kitchener Rangers", "team_id": "10"},
        "visitor": {"name": "Oshawa Generals", "team_id": "4"},
        "venue": "Kitchener Memorial Auditorium",
        "status_value": "Final",
    }

    merged = _merge_clock_with_gamesummary(clock, gamesummary)

    assert merged["scoring"]["home"]["1"] == "4"
    assert merged["shots_on_goal"]["visiting"]["3"] == 12
    assert merged["power_play"]["goals"]["home"] == "1"
    assert merged["period"] == "3"
    assert merged["game_number"] == "517"
    assert merged["season_id"] == "38"
    assert merged["venue"] == "Kitchener Memorial Auditorium"
    assert merged.get("fow") in (None, {})
    assert _clock_has_core_fields(merged)
