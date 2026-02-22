from __future__ import annotations

from scripts.repair_chl_games_data import _CandidateRow, _is_final_status, _is_non_final_placeholder_row, _is_zero_like


def _candidate(**overrides):
    base = _CandidateRow(
        game_id=1,
        season_id="83",
        season_name="2025-26 Regular Season",
        venue="Arena",
        status="7:00 pm ",
        home_goal_count=0,
        away_goal_count=0,
        scoring_breakdown={"home": {"1": "0"}, "visiting": {"1": "0"}},
        shots_on_goal={"home": {"1": 0}, "visiting": {"1": 0}},
        power_play={"total": {"home": "0", "visiting": "0"}, "goals": {"home": "0", "visiting": "0"}},
        fow={"home": 0, "visiting": 0},
        home_power_play_percentage=0,
        away_power_play_percentage=0,
        home_faceoff_win_percentage=0,
        away_faceoff_win_percentage=0,
        home_shots_on_goal_total=0,
        away_shots_on_goal_total=0,
        scheduled_time_utc=None,
        scorebar_snapshot=None,
    )
    if not overrides:
        return base
    return _CandidateRow(**{**base.__dict__, **overrides})


def test_is_final_status_handles_code_and_text():
    assert _is_final_status("4")
    assert _is_final_status("Final")
    assert _is_final_status("completed")
    assert not _is_final_status("7:00 pm ")
    assert not _is_final_status("In Progress")


def test_is_zero_like_nested_structures():
    assert _is_zero_like({"home": {"1": "0"}, "visiting": {"1": 0}})
    assert _is_zero_like({"total": {"home": "0", "visiting": "0"}})
    assert not _is_zero_like({"home": {"1": "1"}})


def test_non_final_placeholder_row_detected():
    row = _candidate()
    assert _is_non_final_placeholder_row(row)


def test_non_final_with_real_stats_not_placeholder():
    row = _candidate(home_goal_count=1)
    assert not _is_non_final_placeholder_row(row)


def test_final_row_not_placeholder_even_if_zeroes():
    row = _candidate(status="Final")
    assert not _is_non_final_placeholder_row(row)

