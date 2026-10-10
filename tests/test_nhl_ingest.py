from __future__ import annotations

from datetime import date

from services.nhl_client import NhlClient, nhl_report_url
from services.nhl_ingest import normalize_nhl_completed_game, normalize_nhl_schedule_game, parse_nhl_report_metrics


def _team(team_id: int, place: str, common: str, abbrev: str) -> dict[str, object]:
    return {
        "id": team_id,
        "placeName": {"default": place},
        "commonName": {"default": common},
        "abbrev": abbrev,
        "logo": f"https://assets.nhle.com/logos/{abbrev}.svg",
    }


def test_report_url_uses_nhl_season_directory_and_game_suffix():
    assert nhl_report_url(2023020204, "GS", base_url="https://reports.example") == (
        "https://reports.example/20232024/GS020204.HTM"
    )
    assert nhl_report_url("2023020204", "fs", base_url="https://reports.example") == (
        "https://reports.example/20232024/FS020204.HTM"
    )


def test_schedule_client_extracts_game_week_payload_without_provider_specific_ids():
    client = NhlClient(requests_per_second=100)
    client._request_json = lambda path: {  # type: ignore[method-assign]
        "gameWeek": [
            {
                "date": "2023-11-10",
                "games": [
                    {
                        "id": 2023020204,
                        "gameDate": "2023-11-10",
                        "homeTeam": {"id": 7},
                        "awayTeam": {"id": 30},
                    }
                ],
            }
        ]
    }
    games = client.get_schedule_for_date(date(2023, 11, 10))
    assert [game["id"] for game in games] == [2023020204]


def test_schedule_client_does_not_accept_undated_fallback_games():
    client = NhlClient(requests_per_second=100)
    responses = iter(
        [
            {"gameWeek": []},
            {"games": [{"id": 2024020003, "homeTeam": {"id": 1}, "awayTeam": {"id": 2}}]},
        ]
    )
    client._request_json = lambda path: next(responses)  # type: ignore[method-assign]
    assert client.get_schedule_for_date(date(2023, 11, 10)) == []


def test_report_metrics_preserve_existing_power_play_and_faceoff_shapes():
    metrics = parse_nhl_report_metrics(
        "<td>Power Plays (Goals-Opp./PPTime)</td><td>2-5/06:41</td>"
        "<td>Power Plays (Goals-Opp./PPTime)</td><td>0-4/06:08</td>",
        "<td>TOT</td><td>10-20/50%</td><td>5-10/50%</td><td>5-19/26%</td><td>20-49/41%</td>"
        "<td>TOT</td><td>12-22/55%</td><td>8-14/57%</td><td>6-24/25%</td><td>26-60/43%</td>",
    )
    assert metrics["power_play"] == {
        "total": {"home": 4, "visiting": 5},
        "goals": {"home": 0, "visiting": 2},
    }
    assert metrics["fow"] == {"home": 26, "visiting": 20}
    assert metrics["complete"] is True


def test_schedule_normalizer_maps_nhl_ids_to_provider_compatible_schema():
    game = {
        "id": 2023020204,
        "season": 20232024,
        "gameType": 2,
        "gameDate": "2023-11-10",
        "startTimeUTC": "2023-11-11T00:00:00Z",
        "gameScheduleState": "OK",
        "homeTeam": _team(7, "Buffalo", "Sabres", "BUF"),
        "awayTeam": _team(30, "Minnesota", "Wild", "MIN"),
    }
    normalized = normalize_nhl_schedule_game(game)
    assert normalized["values"]["game_id"] == 2023020204
    assert normalized["values"]["home_team_id"] == 7
    assert normalized["values"]["away_team_id"] == 30
    assert normalized["home_team"]["hockeytech_id"] == 7
    assert normalized["away_team"]["name"] == "Minnesota Wild"


def test_completed_normalizer_populates_all_model_features():
    landing = {
        "id": 2023020204,
        "season": 20232024,
        "gameType": 2,
        "gameDate": "2023-11-10",
        "startTimeUTC": "2023-11-11T00:00:00Z",
        "gameState": "OFF",
        "venue": {"default": "KeyBank Center"},
        "periodDescriptor": {"number": 3},
        "homeTeam": {**_team(7, "Buffalo", "Sabres", "BUF"), "score": 3, "sog": 25},
        "awayTeam": {**_team(30, "Minnesota", "Wild", "MIN"), "score": 2, "sog": 35},
        "summary": {
            "scoring": [
                {"periodDescriptor": {"number": 1}, "goals": [{"teamAbbrev": "MIN"}, {"teamAbbrev": "BUF"}]},
                {"periodDescriptor": {"number": 2}, "goals": [{"teamAbbrev": "BUF"}]},
            ]
        },
    }
    normalized = normalize_nhl_completed_game(
        landing,
        game_summary_html=(
            "Power Plays (Goals-Opp./PPTime) 2-5/06:41 "
            "Power Plays (Goals-Opp./PPTime) 0-4/06:08"
        ),
        faceoff_summary_html=(
            "TOT 10-20/50% 5-10/50% 5-19/26% 20-49/41% "
            "TOT 12-22/55% 8-14/57% 6-24/25% 26-60/43%"
        ),
    )
    values = normalized["values"]
    assert normalized["stats_complete"] is True
    assert values["home_goal_count"] == 3
    assert values["away_goal_count"] == 2
    assert values["home_shots_on_goal_total"] == 25
    assert values["away_shots_on_goal_total"] == 35
    assert values["home_power_play_percentage"] == 0.0
    assert values["away_power_play_percentage"] == 0.4
    assert values["home_faceoff_win_percentage"] == 26 / 60
    assert values["away_faceoff_win_percentage"] == 20 / 49
    assert values["scoring_breakdown"] == {"home": {"1": 1, "2": 1}, "visiting": {"1": 1}}
