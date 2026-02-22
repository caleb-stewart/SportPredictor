from __future__ import annotations

from scripts import chl_parity_report


class _FakeMappingsResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, _sql, _params):
        return _FakeMappingsResult(self._rows)


def test_build_parity_report_shapes_per_league_payload(monkeypatch):
    db = _FakeDB(
        [
            {"id": 1, "code": "whl", "name": "Western Hockey League", "active": True},
            {"id": 2, "code": "ohl", "name": "Ontario Hockey League", "active": True},
            {"id": 3, "code": "lhjmq", "name": "Quebec Maritimes Junior Hockey League", "active": True},
        ]
    )

    monkeypatch.setattr(chl_parity_report, "_count_for_table", lambda _db, table_name, league_id: league_id * (len(table_name) + 1))
    monkeypatch.setattr(
        chl_parity_report,
        "_integrity_checks",
        lambda _db, _league_id: {
            "teams_non_empty": True,
            "games_non_empty": True,
            "game_team_league_mismatch_rows": 0,
            "prediction_orphan_game_rows": 0,
            "rolling_orphan_game_rows": 0,
            "custom_prediction_team_mismatch_rows": 0,
        },
    )

    payload = chl_parity_report.build_parity_report(db)

    assert payload["all_pass"] is True
    assert payload["missing_leagues"] == []
    assert set(payload["leagues"].keys()) == {"whl", "ohl", "lhjmq"}
    assert payload["leagues"]["whl"]["pass"] is True
    assert "chl_games" in payload["leagues"]["whl"]["counts"]


def test_build_parity_report_flags_missing_league(monkeypatch):
    db = _FakeDB(
        [
            {"id": 1, "code": "whl", "name": "Western Hockey League", "active": True},
            {"id": 2, "code": "ohl", "name": "Ontario Hockey League", "active": True},
        ]
    )

    monkeypatch.setattr(chl_parity_report, "_count_for_table", lambda _db, _table_name, _league_id: 1)
    monkeypatch.setattr(
        chl_parity_report,
        "_integrity_checks",
        lambda _db, _league_id: {
            "teams_non_empty": True,
            "games_non_empty": True,
            "game_team_league_mismatch_rows": 0,
            "prediction_orphan_game_rows": 0,
            "rolling_orphan_game_rows": 0,
            "custom_prediction_team_mismatch_rows": 0,
        },
    )

    payload = chl_parity_report.build_parity_report(db)

    assert payload["all_pass"] is False
    assert payload["missing_leagues"] == ["lhjmq"]
