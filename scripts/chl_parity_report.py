from __future__ import annotations

import datetime as dt
import json
from typing import Any

try:
    from scripts._cli_utils import ensure_repo_root_on_path
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from _cli_utils import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

from sqlalchemy import text

from db.session import SessionLocal

TABLES_BY_LEAGUE: tuple[str, ...] = (
    "chl_teams",
    "chl_games",
    "chl_rolling_averages",
    "chl_prediction_records",
    "chl_custom_prediction_records",
)

TARGET_LEAGUES: tuple[str, ...] = ("whl", "ohl", "lhjmq")


def _count_for_table(db, table_name: str, league_id: int) -> int:
    sql = text(f"SELECT COUNT(*) FROM {table_name} WHERE league_id = :league_id")
    return int(db.execute(sql, {"league_id": league_id}).scalar() or 0)


def _integrity_checks(db, league_id: int) -> dict[str, Any]:
    return {
        "teams_non_empty": bool(
            db.execute(text("SELECT COUNT(*) > 0 FROM chl_teams WHERE league_id = :league_id"), {"league_id": league_id}).scalar()
        ),
        "games_non_empty": bool(
            db.execute(text("SELECT COUNT(*) > 0 FROM chl_games WHERE league_id = :league_id"), {"league_id": league_id}).scalar()
        ),
        "game_team_league_mismatch_rows": int(
            db.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM chl_games g
                    LEFT JOIN chl_teams ht
                      ON ht.league_id = g.league_id
                     AND ht.hockeytech_id = g.home_team_id
                    LEFT JOIN chl_teams at
                      ON at.league_id = g.league_id
                     AND at.hockeytech_id = g.away_team_id
                    WHERE g.league_id = :league_id
                      AND (
                        (g.home_team_id IS NOT NULL AND ht.id IS NULL)
                        OR (g.away_team_id IS NOT NULL AND at.id IS NULL)
                      )
                    """
                ),
                {"league_id": league_id},
            ).scalar()
            or 0
        ),
        "prediction_orphan_game_rows": int(
            db.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM chl_prediction_records p
                    LEFT JOIN chl_games g
                      ON g.league_id = p.league_id
                     AND g.game_id = p.game_id
                    WHERE p.league_id = :league_id
                      AND g.id IS NULL
                    """
                ),
                {"league_id": league_id},
            ).scalar()
            or 0
        ),
        "rolling_orphan_game_rows": int(
            db.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM chl_rolling_averages r
                    LEFT JOIN chl_games g
                      ON g.league_id = r.league_id
                     AND g.game_id = r.game_id
                    WHERE r.league_id = :league_id
                      AND g.id IS NULL
                    """
                ),
                {"league_id": league_id},
            ).scalar()
            or 0
        ),
        "custom_prediction_team_mismatch_rows": int(
            db.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM chl_custom_prediction_records c
                    LEFT JOIN chl_teams ht ON ht.id = c.home_team_id
                    LEFT JOIN chl_teams at ON at.id = c.away_team_id
                    LEFT JOIN chl_teams pt ON pt.id = c.predicted_winner_id
                    WHERE c.league_id = :league_id
                      AND (
                        (c.home_team_id IS NOT NULL AND (ht.id IS NULL OR ht.league_id <> c.league_id))
                        OR (c.away_team_id IS NOT NULL AND (at.id IS NULL OR at.league_id <> c.league_id))
                        OR (c.predicted_winner_id IS NOT NULL AND (pt.id IS NULL OR pt.league_id <> c.league_id))
                      )
                    """
                ),
                {"league_id": league_id},
            ).scalar()
            or 0
        ),
    }


def _league_pass(checks: dict[str, Any]) -> bool:
    return (
        checks.get("teams_non_empty") is True
        and checks.get("games_non_empty") is True
        and int(checks.get("game_team_league_mismatch_rows", 0)) == 0
        and int(checks.get("prediction_orphan_game_rows", 0)) == 0
        and int(checks.get("rolling_orphan_game_rows", 0)) == 0
        and int(checks.get("custom_prediction_team_mismatch_rows", 0)) == 0
    )


def build_parity_report(db, *, target_leagues: tuple[str, ...] = TARGET_LEAGUES) -> dict[str, Any]:
    league_rows = db.execute(
        text(
            """
            SELECT id, code, name, active
            FROM chl_leagues
            WHERE code = ANY(:codes)
            ORDER BY id ASC
            """
        ),
        {"codes": list(target_leagues)},
    ).mappings().all()

    leagues_out: dict[str, Any] = {}
    for row in league_rows:
        league_id = int(row["id"])
        code = str(row["code"])
        counts = {table: _count_for_table(db, table, league_id) for table in TABLES_BY_LEAGUE}
        checks = _integrity_checks(db, league_id)
        leagues_out[code] = {
            "league_id": league_id,
            "league_name": row["name"],
            "active": bool(row["active"]),
            "counts": counts,
            "checks": checks,
            "pass": _league_pass(checks),
        }

    missing = [code for code in target_leagues if code not in leagues_out]
    all_pass = not missing and all(item.get("pass") is True for item in leagues_out.values())

    return {
        "generated_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "target_leagues": list(target_leagues),
        "missing_leagues": missing,
        "all_pass": all_pass,
        "leagues": leagues_out,
    }


if __name__ == "__main__":
    db = SessionLocal()
    try:
        payload = build_parity_report(db)
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        db.close()
