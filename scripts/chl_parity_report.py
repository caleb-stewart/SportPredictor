from __future__ import annotations

import json

from sqlalchemy import text

from db.session import SessionLocal


QUERIES = {
    "teams": (
        "SELECT COUNT(*) AS c FROM whl_teams",
        "SELECT COUNT(*) AS c FROM chl_teams WHERE league_id = 1",
    ),
    "games": (
        "SELECT COUNT(*) AS c FROM whl_games",
        "SELECT COUNT(*) AS c FROM chl_games WHERE league_id = 1",
    ),
    "rolling_averages": (
        "SELECT COUNT(*) AS c FROM whl_rolling_averages",
        "SELECT COUNT(*) AS c FROM chl_rolling_averages WHERE league_id = 1",
    ),
    "prediction_records": (
        "SELECT COUNT(*) AS c FROM whl_prediction_records",
        "SELECT COUNT(*) AS c FROM chl_prediction_records WHERE league_id = 1",
    ),
    "custom_prediction_records": (
        "SELECT COUNT(*) AS c FROM whl_custom_prediction_records",
        "SELECT COUNT(*) AS c FROM chl_custom_prediction_records WHERE league_id = 1",
    ),
}


if __name__ == "__main__":
    db = SessionLocal()
    try:
        out: dict[str, dict[str, int | bool]] = {}
        for key, (whl_sql, chl_sql) in QUERIES.items():
            whl_count = int(db.execute(text(whl_sql)).scalar() or 0)
            chl_count = int(db.execute(text(chl_sql)).scalar() or 0)
            out[key] = {
                "whl_count": whl_count,
                "chl_count": chl_count,
                "match": whl_count == chl_count,
                "delta": chl_count - whl_count,
            }

        all_match = all(item["match"] for item in out.values())
        print(json.dumps({"all_match": all_match, "tables": out}, indent=2))
    finally:
        db.close()
