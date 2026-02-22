from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from db.session import SessionLocal
from services.model_compare import run_model_compare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a baseline model superiority proof artifact.")
    parser.add_argument("--candidate-model-version", required=True)
    parser.add_argument("--baseline-model-version", required=True)
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    parser.add_argument("--league-code", default="whl")
    parser.add_argument("--output", default=None, help="Optional path to write summary JSON")
    return parser.parse_args()


def _parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    return dt.date.fromisoformat(value)


def main() -> int:
    args = parse_args()
    db = SessionLocal()
    try:
        result = run_model_compare(
            db=db,
            candidate_model_version=args.candidate_model_version,
            baseline_model_version=args.baseline_model_version,
            date_from=_parse_date(args.date_from),
            date_to=_parse_date(args.date_to),
            mode="frozen_replay",
            league_code=args.league_code,
        )
    finally:
        db.close()

    output_payload = {
        "generated_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "compare_result": result,
        "league_code": args.league_code,
        "sql_snapshot_query": (
            "SELECT model_version, COUNT(*) FROM chl_prediction_records "
            "WHERE league_id = (SELECT id FROM chl_leagues WHERE code = :league_code) "
            "GROUP BY model_version ORDER BY COUNT(*) DESC;"
        ),
    }

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output_payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote baseline proof snapshot to {path}")
    else:
        print(json.dumps(output_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
