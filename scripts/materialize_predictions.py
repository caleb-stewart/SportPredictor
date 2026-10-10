from __future__ import annotations

import argparse
import datetime as dt
import json

from db.session import SessionLocal
from services.materializer import materialize_model_version


def _date(value: str | None) -> dt.date | None:
    return dt.date.fromisoformat(value) if value else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append-only materialization for a model version.")
    parser.add_argument("--league-code", default="whl")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--date-from")
    parser.add_argument("--date-to")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--recompute-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        result = materialize_model_version(
            db,
            league_code=args.league_code,
            model_version=args.model_version,
            date_from=_date(args.date_from),
            date_to=_date(args.date_to),
            batch_size=args.batch_size,
            skip_existing=not args.recompute_existing,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, indent=2, default=str))
    finally:
        db.close()
