from __future__ import annotations

import argparse
import datetime as dt
import json

from scripts._cli_utils import configure_script_logging, ensure_repo_root_on_path, parse_bool

ensure_repo_root_on_path(__file__)

from db.session import SessionLocal
from services.nhl_ingest import NhlIngestError, run_nhl_history_backfill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill NHL game data from the official GameCenter API and GS/FS reports."
    )
    parser.add_argument("--date-from", required=True, type=dt.date.fromisoformat, help="First game date, YYYY-MM-DD")
    parser.add_argument("--date-to", required=True, type=dt.date.fromisoformat, help="Last game date, YYYY-MM-DD")
    parser.add_argument("--include-playoffs", type=parse_bool, default=True)
    parser.add_argument("--include-preseason", type=parse_bool, default=False)
    parser.add_argument("--resume-run-id", help="Resume an interrupted NHL backfill run and retry its failures.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rps-limit", type=float, default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_script_logging(args.log_level, quiet_http_clients=True)
    db = SessionLocal()
    try:
        result = run_nhl_history_backfill(
            db=db,
            date_from=args.date_from,
            date_to=args.date_to,
            include_playoffs=bool(args.include_playoffs),
            include_preseason=bool(args.include_preseason),
            resume_run_id=args.resume_run_id,
            dry_run=bool(args.dry_run),
            rps_limit=args.rps_limit,
        )
        print(json.dumps(result, indent=2, default=str))
    except NhlIngestError as exc:
        raise SystemExit(f"NHL history backfill failed: {exc}") from exc
    finally:
        db.close()
