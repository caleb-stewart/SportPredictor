from __future__ import annotations

import argparse
import json
import logging

try:
    from scripts._cli_utils import (
        configure_script_logging,
        ensure_repo_root_on_path,
        parse_bool,
        parse_csv_list,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from _cli_utils import (
        configure_script_logging,
        ensure_repo_root_on_path,
        parse_bool,
        parse_csv_list,
    )

ensure_repo_root_on_path(__file__)

from db.session import SessionLocal
from services.chl_history_ingest import ChlHistoryIngestError, run_chl_history_backfill

LOGGER = logging.getLogger("chl_history_backfill")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill CHL historical data using HockeyTech season endpoints.")
    parser.add_argument("--leagues", type=lambda v: parse_csv_list(v, label="league code"), default=["ohl", "lhjmq"], help="Comma-separated league codes, e.g. ohl,lhjmq")
    parser.add_argument("--all-available-seasons", type=parse_bool, default=True)
    parser.add_argument("--include-regular", type=parse_bool, default=True)
    parser.add_argument("--include-playoffs", type=parse_bool, default=True)
    parser.add_argument("--include-preseason", type=parse_bool, default=True)
    parser.add_argument("--resume-run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--rps-limit", type=float, default=None)
    parser.add_argument("--retry-max", type=int, default=None)
    parser.add_argument("--recompute-rolling", type=parse_bool, default=True)
    parser.add_argument("--train-after", type=parse_bool, default=False)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_script_logging(args.log_level, quiet_http_clients=True)
    LOGGER.info(
        "Starting CHL history backfill leagues=%s dry_run=%s",
        args.leagues,
        args.dry_run,
    )

    outputs: list[dict[str, object]] = []
    for league_code in args.leagues:
        db = SessionLocal()
        try:
            LOGGER.info("Running league=%s", league_code)
            result = run_chl_history_backfill(
                db=db,
                league_code=league_code,
                all_available_seasons=bool(args.all_available_seasons),
                include_regular=bool(args.include_regular),
                include_playoffs=bool(args.include_playoffs),
                include_preseason=bool(args.include_preseason),
                resume_run_id=args.resume_run_id,
                dry_run=bool(args.dry_run),
                max_workers=args.max_workers,
                rps_limit=args.rps_limit,
                retry_max=args.retry_max,
                recompute_rolling=bool(args.recompute_rolling),
                train_after=bool(args.train_after),
            )
            outputs.append(result)
            LOGGER.info(
                "League complete league=%s status=%s run_id=%s",
                league_code,
                result.get("status"),
                result.get("run_id"),
            )
        except ChlHistoryIngestError as exc:
            LOGGER.exception("Historical backfill failed for league=%s: %s", league_code, exc)
            raise SystemExit(f"Historical backfill failed for {league_code}: {exc}") from exc
        finally:
            db.close()

    LOGGER.info("Backfill finished for all leagues")
    print(json.dumps(outputs, indent=2, default=str))
