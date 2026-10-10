from __future__ import annotations

import argparse
import json

try:
    from scripts._cli_utils import configure_script_logging, ensure_repo_root_on_path, parse_csv_list
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from _cli_utils import configure_script_logging, ensure_repo_root_on_path, parse_csv_list

ensure_repo_root_on_path(__file__)

from db.session import SessionLocal
from services.data_backend import CHL_LEAGUES
from services.team_sync import sync_missing_teams_for_league


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill missing CHL teams referenced by chl_games home/away provider team ids."
    )
    parser.add_argument(
        "--leagues",
        type=lambda value: parse_csv_list(value, label="league code"),
        default=sorted(CHL_LEAGUES.keys()),
        help="Comma-separated league codes (whl,ohl,lhjmq). Defaults to all configured CHL leagues.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show would-be inserted rows without writing.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_script_logging(args.log_level, quiet_http_clients=True)

    db = SessionLocal()
    try:
        outputs = [
            sync_missing_teams_for_league(
                db,
                league_code=league_code,
                dry_run=bool(args.dry_run),
            )
            for league_code in args.leagues
        ]
        print(json.dumps(outputs, indent=2, default=str))
    finally:
        db.close()

