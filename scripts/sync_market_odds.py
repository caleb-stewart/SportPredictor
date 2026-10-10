"""Sync current or historical Odds API sportsbook data into PostgreSQL."""

from __future__ import annotations

import argparse
import datetime as dt
import json

from scripts._cli_utils import ensure_repo_root_on_path, parse_csv_list

ensure_repo_root_on_path(__file__)

from core.config import settings  # noqa: E402
from db.session import SessionLocal  # noqa: E402
from services.the_odds_api import (  # noqa: E402
    TheOddsApiError,
    sync_current_odds,
    sync_historical_odds_snapshot,
)


def _parse_utc(value: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ISO timestamp: {value}") from exc
    return parsed.replace(tzinfo=dt.UTC) if parsed.tzinfo is None else parsed.astimezone(dt.UTC)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    current = subcommands.add_parser("current", help="Store current pregame odds for configured sports.")
    current.add_argument("--sports", type=parse_csv_list, default=None, help="Comma-separated Odds API sport keys.")

    history = subcommands.add_parser("history", help="Store sparse historical snapshots with a bounded request count.")
    history.add_argument("--sport", default="icehockey_nhl")
    history.add_argument("--from", dest="date_from", required=True, type=_parse_utc)
    history.add_argument("--to", dest="date_to", required=True, type=_parse_utc)
    history.add_argument("--interval-hours", type=int, default=24, help="Snapshot spacing; default is daily.")
    history.add_argument("--max-requests", type=int, default=25, help="Safety cap; use 0 only after reviewing account quota.")
    args = parser.parse_args()

    if args.command == "history":
        if args.date_to < args.date_from:
            parser.error("--to must not be before --from")
        if args.interval_hours < 1:
            parser.error("--interval-hours must be at least one")
        if args.max_requests < 0:
            parser.error("--max-requests must be zero or greater")
    if not settings.the_odds_api_key:
        parser.error("THE_ODDS_API_KEY is missing from .env")

    db = SessionLocal()
    try:
        if args.command == "current":
            sports = args.sports or parse_csv_list(settings.the_odds_api_current_sports, label="sport key")
            results = [sync_current_odds(db, sport).as_dict() for sport in sports]
        else:
            results = []
            next_at = args.date_from
            requests = 0
            cap = args.max_requests or None
            while next_at <= args.date_to and (cap is None or requests < cap):
                result = sync_historical_odds_snapshot(db, args.sport, next_at)
                results.append(result.as_dict())
                requests += 1
                requested_next = next_at + dt.timedelta(hours=args.interval_hours)
                # The provider may return the nearest earlier snapshot. Its
                # cursor prevents a repeated request from storing that same
                # source snapshot after an interrupted/rounded import.
                next_at = max(requested_next, result.next_snapshot_at or requested_next)
        print(json.dumps(results, indent=2, default=str))
        return 0
    except TheOddsApiError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 2
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
