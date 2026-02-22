from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure repository root is importable when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.session import SessionLocal
from services.replay import ReplayError, run_frozen_model_replay

LOGGER = logging.getLogger("replay_last_n_completed")


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_leagues(value: str) -> list[str]:
    leagues = [item.strip() for item in value.split(",") if item.strip()]
    if not leagues:
        raise argparse.ArgumentTypeError("At least one league code is required")
    return leagues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay-predict the last N completed CHL games by league.")
    parser.add_argument("--leagues", type=_parse_leagues, default=["ohl", "lhjmq"], help="Comma-separated league codes, e.g. ohl,lhjmq")
    parser.add_argument("--last-n", type=int, default=1500, help="Number of most recent completed games to replay.")
    parser.add_argument("--dry-run", type=_parse_bool, default=False)
    parser.add_argument("--overwrite", type=_parse_bool, default=True)
    parser.add_argument("--rollback-on-proof-failure", type=_parse_bool, default=False)
    parser.add_argument("--archive-label", default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    LOGGER.info(
        "Starting replay for leagues=%s last_n=%s dry_run=%s overwrite=%s rollback_on_proof_failure=%s",
        args.leagues,
        args.last_n,
        args.dry_run,
        args.overwrite,
        args.rollback_on_proof_failure,
    )

    outputs: list[dict[str, object]] = []
    for league_code in args.leagues:
        db = SessionLocal()
        try:
            LOGGER.info("Running replay league=%s", league_code)
            result = run_frozen_model_replay(
                db=db,
                date_from=None,
                date_to=None,
                selection_mode="last_n_completed_games",
                last_n_games=int(args.last_n),
                dry_run=bool(args.dry_run),
                overwrite=bool(args.overwrite),
                rollback_on_proof_failure=bool(args.rollback_on_proof_failure),
                archive_label=args.archive_label,
                league_code=league_code,
            )
            outputs.append(result)
            LOGGER.info(
                "Replay complete league=%s status=%s run_id=%s",
                league_code,
                result.get("status"),
                result.get("run_id"),
            )
        except ReplayError as exc:
            LOGGER.exception("Replay failed for league=%s: %s", league_code, exc)
            raise SystemExit(f"Replay failed for {league_code}: {exc}") from exc
        finally:
            db.close()

    LOGGER.info("Replay run finished for all leagues")
    print(json.dumps(outputs, indent=2, default=str))
