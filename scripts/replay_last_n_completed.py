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
from services.replay import ReplayError, run_frozen_model_replay

LOGGER = logging.getLogger("replay_last_n_completed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay-predict the last N completed CHL games by league.")
    parser.add_argument("--leagues", type=lambda v: parse_csv_list(v, label="league code"), default=["ohl", "lhjmq"], help="Comma-separated league codes, e.g. ohl,lhjmq")
    parser.add_argument("--last-n", type=int, default=1500, help="Number of most recent completed games to replay.")
    parser.add_argument("--dry-run", type=parse_bool, default=False)
    parser.add_argument("--overwrite", type=parse_bool, default=True)
    parser.add_argument("--rollback-on-proof-failure", type=parse_bool, default=False)
    parser.add_argument("--archive-label", default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_script_logging(args.log_level)

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
