from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from sqlalchemy import select

# Ensure repository root is importable when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.session import SessionLocal
from services.data_backend import (
    DataBackendError,
    apply_league_scope,
    hockeytech_api_key_for_league,
    hockeytech_client_code_for_league,
    primary_store,
    require_supported_league_code,
    resolve_league_id_for_store,
)
from services.hockeytech_client import HockeyTechClient, HockeyTechClientError
from services.leagues import default_logo_url_for_provider


def _best_logo_for_team(
    client: HockeyTechClient,
    provider_team_id: int,
    days_back: int,
    client_code: str,
) -> str | None:
    games = client.get_scorebar(
        number_of_days_ahead=0,
        number_of_days_back=days_back,
        team_id=provider_team_id,
        client_code=client_code,
    )
    candidates: Counter[str] = Counter()

    for game in games:
        if str(game.get("HomeID")) == str(provider_team_id):
            logo = game.get("HomeLogo")
            if logo:
                candidates[logo] += 1

        if str(game.get("VisitorID")) == str(provider_team_id):
            logo = game.get("VisitorLogo")
            if logo:
                candidates[logo] += 1

    if not candidates:
        return None

    return candidates.most_common(1)[0][0]


def backfill_team_logos(days_back: int, force: bool, league_code: str = "whl") -> dict[str, int]:
    normalized_league = require_supported_league_code(league_code)
    db = SessionLocal()
    client_code = hockeytech_client_code_for_league(normalized_league)
    client = HockeyTechClient(
        client_code=client_code,
        api_key=hockeytech_api_key_for_league(normalized_league),
    )

    updated = 0
    fallbacked = 0
    skipped = 0

    try:
        store = primary_store()
        team_model = store.team_model
        league_id = resolve_league_id_for_store(db, store, normalized_league)
        stmt = select(team_model).order_by(team_model.name.asc())
        stmt = apply_league_scope(stmt, team_model, league_id)
        teams = db.scalars(stmt).all()
        for team in teams:
            if team.logo_url and not force:
                skipped += 1
                continue

            discovered_logo = _best_logo_for_team(
                client=client,
                provider_team_id=team.hockeytech_id,
                days_back=days_back,
                client_code=client_code,
            )
            if discovered_logo:
                team.logo_url = discovered_logo
                updated += 1
            else:
                team.logo_url = default_logo_url_for_provider(team.hockeytech_id, league_code=normalized_league)
                fallbacked += 1

        db.commit()
    finally:
        db.close()

    return {
        "updated": updated,
        "fallbacked": fallbacked,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill CHL team logo URLs from HockeyTech scorebar data.")
    parser.add_argument("--days-back", type=int, default=1200, help="How many days back to scan scorebar data.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing logo_url values.")
    parser.add_argument("--league-code", default="whl", help="League code (whl|ohl|lhjmq).")
    args = parser.parse_args()

    try:
        summary = backfill_team_logos(days_back=args.days_back, force=args.force, league_code=args.league_code)
    except HockeyTechClientError as exc:
        raise SystemExit(f"Logo backfill failed: {exc}") from exc
    except DataBackendError as exc:
        raise SystemExit(f"Logo backfill failed: {exc}") from exc

    print(
        f"Logo backfill complete: updated={summary['updated']} fallbacked={summary['fallbacked']} skipped={summary['skipped']}"
    )


if __name__ == "__main__":
    main()
