from __future__ import annotations

import datetime as dt
from collections import defaultdict
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from db.models import ChlGame, ChlLeague, ChlTeam
from services.data_backend import CHL_LEAGUES, normalize_league_code


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _league_id_for_code(db: Session, league_code: str) -> int | None:
    league = db.scalar(select(ChlLeague).where(ChlLeague.code == league_code))
    if league is None:
        return None
    return int(league.id)


def _extract_snapshot_name(snapshot: dict[str, Any] | None, *, side: str) -> str | None:
    data = snapshot or {}
    if side == "home":
        candidates = [
            data.get("home_team_name"),
            data.get("HomeLongName"),
            data.get("HomeTeamName"),
            data.get("home_long_name"),
            data.get("home_team"),
        ]
        nested = data.get("home_team")
    else:
        candidates = [
            data.get("visiting_team_name"),
            data.get("VisitorLongName"),
            data.get("VisitorTeamName"),
            data.get("away_team_name"),
            data.get("away_team"),
            data.get("visiting_team"),
        ]
        nested = data.get("visiting_team")

    for candidate in candidates:
        cleaned = _clean_text(candidate)
        if cleaned:
            return cleaned

    if isinstance(nested, dict):
        for key in ("name", "team_name", "long_name"):
            cleaned = _clean_text(nested.get(key))
            if cleaned:
                return cleaned

    return None


def _extract_snapshot_logo(snapshot: dict[str, Any] | None, *, side: str) -> str | None:
    data = snapshot or {}
    if side == "home":
        candidates = [
            data.get("home_team_logo_url"),
            data.get("HomeTeamLogo"),
            data.get("home_logo_url"),
        ]
        nested = data.get("home_team")
    else:
        candidates = [
            data.get("visiting_team_logo_url"),
            data.get("VisitorTeamLogo"),
            data.get("away_team_logo_url"),
            data.get("visitor_logo_url"),
        ]
        nested = data.get("visiting_team")

    for candidate in candidates:
        cleaned = _clean_text(candidate)
        if cleaned:
            return cleaned

    if isinstance(nested, dict):
        for key in ("logo_url", "team_logo_url", "logo"):
            cleaned = _clean_text(nested.get(key))
            if cleaned:
                return cleaned

    return None


def _build_missing_team_candidates(db: Session, league_id: int) -> dict[int, dict[str, Any]]:
    games = db.scalars(
        select(ChlGame)
        .where(ChlGame.league_id == league_id)
        .where(or_(ChlGame.home_team_id.is_not(None), ChlGame.away_team_id.is_not(None)))
    ).all()
    if not games:
        return {}

    existing_provider_ids = {
        int(team.hockeytech_id)
        for team in db.scalars(select(ChlTeam).where(ChlTeam.league_id == league_id)).all()
    }

    candidates: dict[int, dict[str, Any]] = defaultdict(lambda: {"name": None, "logo_url": None, "seen_rows": 0})
    for game in games:
        for side, provider_id in (("home", game.home_team_id), ("away", game.away_team_id)):
            if provider_id is None:
                continue
            provider = int(provider_id)
            if provider in existing_provider_ids:
                continue

            row = candidates[provider]
            row["seen_rows"] += 1

            if row["name"] is None:
                base_name = _clean_text(game.home_team if side == "home" else game.away_team)
                row["name"] = base_name or _extract_snapshot_name(game.scorebar_snapshot, side=side)

            if row["logo_url"] is None:
                row["logo_url"] = _extract_snapshot_logo(game.scorebar_snapshot, side=side)

    return dict(candidates)


def sync_missing_teams_for_league(db: Session, league_code: str, *, dry_run: bool = False) -> dict[str, Any]:
    normalized = normalize_league_code(league_code)
    if normalized not in CHL_LEAGUES:
        return {
            "league_code": normalized,
            "status": "skipped",
            "reason": "unsupported_league",
            "missing_provider_team_ids": [],
            "inserted": 0,
            "dry_run": dry_run,
        }

    league_id = _league_id_for_code(db, normalized)
    if league_id is None:
        return {
            "league_code": normalized,
            "status": "skipped",
            "reason": "league_not_found",
            "missing_provider_team_ids": [],
            "inserted": 0,
            "dry_run": dry_run,
        }

    candidates = _build_missing_team_candidates(db, league_id)
    missing_provider_team_ids = sorted(candidates.keys())
    if not missing_provider_team_ids:
        return {
            "league_code": normalized,
            "status": "ok",
            "missing_provider_team_ids": [],
            "candidates_seen": 0,
            "inserted": 0,
            "dry_run": dry_run,
        }

    now = dt.datetime.now(dt.UTC)
    inserted = 0
    for provider_team_id in missing_provider_team_ids:
        candidate = candidates[provider_team_id]
        payload = {
            "league_id": league_id,
            "hockeytech_id": provider_team_id,
            "name": _clean_text(candidate.get("name")) or f"Unknown Team {provider_team_id}",
            "city": None,
            "team_name": None,
            "conference": None,
            "division": None,
            "logo_url": _clean_text(candidate.get("logo_url")),
            "active": False,
            "created_at": now,
            "updated_at": now,
        }

        if dry_run:
            inserted += 1
            continue

        result = db.execute(
            insert(ChlTeam)
            .values(**payload)
            .on_conflict_do_nothing(index_elements=["league_id", "hockeytech_id"])
        )
        inserted += int(result.rowcount or 0)

    if not dry_run:
        db.commit()

    return {
        "league_code": normalized,
        "status": "ok",
        "missing_provider_team_ids": missing_provider_team_ids,
        "candidates_seen": len(missing_provider_team_ids),
        "inserted": inserted,
        "dry_run": dry_run,
    }
