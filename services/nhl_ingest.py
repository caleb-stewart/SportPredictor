from __future__ import annotations

import datetime as dt
import logging
import re
import uuid
from html.parser import HTMLParser
from typing import Any

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from sqlalchemy import delete, select, text

from db.models import ChlGame, ChlIngestGameFailure, ChlIngestRun, ChlTeam
from services.data_backend import ensure_chl_league, require_supported_league_code
from services.nhl_client import NhlClient, nhl_report_url


class NhlIngestError(RuntimeError):
    pass


LOGGER = logging.getLogger(__name__)
NHL_BACKFILL_ADVISORY_LOCK = 907241631


class _VisibleTextParser(HTMLParser):
    """Turn the NHL's table-heavy reports into searchable visible text."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"br", "tr", "p", "div", "td", "th"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"tr", "p", "div", "td", "th"}:
            self.parts.append(" ")


def _visible_text(html: str | None) -> str:
    parser = _VisibleTextParser()
    parser.feed(html or "")
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def _text(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("default") or next(iter(value.values()), None)
    if value in (None, ""):
        return None
    value = str(value).strip()
    return value or None


def _int(value: Any) -> int | None:
    try:
        return None if value in (None, "") else int(value)
    except (TypeError, ValueError):
        return None


def _date(value: Any) -> dt.date | None:
    try:
        return dt.date.fromisoformat(str(value)[:10]) if value else None
    except ValueError:
        return None


def _datetime(value: Any) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)
    except ValueError:
        return None


def _team_name(team: dict[str, Any]) -> str:
    place = _text(team.get("placeName"))
    common = _text(team.get("commonName"))
    return " ".join(part for part in (place, common) if part) or str(team.get("abbrev") or team.get("id"))


def _team_payload(team: dict[str, Any]) -> dict[str, Any]:
    team_id = _int(team.get("id"))
    if team_id is None:
        raise NhlIngestError(f"NHL game contains a team without an id: {team!r}")
    return {
        "hockeytech_id": team_id,  # provider-neutral id kept for schema compatibility
        "name": _team_name(team),
        "city": _text(team.get("placeName")),
        "team_name": _text(team.get("commonName")),
        "conference": None,
        "division": None,
        "logo_url": _text(team.get("logo")),
        "active": True,
    }


def parse_nhl_report_metrics(game_summary_html: str | None, faceoff_summary_html: str | None) -> dict[str, Any]:
    game_summary = _visible_text(game_summary_html)
    faceoff_summary = _visible_text(faceoff_summary_html)

    # Official GS reports list the visitor first and the home team second.
    power_play_matches = re.findall(
        r"Power Plays\s*\(Goals-Opp\./PPTime\)\s*(\d+)\s*-\s*(\d+)",
        game_summary,
        flags=re.IGNORECASE,
    )
    if len(power_play_matches) < 2:
        # Older NHL GS reports use a compact totals table instead of the
        # newer repeated "Power Plays (Goals-Opp./PPTime)" label. In that
        # table the final two dash-pairs before the zone-time columns are the
        # visitor/home power-play goals and opportunities.
        legacy_power_play = re.findall(
            r"\bT\s+(\d+)-(\d+)\s+(\d+)-(\d+)\s+T\s+\d+:\d+\s+\d+:\d+",
            game_summary,
            flags=re.IGNORECASE,
        )
        if legacy_power_play:
            away_goals, away_opportunities, home_goals, home_opportunities = legacy_power_play[-1]
            power_play_matches = [(away_goals, away_opportunities), (home_goals, home_opportunities)]
    faceoff_pair = r"\d+\s*-\s*\d+\s*/\s*\d+\s*%"
    faceoff_rows = re.findall(
        rf"\bTOT\s+((?:{faceoff_pair}\s*){{4}})",
        faceoff_summary,
        flags=re.IGNORECASE,
    )
    if len(faceoff_rows) >= 2:
        # The official team summary has EV, PP, SH, TOT columns. The fourth
        # pair is the overall team total; later zone/player tables also have
        # TOT rows, so the first two matches are the visitor/home summaries.
        faceoff_matches = [
            tuple(re.search(r"(\d+)\s*-\s*(\d+)\s*/\s*(\d+)\s*%", row_pairs[-1]).groups())
            for row_pairs in (re.findall(faceoff_pair, row) for row in faceoff_rows[:2])
        ]
    else:
        faceoff_matches = re.findall(
            r"\bTOT\s+(\d+)\s*-\s*(\d+)\s*/\s*(\d+)\s*%",
            faceoff_summary,
            flags=re.IGNORECASE,
        )

    power_play: dict[str, dict[str, int]] | None = None
    pp_percentages: dict[str, float] | None = None
    if len(power_play_matches) >= 2:
        away_goals, away_opportunities = (int(v) for v in power_play_matches[0])
        home_goals, home_opportunities = (int(v) for v in power_play_matches[1])
        power_play = {
            "total": {"home": home_opportunities, "visiting": away_opportunities},
            "goals": {"home": home_goals, "visiting": away_goals},
        }
        pp_percentages = {
            "home": home_goals / home_opportunities if home_opportunities else 0.0,
            "visiting": away_goals / away_opportunities if away_opportunities else 0.0,
        }

    fow: dict[str, int] | None = None
    fow_percentages: dict[str, float] | None = None
    if len(faceoff_matches) >= 2:
        away_won, away_total, away_pct = (int(v) for v in faceoff_matches[0])
        home_won, home_total, home_pct = (int(v) for v in faceoff_matches[1])
        fow = {"home": home_won, "visiting": away_won}
        fow_percentages = {
            "home": home_won / home_total if home_total else home_pct / 100.0,
            "visiting": away_won / away_total if away_total else away_pct / 100.0,
        }

    return {
        "power_play": power_play,
        "power_play_percentages": pp_percentages,
        "fow": fow,
        "faceoff_percentages": fow_percentages,
        "complete": power_play is not None and fow is not None,
    }


def _season_name(season: Any, game_type: Any) -> str | None:
    season_text = str(season or "")
    if len(season_text) != 8 or not season_text.isdigit():
        return None
    labels = {1: "Preseason", 2: "Regular Season", 3: "Playoffs", 4: "All-Star"}
    label = labels.get(_int(game_type) or 2, "Season")
    return f"{season_text[:4]}-{season_text[6:]} {label}"


def _scoring_breakdown(landing: dict[str, Any], home_abbrev: str | None, away_abbrev: str | None) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {"home": {}, "visiting": {}}
    scoring = (landing.get("summary") or {}).get("scoring") or []
    for period in scoring:
        period_number = _text((period.get("periodDescriptor") or {}).get("number")) or "0"
        for goal in period.get("goals") or []:
            abbrev = _text((goal.get("teamAbbrev") or goal.get("team") or {}))
            if abbrev == home_abbrev:
                side = "home"
            elif abbrev == away_abbrev:
                side = "visiting"
            elif goal.get("isHome") is True:
                side = "home"
            elif goal.get("isHome") is False:
                side = "visiting"
            else:
                continue
            result[side][period_number] = result[side].get(period_number, 0) + 1
    return result


def _base_game_values(game: dict[str, Any], landing: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = landing or game
    home = source.get("homeTeam") or game.get("homeTeam") or {}
    away = source.get("awayTeam") or game.get("awayTeam") or {}
    game_id = _int(source.get("id") or game.get("id"))
    game_date = _date(source.get("gameDate") or game.get("gameDate"))
    if game_id is None or game_date is None:
        raise NhlIngestError(f"NHL game is missing id/date: {source!r}")

    home_id = _int(home.get("id"))
    away_id = _int(away.get("id"))
    if home_id is None or away_id is None:
        raise NhlIngestError(f"NHL game is missing home/away team ids: {source!r}")
    game_type = _int(source.get("gameType") or game.get("gameType")) or 2
    status_token = str(source.get("gameState") or source.get("gameScheduleState") or game.get("gameState") or "").upper()
    status = {"OFF": "Final", "FUT": "Scheduled", "LIVE": "Live", "PRE": "Pre-game"}.get(status_token, status_token or "Scheduled")
    scheduled_time = _datetime(source.get("startTimeUTC") or game.get("startTimeUTC"))

    values: dict[str, Any] = {
        "game_id": game_id,
        "season_id": str(source.get("season") or game.get("season") or "") or None,
        "season_name": _season_name(source.get("season") or game.get("season"), game_type),
        "game_date": game_date,
        "venue": _text((source.get("venue") or game.get("venue") or {}).get("default")) if isinstance(source.get("venue") or game.get("venue"), dict) else _text(source.get("venue") or game.get("venue")),
        "status": status,
        "home_team": _team_name(home),
        "away_team": _team_name(away),
        "home_team_id": home_id,
        "away_team_id": away_id,
        "game_number": int(str(game_id)[-4:]),
        "scheduled_time_utc": scheduled_time,
        "scorebar_snapshot": source,
    }
    update_values = {key: value for key, value in values.items() if key not in {"game_id", "scorebar_snapshot"} and value is not None}
    update_values["scorebar_snapshot"] = source
    return _team_payload(home), _team_payload(away), {"values": values, "update_values": update_values}


def normalize_nhl_schedule_game(game: dict[str, Any]) -> dict[str, Any]:
    home_team, away_team, payload = _base_game_values(game)
    return {"home_team": home_team, "away_team": away_team, **payload}


def normalize_nhl_completed_game(
    landing: dict[str, Any],
    *,
    game_summary_html: str | None = None,
    faceoff_summary_html: str | None = None,
) -> dict[str, Any]:
    home_team, away_team, payload = _base_game_values(landing, landing)
    values = payload["values"]
    metrics = parse_nhl_report_metrics(game_summary_html, faceoff_summary_html)
    home = landing.get("homeTeam") or {}
    away = landing.get("awayTeam") or {}
    home_score = _int(home.get("score"))
    away_score = _int(away.get("score"))
    home_sog = _int(home.get("sog"))
    away_sog = _int(away.get("sog"))
    values.update(
        {
            "home_goal_count": home_score,
            "away_goal_count": away_score,
            "scoring_breakdown": _scoring_breakdown(landing, _text(home.get("abbrev")), _text(away.get("abbrev"))),
            "shots_on_goal": {"home": {"total": home_sog}, "visiting": {"total": away_sog}},
            "period": _text((landing.get("periodDescriptor") or {}).get("number")),
            "power_play": metrics["power_play"],
            "fow": metrics["fow"],
            "home_power_play_percentage": (metrics["power_play_percentages"] or {}).get("home"),
            "away_power_play_percentage": (metrics["power_play_percentages"] or {}).get("visiting"),
            "home_faceoff_win_percentage": (metrics["faceoff_percentages"] or {}).get("home"),
            "away_faceoff_win_percentage": (metrics["faceoff_percentages"] or {}).get("visiting"),
            "home_shots_on_goal_total": home_sog,
            "away_shots_on_goal_total": away_sog,
            "scorebar_snapshot": {
                **landing,
                "_provider": "nhl",
                "_reports": {
                    "game_summary_url": nhl_report_url(landing["id"], "GS"),
                    "faceoff_summary_url": nhl_report_url(landing["id"], "FS"),
                    "game_summary_available": game_summary_html is not None,
                    "faceoff_summary_available": faceoff_summary_html is not None,
                    "complete": metrics["complete"],
                },
            },
        }
    )
    payload["update_values"] = {key: value for key, value in values.items() if key != "game_id"}
    return {"home_team": home_team, "away_team": away_team, **payload, "stats_complete": bool(metrics["complete"] and home_score is not None and away_score is not None and home_sog is not None and away_sog is not None)}


def _league_id(db: Session) -> int:
    try:
        return int(ensure_chl_league(db, "nhl").id)
    except Exception as exc:  # noqa: BLE001
        raise NhlIngestError("NHL league row is missing or the canonical tables are not migrated.") from exc


def _write_team(db: Session, league_id: int, payload: dict[str, Any]) -> None:
    db.execute(
        insert(ChlTeam)
        .values(league_id=league_id, **payload)
        .on_conflict_do_update(
            index_elements=["league_id", "hockeytech_id"],
            set_={key: value for key, value in payload.items() if key != "hockeytech_id"},
        )
    )


def _write_game(db: Session, league_id: int, payload: dict[str, Any]) -> None:
    db.execute(
        insert(ChlGame)
        .values(league_id=league_id, **payload["values"])
        .on_conflict_do_update(
            index_elements=["league_id", "game_id"],
            set_={"league_id": league_id, **payload["update_values"]},
        )
    )


def _write_normalized_game(db: Session, league_id: int, normalized: dict[str, Any]) -> None:
    _write_team(db, league_id, normalized["home_team"])
    _write_team(db, league_id, normalized["away_team"])
    _write_game(db, league_id, normalized)


def _record_failure(
    db: Session,
    *,
    run_id: uuid.UUID,
    league_id: int,
    game_id: int,
    season_id: str | None,
    stage: str,
    error: Exception,
) -> None:
    """Keep an actionable retry ledger instead of silently leaving data gaps."""
    now = dt.datetime.now(dt.UTC)
    db.execute(
        insert(ChlIngestGameFailure)
        .values(
            run_id=run_id,
            league_id=league_id,
            game_id=game_id,
            season_id=season_id,
            stage=stage,
            attempts=1,
            last_error=str(error)[:4000],
            last_seen_at=now,
        )
        .on_conflict_do_update(
            index_elements=["run_id", "league_id", "game_id", "stage"],
            set_={
                "season_id": season_id,
                "attempts": ChlIngestGameFailure.attempts + 1,
                "last_error": str(error)[:4000],
                "last_seen_at": now,
            },
        )
    )


def _clear_failure(db: Session, *, run_id: uuid.UUID, league_id: int, game_id: int, stage: str) -> None:
    db.execute(
        delete(ChlIngestGameFailure).where(
            ChlIngestGameFailure.run_id == run_id,
            ChlIngestGameFailure.league_id == league_id,
            ChlIngestGameFailure.game_id == game_id,
            ChlIngestGameFailure.stage == stage,
        )
    )


def _optional_report(client: NhlClient, game_id: int | str, report_code: str) -> str | None:
    """Fetch an auxiliary report without discarding the primary game payload.

    NHL's older archive does not have every GS/FS report. Scores, teams, and
    shots from GameCenter are still valuable, so a missing report should make
    the row incomplete for model training rather than turn the whole game into
    a schedule-only record.
    """
    try:
        if report_code == "GS":
            return client.get_game_summary_report(game_id)
        return client.get_faceoff_summary_report(game_id)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "Optional NHL report unavailable game_id=%s report=%s error=%s",
            game_id,
            report_code,
            exc,
        )
        return None


def _acquire_backfill_lock(db: Session) -> bool:
    """Allow one historical NHL writer across CLI processes and app instances."""
    return bool(db.scalar(text("SELECT pg_try_advisory_lock(:lock_id)"), {"lock_id": NHL_BACKFILL_ADVISORY_LOCK}))


def _release_backfill_lock(db: Session) -> None:
    db.execute(text("SELECT pg_advisory_unlock(:lock_id)"), {"lock_id": NHL_BACKFILL_ADVISORY_LOCK})


def upsert_nhl_schedule(db: Session, target_date: dt.date, *, client: NhlClient | None = None) -> int:
    require_supported_league_code("nhl")
    client = client or NhlClient()
    league_id = _league_id(db)
    written = 0
    for game in client.get_schedule_for_date(target_date):
        normalized = normalize_nhl_schedule_game(game)
        _write_normalized_game(db, league_id, normalized)
        written += 1
    db.commit()
    return written


def update_nhl_completed_games(db: Session, target_date: dt.date, *, client: NhlClient | None = None) -> int:
    require_supported_league_code("nhl")
    client = client or NhlClient()
    league_id = _league_id(db)
    updated = 0
    for game in client.get_schedule_for_date(target_date):
        game_id = game.get("id")
        if game_id is None:
            continue
        landing = client.get_landing(game_id)
        if str(landing.get("gameState") or "").upper() not in {"OFF", "FINAL"}:
            continue
        normalized = normalize_nhl_completed_game(
            landing,
            game_summary_html=_optional_report(client, game_id, "GS"),
            faceoff_summary_html=_optional_report(client, game_id, "FS"),
        )
        _write_normalized_game(db, league_id, normalized)
        updated += 1
    db.commit()
    return updated


def run_nhl_history_backfill(
    db: Session,
    *,
    date_from: dt.date,
    date_to: dt.date,
    include_playoffs: bool = True,
    include_preseason: bool = False,
    resume_run_id: str | None = None,
    dry_run: bool = False,
    rps_limit: float | None = None,
) -> dict[str, Any]:
    if date_to < date_from:
        raise NhlIngestError("date_to must be on or after date_from")
    require_supported_league_code("nhl")
    league_id = _league_id(db)
    run_id: uuid.UUID | None = None
    run_row: ChlIngestRun | None = None
    lock_acquired = False
    if not dry_run:
        lock_acquired = _acquire_backfill_lock(db)
        if not lock_acquired:
            raise NhlIngestError("Another NHL historical backfill is already writing to this database")
        if resume_run_id:
            try:
                run_id = uuid.UUID(resume_run_id)
            except ValueError as exc:
                raise NhlIngestError("resume_run_id must be a valid UUID") from exc
            run_row = db.scalar(select(ChlIngestRun).where(ChlIngestRun.id == run_id, ChlIngestRun.league_id == league_id))
            if run_row is None or run_row.mode != "nhl_historical_backfill":
                raise NhlIngestError(f"NHL ingest run {resume_run_id} was not found")
        else:
            other_running = db.scalar(
                select(ChlIngestRun.id).where(ChlIngestRun.league_id == league_id, ChlIngestRun.status == "running")
            )
            if other_running:
                raise NhlIngestError(f"Another NHL ingest run is already running: {other_running}")
            run_id = uuid.uuid4()
            run_row = ChlIngestRun(
                id=run_id,
                league_id=league_id,
                status="running",
                mode="nhl_historical_backfill",
                started_at=dt.datetime.now(dt.UTC),
                selected_season_ids=[],
            )
            db.add(run_row)
        run_row.status = "running"
        run_row.completed_at = None
        run_row.error_text = None
        db.commit()

    scheduled = hydrated = incomplete = failed = schedule_failures = 0
    seen_game_ids: set[str] = set()
    client = NhlClient(requests_per_second=rps_limit)
    try:
        current = date_from
        while current <= date_to:
            try:
                games = client.get_schedule_for_date(current)
            except Exception as exc:  # noqa: BLE001
                schedule_failures += 1
                LOGGER.warning("NHL schedule request failed date=%s error=%s", current, exc)
                current += dt.timedelta(days=1)
                continue
            for game in games:
                game_id = str(game.get("id") or "")
                if not game_id or game_id in seen_game_ids:
                    continue
                seen_game_ids.add(game_id)
                season_id = str(game.get("season") or "") or None
                normalized_schedule: dict[str, Any] | None = None
                try:
                    normalized_schedule = normalize_nhl_schedule_game(game)
                    landing = client.get_landing(game_id)
                    game_state = str(landing.get("gameState") or "").upper()
                    game_type = _int(landing.get("gameType") or game.get("gameType")) or 2
                    if game_state not in {"OFF", "FINAL"}:
                        if not dry_run:
                            _write_normalized_game(db, league_id, normalized_schedule)
                            db.commit()
                        scheduled += 1
                        continue
                    if game_type == 1 and not include_preseason:
                        continue
                    if game_type == 3 and not include_playoffs:
                        continue
                    normalized = normalize_nhl_completed_game(
                        landing,
                        game_summary_html=_optional_report(client, game_id, "GS"),
                        faceoff_summary_html=_optional_report(client, game_id, "FS"),
                    )
                    if not normalized["stats_complete"]:
                        incomplete += 1
                    if not dry_run:
                        _write_normalized_game(db, league_id, normalized)
                        _clear_failure(db, run_id=run_id, league_id=league_id, game_id=int(game_id), stage="hydrate")
                        # Do not hold row/index locks while making the next
                        # remote request. The daily scheduler may also write
                        # NHL rows, and PostgreSQL otherwise can deadlock.
                        db.commit()
                    scheduled += 1
                    hydrated += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    LOGGER.warning("NHL game hydration failed game_id=%s date=%s error=%s", game_id, current, exc)
                    if not dry_run:
                        # A database error aborts the active transaction; clear
                        # it before recording the retryable failure.
                        db.rollback()
                        if normalized_schedule is not None:
                            _write_normalized_game(db, league_id, normalized_schedule)
                        _record_failure(
                            db,
                            run_id=run_id,
                            league_id=league_id,
                            game_id=int(game_id),
                            season_id=season_id,
                            stage="hydrate",
                            error=exc,
                        )
                        db.commit()
            current += dt.timedelta(days=1)
    except Exception as exc:
        if run_row is not None:
            run_row.status = "failed"
            run_row.error_text = str(exc)[:4000]
            run_row.completed_at = dt.datetime.now(dt.UTC)
            db.commit()
        raise
    finally:
        client.close()
        if lock_acquired:
            _release_backfill_lock(db)

    result = {
        "league_code": "nhl",
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "scheduled_games": scheduled,
        "hydrated_games": hydrated,
        "incomplete_stat_games": incomplete,
        "failed_games": failed,
        "schedule_failures": schedule_failures,
        "dry_run": dry_run,
    }
    if run_row is not None:
        run_row.status = "completed" if not failed and not schedule_failures else "completed_with_failures"
        run_row.completed_at = dt.datetime.now(dt.UTC)
        run_row.counts_json = result
        db.commit()
        result["run_id"] = str(run_id)
    return result
