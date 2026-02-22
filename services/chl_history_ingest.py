from __future__ import annotations

import datetime as dt
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from core.config import settings
from db.models import ChlGame, ChlIngestGameFailure, ChlIngestRun, ChlTeam
from services.data_backend import (
    DataBackendError,
    hockeytech_api_key_for_league,
    hockeytech_client_code_for_league,
    primary_store,
    require_supported_league_code,
    resolve_league_id_for_store,
)
from services.hockeytech_client import HockeyTechAccessDeniedError, HockeyTechClient, HockeyTechClientError
from services.prediction_pipeline import (
    build_clock_game_upsert_values,
    build_schedule_game_upsert_values,
    recompute_rolling_averages,
)
from services.training import TrainingError, train_and_maybe_promote

LOGGER = logging.getLogger(__name__)


class ChlHistoryIngestError(RuntimeError):
    pass


@dataclass(frozen=True)
class _FetchResult:
    game_id: int
    season_id: str | None
    attempts: int
    clock: dict[str, Any] | None
    error: str | None


class _RateLimiter:
    def __init__(self, rps_limit: float) -> None:
        self._interval = (1.0 / rps_limit) if rps_limit > 0 else 0.0
        self._next_time = 0.0
        self._lock = Lock()

    def wait(self) -> None:
        if self._interval <= 0:
            return

        sleep_for = 0.0
        with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                sleep_for = self._next_time - now
            self._next_time = max(now, self._next_time) + self._interval

        if sleep_for > 0:
            time.sleep(sleep_for)


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clock_has_core_fields(clock: dict[str, Any]) -> bool:
    return all(
        [
            clock.get("home_goal_count") not in (None, ""),
            clock.get("visiting_goal_count") not in (None, ""),
            bool(clock.get("scoring")),
            bool(clock.get("shots_on_goal")),
            bool(clock.get("power_play")),
        ]
    )


def _merge_clock_with_gamesummary(clock: dict[str, Any], gamesummary: dict[str, Any]) -> dict[str, Any]:
    if not gamesummary:
        return clock

    merged = dict(clock)
    meta = gamesummary.get("meta") or {}

    total_goals = gamesummary.get("totalGoals") or {}
    if merged.get("home_goal_count") in (None, ""):
        candidate = _int_or_none(total_goals.get("home"))
        if candidate is None:
            candidate = _int_or_none(meta.get("home_goal_count"))
        if candidate is not None:
            merged["home_goal_count"] = str(candidate)
    if merged.get("visiting_goal_count") in (None, ""):
        candidate = _int_or_none(total_goals.get("visitor"))
        if candidate is None:
            candidate = _int_or_none(meta.get("visiting_goal_count"))
        if candidate is not None:
            merged["visiting_goal_count"] = str(candidate)

    if not merged.get("scoring"):
        goals_by_period = gamesummary.get("goalsByPeriod") or {}
        home = goals_by_period.get("home")
        visitor = goals_by_period.get("visitor")
        if isinstance(home, dict) and isinstance(visitor, dict):
            merged["scoring"] = {
                "home": {str(k): str(v) for k, v in home.items()},
                "visiting": {str(k): str(v) for k, v in visitor.items()},
            }

    if not merged.get("shots_on_goal"):
        shots_by_period = gamesummary.get("shotsByPeriod") or {}
        home = shots_by_period.get("home")
        visitor = shots_by_period.get("visitor")
        if isinstance(home, dict) and isinstance(visitor, dict):
            merged["shots_on_goal"] = {
                "home": {str(k): v for k, v in home.items()},
                "visiting": {str(k): v for k, v in visitor.items()},
            }

    if not merged.get("power_play"):
        pp_count = gamesummary.get("powerPlayCount") or {}
        pp_goals = gamesummary.get("powerPlayGoals") or {}
        if isinstance(pp_count, dict) and isinstance(pp_goals, dict):
            merged["power_play"] = {
                "total": {
                    "home": str(pp_count.get("home") if pp_count.get("home") is not None else "0"),
                    "visiting": str(pp_count.get("visitor") if pp_count.get("visitor") is not None else "0"),
                },
                "goals": {
                    "home": str(pp_goals.get("home") if pp_goals.get("home") is not None else "0"),
                    "visiting": str(pp_goals.get("visitor") if pp_goals.get("visitor") is not None else "0"),
                },
            }

    if merged.get("fow") in (None, {}):
        faceoffs = gamesummary.get("totalFaceoffs") or {}
        home = (faceoffs.get("home") or {}).get("won")
        visitor = (faceoffs.get("visitor") or {}).get("won")
        home_won = _int_or_none(home)
        visitor_won = _int_or_none(visitor)
        if home_won is not None and visitor_won is not None and (home_won + visitor_won) > 0:
            merged["fow"] = {"home": home_won, "visiting": visitor_won}

    if merged.get("status") in (None, "") and gamesummary.get("status_value") not in (None, ""):
        merged["status"] = str(gamesummary.get("status_value"))
    if merged.get("progress") in (None, "") and gamesummary.get("status_value") not in (None, ""):
        merged["progress"] = str(gamesummary.get("status_value"))
    if merged.get("period") in (None, "") and meta.get("period") not in (None, ""):
        merged["period"] = str(meta.get("period"))
    if merged.get("game_number") in (None, "") and meta.get("game_number") not in (None, ""):
        merged["game_number"] = str(meta.get("game_number"))
    if merged.get("season_id") in (None, "") and meta.get("season_id") not in (None, ""):
        merged["season_id"] = str(meta.get("season_id"))
    if merged.get("venue") in (None, "") and gamesummary.get("venue") not in (None, ""):
        merged["venue"] = str(gamesummary.get("venue"))

    if merged.get("home_team") in (None, {}):
        home = gamesummary.get("home") or {}
        if home:
            merged["home_team"] = {
                "name": home.get("name"),
                "team_id": home.get("team_id"),
            }
    if merged.get("visiting_team") in (None, {}):
        visitor = gamesummary.get("visitor") or {}
        if visitor:
            merged["visiting_team"] = {
                "name": visitor.get("name"),
                "team_id": visitor.get("team_id"),
            }

    return merged


def _season_kind(season_name: str | None) -> str | None:
    name = (season_name or "").lower()
    if "regular season" in name:
        return "regular"
    if "playoff" in name:
        return "playoff"
    if "pre-season" in name:
        return "preseason"
    return None


def _sort_seasons_desc(seasons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(seasons, key=lambda row: int(str(row.get("season_id") or "0")), reverse=True)


def _select_target_seasons(
    seasons: list[dict[str, Any]],
    *,
    include_regular: bool,
    include_playoffs: bool,
    include_preseason: bool,
    all_available: bool,
) -> list[dict[str, Any]]:
    allowed = {
        "regular": include_regular,
        "playoff": include_playoffs,
        "preseason": include_preseason,
    }

    selected: list[dict[str, Any]] = []
    for season in _sort_seasons_desc(seasons):
        kind = _season_kind(str(season.get("season_name") or ""))
        if kind is None:
            continue
        if allowed.get(kind, False):
            selected.append(season)

    if all_available:
        return selected

    # Non-all mode keeps a bounded recent window for safety.
    return selected[:8]


def _chunked(values: list[int], size: int = 1000) -> list[list[int]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def _existing_games_by_id(db: Session, league_id: int, game_ids: list[int]) -> dict[int, ChlGame]:
    if not game_ids:
        return {}

    out: dict[int, ChlGame] = {}
    for chunk in _chunked(game_ids):
        rows = db.scalars(select(ChlGame).where(ChlGame.league_id == league_id, ChlGame.game_id.in_(chunk))).all()
        for row in rows:
            out[int(row.game_id)] = row
    return out


def _game_needs_hydration(game: ChlGame | None) -> bool:
    if game is None:
        return True
    return any(
        [
            game.home_goal_count is None,
            game.away_goal_count is None,
            game.shots_on_goal is None,
            game.power_play is None,
            game.scoring_breakdown is None,
        ]
    )


def _upsert_team_row(db: Session, league_id: int, team_data: dict[str, Any], now: dt.datetime) -> int:
    provider_team_id = team_data.get("id")
    if provider_team_id in (None, ""):
        return 0

    values = {
        "league_id": league_id,
        "hockeytech_id": int(provider_team_id),
        "name": team_data.get("name") or team_data.get("city") or f"Team {provider_team_id}",
        "city": team_data.get("city"),
        "team_name": team_data.get("nickname") or team_data.get("team_caption"),
        "conference": team_data.get("conference_long_name") or team_data.get("conference_short_name"),
        "division": team_data.get("division_long_name") or team_data.get("division_short_name"),
        "logo_url": team_data.get("team_logo_url"),
        "active": True,
        "created_at": now,
        "updated_at": now,
    }

    db.execute(
        insert(ChlTeam)
        .values(**values)
        .on_conflict_do_update(
            index_elements=["league_id", "hockeytech_id"],
            set_={
                "name": values["name"],
                "city": values["city"],
                "team_name": values["team_name"],
                "conference": values["conference"],
                "division": values["division"],
                "logo_url": values["logo_url"],
                "active": True,
                "updated_at": now,
            },
        )
    )
    return 1


def _upsert_game_row(db: Session, league_id: int, values: dict[str, Any], update_values: dict[str, Any]) -> None:
    db.execute(
        insert(ChlGame)
        .values(**{**values, "league_id": league_id})
        .on_conflict_do_update(
            index_elements=["league_id", "game_id"],
            set_={**update_values, "league_id": league_id},
        )
    )


def _mark_failure(
    db: Session,
    *,
    run_id: uuid.UUID,
    league_id: int,
    game_id: int,
    season_id: str | None,
    stage: str,
    attempts: int,
    error_text: str,
) -> None:
    now = dt.datetime.now(dt.UTC)
    db.execute(
        insert(ChlIngestGameFailure)
        .values(
            run_id=run_id,
            league_id=league_id,
            game_id=game_id,
            season_id=season_id,
            stage=stage,
            attempts=attempts,
            last_error=error_text,
            last_seen_at=now,
        )
        .on_conflict_do_update(
            index_elements=["run_id", "league_id", "game_id", "stage"],
            set_={
                "season_id": season_id,
                "attempts": attempts,
                "last_error": error_text,
                "last_seen_at": now,
            },
        )
    )


def _clear_failure(db: Session, run_id: uuid.UUID, league_id: int, game_id: int, stage: str) -> None:
    db.execute(
        delete(ChlIngestGameFailure).where(
            ChlIngestGameFailure.run_id == run_id,
            ChlIngestGameFailure.league_id == league_id,
            ChlIngestGameFailure.game_id == game_id,
            ChlIngestGameFailure.stage == stage,
        )
    )


def _fetch_clock_with_retry(
    *,
    client: HockeyTechClient,
    limiter: _RateLimiter,
    game_id: int,
    season_id: str | None,
    retry_max: int,
) -> _FetchResult:
    attempts = 0
    last_error: str | None = None

    for attempt in range(1, retry_max + 1):
        attempts = attempt
        try:
            limiter.wait()
            clock = client.get_clock(game_id)
            if clock:
                return _FetchResult(game_id=game_id, season_id=season_id, attempts=attempts, clock=clock, error=None)
            last_error = "Empty clock payload"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)

        if attempt < retry_max:
            time.sleep(min(2.0, 0.25 * (2 ** (attempt - 1))))

    return _FetchResult(game_id=game_id, season_id=season_id, attempts=attempts, clock=None, error=last_error or "Unknown error")


def _failure_game_ids_for_run(db: Session, run_id: uuid.UUID, league_id: int) -> set[int]:
    rows = db.scalars(
        select(ChlIngestGameFailure.game_id).where(
            ChlIngestGameFailure.run_id == run_id,
            ChlIngestGameFailure.league_id == league_id,
        )
    ).all()
    return {int(x) for x in rows}


def _run_counts_template() -> dict[str, int]:
    return {
        "seasons_seen": 0,
        "teams_upserted": 0,
        "games_seen": 0,
        "games_upserted": 0,
        "clock_success": 0,
        "clock_failed": 0,
    }


def _season_inventory(
    client: HockeyTechClient,
    season_id: str,
) -> tuple[int, int]:
    teams = client.get_teams_by_season(season_id=season_id)
    schedule = client.get_schedule_by_season(season_id=season_id)
    return len(teams), len(schedule)


def run_chl_history_backfill(
    db: Session,
    *,
    league_code: str,
    all_available_seasons: bool = True,
    include_regular: bool = True,
    include_playoffs: bool = True,
    include_preseason: bool = True,
    resume_run_id: str | None = None,
    dry_run: bool = False,
    max_workers: int | None = None,
    rps_limit: float | None = None,
    retry_max: int | None = None,
    recompute_rolling: bool = True,
    train_after: bool = False,
) -> dict[str, Any]:
    LOGGER.info(
        "CHL history backfill starting league=%s dry_run=%s resume_run_id=%s",
        league_code,
        dry_run,
        resume_run_id,
    )

    try:
        normalized = require_supported_league_code(league_code)
    except DataBackendError as exc:
        raise ChlHistoryIngestError(str(exc)) from exc

    store = primary_store()
    league_id = resolve_league_id_for_store(db, store, normalized)
    if league_id is None:
        raise ChlHistoryIngestError(f"Unable to resolve league_id for league_code={normalized}")

    client = HockeyTechClient(
        client_code=hockeytech_client_code_for_league(normalized),
        api_key=hockeytech_api_key_for_league(normalized),
        timeout_seconds=settings.chl_ingest_timeout_seconds,
    )

    try:
        seasons = client.get_seasons()
    except HockeyTechAccessDeniedError as exc:
        raise ChlHistoryIngestError(str(exc)) from exc
    except HockeyTechClientError as exc:
        raise ChlHistoryIngestError(f"Failed to fetch seasons for league_code={normalized}: {exc}") from exc

    season_map = {str(row.get("season_id")): row for row in seasons if row.get("season_id") is not None}

    resume_uuid: uuid.UUID | None = None
    selected_seasons: list[dict[str, Any]]
    if resume_run_id:
        try:
            resume_uuid = uuid.UUID(resume_run_id)
        except ValueError as exc:
            raise ChlHistoryIngestError("resume_run_id must be a valid UUID") from exc

        existing_run = db.scalar(
            select(ChlIngestRun).where(ChlIngestRun.id == resume_uuid, ChlIngestRun.league_id == league_id)
        )
        if existing_run is None:
            raise ChlHistoryIngestError(
                f"resume_run_id={resume_run_id} not found for league_code={normalized}"
            )

        season_ids = [str(x) for x in (existing_run.selected_season_ids or [])]
        selected_seasons = [season_map[sid] for sid in season_ids if sid in season_map]
        if not selected_seasons:
            raise ChlHistoryIngestError(
                f"resume_run_id={resume_run_id} has no selectable seasons in current HockeyTech response"
            )
    else:
        selected_seasons = _select_target_seasons(
            seasons,
            include_regular=include_regular,
            include_playoffs=include_playoffs,
            include_preseason=include_preseason,
            all_available=all_available_seasons,
        )

    if not selected_seasons:
        raise ChlHistoryIngestError(
            "No seasons selected for ingest. Verify include flags and HockeyTech season payload."
        )

    selected_season_ids = [str(row.get("season_id")) for row in selected_seasons if row.get("season_id") is not None]
    LOGGER.info(
        "Selected %s seasons for league=%s (all_available=%s regular=%s playoffs=%s preseason=%s)",
        len(selected_season_ids),
        normalized,
        all_available_seasons,
        include_regular,
        include_playoffs,
        include_preseason,
    )

    running_stmt = select(ChlIngestRun.id).where(
        ChlIngestRun.league_id == league_id,
        ChlIngestRun.status == "running",
    )
    if resume_uuid is not None:
        running_stmt = running_stmt.where(ChlIngestRun.id != resume_uuid)
    other_running_ids = [str(x) for x in db.scalars(running_stmt).all()]
    if other_running_ids:
        raise ChlHistoryIngestError(
            "Another ingest run is already running for "
            f"league_code={normalized}: {', '.join(other_running_ids)}. "
            "Stop it or resume that run_id."
        )

    if dry_run:
        inventory: list[dict[str, Any]] = []
        for season in selected_seasons:
            sid = str(season.get("season_id"))
            teams_count, schedule_count = _season_inventory(client, sid)
            LOGGER.info(
                "Dry-run league=%s season_id=%s season_name=%s kind=%s teams=%s schedule_games=%s",
                normalized,
                sid,
                season.get("season_name"),
                _season_kind(str(season.get("season_name") or "")),
                teams_count,
                schedule_count,
            )
            inventory.append(
                {
                    "season_id": sid,
                    "season_name": season.get("season_name"),
                    "kind": _season_kind(str(season.get("season_name") or "")),
                    "teams_count": teams_count,
                    "schedule_count": schedule_count,
                }
            )

        return {
            "league_code": normalized,
            "dry_run": True,
            "selected_season_count": len(selected_season_ids),
            "selected_season_ids": selected_season_ids,
            "inventory": inventory,
        }

    effective_run_id = resume_uuid or uuid.uuid4()
    run_row = db.scalar(select(ChlIngestRun).where(ChlIngestRun.id == effective_run_id))
    if run_row is None:
        run_row = ChlIngestRun(
            id=effective_run_id,
            league_id=league_id,
            status="running",
            mode="historical_backfill",
            started_at=dt.datetime.now(dt.UTC),
            selected_season_ids=selected_season_ids,
            counts_json=_run_counts_template(),
        )
        db.add(run_row)
    else:
        run_row.status = "running"
        run_row.completed_at = None
        run_row.error_text = None
        run_row.selected_season_ids = selected_season_ids
        run_row.counts_json = run_row.counts_json or _run_counts_template()

    db.commit()

    failure_targets = _failure_game_ids_for_run(db, effective_run_id, league_id)
    counts = dict(run_row.counts_json or _run_counts_template())

    limiter = _RateLimiter(rps_limit=float(rps_limit or settings.chl_ingest_rps_limit))
    workers = max(1, int(max_workers or settings.chl_ingest_max_workers))
    retry_limit = max(1, int(retry_max or settings.chl_ingest_retry_max))
    LOGGER.info(
        "Run initialized league=%s run_id=%s workers=%s rps_limit=%s retry_max=%s resume_mode=%s prior_failures=%s",
        normalized,
        effective_run_id,
        workers,
        float(rps_limit or settings.chl_ingest_rps_limit),
        retry_limit,
        resume_uuid is not None,
        len(failure_targets),
    )

    try:
        season_total = len(selected_seasons)
        for season_index, season in enumerate(selected_seasons, start=1):
            sid = str(season.get("season_id"))
            season_name = str(season.get("season_name") or "")
            season_kind = _season_kind(season_name)
            LOGGER.info(
                "Season %s/%s starting league=%s season_id=%s season_name=%s kind=%s",
                season_index,
                season_total,
                normalized,
                sid,
                season_name,
                season_kind,
            )
            counts["seasons_seen"] = int(counts.get("seasons_seen", 0)) + 1

            now = dt.datetime.now(dt.UTC)
            teams = client.get_teams_by_season(season_id=sid)
            for team_data in teams:
                counts["teams_upserted"] = int(counts.get("teams_upserted", 0)) + _upsert_team_row(
                    db=db,
                    league_id=league_id,
                    team_data=team_data,
                    now=now,
                )
            db.commit()
            LOGGER.info(
                "Season %s/%s teams upsert complete league=%s season_id=%s teams_payload=%s teams_upserted_total=%s",
                season_index,
                season_total,
                normalized,
                sid,
                len(teams),
                counts.get("teams_upserted", 0),
            )

            schedule_rows = client.get_schedule_by_season(season_id=sid)
            schedule_by_game_id: dict[int, dict[str, Any]] = {}
            for game_data in schedule_rows:
                try:
                    values, update_values = build_schedule_game_upsert_values(
                        game_data=game_data,
                        now=now,
                        season_name_fallback=season_name,
                    )
                except Exception:
                    continue

                game_id = int(values["game_id"])
                schedule_by_game_id[game_id] = game_data
                _upsert_game_row(db=db, league_id=league_id, values=values, update_values=update_values)
                counts["games_upserted"] = int(counts.get("games_upserted", 0)) + 1
            db.commit()

            game_ids = sorted(schedule_by_game_id.keys())
            counts["games_seen"] = int(counts.get("games_seen", 0)) + len(game_ids)
            LOGGER.info(
                "Season %s/%s schedule upsert complete league=%s season_id=%s schedule_rows=%s unique_games=%s games_seen_total=%s",
                season_index,
                season_total,
                normalized,
                sid,
                len(schedule_rows),
                len(game_ids),
                counts.get("games_seen", 0),
            )

            existing_map = _existing_games_by_id(db, league_id=league_id, game_ids=game_ids)
            hydrate_targets = [
                game_id
                for game_id in game_ids
                if game_id in failure_targets or _game_needs_hydration(existing_map.get(game_id))
            ]
            LOGGER.info(
                "Season %s/%s hydration targets league=%s season_id=%s targets=%s",
                season_index,
                season_total,
                normalized,
                sid,
                len(hydrate_targets),
            )

            if hydrate_targets:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(
                            _fetch_clock_with_retry,
                            client=client,
                            limiter=limiter,
                            game_id=game_id,
                            season_id=sid,
                            retry_max=retry_limit,
                        ): game_id
                        for game_id in hydrate_targets
                    }

                    batch_counter = 0
                    season_clock_success = 0
                    season_clock_failed = 0
                    for future in as_completed(futures):
                        result = future.result()
                        batch_counter += 1

                        if result.clock is None:
                            counts["clock_failed"] = int(counts.get("clock_failed", 0)) + 1
                            season_clock_failed += 1
                            _mark_failure(
                                db,
                                run_id=effective_run_id,
                                league_id=league_id,
                                game_id=result.game_id,
                                season_id=result.season_id,
                                stage="clock_fetch",
                                attempts=result.attempts,
                                error_text=result.error or "clock fetch failed",
                            )
                        else:
                            counts["clock_success"] = int(counts.get("clock_success", 0)) + 1
                            season_clock_success += 1
                            try:
                                clock_payload = result.clock
                                if clock_payload is None:
                                    raise ValueError("Empty clock payload")
                                if not _clock_has_core_fields(clock_payload):
                                    try:
                                        gamesummary = client.get_gamesummary(result.game_id)
                                        clock_payload = _merge_clock_with_gamesummary(clock_payload, gamesummary)
                                    except Exception as summary_exc:  # noqa: BLE001
                                        LOGGER.warning(
                                            "Gamesummary fallback failed league=%s game_id=%s season_id=%s err=%s",
                                            normalized,
                                            result.game_id,
                                            sid,
                                            summary_exc,
                                        )
                                values, update_values = build_clock_game_upsert_values(
                                    game_data=schedule_by_game_id[result.game_id],
                                    clock=clock_payload,
                                    now=dt.datetime.now(dt.UTC),
                                )
                                _upsert_game_row(db=db, league_id=league_id, values=values, update_values=update_values)
                                _clear_failure(
                                    db,
                                    run_id=effective_run_id,
                                    league_id=league_id,
                                    game_id=result.game_id,
                                    stage="clock_fetch",
                                )
                            except Exception as exc:  # noqa: BLE001
                                counts["clock_failed"] = int(counts.get("clock_failed", 0)) + 1
                                season_clock_failed += 1
                                _mark_failure(
                                    db,
                                    run_id=effective_run_id,
                                    league_id=league_id,
                                    game_id=result.game_id,
                                    season_id=result.season_id,
                                    stage="normalize",
                                    attempts=1,
                                    error_text=str(exc),
                                )

                        if batch_counter % 100 == 0:
                            run_row.counts_json = dict(counts)
                            db.commit()
                            LOGGER.info(
                                "Season %s/%s hydration progress league=%s season_id=%s processed=%s/%s success=%s failed=%s",
                                season_index,
                                season_total,
                                normalized,
                                sid,
                                batch_counter,
                                len(hydrate_targets),
                                season_clock_success,
                                season_clock_failed,
                            )

                    LOGGER.info(
                        "Season %s/%s hydration complete league=%s season_id=%s processed=%s success=%s failed=%s",
                        season_index,
                        season_total,
                        normalized,
                        sid,
                        batch_counter,
                        season_clock_success,
                        season_clock_failed,
                    )

            run_row.counts_json = dict(counts)
            db.commit()
            LOGGER.info(
                "Season %s/%s complete league=%s season_id=%s cumulative=%s",
                season_index,
                season_total,
                normalized,
                sid,
                counts,
            )

        rolling_rows = 0
        if recompute_rolling:
            LOGGER.info("Recomputing rolling averages league=%s", normalized)
            rolling_rows = int(recompute_rolling_averages(db=db, league_code=normalized))
            LOGGER.info("Rolling averages recomputed league=%s rows=%s", normalized, rolling_rows)

        train_result: dict[str, Any] | None = None
        if train_after:
            LOGGER.info("Training requested after backfill league=%s", normalized)
            try:
                train_result = train_and_maybe_promote(promote=True, league_code=normalized)
            except TrainingError as exc:
                train_result = {"ok": False, "error": str(exc)}
            LOGGER.info("Training finished league=%s result=%s", normalized, train_result)

        run_row.status = "completed"
        run_row.completed_at = dt.datetime.now(dt.UTC)
        run_row.counts_json = dict(counts)
        db.commit()
        LOGGER.info(
            "CHL history backfill completed league=%s run_id=%s rolling_rows=%s final_counts=%s",
            normalized,
            effective_run_id,
            rolling_rows,
            counts,
        )

        return {
            "run_id": str(effective_run_id),
            "league_code": normalized,
            "status": run_row.status,
            "selected_season_ids": selected_season_ids,
            "counts": counts,
            "rolling_rows": rolling_rows,
            "train_result": train_result,
            "resume_mode": resume_uuid is not None,
        }
    except Exception as exc:  # noqa: BLE001
        run_row.status = "failed"
        run_row.completed_at = dt.datetime.now(dt.UTC)
        run_row.error_text = str(exc)
        run_row.counts_json = dict(counts)
        db.commit()
        LOGGER.exception(
            "CHL history backfill failed league=%s run_id=%s error=%s",
            normalized,
            effective_run_id,
            exc,
        )
        raise ChlHistoryIngestError(str(exc)) from exc
