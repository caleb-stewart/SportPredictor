from __future__ import annotations

import datetime as dt
import logging
import uuid
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from core.config import settings
from db.models import ChlMaintenanceCheckpoint, ChlMaintenanceRun
from services.data_backend import (
    DataBackendError,
    primary_store,
    require_supported_league_code,
    resolve_league_id_for_store,
)
from services.prediction_pipeline import (
    recompute_rolling_averages,
    run_upcoming_predictions,
    update_completed_games,
    upsert_upcoming_schedule,
)
from services.league_time import league_today as _shared_league_today

LOGGER = logging.getLogger(__name__)


class MaintenanceError(RuntimeError):
    pass


class MaintenanceRefreshDisabledError(MaintenanceError):
    pass


class MaintenanceRunFailedError(MaintenanceError):
    pass


def _league_today(league_code: str, *, now_utc: dt.datetime | None = None) -> dt.date:
    return _shared_league_today(league_code, now_utc=now_utc)


def _empty_totals() -> dict[str, int]:
    return {
        "schedule_rows": 0,
        "completed_rows": 0,
        "rolling_rows": 0,
        "predictions_written": 0,
        "skipped_games": 0,
    }


def _merge_totals(aggregate: dict[str, int], day_result: dict[str, Any]) -> dict[str, int]:
    aggregate["schedule_rows"] += int(day_result.get("schedule_rows", 0) or 0)
    aggregate["completed_rows"] += int(day_result.get("completed_rows", 0) or 0)
    aggregate["rolling_rows"] += int(day_result.get("rolling_rows", 0) or 0)
    aggregate["predictions_written"] += int(day_result.get("predictions_written", 0) or 0)
    aggregate["skipped_games"] += int(day_result.get("skipped_games", 0) or 0)
    return aggregate


def compute_catchup_window(
    *,
    today: dt.date,
    last_successful_run_date: dt.date | None,
    max_catchup_days: int,
) -> tuple[dt.date | None, dt.date | None, int]:
    safe_max_days = max(1, int(max_catchup_days or 1))
    start = today if last_successful_run_date is None else (last_successful_run_date + dt.timedelta(days=1))

    if start > today:
        return None, None, 0

    if (today - start).days + 1 > safe_max_days:
        start = today - dt.timedelta(days=safe_max_days - 1)

    return start, today, (today - start).days + 1


def _resolve_league(db: Session, league_code: str) -> tuple[str, int]:
    try:
        normalized = require_supported_league_code(league_code)
        league_id = resolve_league_id_for_store(db, primary_store(), normalized)
    except DataBackendError as exc:
        raise MaintenanceError(str(exc)) from exc

    if league_id is None:
        raise MaintenanceError(f"Unable to resolve league_id for league_code={normalized}")
    return normalized, int(league_id)


def _serialize_run(row: ChlMaintenanceRun, league_code: str) -> dict[str, Any]:
    return {
        "run_id": str(row.id),
        "league_code": league_code,
        "trigger_source": row.trigger_source,
        "status": row.status,
        "from_date": row.from_date,
        "to_date": row.to_date,
        "days_attempted": int(row.days_attempted or 0),
        "days_succeeded": int(row.days_succeeded or 0),
        "totals_json": row.totals_json or _empty_totals(),
        "error_text": row.error_text,
        "started_at": row.started_at,
        "completed_at": row.completed_at,
    }


def _load_or_create_checkpoint(db: Session, league_id: int) -> ChlMaintenanceCheckpoint:
    checkpoint = db.scalar(select(ChlMaintenanceCheckpoint).where(ChlMaintenanceCheckpoint.league_id == league_id))
    if checkpoint is not None:
        return checkpoint

    checkpoint = ChlMaintenanceCheckpoint(
        league_id=league_id,
        updated_at=dt.datetime.now(dt.UTC),
    )
    db.add(checkpoint)
    db.flush()
    return checkpoint


def _update_checkpoint(
    db: Session,
    *,
    league_id: int,
    run_id: uuid.UUID | None,
    successful_date: dt.date,
) -> ChlMaintenanceCheckpoint:
    checkpoint = _load_or_create_checkpoint(db, league_id)
    if checkpoint.last_successful_run_date is None or successful_date >= checkpoint.last_successful_run_date:
        checkpoint.last_successful_run_date = successful_date
        checkpoint.last_successful_run_id = run_id
    checkpoint.updated_at = dt.datetime.now(dt.UTC)
    db.flush()
    return checkpoint


def run_day_maintenance(
    db: Session,
    league_code: str,
    run_date: dt.date,
    include_training: bool = False,
) -> dict[str, Any]:
    normalized = require_supported_league_code(league_code)

    schedule_target = run_date + dt.timedelta(days=1)
    completed_target = run_date - dt.timedelta(days=1)

    schedule_rows = int(upsert_upcoming_schedule(db, schedule_target, league_code=normalized))
    completed_rows = int(update_completed_games(db, completed_target, league_code=normalized))
    rolling_rows = int(recompute_rolling_averages(db, league_code=normalized))
    upcoming = run_upcoming_predictions(db, schedule_target, league_code=normalized)

    result: dict[str, Any] = {
        "run_date": run_date,
        "schedule_target_date": schedule_target,
        "completed_target_date": completed_target,
        "schedule_rows": schedule_rows,
        "completed_rows": completed_rows,
        "rolling_rows": rolling_rows,
        "predictions_written": int(upcoming.predictions_written),
        "skipped_games": int(upcoming.skipped_games),
    }

    if include_training:
        from services.training import train_and_maybe_promote

        result["training"] = train_and_maybe_promote(promote=True, league_code=normalized)

    return result


def record_scheduler_maintenance_success(
    db: Session,
    *,
    league_code: str,
    run_date: dt.date,
    totals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized, league_id = _resolve_league(db, league_code)
    now = dt.datetime.now(dt.UTC)
    run_row = ChlMaintenanceRun(
        id=uuid.uuid4(),
        league_id=league_id,
        trigger_source="scheduler",
        status="completed",
        from_date=run_date,
        to_date=run_date,
        days_attempted=1,
        days_succeeded=1,
        totals_json={
            **_empty_totals(),
            **(totals or {}),
        },
        started_at=now,
        completed_at=now,
        created_at=now,
        updated_at=now,
    )
    db.add(run_row)
    db.flush()
    _update_checkpoint(db, league_id=league_id, run_id=run_row.id, successful_date=run_date)
    db.commit()
    return _serialize_run(run_row, normalized)


def get_maintenance_status(db: Session, league_code: str) -> dict[str, Any]:
    normalized, league_id = _resolve_league(db, league_code)

    checkpoint = db.scalar(select(ChlMaintenanceCheckpoint).where(ChlMaintenanceCheckpoint.league_id == league_id))
    latest_run = db.scalar(
        select(ChlMaintenanceRun)
        .where(ChlMaintenanceRun.league_id == league_id)
        .order_by(desc(ChlMaintenanceRun.started_at), desc(ChlMaintenanceRun.created_at))
        .limit(1)
    )

    return {
        "league_code": normalized,
        "enabled": bool(settings.local_maintenance_refresh_enabled),
        "max_catchup_days": int(settings.local_maintenance_max_catchup_days),
        "checkpoint_date": checkpoint.last_successful_run_date if checkpoint else None,
        "last_successful_run_id": str(checkpoint.last_successful_run_id) if checkpoint and checkpoint.last_successful_run_id else None,
        "latest_run": _serialize_run(latest_run, normalized) if latest_run is not None else None,
    }


def run_catchup_maintenance(
    db: Session,
    *,
    league_code: str,
    trigger_source: str = "manual",
    include_training: bool = False,
) -> dict[str, Any]:
    normalized, league_id = _resolve_league(db, league_code)

    manual_trigger = trigger_source.strip().lower() == "manual"
    if manual_trigger and not settings.local_maintenance_refresh_enabled:
        raise MaintenanceRefreshDisabledError("Local maintenance refresh is disabled for this environment.")

    checkpoint = db.scalar(select(ChlMaintenanceCheckpoint).where(ChlMaintenanceCheckpoint.league_id == league_id))
    today = _league_today(normalized)
    from_date, to_date, days_attempted = compute_catchup_window(
        today=today,
        last_successful_run_date=checkpoint.last_successful_run_date if checkpoint else None,
        max_catchup_days=settings.local_maintenance_max_catchup_days,
    )

    now = dt.datetime.now(dt.UTC)
    run_row = ChlMaintenanceRun(
        id=uuid.uuid4(),
        league_id=league_id,
        trigger_source=trigger_source,
        status="running",
        from_date=from_date,
        to_date=to_date,
        days_attempted=days_attempted,
        days_succeeded=0,
        totals_json=_empty_totals(),
        started_at=now,
        created_at=now,
        updated_at=now,
    )
    db.add(run_row)
    db.commit()

    totals = _empty_totals()
    days_succeeded = 0
    last_successful_day: dt.date | None = None

    try:
        current = from_date
        while current is not None and to_date is not None and current <= to_date:
            day_result = run_day_maintenance(
                db=db,
                league_code=normalized,
                run_date=current,
                include_training=include_training,
            )
            totals = _merge_totals(totals, day_result)
            days_succeeded += 1
            last_successful_day = current
            run_row.days_succeeded = days_succeeded
            run_row.totals_json = dict(totals)
            run_row.updated_at = dt.datetime.now(dt.UTC)
            db.commit()
            current = current + dt.timedelta(days=1)

        run_row.status = "completed"
        run_row.completed_at = dt.datetime.now(dt.UTC)
        run_row.days_succeeded = days_succeeded
        run_row.totals_json = dict(totals)
        run_row.updated_at = dt.datetime.now(dt.UTC)

        if last_successful_day is not None:
            _update_checkpoint(db, league_id=league_id, run_id=run_row.id, successful_date=last_successful_day)

        db.commit()
        return _serialize_run(run_row, normalized)
    except Exception as exc:  # noqa: BLE001
        run_row.status = "failed"
        run_row.error_text = str(exc)
        run_row.completed_at = dt.datetime.now(dt.UTC)
        run_row.days_succeeded = days_succeeded
        run_row.totals_json = dict(totals)
        run_row.updated_at = dt.datetime.now(dt.UTC)

        if last_successful_day is not None:
            _update_checkpoint(db, league_id=league_id, run_id=run_row.id, successful_date=last_successful_day)

        db.commit()
        LOGGER.exception(
            "League maintenance refresh failed league=%s run_id=%s error=%s",
            normalized,
            run_row.id,
            exc,
        )
        raise MaintenanceRunFailedError(str(exc)) from exc
