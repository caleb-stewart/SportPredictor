from __future__ import annotations

import datetime as dt
import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select

from core.config import settings
from db.models import ChlLeague
from db.session import SessionLocal
from services.data_backend import CHL_LEAGUES
from services.model_compare import run_model_compare
from services.prediction_pipeline import (
    recompute_rolling_averages,
    run_upcoming_predictions,
    update_completed_games,
    upsert_upcoming_schedule,
)
from services.predictor import get_model_status
from services.training import promote_staged_model, train_and_maybe_promote

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def _active_league_codes() -> list[str]:
    db = SessionLocal()
    try:
        rows = db.scalars(select(ChlLeague).where(ChlLeague.active.is_(True)).order_by(ChlLeague.code.asc())).all()
        if rows:
            return [str(row.code).lower() for row in rows]
    except Exception:  # noqa: BLE001
        logger.exception("Failed to load active leagues from chl_leagues; using defaults.")
    finally:
        db.close()

    return sorted(CHL_LEAGUES.keys())


def _run_db_job(fn):
    db = SessionLocal()
    try:
        fn(db)
    except Exception:  # noqa: BLE001
        logger.exception("Scheduled job failed: %s", fn.__name__)
    finally:
        db.close()


def _job_upcoming_schedule(db, league_code: str):
    target = dt.date.today() + dt.timedelta(days=1)
    rows = upsert_upcoming_schedule(db, target, league_code=league_code)
    logger.info("%s upcoming schedule refreshed for %s (rows=%s)", league_code, target, rows)


def _job_yesterday_updates(db, league_code: str):
    target = dt.date.today() - dt.timedelta(days=1)
    rows = update_completed_games(db, target, league_code=league_code)
    logger.info("%s yesterday games refreshed for %s (rows=%s)", league_code, target, rows)


def _job_rolling(db, league_code: str):
    rows = recompute_rolling_averages(db, league_code=league_code)
    logger.info("%s rolling averages recomputed (rows touched=%s)", league_code, rows)


def _job_train(_db, league_code: str):
    details = train_and_maybe_promote(promote=False, league_code=league_code)
    logger.info("%s model training complete: %s", league_code, details)


def _job_promote(_db, league_code: str):
    details = promote_staged_model(league_code=league_code)
    logger.info("%s model promotion complete: %s", league_code, details)


def _job_predict_upcoming(db, league_code: str):
    target = dt.date.today() + dt.timedelta(days=1)
    stats = run_upcoming_predictions(db, target, league_code=league_code)
    logger.info("%s upcoming predictions done: %s", league_code, stats)


def _job_weekly_compare(db, league_code: str):
    if not settings.weekly_compare_enabled:
        logger.info("weekly model compare disabled; skipping.")
        return

    status = get_model_status(league_code=league_code)
    if not status.get("active"):
        logger.warning("weekly compare skipped: no active model.")
        return
    candidate = str(status.get("model_version") or "")
    baseline = settings.weekly_compare_baseline_model_version.strip()
    if not candidate or not baseline:
        logger.warning("weekly compare skipped: candidate or baseline model version missing.")
        return
    if candidate == baseline:
        logger.info("weekly compare skipped: candidate equals baseline (%s).", candidate)
        return

    today = dt.date.today()
    date_to = today - dt.timedelta(days=1)
    date_from = date_to - dt.timedelta(days=max(settings.weekly_compare_lookback_days, 30))

    result = run_model_compare(
        db=db,
        candidate_model_version=candidate,
        baseline_model_version=baseline,
        date_from=date_from,
        date_to=date_to,
        mode="frozen_replay",
        league_code=league_code,
    )
    logger.info("%s weekly model compare complete: %s", league_code, result)


def start_scheduler() -> BackgroundScheduler:
    global _scheduler

    if _scheduler and _scheduler.running:
        return _scheduler

    scheduler = BackgroundScheduler(timezone=settings.scheduler_timezone)

    for league_code in _active_league_codes():
        scheduler.add_job(
            lambda lc=league_code: _run_db_job(lambda db: _job_upcoming_schedule(db, lc)),
            trigger=CronTrigger(hour=9, minute=0, timezone=settings.scheduler_timezone),
            id=f"{league_code}_upcoming_schedule",
            replace_existing=True,
        )
        scheduler.add_job(
            lambda lc=league_code: _run_db_job(lambda db: _job_yesterday_updates(db, lc)),
            trigger=CronTrigger(hour=9, minute=10, timezone=settings.scheduler_timezone),
            id=f"{league_code}_yesterday_updates",
            replace_existing=True,
        )
        scheduler.add_job(
            lambda lc=league_code: _run_db_job(lambda db: _job_rolling(db, lc)),
            trigger=CronTrigger(hour=9, minute=20, timezone=settings.scheduler_timezone),
            id=f"{league_code}_rolling_recompute",
            replace_existing=True,
        )
        scheduler.add_job(
            lambda lc=league_code: _run_db_job(lambda db: _job_train(db, lc)),
            trigger=CronTrigger(hour=9, minute=30, timezone=settings.scheduler_timezone),
            id=f"{league_code}_model_train",
            replace_existing=True,
        )
        scheduler.add_job(
            lambda lc=league_code: _run_db_job(lambda db: _job_promote(db, lc)),
            trigger=CronTrigger(hour=9, minute=40, timezone=settings.scheduler_timezone),
            id=f"{league_code}_model_promote",
            replace_existing=True,
        )
        scheduler.add_job(
            lambda lc=league_code: _run_db_job(lambda db: _job_predict_upcoming(db, lc)),
            trigger=CronTrigger(hour=9, minute=45, timezone=settings.scheduler_timezone),
            id=f"{league_code}_predict_upcoming",
            replace_existing=True,
        )
        scheduler.add_job(
            lambda lc=league_code: _run_db_job(lambda db: _job_weekly_compare(db, lc)),
            trigger=CronTrigger(day_of_week="sun", hour=10, minute=0, timezone=settings.scheduler_timezone),
            id=f"{league_code}_model_weekly_compare",
            replace_existing=True,
        )

    scheduler.start()
    _scheduler = scheduler
    logger.info("APScheduler started with timezone=%s", settings.scheduler_timezone)
    return scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("APScheduler stopped")
    _scheduler = None
