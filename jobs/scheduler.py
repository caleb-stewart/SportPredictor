from __future__ import annotations

import datetime as dt
import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from core.config import settings
from db.session import SessionLocal
from services.prediction_pipeline import (
    recompute_rolling_averages,
    run_upcoming_predictions,
    update_completed_games,
    upsert_upcoming_schedule,
)
from services.training import promote_staged_model, train_and_maybe_promote

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def _run_db_job(fn):
    db = SessionLocal()
    try:
        fn(db)
    except Exception:  # noqa: BLE001
        logger.exception("Scheduled job failed: %s", fn.__name__)
    finally:
        db.close()


def _job_upcoming_schedule(db):
    target = dt.date.today() + dt.timedelta(days=1)
    rows = upsert_upcoming_schedule(db, target)
    logger.info("upcoming schedule refreshed for %s (rows=%s)", target, rows)


def _job_yesterday_updates(db):
    target = dt.date.today() - dt.timedelta(days=1)
    rows = update_completed_games(db, target)
    logger.info("yesterday games refreshed for %s (rows=%s)", target, rows)


def _job_rolling(db):
    rows = recompute_rolling_averages(db)
    logger.info("rolling averages recomputed (rows touched=%s)", rows)


def _job_train(_db):
    details = train_and_maybe_promote(promote=False)
    logger.info("model training complete: %s", details)


def _job_promote(_db):
    details = promote_staged_model()
    logger.info("model promotion complete: %s", details)


def _job_predict_upcoming(db):
    target = dt.date.today() + dt.timedelta(days=1)
    stats = run_upcoming_predictions(db, target)
    logger.info("upcoming predictions done: %s", stats)


def start_scheduler() -> BackgroundScheduler:
    global _scheduler

    if _scheduler and _scheduler.running:
        return _scheduler

    scheduler = BackgroundScheduler(timezone=settings.scheduler_timezone)

    scheduler.add_job(
        lambda: _run_db_job(_job_upcoming_schedule),
        trigger=CronTrigger(hour=9, minute=0, timezone=settings.scheduler_timezone),
        id="whl_upcoming_schedule",
        replace_existing=True,
    )
    scheduler.add_job(
        lambda: _run_db_job(_job_yesterday_updates),
        trigger=CronTrigger(hour=9, minute=10, timezone=settings.scheduler_timezone),
        id="whl_yesterday_updates",
        replace_existing=True,
    )
    scheduler.add_job(
        lambda: _run_db_job(_job_rolling),
        trigger=CronTrigger(hour=9, minute=20, timezone=settings.scheduler_timezone),
        id="whl_rolling_recompute",
        replace_existing=True,
    )
    scheduler.add_job(
        lambda: _run_db_job(_job_train),
        trigger=CronTrigger(hour=9, minute=30, timezone=settings.scheduler_timezone),
        id="whl_model_train",
        replace_existing=True,
    )
    scheduler.add_job(
        lambda: _run_db_job(_job_promote),
        trigger=CronTrigger(hour=9, minute=40, timezone=settings.scheduler_timezone),
        id="whl_model_promote",
        replace_existing=True,
    )
    scheduler.add_job(
        lambda: _run_db_job(_job_predict_upcoming),
        trigger=CronTrigger(hour=9, minute=45, timezone=settings.scheduler_timezone),
        id="whl_predict_upcoming",
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
