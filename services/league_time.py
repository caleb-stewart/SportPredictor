from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from services.data_backend import CHL_LEAGUES


def league_today(league_code: str, *, now_utc: dt.datetime | None = None) -> dt.date:
    normalized = str(league_code or "").strip().lower()
    league_cfg = CHL_LEAGUES.get(normalized) or {}
    timezone_name = str(league_cfg.get("timezone") or "UTC")
    timezone = ZoneInfo(timezone_name)

    now = now_utc or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    return now.astimezone(timezone).date()
