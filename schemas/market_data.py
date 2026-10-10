from __future__ import annotations

import datetime as dt
from typing import Optional

from pydantic import BaseModel


class MarketDataSyncResponse(BaseModel):
    run_id: str
    provider: str
    sport_key: str
    operation: str
    events_written: int
    quotes_written: int
    quota_remaining: Optional[int] = None
    quota_cost: Optional[int] = None
    snapshot_at: Optional[dt.datetime] = None
    next_snapshot_at: Optional[dt.datetime] = None


class MarketDataStatusResponse(BaseModel):
    provider: str
    configured: bool
    current_sports: list[str]
    markets: list[str]
    regions: list[str]
    event_count: int
    quote_count: int
    latest_run: Optional[dict] = None
