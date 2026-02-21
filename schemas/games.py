from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from pydantic import BaseModel


class UpcomingGameResponse(BaseModel):
    game_id: int
    game_date: Optional[dt.date]
    status: Optional[str]
    venue: Optional[str]
    home_team_id: Optional[int]
    away_team_id: Optional[int]
    home_team: Optional[str]
    away_team: Optional[str]
    scheduled_time_utc: Optional[dt.datetime]
    scorebar_snapshot: Optional[dict[str, Any]]
