from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class TeamResponse(BaseModel):
    league_code: str = "whl"
    id: int
    name: str
    hockeytech_id: int
    city: Optional[str]
    conference: Optional[str]
    division: Optional[str]
    logo_url: Optional[str]
    active: bool
