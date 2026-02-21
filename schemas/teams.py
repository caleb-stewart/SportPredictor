from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class TeamResponse(BaseModel):
    id: int
    name: str
    hockeytech_id: int
    city: Optional[str]
    conference: Optional[str]
    division: Optional[str]
    active: bool
