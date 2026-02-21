from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import WhlTeam
from db.session import get_db
from schemas.teams import TeamResponse

router = APIRouter(prefix="/teams", tags=["teams"])


@router.get("", response_model=list[TeamResponse])
def list_teams(
    active_only: bool = Query(default=True),
    db: Session = Depends(get_db),
) -> list[TeamResponse]:
    stmt = select(WhlTeam).order_by(WhlTeam.name.asc())
    if active_only:
        stmt = stmt.where(WhlTeam.active.is_(True))

    rows = db.scalars(stmt).all()
    return [
        TeamResponse(
            id=team.id,
            name=team.name,
            hockeytech_id=team.hockeytech_id,
            city=team.city,
            conference=team.conference,
            division=team.division,
            active=bool(team.active),
        )
        for team in rows
    ]
