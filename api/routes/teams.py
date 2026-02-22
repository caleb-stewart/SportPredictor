from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.session import get_db
from schemas.teams import TeamResponse
from services.data_backend import (
    DataBackendError,
    apply_league_scope,
    primary_store,
    require_supported_league_code,
    resolve_league_id_for_store,
)
from fastapi import HTTPException

router = APIRouter(prefix="/teams", tags=["teams"])


@router.get("", response_model=list[TeamResponse])
def list_teams(
    league_code: str = Query(default="whl", description="League code (whl|ohl|lhjmq)"),
    active_only: bool = Query(default=True),
    db: Session = Depends(get_db),
) -> list[TeamResponse]:
    try:
        normalized_league = require_supported_league_code(league_code)
    except DataBackendError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    store = primary_store()
    team_model = store.team_model
    league_id = resolve_league_id_for_store(db, store, normalized_league)

    stmt = select(team_model).order_by(team_model.name.asc())
    stmt = apply_league_scope(stmt, team_model, league_id)
    if active_only:
        stmt = stmt.where(team_model.active.is_(True))

    rows = db.scalars(stmt).all()
    return [
        TeamResponse(
            league_code=normalized_league,
            id=team.id,
            name=team.name,
            hockeytech_id=team.hockeytech_id,
            city=team.city,
            conference=team.conference,
            division=team.division,
            logo_url=team.logo_url,
            active=bool(team.active),
        )
        for team in rows
    ]
