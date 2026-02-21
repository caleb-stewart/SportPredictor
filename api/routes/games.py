from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from db.session import get_db
from schemas.games import UpcomingGameResponse
from services.prediction_pipeline import get_upcoming_games, upsert_upcoming_schedule
from services.hockeytech_client import HockeyTechClientError
from fastapi import HTTPException

router = APIRouter(prefix="/games", tags=["games"])


@router.get("/upcoming", response_model=list[UpcomingGameResponse])
def upcoming_games(
    date: dt.date | None = Query(default=None, description="Target date YYYY-MM-DD"),
    refresh: bool = Query(default=True, description="Refresh schedule from HockeyTech before querying DB"),
    db: Session = Depends(get_db),
) -> list[UpcomingGameResponse]:
    target_date = date or (dt.date.today() + dt.timedelta(days=1))

    if refresh:
        try:
            upsert_upcoming_schedule(db, target_date)
        except HockeyTechClientError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    games = get_upcoming_games(db, target_date)
    return [
        UpcomingGameResponse(
            game_id=g.game_id,
            game_date=g.game_date,
            status=g.status,
            venue=g.venue,
            home_team_id=g.home_team_id,
            away_team_id=g.away_team_id,
            home_team=g.home_team,
            away_team=g.away_team,
            scheduled_time_utc=g.scheduled_time_utc,
            scorebar_snapshot=g.scorebar_snapshot,
        )
        for g in games
    ]
