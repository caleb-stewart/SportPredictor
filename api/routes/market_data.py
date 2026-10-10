from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from db.session import get_db
from schemas.market_data import MarketDataStatusResponse, MarketDataSyncResponse
from services.the_odds_api import TheOddsApiError, market_data_status, sync_current_odds

router = APIRouter(prefix="/market-data", tags=["market-data"])


@router.get("/status", response_model=MarketDataStatusResponse)
def status(db: Session = Depends(get_db)) -> MarketDataStatusResponse:
    return MarketDataStatusResponse.model_validate(market_data_status(db))


@router.post("/the-odds-api/sync/current", response_model=MarketDataSyncResponse)
def sync_current(
    sport_key: str = Query(default="icehockey_nhl"),
    db: Session = Depends(get_db),
) -> MarketDataSyncResponse:
    try:
        return MarketDataSyncResponse.model_validate(sync_current_odds(db, sport_key).as_dict())
    except TheOddsApiError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
