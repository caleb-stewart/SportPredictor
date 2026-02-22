from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from db.session import get_db
from schemas.leagues import (
    DailyReportDetailResponse,
    DailyReportSummaryResponse,
    LeagueGameDetailResponse,
    LeagueResponse,
    LeagueTeamResponse,
    NextSlateResponse,
    PredictionPageResponse,
    PredictedGameResponse,
    RecentResultDaysResponse,
)
from services.leagues import (
    DailyReportNotFoundError,
    GameNotFoundError,
    UnknownLeagueError,
    get_daily_report_detail,
    get_game_detail,
    get_next_predicted_slate,
    list_daily_reports,
    list_league_teams,
    list_prediction_results,
    list_predicted_games,
    list_recent_result_days,
    list_supported_leagues,
    list_upcoming_predictions,
)

router = APIRouter(prefix="/leagues", tags=["leagues"])


@router.get("", response_model=list[LeagueResponse])
def leagues() -> list[LeagueResponse]:
    return [LeagueResponse.model_validate(item) for item in list_supported_leagues()]


@router.get("/{league_code}/teams", response_model=list[LeagueTeamResponse])
def league_teams(
    league_code: str,
    active_only: bool = Query(default=True),
    db: Session = Depends(get_db),
) -> list[LeagueTeamResponse]:
    try:
        rows = list_league_teams(db=db, league_code=league_code, active_only=active_only)
    except UnknownLeagueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return [LeagueTeamResponse.model_validate(row) for row in rows]


@router.get("/{league_code}/predicted-games", response_model=list[PredictedGameResponse])
def league_predicted_games(
    league_code: str,
    prediction_date_from: dt.date | None = Query(default=None),
    prediction_date_to: dt.date | None = Query(default=None),
    game_date_from: dt.date | None = Query(default=None),
    game_date_to: dt.date | None = Query(default=None),
    team_provider_id: int | None = Query(default=None),
    result: str = Query(default="all", description="One of: all, correct, incorrect, pending"),
    limit: int = Query(default=200, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    sort: str = Query(default="prediction_date_desc", description="prediction_date_desc|prediction_date_asc|game_date_desc|game_date_asc"),
    db: Session = Depends(get_db),
) -> list[PredictedGameResponse]:
    try:
        rows = list_predicted_games(
            db=db,
            league_code=league_code,
            prediction_date_from=prediction_date_from,
            prediction_date_to=prediction_date_to,
            game_date_from=game_date_from,
            game_date_to=game_date_to,
            team_provider_id=team_provider_id,
            result=result,
            limit=limit,
            offset=offset,
            sort=sort,
        )
    except UnknownLeagueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return [PredictedGameResponse.model_validate(row) for row in rows]


@router.get("/{league_code}/predictions/next-slate", response_model=NextSlateResponse)
def league_predictions_next_slate(
    league_code: str,
    as_of_date: dt.date | None = Query(default=None),
    team_provider_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
) -> NextSlateResponse:
    try:
        payload = get_next_predicted_slate(
            db=db,
            league_code=league_code,
            as_of_date=as_of_date,
            team_provider_id=team_provider_id,
        )
    except UnknownLeagueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return NextSlateResponse.model_validate(payload)


@router.get("/{league_code}/predictions/upcoming", response_model=PredictionPageResponse)
def league_predictions_upcoming(
    league_code: str,
    as_of_date: dt.date | None = Query(default=None),
    game_date_from: dt.date | None = Query(default=None),
    game_date_to: dt.date | None = Query(default=None),
    team_provider_id: int | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> PredictionPageResponse:
    try:
        payload = list_upcoming_predictions(
            db=db,
            league_code=league_code,
            as_of_date=as_of_date,
            game_date_from=game_date_from,
            game_date_to=game_date_to,
            team_provider_id=team_provider_id,
            limit=limit,
            offset=offset,
        )
    except UnknownLeagueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictionPageResponse.model_validate(payload)


@router.get("/{league_code}/predictions/results", response_model=PredictionPageResponse)
def league_predictions_results(
    league_code: str,
    game_date_from: dt.date | None = Query(default=None),
    game_date_to: dt.date | None = Query(default=None),
    team_provider_id: int | None = Query(default=None),
    result: str = Query(default="all", description="One of: all, correct, incorrect"),
    sort: str = Query(default="game_date_desc", description="prediction_date_desc|prediction_date_asc|game_date_desc|game_date_asc"),
    limit: int = Query(default=20, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> PredictionPageResponse:
    try:
        payload = list_prediction_results(
            db=db,
            league_code=league_code,
            game_date_from=game_date_from,
            game_date_to=game_date_to,
            team_provider_id=team_provider_id,
            result=result,
            sort=sort,
            limit=limit,
            offset=offset,
        )
    except UnknownLeagueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictionPageResponse.model_validate(payload)


@router.get("/{league_code}/predictions/results/recent-days", response_model=RecentResultDaysResponse)
def league_prediction_result_recent_days(
    league_code: str,
    days: int = Query(default=3, ge=1, le=31),
    as_of_date: dt.date | None = Query(default=None),
    team_provider_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
) -> RecentResultDaysResponse:
    try:
        payload = list_recent_result_days(
            db=db,
            league_code=league_code,
            days=days,
            team_provider_id=team_provider_id,
            as_of_date=as_of_date,
        )
    except UnknownLeagueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return RecentResultDaysResponse.model_validate(payload)


@router.get("/{league_code}/reports/daily", response_model=list[DailyReportSummaryResponse])
def league_daily_reports(
    league_code: str,
    date_from: dt.date | None = Query(default=None),
    date_to: dt.date | None = Query(default=None),
    team_provider_id: int | None = Query(default=None),
    limit: int = Query(default=120, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[DailyReportSummaryResponse]:
    try:
        rows = list_daily_reports(
            db=db,
            league_code=league_code,
            date_from=date_from,
            date_to=date_to,
            team_provider_id=team_provider_id,
            limit=limit,
            offset=offset,
        )
    except UnknownLeagueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return [DailyReportSummaryResponse.model_validate(row) for row in rows]


@router.get("/{league_code}/reports/daily/{prediction_date}", response_model=DailyReportDetailResponse)
def league_daily_report_detail(
    league_code: str,
    prediction_date: dt.date,
    team_provider_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
) -> DailyReportDetailResponse:
    try:
        payload = get_daily_report_detail(
            db=db,
            league_code=league_code,
            prediction_date=prediction_date,
            team_provider_id=team_provider_id,
        )
    except UnknownLeagueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DailyReportNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return DailyReportDetailResponse.model_validate(payload)


@router.get("/{league_code}/games/{game_id}", response_model=LeagueGameDetailResponse)
def league_game_detail(
    league_code: str,
    game_id: int,
    db: Session = Depends(get_db),
) -> LeagueGameDetailResponse:
    try:
        payload = get_game_detail(db=db, league_code=league_code, game_id=game_id)
    except UnknownLeagueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GameNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return LeagueGameDetailResponse.model_validate(payload)
