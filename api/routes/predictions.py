from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from db.session import get_db
from schemas.predictions import (
    CustomPredictionApiResponse,
    CustomPredictionRequest,
    CustomPredictionStoredResponse,
    PredictionHistoryRecord,
    ReplayReportResponse,
    ReplayRunRequest,
    ReplayRunResponse,
    UpcomingPredictionRunResponse,
)
from services.feature_builder import InsufficientHistoryError, TeamNotFoundError
from services.hockeytech_client import HockeyTechClientError
from services.predictor import ModelNotAvailableError, PayloadContractError
from services.prediction_pipeline import (
    list_prediction_history,
    run_custom_prediction,
    run_upcoming_predictions,
)
from services.replay import (
    ReplayNotFoundError,
    ReplayValidationError,
    get_replay_report,
    run_frozen_model_replay,
)

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/upcoming/run", response_model=UpcomingPredictionRunResponse)
def run_upcoming(
    date: dt.date | None = Query(default=None, description="Target date YYYY-MM-DD"),
    db: Session = Depends(get_db),
) -> UpcomingPredictionRunResponse:
    target_date = date or (dt.date.today() + dt.timedelta(days=1))

    try:
        stats = run_upcoming_predictions(db, target_date)
    except HockeyTechClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return UpcomingPredictionRunResponse(
        target_date=target_date,
        predictions_written=stats.predictions_written,
        skipped_games=stats.skipped_games,
    )


@router.post("/custom", response_model=CustomPredictionApiResponse)
def predict_custom(
    request: CustomPredictionRequest,
    db: Session = Depends(get_db),
) -> CustomPredictionApiResponse:
    try:
        prediction, stored = run_custom_prediction(
            db=db,
            home_team_hockeytech_id=request.home_team_id,
            away_team_hockeytech_id=request.away_team_id,
            game_date=request.game_date,
            store_result=request.store_result,
        )
    except TeamNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InsufficientHistoryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ModelNotAvailableError, PayloadContractError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    response = CustomPredictionApiResponse.model_validate(prediction)

    if stored is not None:
        response.stored_record = CustomPredictionStoredResponse(
            id=str(stored.id),
            home_team_id=stored.home_team_id,
            away_team_id=stored.away_team_id,
            game_date=stored.game_date,
            home_team_probability=stored.home_team_probability,
            away_team_probability=stored.away_team_probability,
            predicted_winner_id=stored.predicted_winner_id,
            model_version=stored.model_version,
            model_family=stored.model_family,
            k_components=stored.k_components,
            created_at=stored.created_at,
        )

    return response


@router.get("/history", response_model=list[PredictionHistoryRecord])
def prediction_history(
    date_from: dt.date | None = Query(default=None),
    date_to: dt.date | None = Query(default=None),
    team_id: int | None = Query(default=None, description="Filter by HockeyTech team id"),
    k_value: int | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
    db: Session = Depends(get_db),
) -> list[PredictionHistoryRecord]:
    records = list_prediction_history(
        db=db,
        date_from=date_from,
        date_to=date_to,
        team_hockeytech_id=team_id,
        k_value=k_value,
        limit=limit,
    )

    return [
        PredictionHistoryRecord(
            id=r.id,
            game_id=r.game_id,
            k_value=r.k_value,
            home_team_id=r.home_team_id,
            away_team_id=r.away_team_id,
            predicted_winner_id=r.predicted_winner_id,
            home_team_probability=r.home_team_probability,
            away_team_probability=r.away_team_probability,
            actual_winner_id=r.actual_winner_id,
            correct=r.correct,
            prediction_date=r.prediction_date,
            model_version=r.model_version,
            model_family=r.model_family,
            raw_model_outputs=r.raw_model_outputs,
        )
        for r in records
    ]


@router.post("/replay/run", response_model=ReplayRunResponse)
def run_replay(
    request: ReplayRunRequest,
    db: Session = Depends(get_db),
) -> ReplayRunResponse:
    try:
        result = run_frozen_model_replay(
            db=db,
            date_from=request.date_from,
            date_to=request.date_to,
            dry_run=request.dry_run,
            overwrite=request.overwrite,
            archive_label=request.archive_label,
        )
    except ReplayValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ReplayRunResponse.model_validate(result)


@router.get("/replay/report/{run_id}", response_model=ReplayReportResponse)
def replay_report(
    run_id: str,
    db: Session = Depends(get_db),
) -> ReplayReportResponse:
    try:
        report = get_replay_report(db=db, run_id=run_id)
    except ReplayValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ReplayNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ReplayReportResponse.model_validate(report)
