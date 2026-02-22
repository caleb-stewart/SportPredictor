from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
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
from services.data_backend import (
    DataBackendError,
    apply_league_scope,
    primary_store,
    require_supported_league_code,
    resolve_league_id_for_store,
)

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/upcoming/run", response_model=UpcomingPredictionRunResponse)
def run_upcoming(
    league_code: str = Query(default="whl", description="League code (whl|ohl|lhjmq)"),
    date: dt.date | None = Query(default=None, description="Target date YYYY-MM-DD"),
    db: Session = Depends(get_db),
) -> UpcomingPredictionRunResponse:
    try:
        normalized_league = require_supported_league_code(league_code)
    except DataBackendError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    target_date = date or (dt.date.today() + dt.timedelta(days=1))

    try:
        stats = run_upcoming_predictions(db, target_date, league_code=normalized_league)
    except HockeyTechClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return UpcomingPredictionRunResponse(
        league_code=normalized_league,
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
        normalized_league = require_supported_league_code(request.league_code)
    except DataBackendError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        prediction, stored = run_custom_prediction(
            db=db,
            home_team_hockeytech_id=request.home_team_id,
            away_team_hockeytech_id=request.away_team_id,
            game_date=request.game_date,
            store_result=request.store_result,
            league_code=normalized_league,
        )
    except TeamNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InsufficientHistoryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DataBackendError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ModelNotAvailableError, PayloadContractError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    response = CustomPredictionApiResponse.model_validate(prediction)
    response.league_code = normalized_league

    if stored is not None:
        response.stored_record = CustomPredictionStoredResponse(
            id=str(stored.id),
            league_code=normalized_league,
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
    league_code: str = Query(default="whl", description="League code (whl|ohl|lhjmq)"),
    date_from: dt.date | None = Query(default=None),
    date_to: dt.date | None = Query(default=None),
    team_id: int | None = Query(default=None, description="Filter by HockeyTech team id"),
    k_value: int | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
    db: Session = Depends(get_db),
) -> list[PredictionHistoryRecord]:
    try:
        normalized_league = require_supported_league_code(league_code)
    except DataBackendError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    records = list_prediction_history(
        db=db,
        date_from=date_from,
        date_to=date_to,
        team_hockeytech_id=team_id,
        k_value=k_value,
        limit=limit,
        league_code=normalized_league,
    )

    team_ids: set[int] = set()
    for record in records:
        team_ids.add(record.home_team_id)
        team_ids.add(record.away_team_id)
        if record.predicted_winner_id:
            team_ids.add(record.predicted_winner_id)
        if record.actual_winner_id:
            team_ids.add(record.actual_winner_id)

    store = primary_store()
    team_model = store.team_model
    league_id = resolve_league_id_for_store(db, store, normalized_league)

    teams_by_db_id = {}
    if team_ids:
        team_stmt = select(team_model).where(team_model.id.in_(team_ids))
        team_stmt = apply_league_scope(team_stmt, team_model, league_id)
        teams_by_db_id = {team.id: team for team in db.scalars(team_stmt).all()}

    return [
        PredictionHistoryRecord(
            league_code=normalized_league,
            id=r.id,
            game_id=r.game_id,
            k_value=r.k_value,
            home_team_id=r.home_team_id,
            away_team_id=r.away_team_id,
            home_team_provider_id=teams_by_db_id.get(r.home_team_id).hockeytech_id if teams_by_db_id.get(r.home_team_id) else None,
            away_team_provider_id=teams_by_db_id.get(r.away_team_id).hockeytech_id if teams_by_db_id.get(r.away_team_id) else None,
            home_team_name=teams_by_db_id.get(r.home_team_id).name if teams_by_db_id.get(r.home_team_id) else None,
            away_team_name=teams_by_db_id.get(r.away_team_id).name if teams_by_db_id.get(r.away_team_id) else None,
            predicted_winner_id=r.predicted_winner_id,
            predicted_winner_provider_id=teams_by_db_id.get(r.predicted_winner_id).hockeytech_id if r.predicted_winner_id and teams_by_db_id.get(r.predicted_winner_id) else None,
            predicted_winner_name=teams_by_db_id.get(r.predicted_winner_id).name if r.predicted_winner_id and teams_by_db_id.get(r.predicted_winner_id) else None,
            home_team_probability=r.home_team_probability,
            away_team_probability=r.away_team_probability,
            actual_winner_id=r.actual_winner_id,
            actual_winner_provider_id=teams_by_db_id.get(r.actual_winner_id).hockeytech_id if r.actual_winner_id and teams_by_db_id.get(r.actual_winner_id) else None,
            actual_winner_name=teams_by_db_id.get(r.actual_winner_id).name if r.actual_winner_id and teams_by_db_id.get(r.actual_winner_id) else None,
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
        normalized_league = require_supported_league_code(request.league_code)
    except DataBackendError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        result = run_frozen_model_replay(
            db=db,
            date_from=request.date_from,
            date_to=request.date_to,
            selection_mode=request.selection_mode,
            last_n_games=request.last_n_games,
            dry_run=request.dry_run,
            overwrite=request.overwrite,
            rollback_on_proof_failure=request.rollback_on_proof_failure,
            archive_label=request.archive_label,
            league_code=normalized_league,
        )
    except ReplayValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    response = ReplayRunResponse.model_validate(result)
    response.league_code = normalized_league
    return response


@router.get("/replay/report/{run_id}", response_model=ReplayReportResponse)
def replay_report(
    run_id: str,
    league_code: str = Query(default="whl", description="League code (whl|ohl|lhjmq)"),
    db: Session = Depends(get_db),
) -> ReplayReportResponse:
    try:
        normalized_league = require_supported_league_code(league_code)
    except DataBackendError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        report = get_replay_report(db=db, run_id=run_id, league_code=normalized_league)
    except ReplayValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ReplayNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    response = ReplayReportResponse.model_validate(report)
    response.league_code = normalized_league
    return response
