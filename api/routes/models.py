from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from schemas.modeling import (
    ModelCompareReportResponse,
    ModelCompareRunRequest,
    ModelCompareRunResponse,
    ModelStatusResponse,
    ModelTrainResponse,
)
from services.model_compare import (
    ModelCompareNotFoundError,
    ModelCompareValidationError,
    get_model_compare_report,
    run_model_compare,
)
from services.predictor import ModelNotAvailableError, get_model_status
from services.training import TrainingError, train_and_maybe_promote
from db.session import get_db
from fastapi import Depends
from sqlalchemy.orm import Session
from services.data_backend import DataBackendError, require_supported_league_code

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/active", response_model=ModelStatusResponse)
def active_model(
    league_code: str = Query(default="whl", description="League code (whl|ohl|lhjmq)"),
) -> ModelStatusResponse:
    try:
        normalized_league = require_supported_league_code(league_code)
    except DataBackendError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    status = get_model_status(league_code=normalized_league)
    status["league_code"] = normalized_league
    return ModelStatusResponse.model_validate(status)


@router.post("/train", response_model=ModelTrainResponse)
def train_model(
    promote: bool = Query(default=True),
    league_code: str = Query(default="whl", description="League code (whl|ohl|lhjmq)"),
) -> ModelTrainResponse:
    try:
        normalized_league = require_supported_league_code(league_code)
    except DataBackendError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        details = train_and_maybe_promote(promote=promote, league_code=normalized_league)
    except TrainingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    details["league_code"] = normalized_league
    return ModelTrainResponse(ok=bool(details.get("ok", False)), promoted=bool(details.get("promoted", promote)), details=details)


@router.post("/compare/run", response_model=ModelCompareRunResponse)
def compare_models(
    request: ModelCompareRunRequest,
    league_code: str = Query(default="whl", description="League code (whl|ohl|lhjmq)"),
    db: Session = Depends(get_db),
) -> ModelCompareRunResponse:
    try:
        normalized_league = require_supported_league_code(league_code)
    except DataBackendError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        result = run_model_compare(
            db=db,
            candidate_model_version=request.candidate_model_version,
            baseline_model_version=request.baseline_model_version,
            date_from=request.date_from,
            date_to=request.date_to,
            mode=request.mode,
            league_code=normalized_league,
        )
    except ModelCompareValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TrainingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response = ModelCompareRunResponse.model_validate(result)
    response.league_code = normalized_league
    return response


@router.get("/compare/report/{run_id}", response_model=ModelCompareReportResponse)
def compare_report(
    run_id: str,
    league_code: str = Query(default="whl", description="League code (whl|ohl|lhjmq)"),
    db: Session = Depends(get_db),
) -> ModelCompareReportResponse:
    try:
        normalized_league = require_supported_league_code(league_code)
    except DataBackendError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        report = get_model_compare_report(db=db, run_id=run_id, league_code=normalized_league)
    except ModelCompareValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelCompareNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    response = ModelCompareReportResponse.model_validate(report)
    response.league_code = normalized_league
    return response
