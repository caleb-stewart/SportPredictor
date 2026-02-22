from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from db.session import get_db
from schemas.modeling import (
    ExperimentReportResponse,
    FeatureProposalRunRequest,
    FeatureProposalRunResponse,
)
from services.experiments import (
    ExperimentNotFoundError,
    ExperimentValidationError,
    get_experiment,
    run_feature_proposal_experiment,
)
from services.data_backend import DataBackendError, require_supported_league_code

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("/feature-proposal/run", response_model=FeatureProposalRunResponse)
def run_feature_proposal(
    request: FeatureProposalRunRequest,
    league_code: str = Query(default="whl", description="League code (whl|ohl|lhjmq)"),
    db: Session = Depends(get_db),
) -> FeatureProposalRunResponse:
    try:
        normalized_league = require_supported_league_code(league_code)
    except DataBackendError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        result = run_feature_proposal_experiment(
            db=db,
            proposal_name=request.proposal_name,
            proposals=[item.model_dump() for item in request.proposals],
            seed=request.seed,
            league_code=normalized_league,
        )
    except ExperimentValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    response = FeatureProposalRunResponse.model_validate(result)
    response.league_code = normalized_league
    return response


@router.get("/{experiment_id}", response_model=ExperimentReportResponse)
def get_experiment_report(
    experiment_id: str,
    league_code: str = Query(default="whl", description="League code (whl|ohl|lhjmq)"),
    db: Session = Depends(get_db),
) -> ExperimentReportResponse:
    try:
        normalized_league = require_supported_league_code(league_code)
    except DataBackendError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        payload = get_experiment(db=db, experiment_id=experiment_id, league_code=normalized_league)
    except ExperimentValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    response = ExperimentReportResponse.model_validate(payload)
    response.league_code = normalized_league
    return response
