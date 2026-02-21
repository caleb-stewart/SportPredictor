from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from schemas.modeling import ModelStatusResponse, ModelTrainResponse
from services.predictor import get_model_status
from services.training import TrainingError, train_and_maybe_promote

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/active", response_model=ModelStatusResponse)
def active_model() -> ModelStatusResponse:
    status = get_model_status()
    return ModelStatusResponse.model_validate(status)


@router.post("/train", response_model=ModelTrainResponse)
def train_model(promote: bool = Query(default=True)) -> ModelTrainResponse:
    try:
        details = train_and_maybe_promote(promote=promote)
    except TrainingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ModelTrainResponse(ok=bool(details.get("ok", False)), promoted=bool(details.get("promoted", promote)), details=details)
