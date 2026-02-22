from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from pydantic import BaseModel, Field


class ModelStatusResponse(BaseModel):
    league_code: str = "whl"
    active: bool
    model_version: Optional[str] = None
    model_family: Optional[str] = None
    k_values: Optional[list[int]] = None
    error: Optional[str] = None


class ModelTrainResponse(BaseModel):
    ok: bool
    promoted: bool
    details: dict[str, Any]


class ModelCompareRunRequest(BaseModel):
    candidate_model_version: str
    baseline_model_version: str
    date_from: Optional[dt.date] = None
    date_to: Optional[dt.date] = None
    mode: str = Field(default="frozen_replay")


class ModelCompareRunResponse(BaseModel):
    league_code: str = "whl"
    run_id: str
    status: str
    mode: str
    baseline_model_version: str
    candidate_model_version: str
    games_scanned: int
    games_compared: int
    proof_summary: Optional[dict[str, Any]] = None


class ModelCompareReportResponse(BaseModel):
    league_code: str = "whl"
    run_id: str
    status: str
    mode: str
    baseline_model_version: str
    candidate_model_version: str
    date_from: Optional[dt.date]
    date_to: Optional[dt.date]
    started_at: Optional[dt.datetime]
    completed_at: Optional[dt.datetime]
    games_scanned: int
    games_compared: int
    proof_summary: Optional[dict[str, Any]] = None
    error_text: Optional[str] = None


class FeatureProposalItem(BaseModel):
    name: str
    left_feature: str
    op: str
    right_feature: str


class FeatureProposalRunRequest(BaseModel):
    proposal_name: str
    seed: int = 42
    proposals: list[FeatureProposalItem]


class FeatureProposalRunResponse(BaseModel):
    league_code: str = "whl"
    experiment_id: str
    status: str
    experiment_type: str
    result: dict[str, Any]


class ExperimentReportResponse(BaseModel):
    league_code: str = "whl"
    experiment_id: str
    status: str
    experiment_type: str
    created_at: Optional[dt.datetime]
    completed_at: Optional[dt.datetime]
    proposal: Optional[dict[str, Any]] = None
    result: Optional[dict[str, Any]] = None
    error_text: Optional[str] = None
