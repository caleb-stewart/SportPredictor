from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any, Optional

from pydantic import BaseModel, Field


class KComponentProbability(BaseModel):
    home_team_prob: float
    away_team_prob: float


class PredictionResponse(BaseModel):
    home_team_prob: float
    away_team_prob: float
    predicted_winner_id: Optional[int]
    model_version: Optional[str]
    model_family: Optional[str]
    k_components: dict[str, KComponentProbability]


class CustomPredictionRequest(BaseModel):
    home_team_id: int = Field(description="HockeyTech WHL team id")
    away_team_id: int = Field(description="HockeyTech WHL team id")
    game_date: dt.date
    store_result: bool = False


class UpcomingPredictionRunResponse(BaseModel):
    target_date: dt.date
    predictions_written: int
    skipped_games: int


class PredictionHistoryRecord(BaseModel):
    id: int
    game_id: int
    k_value: int
    home_team_id: int
    away_team_id: int
    predicted_winner_id: Optional[int]
    home_team_probability: Optional[Decimal]
    away_team_probability: Optional[Decimal]
    actual_winner_id: Optional[int]
    correct: Optional[bool]
    prediction_date: Optional[dt.datetime]
    model_version: Optional[str]
    model_family: Optional[str]
    raw_model_outputs: Optional[dict[str, Any]]


class CustomPredictionStoredResponse(BaseModel):
    id: str
    home_team_id: int
    away_team_id: int
    game_date: dt.date
    home_team_probability: Decimal
    away_team_probability: Decimal
    predicted_winner_id: Optional[int]
    model_version: Optional[str]
    model_family: Optional[str]
    k_components: Optional[dict[str, Any]]
    created_at: dt.datetime


class CustomPredictionApiResponse(PredictionResponse):
    stored_record: Optional[CustomPredictionStoredResponse] = None


class ReplayRunRequest(BaseModel):
    date_from: Optional[dt.date] = None
    date_to: Optional[dt.date] = None
    dry_run: bool
    overwrite: bool
    archive_label: Optional[str] = None


class ReplayRunResponse(BaseModel):
    run_id: str
    status: str
    active_model_version: Optional[str]
    games_scanned: int
    games_predicted: int
    games_skipped: int
    rows_upserted: int
    skip_reasons: dict[str, int]
    proof_summary: Optional[dict[str, Any]] = None


class ReplayReportResponse(BaseModel):
    run_id: str
    status: str
    date_from: Optional[dt.date]
    date_to: Optional[dt.date]
    started_at: Optional[dt.datetime]
    completed_at: Optional[dt.datetime]
    active_model_version: Optional[str]
    games_scanned: int
    games_predicted: int
    games_skipped: int
    rows_upserted: int
    skip_reasons: dict[str, int]
    proof_summary: Optional[dict[str, Any]] = None
    error_text: Optional[str] = None
