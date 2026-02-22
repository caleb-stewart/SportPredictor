from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class KComponentProbability(BaseModel):
    home_team_prob: float
    away_team_prob: float


class PredictionResponse(BaseModel):
    league_code: str = "whl"
    home_team_prob: float
    away_team_prob: float
    predicted_winner_id: Optional[int]
    model_version: Optional[str]
    model_family: Optional[str]
    k_components: dict[str, KComponentProbability]


class CustomPredictionRequest(BaseModel):
    league_code: str = Field(default="whl", description="League code (whl|ohl|lhjmq)")
    home_team_id: int = Field(description="HockeyTech team id")
    away_team_id: int = Field(description="HockeyTech team id")
    game_date: dt.date
    store_result: bool = False


class UpcomingPredictionRunResponse(BaseModel):
    league_code: str = "whl"
    target_date: dt.date
    predictions_written: int
    skipped_games: int


class PredictionHistoryRecord(BaseModel):
    league_code: str = "whl"
    id: int
    game_id: int
    k_value: int
    home_team_id: int
    away_team_id: int
    home_team_provider_id: Optional[int] = None
    away_team_provider_id: Optional[int] = None
    home_team_name: Optional[str] = None
    away_team_name: Optional[str] = None
    predicted_winner_id: Optional[int]
    predicted_winner_provider_id: Optional[int] = None
    predicted_winner_name: Optional[str] = None
    home_team_probability: Optional[Decimal]
    away_team_probability: Optional[Decimal]
    actual_winner_id: Optional[int]
    actual_winner_provider_id: Optional[int] = None
    actual_winner_name: Optional[str] = None
    correct: Optional[bool]
    prediction_date: Optional[dt.datetime]
    model_version: Optional[str]
    model_family: Optional[str]
    raw_model_outputs: Optional[dict[str, Any]]


class CustomPredictionStoredResponse(BaseModel):
    id: str
    league_code: str = "whl"
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
    league_code: str = "whl"
    date_from: Optional[dt.date] = None
    date_to: Optional[dt.date] = None
    selection_mode: Literal["date_range", "last_n_completed_games"] = "date_range"
    last_n_games: Optional[int] = Field(default=None, ge=1)
    dry_run: bool
    overwrite: bool
    rollback_on_proof_failure: bool = True
    archive_label: Optional[str] = None


class ReplayRunResponse(BaseModel):
    league_code: str = "whl"
    run_id: str
    status: str
    selection_mode: Optional[Literal["date_range", "last_n_completed_games"]] = None
    last_n_games: Optional[int] = None
    active_model_version: Optional[str]
    games_scanned: int
    games_predicted: int
    games_skipped: int
    rows_upserted: int
    skip_reasons: dict[str, int]
    proof_summary: Optional[dict[str, Any]] = None


class ReplayReportResponse(BaseModel):
    league_code: str = "whl"
    run_id: str
    status: str
    selection_mode: Optional[Literal["date_range", "last_n_completed_games"]] = None
    last_n_games: Optional[int] = None
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
