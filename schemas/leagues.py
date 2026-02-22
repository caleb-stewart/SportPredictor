from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from pydantic import BaseModel


class LeagueResponse(BaseModel):
    code: str
    name: str
    sport: str
    active: bool
    timezone: str


class LeagueTeamResponse(BaseModel):
    db_team_id: int
    provider_team_id: int
    name: str
    city: Optional[str]
    conference: Optional[str]
    division: Optional[str]
    logo_url: Optional[str]
    active: bool


class GameScoreResponse(BaseModel):
    home: Optional[int]
    away: Optional[int]


class PredictionByKResponse(BaseModel):
    k_value: int
    home_team_probability: Optional[float]
    away_team_probability: Optional[float]
    predicted_winner_db_team_id: Optional[int]
    predicted_winner_provider_team_id: Optional[int]
    predicted_winner_name: Optional[str]
    actual_winner_db_team_id: Optional[int]
    actual_winner_provider_team_id: Optional[int]
    actual_winner_name: Optional[str]
    correct: Optional[bool]
    model_version: Optional[str]
    model_family: Optional[str]
    raw_model_outputs: Optional[dict[str, Any]]


class PredictedGameResponse(BaseModel):
    league_code: str
    game_id: int
    game_date: Optional[dt.date]
    prediction_timestamp: Optional[dt.datetime]
    prediction_date: Optional[dt.date]
    status: Optional[str]
    period: Optional[str]
    venue: Optional[str]
    scheduled_time_utc: Optional[dt.datetime]
    final_score: Optional[GameScoreResponse]
    home_team: LeagueTeamResponse
    away_team: LeagueTeamResponse
    predictions_by_k: list[PredictionByKResponse]
    consensus_k_value: Optional[int]
    consensus_predicted_winner_db_team_id: Optional[int]
    consensus_predicted_winner_provider_team_id: Optional[int]
    consensus_predicted_winner_name: Optional[str]
    consensus_correct: Optional[bool]
    resolved_count: int
    correct_count: int


class DailyReportSummaryResponse(BaseModel):
    prediction_date: dt.date
    games_count: int
    predictions_count: int
    resolved_count: int
    correct_count: int
    accuracy_pct: Optional[float]


class DailyReportDetailResponse(BaseModel):
    league_code: str
    prediction_date: dt.date
    summary: DailyReportSummaryResponse
    games: list[PredictedGameResponse]


class PredictionPageResponse(BaseModel):
    items: list[PredictedGameResponse]
    total: int
    limit: int
    offset: int
    has_more: bool


class NextSlateResponse(BaseModel):
    league_code: str
    as_of_date: dt.date
    target_game_date: Optional[dt.date]
    games_count: int
    predictions_count: int
    games: list[PredictedGameResponse]


class RecentResultDayResponse(BaseModel):
    game_date: dt.date
    games_count: int
    resolved_count: int
    correct_count: int
    accuracy_pct: Optional[float]
    games: list[PredictedGameResponse]


class RecentResultDaysResponse(BaseModel):
    league_code: str
    as_of_date: dt.date
    days: list[RecentResultDayResponse]


class GameLinksResponse(BaseModel):
    ticket_url: Optional[str]
    game_summary_url: Optional[str]
    home_audio_url: Optional[str]
    away_audio_url: Optional[str]
    home_video_url: Optional[str]
    away_video_url: Optional[str]
    home_webcast_url: Optional[str]
    away_webcast_url: Optional[str]


class LeagueGameDetailResponse(BaseModel):
    league_code: str
    game_id: int
    game_date: Optional[dt.date]
    status: Optional[str]
    venue: Optional[str]
    scheduled_time_utc: Optional[dt.datetime]
    period: Optional[str]
    final_score: Optional[GameScoreResponse]
    home_team: LeagueTeamResponse
    away_team: LeagueTeamResponse
    scoring_breakdown: Optional[dict[str, Any]]
    shots_on_goal: Optional[dict[str, Any]]
    power_play: Optional[dict[str, Any]]
    fow: Optional[dict[str, Any]]
    links: GameLinksResponse
    predictions_by_k: list[PredictionByKResponse]
    consensus_k_value: Optional[int]
    consensus_predicted_winner_db_team_id: Optional[int]
    consensus_predicted_winner_provider_team_id: Optional[int]
    consensus_predicted_winner_name: Optional[str]
    consensus_correct: Optional[bool]
