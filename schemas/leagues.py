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


class TeamPerformanceStatsResponse(BaseModel):
    games_with_predictions: int
    resolved_games: int
    pending_games: int
    correct_predictions: int
    accuracy_pct: Optional[float]
    actual_wins: int
    actual_losses: int
    predicted_wins: int
    predicted_losses: int
    win_delta: int


class LeagueTeamPerformanceResponse(BaseModel):
    team: LeagueTeamResponse
    stats: TeamPerformanceStatsResponse


class LeagueTeamPerformanceListResponse(BaseModel):
    league_code: str
    date_from: Optional[dt.date]
    date_to: Optional[dt.date]
    items: list[LeagueTeamPerformanceResponse]


class LeagueTeamPerformanceDetailResponse(BaseModel):
    league_code: str
    date_from: Optional[dt.date]
    date_to: Optional[dt.date]
    item: LeagueTeamPerformanceResponse


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
    consensus_home_probability: Optional[float] = None
    consensus_away_probability: Optional[float] = None
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


class PredictionResultsByDayPageResponse(BaseModel):
    league_code: str
    items: list[RecentResultDayResponse]
    page: int
    days_per_page: int
    total_days: int
    total_pages: int
    has_more: bool
    has_prev: bool


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
    consensus_home_probability: Optional[float] = None
    consensus_away_probability: Optional[float] = None
    consensus_correct: Optional[bool]


class MaintenanceRunResponse(BaseModel):
    run_id: str
    league_code: str
    trigger_source: str
    status: str
    from_date: Optional[dt.date]
    to_date: Optional[dt.date]
    days_attempted: int
    days_succeeded: int
    totals_json: dict[str, int]
    error_text: Optional[str]
    started_at: dt.datetime
    completed_at: Optional[dt.datetime]


class LeagueMaintenanceStatusResponse(BaseModel):
    league_code: str
    enabled: bool
    max_catchup_days: int
    checkpoint_date: Optional[dt.date]
    last_successful_run_id: Optional[str]
    latest_run: Optional[MaintenanceRunResponse]
