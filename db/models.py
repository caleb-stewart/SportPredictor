from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ChlLeague(Base):
    __tablename__ = "chl_leagues"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    sport: Mapped[str] = mapped_column(String, nullable=False, default="hockey")
    provider: Mapped[str] = mapped_column(String, nullable=False, default="hockeytech")
    provider_league_code: Mapped[str] = mapped_column(String, nullable=False)
    timezone: Mapped[Optional[str]] = mapped_column(String)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))


class ChlTeam(Base):
    __tablename__ = "chl_teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    league_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("chl_leagues.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    hockeytech_id: Mapped[int] = mapped_column(Integer, nullable=False)
    city: Mapped[Optional[str]] = mapped_column(String)
    team_name: Mapped[Optional[str]] = mapped_column(String)
    conference: Mapped[Optional[str]] = mapped_column(String)
    division: Mapped[Optional[str]] = mapped_column(String)
    logo_url: Mapped[Optional[str]] = mapped_column(String)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("league_id", "hockeytech_id", name="uq_chl_teams_league_id_hockeytech_id"),
    )


class ChlGame(Base):
    __tablename__ = "chl_games"

    id: Mapped[int] = mapped_column(primary_key=True)
    league_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("chl_leagues.id"), nullable=False)
    game_id: Mapped[int] = mapped_column(Integer, nullable=False)
    season_id: Mapped[Optional[str]] = mapped_column(String)
    season_name: Mapped[Optional[str]] = mapped_column(String)
    game_date: Mapped[Optional[dt.date]] = mapped_column(Date)
    venue: Mapped[Optional[str]] = mapped_column(String)
    status: Mapped[Optional[str]] = mapped_column(String)
    home_goal_count: Mapped[Optional[int]] = mapped_column(Integer)
    away_goal_count: Mapped[Optional[int]] = mapped_column(Integer)
    scoring_breakdown: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    shots_on_goal: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    period: Mapped[Optional[str]] = mapped_column(String)
    game_number: Mapped[Optional[int]] = mapped_column(Integer)
    power_play: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    fow: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    home_power_play_percentage: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    away_power_play_percentage: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    home_faceoff_win_percentage: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    away_faceoff_win_percentage: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    home_shots_on_goal_total: Mapped[Optional[int]] = mapped_column(Integer)
    away_shots_on_goal_total: Mapped[Optional[int]] = mapped_column(Integer)
    home_team: Mapped[Optional[str]] = mapped_column(String)
    away_team: Mapped[Optional[str]] = mapped_column(String)
    home_team_id: Mapped[Optional[int]] = mapped_column(Integer)
    away_team_id: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    scorebar_snapshot: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    scheduled_time_utc: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("league_id", "game_id", name="uq_chl_games_league_id_game_id"),
    )


class ChlRollingAverage(Base):
    __tablename__ = "chl_rolling_averages"

    id: Mapped[int] = mapped_column(primary_key=True)
    league_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("chl_leagues.id"), nullable=False)
    game_id: Mapped[int] = mapped_column(Integer, nullable=False)
    team_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("chl_teams.id"), nullable=False)
    k_value: Mapped[int] = mapped_column(Integer, nullable=False)
    goals_for_avg: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 2))
    goals_against_avg: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 2))
    shots_for_avg: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 2))
    shots_against_avg: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 2))
    power_play_percentage_avg: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    power_play_percentage_against_avg: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    faceoff_win_percentage_avg: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    faceoff_win_percentage_against_avg: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    home_away: Mapped[Optional[int]] = mapped_column(Integer)
    goals_diff: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 2))
    ppp_diff: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    sog_diff: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 2))
    fowp_diff: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    target_win: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint(
            "league_id",
            "game_id",
            "k_value",
            "team_id",
            name="uq_chl_rolling_averages_league_game_k_team",
        ),
    )


class ChlPredictionRecord(Base):
    __tablename__ = "chl_prediction_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    league_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("chl_leagues.id"), nullable=False)
    game_id: Mapped[int] = mapped_column(Integer, nullable=False)
    k_value: Mapped[int] = mapped_column(Integer, nullable=False)
    home_team_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("chl_teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("chl_teams.id"), nullable=False)
    predicted_winner_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("chl_teams.id"))
    home_team_probability: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    away_team_probability: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    actual_winner_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("chl_teams.id"))
    correct: Mapped[Optional[bool]] = mapped_column(Boolean)
    prediction_date: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    model_version: Mapped[Optional[str]] = mapped_column(String)
    model_family: Mapped[Optional[str]] = mapped_column(String)
    raw_model_outputs: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    created_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint(
            "league_id",
            "game_id",
            "k_value",
            name="uq_chl_prediction_records_league_game_k",
        ),
    )


class ChlCustomPredictionRecord(Base):
    __tablename__ = "chl_custom_prediction_records"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    league_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("chl_leagues.id"), nullable=False)
    home_team_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("chl_teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("chl_teams.id"), nullable=False)
    game_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    home_team_probability: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    away_team_probability: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    predicted_winner_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("chl_teams.id"))
    model_version: Mapped[Optional[str]] = mapped_column(String)
    model_family: Mapped[Optional[str]] = mapped_column(String)
    k_components: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.UTC))


Index("idx_chl_custom_prediction_records_game_date", ChlCustomPredictionRecord.game_date)
Index("idx_chl_custom_prediction_records_created_at", ChlCustomPredictionRecord.created_at)


class ChlPredictionRecordArchive(Base):
    __tablename__ = "chl_prediction_records_archive"

    archive_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    league_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("chl_leagues.id"), nullable=False)
    id: Mapped[Optional[int]] = mapped_column(BigInteger)
    game_id: Mapped[int] = mapped_column(Integer, nullable=False)
    k_value: Mapped[int] = mapped_column(Integer, nullable=False)
    home_team_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    away_team_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    predicted_winner_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    home_team_probability: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    away_team_probability: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    actual_winner_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    correct: Mapped[Optional[bool]] = mapped_column(Boolean)
    prediction_date: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    model_version: Mapped[Optional[str]] = mapped_column(String)
    model_family: Mapped[Optional[str]] = mapped_column(String)
    raw_model_outputs: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    created_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    archive_run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    archive_label: Mapped[Optional[str]] = mapped_column(String)
    archived_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: dt.datetime.now(dt.UTC))


class ChlReplayRun(Base):
    __tablename__ = "chl_replay_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    league_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("chl_leagues.id"), nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: dt.datetime.now(dt.UTC))
    completed_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    date_from: Mapped[Optional[dt.date]] = mapped_column(Date)
    date_to: Mapped[Optional[dt.date]] = mapped_column(Date)
    active_model_version: Mapped[Optional[str]] = mapped_column(String)
    games_scanned: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    games_predicted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    games_skipped: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rows_upserted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    proof_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    error_text: Mapped[Optional[str]] = mapped_column(String)


class ChlModelCompareRun(Base):
    __tablename__ = "chl_model_compare_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    league_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("chl_leagues.id"), nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: dt.datetime.now(dt.UTC))
    completed_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    mode: Mapped[str] = mapped_column(String, nullable=False, default="frozen_replay")
    baseline_model_version: Mapped[str] = mapped_column(String, nullable=False)
    candidate_model_version: Mapped[str] = mapped_column(String, nullable=False)
    date_from: Mapped[Optional[dt.date]] = mapped_column(Date)
    date_to: Mapped[Optional[dt.date]] = mapped_column(Date)
    games_scanned: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    games_compared: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    proof_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    error_text: Mapped[Optional[str]] = mapped_column(String)


class ChlExperiment(Base):
    __tablename__ = "chl_experiments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    league_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("chl_leagues.id"), nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    experiment_type: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: dt.datetime.now(dt.UTC))
    completed_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    proposal_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    result_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    error_text: Mapped[Optional[str]] = mapped_column(String)


class ChlFeatureRegistry(Base):
    __tablename__ = "chl_feature_registry"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    league_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("chl_leagues.id"), nullable=False)
    feature_name: Mapped[str] = mapped_column(String, nullable=False)
    feature_group: Mapped[Optional[str]] = mapped_column(String)
    description: Mapped[Optional[str]] = mapped_column(String)
    spec_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    no_leakage_rule: Mapped[Optional[str]] = mapped_column(String)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: dt.datetime.now(dt.UTC))
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: dt.datetime.now(dt.UTC))

    __table_args__ = (
        UniqueConstraint("league_id", "feature_name", name="uq_chl_feature_registry_league_feature_name"),
    )


class ChlIngestRun(Base):
    __tablename__ = "chl_ingest_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    league_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("chl_leagues.id"), nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    mode: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: dt.datetime.now(dt.UTC))
    completed_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    selected_season_ids: Mapped[Optional[list[str]]] = mapped_column(JSONB)
    counts_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    error_text: Mapped[Optional[str]] = mapped_column(String)


class ChlIngestGameFailure(Base):
    __tablename__ = "chl_ingest_game_failures"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chl_ingest_runs.id"), nullable=False)
    league_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("chl_leagues.id"), nullable=False)
    game_id: Mapped[int] = mapped_column(Integer, nullable=False)
    season_id: Mapped[Optional[str]] = mapped_column(String)
    stage: Mapped[str] = mapped_column(String, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    last_error: Mapped[Optional[str]] = mapped_column(String)
    last_seen_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: dt.datetime.now(dt.UTC))

    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "league_id",
            "game_id",
            "stage",
            name="uq_chl_ingest_game_failures_run_league_game_stage",
        ),
    )


Index(
    "idx_chl_prediction_records_archive_archive_run_id",
    ChlPredictionRecordArchive.archive_run_id,
)
Index("idx_chl_games_league_id_game_date", ChlGame.league_id, ChlGame.game_date)
Index(
    "idx_chl_prediction_records_archive_game_k_run",
    ChlPredictionRecordArchive.game_id,
    ChlPredictionRecordArchive.k_value,
    ChlPredictionRecordArchive.archive_run_id,
)
Index("idx_chl_model_compare_runs_started_at", ChlModelCompareRun.started_at)
Index("idx_chl_model_compare_runs_status", ChlModelCompareRun.status)
Index("idx_chl_experiments_created_at", ChlExperiment.created_at)
Index("idx_chl_experiments_status", ChlExperiment.status)
Index("idx_chl_ingest_runs_league_status_started_at", ChlIngestRun.league_id, ChlIngestRun.status, ChlIngestRun.started_at)
Index("idx_chl_ingest_failures_run_id", ChlIngestGameFailure.run_id)
Index("idx_chl_ingest_failures_league_game", ChlIngestGameFailure.league_id, ChlIngestGameFailure.game_id)
