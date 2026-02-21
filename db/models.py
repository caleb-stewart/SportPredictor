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


class WhlTeam(Base):
    __tablename__ = "whl_teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    hockeytech_id: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    city: Mapped[Optional[str]] = mapped_column(String)
    team_name: Mapped[Optional[str]] = mapped_column(String)
    conference: Mapped[Optional[str]] = mapped_column(String)
    division: Mapped[Optional[str]] = mapped_column(String)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))


class WhlGame(Base):
    __tablename__ = "whl_games"

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
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


class WhlRollingAverage(Base):
    __tablename__ = "whl_rolling_averages"

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(Integer, nullable=False)
    whl_team_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("whl_teams.id"), nullable=False)
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
            "game_id",
            "k_value",
            "whl_team_id",
            name="index_whl_rolling_averages_on_game_id_k_value_whl_team_id",
        ),
    )


class WhlPredictionRecord(Base):
    __tablename__ = "whl_prediction_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(Integer, nullable=False)
    k_value: Mapped[int] = mapped_column(Integer, nullable=False)
    home_team_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("whl_teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("whl_teams.id"), nullable=False)
    predicted_winner_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("whl_teams.id"))
    home_team_probability: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    away_team_probability: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    actual_winner_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("whl_teams.id"))
    correct: Mapped[Optional[bool]] = mapped_column(Boolean)
    prediction_date: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    model_version: Mapped[Optional[str]] = mapped_column(String)
    model_family: Mapped[Optional[str]] = mapped_column(String)
    raw_model_outputs: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    created_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("game_id", "k_value", name="index_whl_prediction_records_on_game_id_and_k_value"),
    )


class WhlCustomPredictionRecord(Base):
    __tablename__ = "whl_custom_prediction_records"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    home_team_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("whl_teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("whl_teams.id"), nullable=False)
    game_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    home_team_probability: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    away_team_probability: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    predicted_winner_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("whl_teams.id"))
    model_version: Mapped[Optional[str]] = mapped_column(String)
    model_family: Mapped[Optional[str]] = mapped_column(String)
    k_components: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.UTC))


Index("idx_whl_custom_prediction_records_game_date", WhlCustomPredictionRecord.game_date)
Index("idx_whl_custom_prediction_records_created_at", WhlCustomPredictionRecord.created_at)


class WhlPredictionRecordArchive(Base):
    __tablename__ = "whl_prediction_records_archive"

    archive_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
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


class WhlReplayRun(Base):
    __tablename__ = "whl_replay_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
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


Index(
    "idx_whl_prediction_records_archive_archive_run_id",
    WhlPredictionRecordArchive.archive_run_id,
)
Index(
    "idx_whl_prediction_records_archive_game_k_run",
    WhlPredictionRecordArchive.game_id,
    WhlPredictionRecordArchive.k_value,
    WhlPredictionRecordArchive.archive_run_id,
)
