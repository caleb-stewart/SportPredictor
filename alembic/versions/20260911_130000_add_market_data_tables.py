"""add durable sportsbook and exchange market-data tables

Revision ID: 20260911_130000
Revises: 20260911_120000
Create Date: 2026-09-11 13:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260911_130000"
down_revision = "20260911_120000"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "market_data_providers",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("code", sa.String(), nullable=False, unique=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "market_data_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("provider_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("market_data_providers.id"), nullable=False),
        sa.Column("provider_event_id", sa.String(), nullable=False),
        sa.Column("sport_key", sa.String(), nullable=False),
        sa.Column("sport_title", sa.String()),
        sa.Column("commence_time", sa.DateTime(timezone=True)),
        sa.Column("home_team", sa.String()),
        sa.Column("away_team", sa.String()),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("latest_payload", postgresql.JSONB()),
        sa.UniqueConstraint("provider_id", "provider_event_id", name="uq_market_data_events_provider_event"),
    )
    op.create_table(
        "market_data_quotes",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("event_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("market_data_events.id"), nullable=False),
        sa.Column("bookmaker_key", sa.String(), nullable=False),
        sa.Column("bookmaker_title", sa.String()),
        sa.Column("market_key", sa.String(), nullable=False),
        sa.Column("outcome_name", sa.String(), nullable=False),
        sa.Column("outcome_description", sa.String()),
        sa.Column("point", sa.Numeric(10, 3)),
        sa.Column("decimal_price", sa.Numeric(12, 5), nullable=False),
        sa.Column("source_updated_at", sa.DateTime(timezone=True)),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_live", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("raw_quote", postgresql.JSONB()),
    )
    op.create_table(
        "market_data_ingestion_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("provider_code", sa.String(), nullable=False),
        sa.Column("operation", sa.String(), nullable=False),
        sa.Column("sport_key", sa.String()),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("events_written", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("quotes_written", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("quota_remaining", sa.Integer()),
        sa.Column("quota_used", sa.Integer()),
        sa.Column("quota_cost", sa.Integer()),
        sa.Column("details", postgresql.JSONB()),
        sa.Column("error_text", sa.String()),
    )
    op.create_index("idx_market_data_events_sport_commence", "market_data_events", ["sport_key", "commence_time"])
    op.create_index("idx_market_data_quotes_event_captured", "market_data_quotes", ["event_id", "captured_at"])
    op.create_index("idx_market_data_quotes_book_market_captured", "market_data_quotes", ["bookmaker_key", "market_key", "captured_at"])
    op.create_index("idx_market_data_ingestion_runs_provider_started", "market_data_ingestion_runs", ["provider_code", "started_at"])


def downgrade() -> None:
    op.drop_table("market_data_ingestion_runs")
    op.drop_table("market_data_quotes")
    op.drop_table("market_data_events")
    op.drop_table("market_data_providers")
