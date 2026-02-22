"""add CHL maintenance run and checkpoint tables

Revision ID: 20260222_120000
Revises: 20260221_190000
Create Date: 2026-02-22 12:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260222_120000"
down_revision = "20260221_190000"
branch_labels = None
depends_on = None


def _table_exists(bind, table_name: str) -> bool:
    return table_name in sa.inspect(bind).get_table_names()


def _index_exists(bind, table_name: str, index_name: str) -> bool:
    indexes = sa.inspect(bind).get_indexes(table_name)
    return any(index.get("name") == index_name for index in indexes)


def upgrade() -> None:
    bind = op.get_bind()

    if not _table_exists(bind, "chl_maintenance_runs"):
        op.create_table(
            "chl_maintenance_runs",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
            sa.Column("league_id", sa.BigInteger(), nullable=False),
            sa.Column("trigger_source", sa.Text(), nullable=False),
            sa.Column("status", sa.Text(), nullable=False),
            sa.Column("from_date", sa.Date(), nullable=True),
            sa.Column("to_date", sa.Date(), nullable=True),
            sa.Column("days_attempted", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("days_succeeded", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("totals_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("error_text", sa.Text(), nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.ForeignKeyConstraint(["league_id"], ["chl_leagues.id"]),
        )

    if not _table_exists(bind, "chl_maintenance_checkpoints"):
        op.create_table(
            "chl_maintenance_checkpoints",
            sa.Column("league_id", sa.BigInteger(), primary_key=True, nullable=False),
            sa.Column("last_successful_run_date", sa.Date(), nullable=True),
            sa.Column("last_successful_run_id", postgresql.UUID(as_uuid=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.ForeignKeyConstraint(["league_id"], ["chl_leagues.id"]),
            sa.ForeignKeyConstraint(["last_successful_run_id"], ["chl_maintenance_runs.id"]),
        )

    if _table_exists(bind, "chl_maintenance_runs") and not _index_exists(bind, "chl_maintenance_runs", "idx_chl_maintenance_runs_league_started_at"):
        op.create_index(
            "idx_chl_maintenance_runs_league_started_at",
            "chl_maintenance_runs",
            ["league_id", "started_at"],
            unique=False,
        )

    if _table_exists(bind, "chl_maintenance_runs") and not _index_exists(bind, "chl_maintenance_runs", "idx_chl_maintenance_runs_status_completed_at"):
        op.create_index(
            "idx_chl_maintenance_runs_status_completed_at",
            "chl_maintenance_runs",
            ["status", "completed_at"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()

    if _table_exists(bind, "chl_maintenance_runs") and _index_exists(bind, "chl_maintenance_runs", "idx_chl_maintenance_runs_status_completed_at"):
        op.drop_index("idx_chl_maintenance_runs_status_completed_at", table_name="chl_maintenance_runs")

    if _table_exists(bind, "chl_maintenance_runs") and _index_exists(bind, "chl_maintenance_runs", "idx_chl_maintenance_runs_league_started_at"):
        op.drop_index("idx_chl_maintenance_runs_league_started_at", table_name="chl_maintenance_runs")

    if _table_exists(bind, "chl_maintenance_checkpoints"):
        op.drop_table("chl_maintenance_checkpoints")

    if _table_exists(bind, "chl_maintenance_runs"):
        op.drop_table("chl_maintenance_runs")
