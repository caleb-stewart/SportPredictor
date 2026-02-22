"""add CHL ingest run and failure checkpoint tables

Revision ID: 20260221_190000
Revises: 20260221_180000
Create Date: 2026-02-21 19:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260221_190000"
down_revision = "20260221_180000"
branch_labels = None
depends_on = None


def _table_exists(bind, table_name: str) -> bool:
    return table_name in sa.inspect(bind).get_table_names()


def _index_exists(bind, table_name: str, index_name: str) -> bool:
    indexes = sa.inspect(bind).get_indexes(table_name)
    return any(index.get("name") == index_name for index in indexes)


def upgrade() -> None:
    bind = op.get_bind()

    if not _table_exists(bind, "chl_ingest_runs"):
        op.create_table(
            "chl_ingest_runs",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
            sa.Column("league_id", sa.BigInteger(), nullable=False),
            sa.Column("status", sa.Text(), nullable=False),
            sa.Column("mode", sa.Text(), nullable=False),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("selected_season_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("counts_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("error_text", sa.Text(), nullable=True),
            sa.ForeignKeyConstraint(["league_id"], ["chl_leagues.id"]),
        )

    if not _table_exists(bind, "chl_ingest_game_failures"):
        op.create_table(
            "chl_ingest_game_failures",
            sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True, nullable=False),
            sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("league_id", sa.BigInteger(), nullable=False),
            sa.Column("game_id", sa.Integer(), nullable=False),
            sa.Column("season_id", sa.Text(), nullable=True),
            sa.Column("stage", sa.Text(), nullable=False),
            sa.Column("attempts", sa.Integer(), nullable=False, server_default=sa.text("1")),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.ForeignKeyConstraint(["run_id"], ["chl_ingest_runs.id"]),
            sa.ForeignKeyConstraint(["league_id"], ["chl_leagues.id"]),
            sa.UniqueConstraint(
                "run_id",
                "league_id",
                "game_id",
                "stage",
                name="uq_chl_ingest_game_failures_run_league_game_stage",
            ),
        )

    if _table_exists(bind, "chl_ingest_runs") and not _index_exists(bind, "chl_ingest_runs", "idx_chl_ingest_runs_league_status_started_at"):
        op.create_index(
            "idx_chl_ingest_runs_league_status_started_at",
            "chl_ingest_runs",
            ["league_id", "status", "started_at"],
            unique=False,
        )

    if _table_exists(bind, "chl_ingest_game_failures") and not _index_exists(bind, "chl_ingest_game_failures", "idx_chl_ingest_failures_run_id"):
        op.create_index(
            "idx_chl_ingest_failures_run_id",
            "chl_ingest_game_failures",
            ["run_id"],
            unique=False,
        )

    if _table_exists(bind, "chl_ingest_game_failures") and not _index_exists(bind, "chl_ingest_game_failures", "idx_chl_ingest_failures_league_game"):
        op.create_index(
            "idx_chl_ingest_failures_league_game",
            "chl_ingest_game_failures",
            ["league_id", "game_id"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()

    if _table_exists(bind, "chl_ingest_game_failures") and _index_exists(bind, "chl_ingest_game_failures", "idx_chl_ingest_failures_league_game"):
        op.drop_index("idx_chl_ingest_failures_league_game", table_name="chl_ingest_game_failures")
    if _table_exists(bind, "chl_ingest_game_failures") and _index_exists(bind, "chl_ingest_game_failures", "idx_chl_ingest_failures_run_id"):
        op.drop_index("idx_chl_ingest_failures_run_id", table_name="chl_ingest_game_failures")
    if _table_exists(bind, "chl_ingest_runs") and _index_exists(bind, "chl_ingest_runs", "idx_chl_ingest_runs_league_status_started_at"):
        op.drop_index("idx_chl_ingest_runs_league_status_started_at", table_name="chl_ingest_runs")

    if _table_exists(bind, "chl_ingest_game_failures"):
        op.drop_table("chl_ingest_game_failures")
    if _table_exists(bind, "chl_ingest_runs"):
        op.drop_table("chl_ingest_runs")
