"""add replay run tracking and prediction archive tables

Revision ID: 20260217_100000
Revises: 20260214_190000
Create Date: 2026-02-17 10:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260217_100000"
down_revision = "20260214_190000"
branch_labels = None
depends_on = None


def _table_exists(bind, table_name: str) -> bool:
    inspector = sa.inspect(bind)
    return table_name in inspector.get_table_names()


def _index_exists(bind, table_name: str, index_name: str) -> bool:
    inspector = sa.inspect(bind)
    indexes = [idx["name"] for idx in inspector.get_indexes(table_name)]
    return index_name in indexes


def upgrade() -> None:
    bind = op.get_bind()

    if not _table_exists(bind, "whl_prediction_records_archive"):
        op.create_table(
            "whl_prediction_records_archive",
            sa.Column("archive_id", sa.BigInteger(), primary_key=True, nullable=False, autoincrement=True),
            sa.Column("id", sa.BigInteger(), nullable=True),
            sa.Column("game_id", sa.Integer(), nullable=False),
            sa.Column("k_value", sa.Integer(), nullable=False),
            sa.Column("home_team_id", sa.BigInteger(), nullable=False),
            sa.Column("away_team_id", sa.BigInteger(), nullable=False),
            sa.Column("predicted_winner_id", sa.BigInteger(), nullable=True),
            sa.Column("home_team_probability", sa.Numeric(5, 4), nullable=True),
            sa.Column("away_team_probability", sa.Numeric(5, 4), nullable=True),
            sa.Column("actual_winner_id", sa.BigInteger(), nullable=True),
            sa.Column("correct", sa.Boolean(), nullable=True),
            sa.Column("prediction_date", sa.DateTime(timezone=True), nullable=True),
            sa.Column("model_version", sa.Text(), nullable=True),
            sa.Column("model_family", sa.Text(), nullable=True),
            sa.Column("raw_model_outputs", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("archive_run_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("archive_label", sa.Text(), nullable=True),
            sa.Column(
                "archived_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
        )

    if not _index_exists(bind, "whl_prediction_records_archive", "idx_whl_prediction_records_archive_archive_run_id"):
        op.create_index(
            "idx_whl_prediction_records_archive_archive_run_id",
            "whl_prediction_records_archive",
            ["archive_run_id"],
            unique=False,
        )

    if not _index_exists(bind, "whl_prediction_records_archive", "idx_whl_prediction_records_archive_game_k_run"):
        op.create_index(
            "idx_whl_prediction_records_archive_game_k_run",
            "whl_prediction_records_archive",
            ["game_id", "k_value", "archive_run_id"],
            unique=False,
        )

    if not _table_exists(bind, "whl_replay_runs"):
        op.create_table(
            "whl_replay_runs",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
            sa.Column("status", sa.Text(), nullable=False),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("date_from", sa.Date(), nullable=True),
            sa.Column("date_to", sa.Date(), nullable=True),
            sa.Column("active_model_version", sa.Text(), nullable=True),
            sa.Column("games_scanned", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("games_predicted", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("games_skipped", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("rows_upserted", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("proof_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("error_text", sa.Text(), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()

    if _table_exists(bind, "whl_replay_runs"):
        op.drop_table("whl_replay_runs")

    if _index_exists(bind, "whl_prediction_records_archive", "idx_whl_prediction_records_archive_game_k_run"):
        op.drop_index(
            "idx_whl_prediction_records_archive_game_k_run",
            table_name="whl_prediction_records_archive",
        )

    if _index_exists(bind, "whl_prediction_records_archive", "idx_whl_prediction_records_archive_archive_run_id"):
        op.drop_index(
            "idx_whl_prediction_records_archive_archive_run_id",
            table_name="whl_prediction_records_archive",
        )

    if _table_exists(bind, "whl_prediction_records_archive"):
        op.drop_table("whl_prediction_records_archive")
