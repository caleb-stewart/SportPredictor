"""add model compare, experiments, and feature registry tables

Revision ID: 20260221_120000
Revises: 20260217_100000
Create Date: 2026-02-21 12:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260221_120000"
down_revision = "20260217_100000"
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

    if not _table_exists(bind, "whl_model_compare_runs"):
        op.create_table(
            "whl_model_compare_runs",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
            sa.Column("status", sa.Text(), nullable=False),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("mode", sa.Text(), nullable=False, server_default=sa.text("'frozen_replay'")),
            sa.Column("baseline_model_version", sa.Text(), nullable=False),
            sa.Column("candidate_model_version", sa.Text(), nullable=False),
            sa.Column("date_from", sa.Date(), nullable=True),
            sa.Column("date_to", sa.Date(), nullable=True),
            sa.Column("games_scanned", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("games_compared", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("proof_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("error_text", sa.Text(), nullable=True),
        )

    if not _index_exists(bind, "whl_model_compare_runs", "idx_whl_model_compare_runs_started_at"):
        op.create_index(
            "idx_whl_model_compare_runs_started_at",
            "whl_model_compare_runs",
            ["started_at"],
            unique=False,
        )

    if not _index_exists(bind, "whl_model_compare_runs", "idx_whl_model_compare_runs_status"):
        op.create_index(
            "idx_whl_model_compare_runs_status",
            "whl_model_compare_runs",
            ["status"],
            unique=False,
        )

    if not _table_exists(bind, "whl_experiments"):
        op.create_table(
            "whl_experiments",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
            sa.Column("status", sa.Text(), nullable=False),
            sa.Column("experiment_type", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("proposal_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("result_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("error_text", sa.Text(), nullable=True),
        )

    if not _index_exists(bind, "whl_experiments", "idx_whl_experiments_created_at"):
        op.create_index(
            "idx_whl_experiments_created_at",
            "whl_experiments",
            ["created_at"],
            unique=False,
        )

    if not _index_exists(bind, "whl_experiments", "idx_whl_experiments_status"):
        op.create_index(
            "idx_whl_experiments_status",
            "whl_experiments",
            ["status"],
            unique=False,
        )

    if not _table_exists(bind, "whl_feature_registry"):
        op.create_table(
            "whl_feature_registry",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
            sa.Column("feature_name", sa.Text(), nullable=False, unique=True),
            sa.Column("feature_group", sa.Text(), nullable=True),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("spec_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("no_leakage_rule", sa.Text(), nullable=True),
            sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        )


def downgrade() -> None:
    bind = op.get_bind()

    if _table_exists(bind, "whl_feature_registry"):
        op.drop_table("whl_feature_registry")

    if _index_exists(bind, "whl_experiments", "idx_whl_experiments_status"):
        op.drop_index("idx_whl_experiments_status", table_name="whl_experiments")
    if _index_exists(bind, "whl_experiments", "idx_whl_experiments_created_at"):
        op.drop_index("idx_whl_experiments_created_at", table_name="whl_experiments")
    if _table_exists(bind, "whl_experiments"):
        op.drop_table("whl_experiments")

    if _index_exists(bind, "whl_model_compare_runs", "idx_whl_model_compare_runs_status"):
        op.drop_index("idx_whl_model_compare_runs_status", table_name="whl_model_compare_runs")
    if _index_exists(bind, "whl_model_compare_runs", "idx_whl_model_compare_runs_started_at"):
        op.drop_index("idx_whl_model_compare_runs_started_at", table_name="whl_model_compare_runs")
    if _table_exists(bind, "whl_model_compare_runs"):
        op.drop_table("whl_model_compare_runs")
