"""add fastapi schedule snapshot and custom prediction table

Revision ID: 20260214_190000
Revises: 20260214_180000
Create Date: 2026-02-14 19:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260214_190000"
down_revision = "20260214_180000"
branch_labels = None
depends_on = None


def _column_exists(bind, table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(bind)
    columns = [c["name"] for c in inspector.get_columns(table_name)]
    return column_name in columns


def _table_exists(bind, table_name: str) -> bool:
    inspector = sa.inspect(bind)
    return table_name in inspector.get_table_names()


def _index_exists(bind, table_name: str, index_name: str) -> bool:
    inspector = sa.inspect(bind)
    indexes = [idx["name"] for idx in inspector.get_indexes(table_name)]
    return index_name in indexes


def upgrade() -> None:
    bind = op.get_bind()

    if not _column_exists(bind, "whl_games", "scorebar_snapshot"):
        op.add_column("whl_games", sa.Column("scorebar_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=True))

    if not _column_exists(bind, "whl_games", "scheduled_time_utc"):
        op.add_column("whl_games", sa.Column("scheduled_time_utc", sa.DateTime(timezone=True), nullable=True))

    if not _table_exists(bind, "whl_custom_prediction_records"):
        op.create_table(
            "whl_custom_prediction_records",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
            sa.Column("home_team_id", sa.BigInteger(), nullable=False),
            sa.Column("away_team_id", sa.BigInteger(), nullable=False),
            sa.Column("game_date", sa.Date(), nullable=False),
            sa.Column("home_team_probability", sa.Numeric(precision=5, scale=4), nullable=False),
            sa.Column("away_team_probability", sa.Numeric(precision=5, scale=4), nullable=False),
            sa.Column("predicted_winner_id", sa.BigInteger(), nullable=True),
            sa.Column("model_version", sa.Text(), nullable=True),
            sa.Column("model_family", sa.Text(), nullable=True),
            sa.Column("k_components", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.ForeignKeyConstraint(["home_team_id"], ["whl_teams.id"]),
            sa.ForeignKeyConstraint(["away_team_id"], ["whl_teams.id"]),
            sa.ForeignKeyConstraint(["predicted_winner_id"], ["whl_teams.id"]),
        )

    if not _index_exists(bind, "whl_custom_prediction_records", "idx_whl_custom_prediction_records_game_date"):
        op.create_index(
            "idx_whl_custom_prediction_records_game_date",
            "whl_custom_prediction_records",
            ["game_date"],
            unique=False,
        )

    if not _index_exists(bind, "whl_custom_prediction_records", "idx_whl_custom_prediction_records_created_at"):
        op.create_index(
            "idx_whl_custom_prediction_records_created_at",
            "whl_custom_prediction_records",
            ["created_at"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()

    if _index_exists(bind, "whl_custom_prediction_records", "idx_whl_custom_prediction_records_created_at"):
        op.drop_index("idx_whl_custom_prediction_records_created_at", table_name="whl_custom_prediction_records")

    if _index_exists(bind, "whl_custom_prediction_records", "idx_whl_custom_prediction_records_game_date"):
        op.drop_index("idx_whl_custom_prediction_records_game_date", table_name="whl_custom_prediction_records")

    if _table_exists(bind, "whl_custom_prediction_records"):
        op.drop_table("whl_custom_prediction_records")

    if _column_exists(bind, "whl_games", "scheduled_time_utc"):
        op.drop_column("whl_games", "scheduled_time_utc")

    if _column_exists(bind, "whl_games", "scorebar_snapshot"):
        op.drop_column("whl_games", "scorebar_snapshot")
