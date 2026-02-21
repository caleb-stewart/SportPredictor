"""create python baseline schema for whl predictor

Revision ID: 20260214_180000
Revises:
Create Date: 2026-02-14 18:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260214_180000"
down_revision = None
branch_labels = None
depends_on = None


def _table_exists(bind, table_name: str) -> bool:
    inspector = sa.inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    bind = op.get_bind()

    if not _table_exists(bind, "whl_teams"):
        op.create_table(
            "whl_teams",
            sa.Column("id", sa.BigInteger(), primary_key=True, nullable=False),
            sa.Column("name", sa.Text(), nullable=False),
            sa.Column("hockeytech_id", sa.Integer(), nullable=False, unique=True),
            sa.Column("city", sa.Text(), nullable=True),
            sa.Column("team_name", sa.Text(), nullable=True),
            sa.Column("conference", sa.Text(), nullable=True),
            sa.Column("division", sa.Text(), nullable=True),
            sa.Column("active", sa.Boolean(), nullable=True, server_default=sa.text("true")),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )

    if not _table_exists(bind, "whl_games"):
        op.create_table(
            "whl_games",
            sa.Column("id", sa.BigInteger(), primary_key=True, nullable=False),
            sa.Column("game_id", sa.Integer(), nullable=False, unique=True),
            sa.Column("season_id", sa.Text(), nullable=True),
            sa.Column("season_name", sa.Text(), nullable=True),
            sa.Column("game_date", sa.Date(), nullable=True),
            sa.Column("venue", sa.Text(), nullable=True),
            sa.Column("status", sa.Text(), nullable=True),
            sa.Column("home_goal_count", sa.Integer(), nullable=True),
            sa.Column("away_goal_count", sa.Integer(), nullable=True),
            sa.Column("scoring_breakdown", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("shots_on_goal", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("period", sa.Text(), nullable=True),
            sa.Column("game_number", sa.Integer(), nullable=True),
            sa.Column("power_play", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("fow", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("home_power_play_percentage", sa.Numeric(8, 4), nullable=True),
            sa.Column("away_power_play_percentage", sa.Numeric(8, 4), nullable=True),
            sa.Column("home_faceoff_win_percentage", sa.Numeric(8, 4), nullable=True),
            sa.Column("away_faceoff_win_percentage", sa.Numeric(8, 4), nullable=True),
            sa.Column("home_shots_on_goal_total", sa.Integer(), nullable=True),
            sa.Column("away_shots_on_goal_total", sa.Integer(), nullable=True),
            sa.Column("home_team", sa.Text(), nullable=True),
            sa.Column("away_team", sa.Text(), nullable=True),
            sa.Column("home_team_id", sa.Integer(), nullable=True),
            sa.Column("away_team_id", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )

    if not _table_exists(bind, "whl_rolling_averages"):
        op.create_table(
            "whl_rolling_averages",
            sa.Column("id", sa.BigInteger(), primary_key=True, nullable=False),
            sa.Column("game_id", sa.Integer(), nullable=False),
            sa.Column("whl_team_id", sa.BigInteger(), nullable=False),
            sa.Column("k_value", sa.Integer(), nullable=False),
            sa.Column("goals_for_avg", sa.Numeric(8, 2), nullable=True),
            sa.Column("goals_against_avg", sa.Numeric(8, 2), nullable=True),
            sa.Column("shots_for_avg", sa.Numeric(8, 2), nullable=True),
            sa.Column("shots_against_avg", sa.Numeric(8, 2), nullable=True),
            sa.Column("power_play_percentage_avg", sa.Numeric(8, 4), nullable=True),
            sa.Column("power_play_percentage_against_avg", sa.Numeric(8, 4), nullable=True),
            sa.Column("faceoff_win_percentage_avg", sa.Numeric(8, 4), nullable=True),
            sa.Column("faceoff_win_percentage_against_avg", sa.Numeric(8, 4), nullable=True),
            sa.Column("home_away", sa.Integer(), nullable=True),
            sa.Column("goals_diff", sa.Numeric(8, 2), nullable=True),
            sa.Column("ppp_diff", sa.Numeric(8, 4), nullable=True),
            sa.Column("sog_diff", sa.Numeric(8, 2), nullable=True),
            sa.Column("fowp_diff", sa.Numeric(8, 4), nullable=True),
            sa.Column("target_win", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["whl_team_id"], ["whl_teams.id"]),
            sa.UniqueConstraint(
                "game_id",
                "k_value",
                "whl_team_id",
                name="index_whl_rolling_averages_on_game_id_k_value_whl_team_id",
            ),
        )

    if not _table_exists(bind, "whl_prediction_records"):
        op.create_table(
            "whl_prediction_records",
            sa.Column("id", sa.BigInteger(), primary_key=True, nullable=False),
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
            sa.ForeignKeyConstraint(["home_team_id"], ["whl_teams.id"]),
            sa.ForeignKeyConstraint(["away_team_id"], ["whl_teams.id"]),
            sa.ForeignKeyConstraint(["predicted_winner_id"], ["whl_teams.id"]),
            sa.ForeignKeyConstraint(["actual_winner_id"], ["whl_teams.id"]),
            sa.UniqueConstraint("game_id", "k_value", name="index_whl_prediction_records_on_game_id_and_k_value"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    if _table_exists(bind, "whl_prediction_records"):
        op.drop_table("whl_prediction_records")
    if _table_exists(bind, "whl_rolling_averages"):
        op.drop_table("whl_rolling_averages")
    if _table_exists(bind, "whl_games"):
        op.drop_table("whl_games")
    if _table_exists(bind, "whl_teams"):
        op.drop_table("whl_teams")
