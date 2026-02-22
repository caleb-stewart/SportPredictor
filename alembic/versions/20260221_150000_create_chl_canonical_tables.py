"""create CHL canonical tables

Revision ID: 20260221_150000
Revises: 20260221_140000
Create Date: 2026-02-21 15:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260221_150000"
down_revision = "20260221_140000"
branch_labels = None
depends_on = None


def _table_exists(bind, table_name: str) -> bool:
    inspector = sa.inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    bind = op.get_bind()

    if not _table_exists(bind, "chl_leagues"):
        op.create_table(
            "chl_leagues",
            sa.Column("id", sa.BigInteger(), primary_key=True, nullable=False),
            sa.Column("code", sa.Text(), nullable=False, unique=True),
            sa.Column("name", sa.Text(), nullable=False),
            sa.Column("sport", sa.Text(), nullable=False, server_default=sa.text("'hockey'")),
            sa.Column("provider", sa.Text(), nullable=False, server_default=sa.text("'hockeytech'")),
            sa.Column("provider_league_code", sa.Text(), nullable=False),
            sa.Column("timezone", sa.Text(), nullable=True),
            sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )

    if not _table_exists(bind, "chl_teams"):
        op.create_table(
            "chl_teams",
            sa.Column("id", sa.BigInteger(), primary_key=True, nullable=False),
            sa.Column("league_id", sa.BigInteger(), nullable=False),
            sa.Column("name", sa.Text(), nullable=False),
            sa.Column("hockeytech_id", sa.Integer(), nullable=False),
            sa.Column("city", sa.Text(), nullable=True),
            sa.Column("team_name", sa.Text(), nullable=True),
            sa.Column("conference", sa.Text(), nullable=True),
            sa.Column("division", sa.Text(), nullable=True),
            sa.Column("logo_url", sa.Text(), nullable=True),
            sa.Column("active", sa.Boolean(), nullable=True, server_default=sa.text("true")),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["league_id"], ["chl_leagues.id"]),
        )

    if not _table_exists(bind, "chl_games"):
        op.create_table(
            "chl_games",
            sa.Column("id", sa.BigInteger(), primary_key=True, nullable=False),
            sa.Column("league_id", sa.BigInteger(), nullable=False),
            sa.Column("game_id", sa.Integer(), nullable=False),
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
            sa.Column("scorebar_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("scheduled_time_utc", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["league_id"], ["chl_leagues.id"]),
        )

    if not _table_exists(bind, "chl_rolling_averages"):
        op.create_table(
            "chl_rolling_averages",
            sa.Column("id", sa.BigInteger(), primary_key=True, nullable=False),
            sa.Column("league_id", sa.BigInteger(), nullable=False),
            sa.Column("game_id", sa.Integer(), nullable=False),
            sa.Column("team_id", sa.BigInteger(), nullable=False),
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
            sa.ForeignKeyConstraint(["league_id"], ["chl_leagues.id"]),
            sa.ForeignKeyConstraint(["team_id"], ["chl_teams.id"]),
        )

    if not _table_exists(bind, "chl_prediction_records"):
        op.create_table(
            "chl_prediction_records",
            sa.Column("id", sa.BigInteger(), primary_key=True, nullable=False),
            sa.Column("league_id", sa.BigInteger(), nullable=False),
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
            sa.ForeignKeyConstraint(["league_id"], ["chl_leagues.id"]),
            sa.ForeignKeyConstraint(["home_team_id"], ["chl_teams.id"]),
            sa.ForeignKeyConstraint(["away_team_id"], ["chl_teams.id"]),
            sa.ForeignKeyConstraint(["predicted_winner_id"], ["chl_teams.id"]),
            sa.ForeignKeyConstraint(["actual_winner_id"], ["chl_teams.id"]),
        )

    if not _table_exists(bind, "chl_custom_prediction_records"):
        op.create_table(
            "chl_custom_prediction_records",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
            sa.Column("league_id", sa.BigInteger(), nullable=False),
            sa.Column("home_team_id", sa.BigInteger(), nullable=False),
            sa.Column("away_team_id", sa.BigInteger(), nullable=False),
            sa.Column("game_date", sa.Date(), nullable=False),
            sa.Column("home_team_probability", sa.Numeric(5, 4), nullable=False),
            sa.Column("away_team_probability", sa.Numeric(5, 4), nullable=False),
            sa.Column("predicted_winner_id", sa.BigInteger(), nullable=True),
            sa.Column("model_version", sa.Text(), nullable=True),
            sa.Column("model_family", sa.Text(), nullable=True),
            sa.Column("k_components", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.ForeignKeyConstraint(["league_id"], ["chl_leagues.id"]),
            sa.ForeignKeyConstraint(["home_team_id"], ["chl_teams.id"]),
            sa.ForeignKeyConstraint(["away_team_id"], ["chl_teams.id"]),
            sa.ForeignKeyConstraint(["predicted_winner_id"], ["chl_teams.id"]),
        )

    if not _table_exists(bind, "chl_prediction_records_archive"):
        op.create_table(
            "chl_prediction_records_archive",
            sa.Column("archive_id", sa.BigInteger(), primary_key=True, autoincrement=True, nullable=False),
            sa.Column("league_id", sa.BigInteger(), nullable=False),
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
            sa.Column("archived_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.ForeignKeyConstraint(["league_id"], ["chl_leagues.id"]),
        )

    if not _table_exists(bind, "chl_replay_runs"):
        op.create_table(
            "chl_replay_runs",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
            sa.Column("league_id", sa.BigInteger(), nullable=False),
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
            sa.ForeignKeyConstraint(["league_id"], ["chl_leagues.id"]),
        )

    if not _table_exists(bind, "chl_model_compare_runs"):
        op.create_table(
            "chl_model_compare_runs",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
            sa.Column("league_id", sa.BigInteger(), nullable=False),
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
            sa.ForeignKeyConstraint(["league_id"], ["chl_leagues.id"]),
        )

    if not _table_exists(bind, "chl_experiments"):
        op.create_table(
            "chl_experiments",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
            sa.Column("league_id", sa.BigInteger(), nullable=False),
            sa.Column("status", sa.Text(), nullable=False),
            sa.Column("experiment_type", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("proposal_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("result_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("error_text", sa.Text(), nullable=True),
            sa.ForeignKeyConstraint(["league_id"], ["chl_leagues.id"]),
        )

    if not _table_exists(bind, "chl_feature_registry"):
        op.create_table(
            "chl_feature_registry",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
            sa.Column("league_id", sa.BigInteger(), nullable=False),
            sa.Column("feature_name", sa.Text(), nullable=False),
            sa.Column("feature_group", sa.Text(), nullable=True),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("spec_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("no_leakage_rule", sa.Text(), nullable=True),
            sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.ForeignKeyConstraint(["league_id"], ["chl_leagues.id"]),
        )

    # Seed default CHL leagues.
    op.execute(
        """
        INSERT INTO chl_leagues (id, code, name, sport, provider, provider_league_code, timezone, active)
        VALUES
            (1, 'whl', 'Western Hockey League', 'hockey', 'hockeytech', 'whl', 'America/Los_Angeles', true),
            (2, 'ohl', 'Ontario Hockey League', 'hockey', 'hockeytech', 'ohl', 'America/Toronto', true),
            (3, 'lhjmq', 'Quebec Maritimes Junior Hockey League', 'hockey', 'hockeytech', 'lhjmq', 'America/Halifax', true)
        ON CONFLICT (code) DO NOTHING
        """
    )


def downgrade() -> None:
    bind = op.get_bind()

    if _table_exists(bind, "chl_feature_registry"):
        op.drop_table("chl_feature_registry")
    if _table_exists(bind, "chl_experiments"):
        op.drop_table("chl_experiments")
    if _table_exists(bind, "chl_model_compare_runs"):
        op.drop_table("chl_model_compare_runs")
    if _table_exists(bind, "chl_replay_runs"):
        op.drop_table("chl_replay_runs")
    if _table_exists(bind, "chl_prediction_records_archive"):
        op.drop_table("chl_prediction_records_archive")
    if _table_exists(bind, "chl_custom_prediction_records"):
        op.drop_table("chl_custom_prediction_records")
    if _table_exists(bind, "chl_prediction_records"):
        op.drop_table("chl_prediction_records")
    if _table_exists(bind, "chl_rolling_averages"):
        op.drop_table("chl_rolling_averages")
    if _table_exists(bind, "chl_games"):
        op.drop_table("chl_games")
    if _table_exists(bind, "chl_teams"):
        op.drop_table("chl_teams")
    if _table_exists(bind, "chl_leagues"):
        op.drop_table("chl_leagues")
