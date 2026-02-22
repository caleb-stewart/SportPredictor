"""add CHL multi-league uniqueness and indexes

Revision ID: 20260221_160000
Revises: 20260221_150000
Create Date: 2026-02-21 16:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "20260221_160000"
down_revision = "20260221_150000"
branch_labels = None
depends_on = None


def _table_exists(bind, table_name: str) -> bool:
    inspector = sa.inspect(bind)
    return table_name in inspector.get_table_names()


def _index_exists(bind, table_name: str, index_name: str) -> bool:
    inspector = sa.inspect(bind)
    indexes = [idx["name"] for idx in inspector.get_indexes(table_name)]
    return index_name in indexes


def _unique_exists(bind, table_name: str, constraint_name: str) -> bool:
    inspector = sa.inspect(bind)
    constraints = [c["name"] for c in inspector.get_unique_constraints(table_name)]
    return constraint_name in constraints


def upgrade() -> None:
    bind = op.get_bind()

    if _table_exists(bind, "chl_teams") and not _unique_exists(bind, "chl_teams", "uq_chl_teams_league_id_hockeytech_id"):
        op.create_unique_constraint(
            "uq_chl_teams_league_id_hockeytech_id",
            "chl_teams",
            ["league_id", "hockeytech_id"],
        )

    if _table_exists(bind, "chl_games") and not _unique_exists(bind, "chl_games", "uq_chl_games_league_id_game_id"):
        op.create_unique_constraint(
            "uq_chl_games_league_id_game_id",
            "chl_games",
            ["league_id", "game_id"],
        )

    if _table_exists(bind, "chl_rolling_averages") and not _unique_exists(
        bind,
        "chl_rolling_averages",
        "uq_chl_rolling_averages_league_game_k_team",
    ):
        op.create_unique_constraint(
            "uq_chl_rolling_averages_league_game_k_team",
            "chl_rolling_averages",
            ["league_id", "game_id", "k_value", "team_id"],
        )

    if _table_exists(bind, "chl_prediction_records") and not _unique_exists(
        bind,
        "chl_prediction_records",
        "uq_chl_prediction_records_league_game_k",
    ):
        op.create_unique_constraint(
            "uq_chl_prediction_records_league_game_k",
            "chl_prediction_records",
            ["league_id", "game_id", "k_value"],
        )

    if _table_exists(bind, "chl_feature_registry") and not _unique_exists(
        bind,
        "chl_feature_registry",
        "uq_chl_feature_registry_league_feature_name",
    ):
        op.create_unique_constraint(
            "uq_chl_feature_registry_league_feature_name",
            "chl_feature_registry",
            ["league_id", "feature_name"],
        )

    if _table_exists(bind, "chl_games") and not _index_exists(bind, "chl_games", "idx_chl_games_league_id_game_date"):
        op.create_index(
            "idx_chl_games_league_id_game_date",
            "chl_games",
            ["league_id", "game_date"],
            unique=False,
        )

    if _table_exists(bind, "chl_prediction_records") and not _index_exists(bind, "chl_prediction_records", "idx_chl_prediction_records_league_id_prediction_date"):
        op.create_index(
            "idx_chl_prediction_records_league_id_prediction_date",
            "chl_prediction_records",
            ["league_id", "prediction_date"],
            unique=False,
        )

    if _table_exists(bind, "chl_teams") and not _index_exists(bind, "chl_teams", "idx_chl_teams_league_id_active_name"):
        op.create_index(
            "idx_chl_teams_league_id_active_name",
            "chl_teams",
            ["league_id", "active", "name"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()

    if _table_exists(bind, "chl_teams") and _index_exists(bind, "chl_teams", "idx_chl_teams_league_id_active_name"):
        op.drop_index("idx_chl_teams_league_id_active_name", table_name="chl_teams")

    if _table_exists(bind, "chl_prediction_records") and _index_exists(bind, "chl_prediction_records", "idx_chl_prediction_records_league_id_prediction_date"):
        op.drop_index("idx_chl_prediction_records_league_id_prediction_date", table_name="chl_prediction_records")

    if _table_exists(bind, "chl_games") and _index_exists(bind, "chl_games", "idx_chl_games_league_id_game_date"):
        op.drop_index("idx_chl_games_league_id_game_date", table_name="chl_games")

    if _table_exists(bind, "chl_feature_registry") and _unique_exists(bind, "chl_feature_registry", "uq_chl_feature_registry_league_feature_name"):
        op.drop_constraint("uq_chl_feature_registry_league_feature_name", "chl_feature_registry", type_="unique")

    if _table_exists(bind, "chl_prediction_records") and _unique_exists(bind, "chl_prediction_records", "uq_chl_prediction_records_league_game_k"):
        op.drop_constraint("uq_chl_prediction_records_league_game_k", "chl_prediction_records", type_="unique")

    if _table_exists(bind, "chl_rolling_averages") and _unique_exists(bind, "chl_rolling_averages", "uq_chl_rolling_averages_league_game_k_team"):
        op.drop_constraint("uq_chl_rolling_averages_league_game_k_team", "chl_rolling_averages", type_="unique")

    if _table_exists(bind, "chl_games") and _unique_exists(bind, "chl_games", "uq_chl_games_league_id_game_id"):
        op.drop_constraint("uq_chl_games_league_id_game_id", "chl_games", type_="unique")

    if _table_exists(bind, "chl_teams") and _unique_exists(bind, "chl_teams", "uq_chl_teams_league_id_hockeytech_id"):
        op.drop_constraint("uq_chl_teams_league_id_hockeytech_id", "chl_teams", type_="unique")
