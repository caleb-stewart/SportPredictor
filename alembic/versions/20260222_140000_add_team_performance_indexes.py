"""add indexes for team performance aggregation

Revision ID: 20260222_140000
Revises: 20260222_130000
Create Date: 2026-02-22 14:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "20260222_140000"
down_revision = "20260222_130000"
branch_labels = None
depends_on = None


INDEX_CONSENSUS_SCAN = "idx_chl_prediction_records_league_game_k_pred_ts"
INDEX_HOME_TEAM_SCAN = "idx_chl_prediction_records_league_home_game_partial"
INDEX_AWAY_TEAM_SCAN = "idx_chl_prediction_records_league_away_game_partial"


def _table_exists(bind, table_name: str) -> bool:
    return table_name in sa.inspect(bind).get_table_names()


def _index_exists(bind, table_name: str, index_name: str) -> bool:
    indexes = sa.inspect(bind).get_indexes(table_name)
    return any(index.get("name") == index_name for index in indexes)


def upgrade() -> None:
    bind = op.get_bind()

    if _table_exists(bind, "chl_prediction_records") and not _index_exists(bind, "chl_prediction_records", INDEX_CONSENSUS_SCAN):
        op.execute(
            sa.text(
                f"""
                CREATE INDEX {INDEX_CONSENSUS_SCAN}
                ON chl_prediction_records (league_id, game_id, k_value DESC, prediction_date DESC)
                """
            )
        )

    if _table_exists(bind, "chl_prediction_records") and not _index_exists(bind, "chl_prediction_records", INDEX_HOME_TEAM_SCAN):
        op.execute(
            sa.text(
                f"""
                CREATE INDEX {INDEX_HOME_TEAM_SCAN}
                ON chl_prediction_records (league_id, home_team_id, game_id)
                WHERE home_team_id IS NOT NULL
                """
            )
        )

    if _table_exists(bind, "chl_prediction_records") and not _index_exists(bind, "chl_prediction_records", INDEX_AWAY_TEAM_SCAN):
        op.execute(
            sa.text(
                f"""
                CREATE INDEX {INDEX_AWAY_TEAM_SCAN}
                ON chl_prediction_records (league_id, away_team_id, game_id)
                WHERE away_team_id IS NOT NULL
                """
            )
        )


def downgrade() -> None:
    bind = op.get_bind()

    if _table_exists(bind, "chl_prediction_records") and _index_exists(bind, "chl_prediction_records", INDEX_AWAY_TEAM_SCAN):
        op.drop_index(INDEX_AWAY_TEAM_SCAN, table_name="chl_prediction_records")

    if _table_exists(bind, "chl_prediction_records") and _index_exists(bind, "chl_prediction_records", INDEX_HOME_TEAM_SCAN):
        op.drop_index(INDEX_HOME_TEAM_SCAN, table_name="chl_prediction_records")

    if _table_exists(bind, "chl_prediction_records") and _index_exists(bind, "chl_prediction_records", INDEX_CONSENSUS_SCAN):
        op.drop_index(INDEX_CONSENSUS_SCAN, table_name="chl_prediction_records")
