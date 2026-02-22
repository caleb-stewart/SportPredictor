"""add indexes for results by day pagination

Revision ID: 20260222_130000
Revises: 20260222_120000
Create Date: 2026-02-22 13:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "20260222_130000"
down_revision = "20260222_120000"
branch_labels = None
depends_on = None


INDEX_RESOLVED_K15 = "idx_chl_prediction_records_league_correct_game_k15_resolved"
INDEX_GAMES_DAY_SORT = "idx_chl_games_league_date_scheduled_game"


def _table_exists(bind, table_name: str) -> bool:
    return table_name in sa.inspect(bind).get_table_names()


def _index_exists(bind, table_name: str, index_name: str) -> bool:
    indexes = sa.inspect(bind).get_indexes(table_name)
    return any(index.get("name") == index_name for index in indexes)


def upgrade() -> None:
    bind = op.get_bind()

    if _table_exists(bind, "chl_prediction_records") and not _index_exists(bind, "chl_prediction_records", INDEX_RESOLVED_K15):
        op.execute(
            sa.text(
                f"""
                CREATE INDEX {INDEX_RESOLVED_K15}
                ON chl_prediction_records (league_id, correct, game_id)
                WHERE k_value = 15 AND correct IS NOT NULL
                """
            )
        )

    if _table_exists(bind, "chl_games") and not _index_exists(bind, "chl_games", INDEX_GAMES_DAY_SORT):
        op.execute(
            sa.text(
                f"""
                CREATE INDEX {INDEX_GAMES_DAY_SORT}
                ON chl_games (league_id, game_date DESC, scheduled_time_utc, game_id)
                """
            )
        )


def downgrade() -> None:
    bind = op.get_bind()

    if _table_exists(bind, "chl_games") and _index_exists(bind, "chl_games", INDEX_GAMES_DAY_SORT):
        op.drop_index(INDEX_GAMES_DAY_SORT, table_name="chl_games")

    if _table_exists(bind, "chl_prediction_records") and _index_exists(bind, "chl_prediction_records", INDEX_RESOLVED_K15):
        op.drop_index(INDEX_RESOLVED_K15, table_name="chl_prediction_records")
