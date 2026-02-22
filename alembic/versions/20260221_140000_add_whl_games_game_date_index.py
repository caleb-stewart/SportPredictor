"""add index on whl_games.game_date

Revision ID: 20260221_140000
Revises: 20260221_130000
Create Date: 2026-02-21 14:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "20260221_140000"
down_revision = "20260221_130000"
branch_labels = None
depends_on = None


_INDEX_NAME = "idx_whl_games_game_date"


def _index_exists(bind, table_name: str, index_name: str) -> bool:
    inspector = sa.inspect(bind)
    indexes = [idx["name"] for idx in inspector.get_indexes(table_name)]
    return index_name in indexes


def upgrade() -> None:
    bind = op.get_bind()
    if not _index_exists(bind, "whl_games", _INDEX_NAME):
        op.create_index(_INDEX_NAME, "whl_games", ["game_date"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    if _index_exists(bind, "whl_games", _INDEX_NAME):
        op.drop_index(_INDEX_NAME, table_name="whl_games")
