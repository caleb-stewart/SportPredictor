"""add game team lookup indexes

Revision ID: 20260223_150000
Revises: 20260222_140000
Create Date: 2026-02-23 15:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "20260223_150000"
down_revision = "20260222_140000"
branch_labels = None
depends_on = None


INDEX_HOME_DATE = "idx_chl_games_league_home_team_date_game"
INDEX_AWAY_DATE = "idx_chl_games_league_away_team_date_game"
INDEX_HOME_PLAYED_DATE = "idx_chl_games_league_home_team_played_date"
INDEX_AWAY_PLAYED_DATE = "idx_chl_games_league_away_team_played_date"


def _table_exists(bind, table_name: str) -> bool:
    inspector = sa.inspect(bind)
    return table_name in inspector.get_table_names()


def _index_exists(bind, table_name: str, index_name: str) -> bool:
    inspector = sa.inspect(bind)
    indexes = inspector.get_indexes(table_name)
    return any(index.get("name") == index_name for index in indexes)


def upgrade() -> None:
    bind = op.get_bind()
    if not _table_exists(bind, "chl_games"):
        return

    if not _index_exists(bind, "chl_games", INDEX_HOME_DATE):
        op.execute(
            sa.text(
                f"""
                CREATE INDEX {INDEX_HOME_DATE}
                ON chl_games (league_id, home_team_id, game_date, game_id)
                WHERE game_date IS NOT NULL
                """
            )
        )

    if not _index_exists(bind, "chl_games", INDEX_AWAY_DATE):
        op.execute(
            sa.text(
                f"""
                CREATE INDEX {INDEX_AWAY_DATE}
                ON chl_games (league_id, away_team_id, game_date, game_id)
                WHERE game_date IS NOT NULL
                """
            )
        )

    if not _index_exists(bind, "chl_games", INDEX_HOME_PLAYED_DATE):
        op.execute(
            sa.text(
                f"""
                CREATE INDEX {INDEX_HOME_PLAYED_DATE}
                ON chl_games (league_id, home_team_id, game_date, game_id)
                WHERE game_date IS NOT NULL
                  AND home_goal_count IS NOT NULL
                  AND away_goal_count IS NOT NULL
                """
            )
        )

    if not _index_exists(bind, "chl_games", INDEX_AWAY_PLAYED_DATE):
        op.execute(
            sa.text(
                f"""
                CREATE INDEX {INDEX_AWAY_PLAYED_DATE}
                ON chl_games (league_id, away_team_id, game_date, game_id)
                WHERE game_date IS NOT NULL
                  AND home_goal_count IS NOT NULL
                  AND away_goal_count IS NOT NULL
                """
            )
        )


def downgrade() -> None:
    bind = op.get_bind()
    if not _table_exists(bind, "chl_games"):
        return

    if _index_exists(bind, "chl_games", INDEX_AWAY_PLAYED_DATE):
        op.drop_index(INDEX_AWAY_PLAYED_DATE, table_name="chl_games")
    if _index_exists(bind, "chl_games", INDEX_HOME_PLAYED_DATE):
        op.drop_index(INDEX_HOME_PLAYED_DATE, table_name="chl_games")
    if _index_exists(bind, "chl_games", INDEX_AWAY_DATE):
        op.drop_index(INDEX_AWAY_DATE, table_name="chl_games")
    if _index_exists(bind, "chl_games", INDEX_HOME_DATE):
        op.drop_index(INDEX_HOME_DATE, table_name="chl_games")
