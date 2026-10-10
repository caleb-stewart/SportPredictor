"""add game team provider fks as not valid

Revision ID: 20260223_180000
Revises: 20260223_170000
Create Date: 2026-02-23 18:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "20260223_180000"
down_revision = "20260223_170000"
branch_labels = None
depends_on = None


FK_GAME_HOME_PROVIDER = "fk_chl_games_league_home_provider_team"
FK_GAME_AWAY_PROVIDER = "fk_chl_games_league_away_provider_team"


def _table_exists(bind, table_name: str) -> bool:
    inspector = sa.inspect(bind)
    return table_name in inspector.get_table_names()


def _constraint_exists(bind, table_name: str, constraint_name: str) -> bool:
    rows = bind.execute(
        sa.text(
            """
            SELECT 1
            FROM pg_constraint c
            JOIN pg_class t ON t.oid = c.conrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            WHERE n.nspname = 'public'
              AND t.relname = :table_name
              AND c.conname = :constraint_name
            """
        ),
        {"table_name": table_name, "constraint_name": constraint_name},
    ).fetchone()
    return rows is not None


def upgrade() -> None:
    bind = op.get_bind()
    if not (_table_exists(bind, "chl_games") and _table_exists(bind, "chl_teams")):
        return

    if not _constraint_exists(bind, "chl_games", FK_GAME_HOME_PROVIDER):
        op.execute(
            sa.text(
                f"""
                ALTER TABLE chl_games
                ADD CONSTRAINT {FK_GAME_HOME_PROVIDER}
                FOREIGN KEY (league_id, home_team_id)
                REFERENCES chl_teams (league_id, hockeytech_id)
                NOT VALID
                """
            )
        )

    if not _constraint_exists(bind, "chl_games", FK_GAME_AWAY_PROVIDER):
        op.execute(
            sa.text(
                f"""
                ALTER TABLE chl_games
                ADD CONSTRAINT {FK_GAME_AWAY_PROVIDER}
                FOREIGN KEY (league_id, away_team_id)
                REFERENCES chl_teams (league_id, hockeytech_id)
                NOT VALID
                """
            )
        )


def downgrade() -> None:
    bind = op.get_bind()
    if _table_exists(bind, "chl_games") and _constraint_exists(bind, "chl_games", FK_GAME_AWAY_PROVIDER):
        op.drop_constraint(FK_GAME_AWAY_PROVIDER, "chl_games", type_="foreignkey")
    if _table_exists(bind, "chl_games") and _constraint_exists(bind, "chl_games", FK_GAME_HOME_PROVIDER):
        op.drop_constraint(FK_GAME_HOME_PROVIDER, "chl_games", type_="foreignkey")
