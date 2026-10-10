"""add prediction/game composite fks as not valid

Revision ID: 20260223_160000
Revises: 20260223_150000
Create Date: 2026-02-23 16:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "20260223_160000"
down_revision = "20260223_150000"
branch_labels = None
depends_on = None


FK_PREDICTION_GAME = "fk_chl_prediction_records_league_game"
FK_ROLLING_GAME = "fk_chl_rolling_averages_league_game"


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
    if _table_exists(bind, "chl_prediction_records") and _table_exists(bind, "chl_games"):
        if not _constraint_exists(bind, "chl_prediction_records", FK_PREDICTION_GAME):
            op.execute(
                sa.text(
                    f"""
                    ALTER TABLE chl_prediction_records
                    ADD CONSTRAINT {FK_PREDICTION_GAME}
                    FOREIGN KEY (league_id, game_id)
                    REFERENCES chl_games (league_id, game_id)
                    NOT VALID
                    """
                )
            )

    if _table_exists(bind, "chl_rolling_averages") and _table_exists(bind, "chl_games"):
        if not _constraint_exists(bind, "chl_rolling_averages", FK_ROLLING_GAME):
            op.execute(
                sa.text(
                    f"""
                    ALTER TABLE chl_rolling_averages
                    ADD CONSTRAINT {FK_ROLLING_GAME}
                    FOREIGN KEY (league_id, game_id)
                    REFERENCES chl_games (league_id, game_id)
                    NOT VALID
                    """
                )
            )


def downgrade() -> None:
    bind = op.get_bind()
    if _table_exists(bind, "chl_rolling_averages") and _constraint_exists(bind, "chl_rolling_averages", FK_ROLLING_GAME):
        op.drop_constraint(FK_ROLLING_GAME, "chl_rolling_averages", type_="foreignkey")
    if _table_exists(bind, "chl_prediction_records") and _constraint_exists(
        bind,
        "chl_prediction_records",
        FK_PREDICTION_GAME,
    ):
        op.drop_constraint(FK_PREDICTION_GAME, "chl_prediction_records", type_="foreignkey")
