"""validate previously added CHL composite FKs

Revision ID: 20260223_170000
Revises: 20260223_160000
Create Date: 2026-02-23 17:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "20260223_170000"
down_revision = "20260223_160000"
branch_labels = None
depends_on = None


FK_PREDICTION_GAME = "fk_chl_prediction_records_league_game"
FK_ROLLING_GAME = "fk_chl_rolling_averages_league_game"


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
    if _constraint_exists(bind, "chl_prediction_records", FK_PREDICTION_GAME):
        op.execute(sa.text(f"ALTER TABLE chl_prediction_records VALIDATE CONSTRAINT {FK_PREDICTION_GAME}"))
    if _constraint_exists(bind, "chl_rolling_averages", FK_ROLLING_GAME):
        op.execute(sa.text(f"ALTER TABLE chl_rolling_averages VALIDATE CONSTRAINT {FK_ROLLING_GAME}"))


def downgrade() -> None:
    # PostgreSQL does not support "UNVALIDATE"; keep this migration as a one-way integrity step.
    pass
