"""add logo_url column to whl_teams

Revision ID: 20260221_130000
Revises: 20260221_120000
Create Date: 2026-02-21 13:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "20260221_130000"
down_revision = "20260221_120000"
branch_labels = None
depends_on = None


def _column_exists(bind, table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(bind)
    columns = [c["name"] for c in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    bind = op.get_bind()
    if not _column_exists(bind, "whl_teams", "logo_url"):
        op.add_column("whl_teams", sa.Column("logo_url", sa.Text(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    if _column_exists(bind, "whl_teams", "logo_url"):
        op.drop_column("whl_teams", "logo_url")
