"""register the NHL provider in the canonical hockey tables

Revision ID: 20260911_120000
Revises: 20260223_180000
Create Date: 2026-09-11 12:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260911_120000"
down_revision = "20260223_180000"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    tables = sa.inspect(bind).get_table_names()
    if "chl_leagues" not in tables:
        return

    # Existing deployments use stable ids 1..3 for WHL/OHL/LHJMQ. Keep NHL's
    # id stable as well because the current feature/training SQL is league-id
    # scoped through the shared canonical tables.
    op.execute(
        sa.text(
            """
            INSERT INTO chl_leagues
                (id, code, name, sport, provider, provider_league_code, timezone, active)
            VALUES
                (4, 'nhl', 'National Hockey League', 'hockey', 'nhl_api', 'nhl', 'America/New_York', true)
            ON CONFLICT (code) DO UPDATE SET
                name = EXCLUDED.name,
                sport = EXCLUDED.sport,
                provider = EXCLUDED.provider,
                provider_league_code = EXCLUDED.provider_league_code,
                timezone = EXCLUDED.timezone,
                active = EXCLUDED.active
            """
        )
    )

    # The canonical table was originally created with explicit ids. Advance a
    # sequence if the database has one, without assuming a particular name.
    sequence = bind.execute(
        sa.text("SELECT pg_get_serial_sequence('chl_leagues', 'id')")
    ).scalar()
    if sequence:
        bind.execute(sa.text("SELECT setval(:sequence, GREATEST((SELECT MAX(id) FROM chl_leagues), 1), true)"), {"sequence": sequence})


def downgrade() -> None:
    bind = op.get_bind()
    if "chl_leagues" not in sa.inspect(bind).get_table_names():
        return
    op.execute(sa.text("DELETE FROM chl_leagues WHERE code = 'nhl'"))
