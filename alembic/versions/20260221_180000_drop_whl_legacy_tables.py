"""drop legacy whl namespace tables after chl cutover

Revision ID: 20260221_180000
Revises: 20260221_170000
Create Date: 2026-02-21 18:00:00.000000
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = "20260221_180000"
down_revision = "20260221_170000"
branch_labels = None
depends_on = None


def _table_exists(bind, table_name: str) -> bool:
    return table_name in inspect(bind).get_table_names()


def upgrade() -> None:
    bind = op.get_bind()

    drop_order = [
        "whl_prediction_records_archive",
        "whl_replay_runs",
        "whl_model_compare_runs",
        "whl_experiments",
        "whl_feature_registry",
        "whl_prediction_records",
        "whl_custom_prediction_records",
        "whl_rolling_averages",
        "whl_games",
        "whl_teams",
    ]

    for table_name in drop_order:
        if _table_exists(bind, table_name):
            op.drop_table(table_name)


def downgrade() -> None:
    # Irreversible by design: restoring legacy tables should be done from backup,
    # not by auto-recreating stale schema in-place.
    pass
