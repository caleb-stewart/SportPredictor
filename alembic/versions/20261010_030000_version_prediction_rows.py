"""make prediction rows append-only by model version

Revision ID: 20261010_030000
Revises: 20260222_140000
Create Date: 2026-10-10 03:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20261010_030000"
down_revision = "20260222_140000"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chl_prediction_records",
        sa.Column("ensemble_predicted_winner_id", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "chl_prediction_records",
        sa.Column("ensemble_home_team_probability", sa.Numeric(5, 4), nullable=True),
    )
    op.add_column(
        "chl_prediction_records",
        sa.Column("ensemble_away_team_probability", sa.Numeric(5, 4), nullable=True),
    )
    op.add_column(
        "chl_prediction_records",
        sa.Column("ensemble_actual_winner_id", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "chl_prediction_records",
        sa.Column("ensemble_correct", sa.Boolean(), nullable=True),
    )

    op.add_column(
        "chl_prediction_records_archive",
        sa.Column("ensemble_predicted_winner_id", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "chl_prediction_records_archive",
        sa.Column("ensemble_home_team_probability", sa.Numeric(5, 4), nullable=True),
    )
    op.add_column(
        "chl_prediction_records_archive",
        sa.Column("ensemble_away_team_probability", sa.Numeric(5, 4), nullable=True),
    )
    op.add_column(
        "chl_prediction_records_archive",
        sa.Column("ensemble_actual_winner_id", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "chl_prediction_records_archive",
        sa.Column("ensemble_correct", sa.Boolean(), nullable=True),
    )

    # Preserve the ensemble that older rows already stored in raw_model_outputs
    # so activating the version filter does not change historical scoring.
    op.execute(
        sa.text(
            """
            UPDATE chl_prediction_records
            SET ensemble_home_team_probability = COALESCE(
                    ensemble_home_team_probability,
                    NULLIF(raw_model_outputs -> 'ensemble' ->> 'home_team_prob', '')::numeric
                ),
                ensemble_away_team_probability = COALESCE(
                    ensemble_away_team_probability,
                    NULLIF(raw_model_outputs -> 'ensemble' ->> 'away_team_prob', '')::numeric
                ),
                ensemble_actual_winner_id = COALESCE(ensemble_actual_winner_id, actual_winner_id),
                ensemble_predicted_winner_id = COALESCE(
                    ensemble_predicted_winner_id,
                    CASE
                        WHEN NULLIF(raw_model_outputs -> 'ensemble' ->> 'home_team_prob', '')::numeric
                            >= NULLIF(raw_model_outputs -> 'ensemble' ->> 'away_team_prob', '')::numeric
                        THEN home_team_id
                        ELSE away_team_id
                    END
                )
            WHERE raw_model_outputs IS NOT NULL
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE chl_prediction_records
            SET ensemble_correct = CASE
                WHEN ensemble_actual_winner_id IS NULL OR ensemble_predicted_winner_id IS NULL THEN NULL
                ELSE ensemble_predicted_winner_id = ensemble_actual_winner_id
            END
            WHERE ensemble_actual_winner_id IS NOT NULL
            """
        )
    )

    op.drop_constraint(
        "uq_chl_prediction_records_league_game_k",
        "chl_prediction_records",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_chl_prediction_records_league_game_k_model",
        "chl_prediction_records",
        ["league_id", "game_id", "k_value", "model_version"],
    )
    op.create_index(
        "idx_chl_prediction_records_league_model_game",
        "chl_prediction_records",
        ["league_id", "model_version", "game_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_chl_prediction_records_league_model_game",
        table_name="chl_prediction_records",
    )
    op.drop_constraint(
        "uq_chl_prediction_records_league_game_k_model",
        "chl_prediction_records",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_chl_prediction_records_league_game_k",
        "chl_prediction_records",
        ["league_id", "game_id", "k_value"],
    )

    for column in (
        "ensemble_correct",
        "ensemble_actual_winner_id",
        "ensemble_away_team_probability",
        "ensemble_home_team_probability",
        "ensemble_predicted_winner_id",
    ):
        op.drop_column("chl_prediction_records_archive", column)
        op.drop_column("chl_prediction_records", column)
