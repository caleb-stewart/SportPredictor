"""backfill WHL rows into CHL canonical tables

Revision ID: 20260221_170000
Revises: 20260221_160000
Create Date: 2026-02-21 17:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "20260221_170000"
down_revision = "20260221_160000"
branch_labels = None
depends_on = None


WHL_LEAGUE_ID = 1


def _table_exists(bind, table_name: str) -> bool:
    inspector = sa.inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    bind = op.get_bind()

    if not _table_exists(bind, "chl_leagues"):
        return

    if _table_exists(bind, "whl_teams") and _table_exists(bind, "chl_teams"):
        op.execute(
            """
            INSERT INTO chl_teams (
                id, league_id, name, hockeytech_id, city, team_name, conference, division,
                logo_url, active, created_at, updated_at
            )
            SELECT
                w.id, 1, w.name, w.hockeytech_id, w.city, w.team_name, w.conference, w.division,
                w.logo_url, w.active, w.created_at, w.updated_at
            FROM whl_teams w
            ON CONFLICT (league_id, hockeytech_id) DO UPDATE SET
                name = EXCLUDED.name,
                city = EXCLUDED.city,
                team_name = EXCLUDED.team_name,
                conference = EXCLUDED.conference,
                division = EXCLUDED.division,
                logo_url = EXCLUDED.logo_url,
                active = EXCLUDED.active,
                updated_at = EXCLUDED.updated_at
            """
        )
        op.execute("SELECT setval('chl_teams_id_seq', COALESCE((SELECT MAX(id) FROM chl_teams), 1), true)")

    if _table_exists(bind, "whl_games") and _table_exists(bind, "chl_games"):
        op.execute(
            """
            INSERT INTO chl_games (
                id, league_id, game_id, season_id, season_name, game_date, venue, status,
                home_goal_count, away_goal_count, scoring_breakdown, shots_on_goal, period,
                game_number, power_play, fow, home_power_play_percentage, away_power_play_percentage,
                home_faceoff_win_percentage, away_faceoff_win_percentage, home_shots_on_goal_total,
                away_shots_on_goal_total, home_team, away_team, home_team_id, away_team_id,
                created_at, updated_at, scorebar_snapshot, scheduled_time_utc
            )
            SELECT
                w.id, 1, w.game_id, w.season_id, w.season_name, w.game_date, w.venue, w.status,
                w.home_goal_count, w.away_goal_count, w.scoring_breakdown, w.shots_on_goal, w.period,
                w.game_number, w.power_play, w.fow, w.home_power_play_percentage, w.away_power_play_percentage,
                w.home_faceoff_win_percentage, w.away_faceoff_win_percentage, w.home_shots_on_goal_total,
                w.away_shots_on_goal_total, w.home_team, w.away_team, w.home_team_id, w.away_team_id,
                w.created_at, w.updated_at, w.scorebar_snapshot, w.scheduled_time_utc
            FROM whl_games w
            ON CONFLICT (league_id, game_id) DO UPDATE SET
                season_id = EXCLUDED.season_id,
                season_name = EXCLUDED.season_name,
                game_date = EXCLUDED.game_date,
                venue = EXCLUDED.venue,
                status = EXCLUDED.status,
                home_goal_count = EXCLUDED.home_goal_count,
                away_goal_count = EXCLUDED.away_goal_count,
                scoring_breakdown = EXCLUDED.scoring_breakdown,
                shots_on_goal = EXCLUDED.shots_on_goal,
                period = EXCLUDED.period,
                game_number = EXCLUDED.game_number,
                power_play = EXCLUDED.power_play,
                fow = EXCLUDED.fow,
                home_power_play_percentage = EXCLUDED.home_power_play_percentage,
                away_power_play_percentage = EXCLUDED.away_power_play_percentage,
                home_faceoff_win_percentage = EXCLUDED.home_faceoff_win_percentage,
                away_faceoff_win_percentage = EXCLUDED.away_faceoff_win_percentage,
                home_shots_on_goal_total = EXCLUDED.home_shots_on_goal_total,
                away_shots_on_goal_total = EXCLUDED.away_shots_on_goal_total,
                home_team = EXCLUDED.home_team,
                away_team = EXCLUDED.away_team,
                home_team_id = EXCLUDED.home_team_id,
                away_team_id = EXCLUDED.away_team_id,
                updated_at = EXCLUDED.updated_at,
                scorebar_snapshot = EXCLUDED.scorebar_snapshot,
                scheduled_time_utc = EXCLUDED.scheduled_time_utc
            """
        )
        op.execute("SELECT setval('chl_games_id_seq', COALESCE((SELECT MAX(id) FROM chl_games), 1), true)")

    if _table_exists(bind, "whl_rolling_averages") and _table_exists(bind, "chl_rolling_averages"):
        op.execute(
            """
            INSERT INTO chl_rolling_averages (
                id, league_id, game_id, team_id, k_value, goals_for_avg, goals_against_avg,
                shots_for_avg, shots_against_avg, power_play_percentage_avg,
                power_play_percentage_against_avg, faceoff_win_percentage_avg,
                faceoff_win_percentage_against_avg, home_away, goals_diff, ppp_diff,
                sog_diff, fowp_diff, target_win, created_at, updated_at
            )
            SELECT
                w.id, 1, w.game_id, w.whl_team_id, w.k_value, w.goals_for_avg, w.goals_against_avg,
                w.shots_for_avg, w.shots_against_avg, w.power_play_percentage_avg,
                w.power_play_percentage_against_avg, w.faceoff_win_percentage_avg,
                w.faceoff_win_percentage_against_avg, w.home_away, w.goals_diff, w.ppp_diff,
                w.sog_diff, w.fowp_diff, w.target_win, w.created_at, w.updated_at
            FROM whl_rolling_averages w
            ON CONFLICT (league_id, game_id, k_value, team_id) DO UPDATE SET
                goals_for_avg = EXCLUDED.goals_for_avg,
                goals_against_avg = EXCLUDED.goals_against_avg,
                shots_for_avg = EXCLUDED.shots_for_avg,
                shots_against_avg = EXCLUDED.shots_against_avg,
                power_play_percentage_avg = EXCLUDED.power_play_percentage_avg,
                power_play_percentage_against_avg = EXCLUDED.power_play_percentage_against_avg,
                faceoff_win_percentage_avg = EXCLUDED.faceoff_win_percentage_avg,
                faceoff_win_percentage_against_avg = EXCLUDED.faceoff_win_percentage_against_avg,
                home_away = EXCLUDED.home_away,
                goals_diff = EXCLUDED.goals_diff,
                ppp_diff = EXCLUDED.ppp_diff,
                sog_diff = EXCLUDED.sog_diff,
                fowp_diff = EXCLUDED.fowp_diff,
                target_win = EXCLUDED.target_win,
                updated_at = EXCLUDED.updated_at
            """
        )
        op.execute("SELECT setval('chl_rolling_averages_id_seq', COALESCE((SELECT MAX(id) FROM chl_rolling_averages), 1), true)")

    if _table_exists(bind, "whl_prediction_records") and _table_exists(bind, "chl_prediction_records"):
        op.execute(
            """
            INSERT INTO chl_prediction_records (
                id, league_id, game_id, k_value, home_team_id, away_team_id, predicted_winner_id,
                home_team_probability, away_team_probability, actual_winner_id, correct,
                prediction_date, model_version, model_family, raw_model_outputs,
                created_at, updated_at
            )
            SELECT
                w.id, 1, w.game_id, w.k_value, w.home_team_id, w.away_team_id, w.predicted_winner_id,
                w.home_team_probability, w.away_team_probability, w.actual_winner_id, w.correct,
                w.prediction_date, w.model_version, w.model_family, w.raw_model_outputs,
                w.created_at, w.updated_at
            FROM whl_prediction_records w
            ON CONFLICT (league_id, game_id, k_value) DO UPDATE SET
                home_team_id = EXCLUDED.home_team_id,
                away_team_id = EXCLUDED.away_team_id,
                predicted_winner_id = EXCLUDED.predicted_winner_id,
                home_team_probability = EXCLUDED.home_team_probability,
                away_team_probability = EXCLUDED.away_team_probability,
                actual_winner_id = EXCLUDED.actual_winner_id,
                correct = EXCLUDED.correct,
                prediction_date = EXCLUDED.prediction_date,
                model_version = EXCLUDED.model_version,
                model_family = EXCLUDED.model_family,
                raw_model_outputs = EXCLUDED.raw_model_outputs,
                updated_at = EXCLUDED.updated_at
            """
        )
        op.execute("SELECT setval('chl_prediction_records_id_seq', COALESCE((SELECT MAX(id) FROM chl_prediction_records), 1), true)")

    if _table_exists(bind, "whl_custom_prediction_records") and _table_exists(bind, "chl_custom_prediction_records"):
        op.execute(
            """
            INSERT INTO chl_custom_prediction_records (
                id, league_id, home_team_id, away_team_id, game_date,
                home_team_probability, away_team_probability, predicted_winner_id,
                model_version, model_family, k_components, created_at
            )
            SELECT
                w.id, 1, w.home_team_id, w.away_team_id, w.game_date,
                w.home_team_probability, w.away_team_probability, w.predicted_winner_id,
                w.model_version, w.model_family, w.k_components, w.created_at
            FROM whl_custom_prediction_records w
            ON CONFLICT (id) DO NOTHING
            """
        )

    if _table_exists(bind, "whl_prediction_records_archive") and _table_exists(bind, "chl_prediction_records_archive"):
        op.execute(
            """
            INSERT INTO chl_prediction_records_archive (
                archive_id, league_id, id, game_id, k_value, home_team_id, away_team_id,
                predicted_winner_id, home_team_probability, away_team_probability, actual_winner_id,
                correct, prediction_date, model_version, model_family, raw_model_outputs,
                created_at, updated_at, archive_run_id, archive_label, archived_at
            )
            SELECT
                w.archive_id, 1, w.id, w.game_id, w.k_value, w.home_team_id, w.away_team_id,
                w.predicted_winner_id, w.home_team_probability, w.away_team_probability, w.actual_winner_id,
                w.correct, w.prediction_date, w.model_version, w.model_family, w.raw_model_outputs,
                w.created_at, w.updated_at, w.archive_run_id, w.archive_label, w.archived_at
            FROM whl_prediction_records_archive w
            ON CONFLICT (archive_id) DO NOTHING
            """
        )
        op.execute("SELECT setval('chl_prediction_records_archive_archive_id_seq', COALESCE((SELECT MAX(archive_id) FROM chl_prediction_records_archive), 1), true)")

    if _table_exists(bind, "whl_replay_runs") and _table_exists(bind, "chl_replay_runs"):
        op.execute(
            """
            INSERT INTO chl_replay_runs (
                id, league_id, status, started_at, completed_at, date_from, date_to,
                active_model_version, games_scanned, games_predicted, games_skipped,
                rows_upserted, proof_json, error_text
            )
            SELECT
                w.id, 1, w.status, w.started_at, w.completed_at, w.date_from, w.date_to,
                w.active_model_version, w.games_scanned, w.games_predicted, w.games_skipped,
                w.rows_upserted, w.proof_json, w.error_text
            FROM whl_replay_runs w
            ON CONFLICT (id) DO NOTHING
            """
        )

    if _table_exists(bind, "whl_model_compare_runs") and _table_exists(bind, "chl_model_compare_runs"):
        op.execute(
            """
            INSERT INTO chl_model_compare_runs (
                id, league_id, status, started_at, completed_at, mode,
                baseline_model_version, candidate_model_version, date_from, date_to,
                games_scanned, games_compared, proof_json, error_text
            )
            SELECT
                w.id, 1, w.status, w.started_at, w.completed_at, w.mode,
                w.baseline_model_version, w.candidate_model_version, w.date_from, w.date_to,
                w.games_scanned, w.games_compared, w.proof_json, w.error_text
            FROM whl_model_compare_runs w
            ON CONFLICT (id) DO NOTHING
            """
        )

    if _table_exists(bind, "whl_experiments") and _table_exists(bind, "chl_experiments"):
        op.execute(
            """
            INSERT INTO chl_experiments (
                id, league_id, status, experiment_type, created_at,
                completed_at, proposal_json, result_json, error_text
            )
            SELECT
                w.id, 1, w.status, w.experiment_type, w.created_at,
                w.completed_at, w.proposal_json, w.result_json, w.error_text
            FROM whl_experiments w
            ON CONFLICT (id) DO NOTHING
            """
        )

    if _table_exists(bind, "whl_feature_registry") and _table_exists(bind, "chl_feature_registry"):
        op.execute(
            """
            INSERT INTO chl_feature_registry (
                id, league_id, feature_name, feature_group, description,
                spec_json, no_leakage_rule, active, created_at, updated_at
            )
            SELECT
                w.id, 1, w.feature_name, w.feature_group, w.description,
                w.spec_json, w.no_leakage_rule, w.active, w.created_at, w.updated_at
            FROM whl_feature_registry w
            ON CONFLICT (league_id, feature_name) DO UPDATE SET
                feature_group = EXCLUDED.feature_group,
                description = EXCLUDED.description,
                spec_json = EXCLUDED.spec_json,
                no_leakage_rule = EXCLUDED.no_leakage_rule,
                active = EXCLUDED.active,
                updated_at = EXCLUDED.updated_at
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    if not _table_exists(bind, "chl_leagues"):
        return

    op.execute("DELETE FROM chl_feature_registry WHERE league_id = 1")
    op.execute("DELETE FROM chl_experiments WHERE league_id = 1")
    op.execute("DELETE FROM chl_model_compare_runs WHERE league_id = 1")
    op.execute("DELETE FROM chl_replay_runs WHERE league_id = 1")
    op.execute("DELETE FROM chl_prediction_records_archive WHERE league_id = 1")
    op.execute("DELETE FROM chl_custom_prediction_records WHERE league_id = 1")
    op.execute("DELETE FROM chl_prediction_records WHERE league_id = 1")
    op.execute("DELETE FROM chl_rolling_averages WHERE league_id = 1")
    op.execute("DELETE FROM chl_games WHERE league_id = 1")
    op.execute("DELETE FROM chl_teams WHERE league_id = 1")
