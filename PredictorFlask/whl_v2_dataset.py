"""Canonical WHL dataset extraction for Predictor V2 training."""

from __future__ import annotations

import csv
import os
import subprocess
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from whl_v2_features import FEATURE_COLUMNS, K_VALUES, build_pairwise_features

CANONICAL_SQL = """
SELECT
  g.game_id,
  g.game_date,
  g.season_id,
  g.home_team_id,
  g.away_team_id,
  h.k_value,
  CASE WHEN g.home_goal_count > g.away_goal_count THEN 1 ELSE 0 END AS home_win,
  h.goals_for_avg AS h_goals_for_avg,
  h.goals_against_avg AS h_goals_against_avg,
  h.shots_for_avg AS h_shots_for_avg,
  h.shots_against_avg AS h_shots_against_avg,
  h.power_play_percentage_avg AS h_power_play_percentage_avg,
  h.power_play_percentage_against_avg AS h_power_play_percentage_against_avg,
  h.faceoff_win_percentage_avg AS h_faceoff_win_percentage_avg,
  h.faceoff_win_percentage_against_avg AS h_faceoff_win_percentage_against_avg,
  h.goals_diff AS h_goals_diff,
  h.ppp_diff AS h_ppp_diff,
  h.sog_diff AS h_sog_diff,
  h.fowp_diff AS h_fowp_diff,
  a.goals_for_avg AS a_goals_for_avg,
  a.goals_against_avg AS a_goals_against_avg,
  a.shots_for_avg AS a_shots_for_avg,
  a.shots_against_avg AS a_shots_against_avg,
  a.power_play_percentage_avg AS a_power_play_percentage_avg,
  a.power_play_percentage_against_avg AS a_power_play_percentage_against_avg,
  a.faceoff_win_percentage_avg AS a_faceoff_win_percentage_avg,
  a.faceoff_win_percentage_against_avg AS a_faceoff_win_percentage_against_avg,
  a.goals_diff AS a_goals_diff,
  a.ppp_diff AS a_ppp_diff,
  a.sog_diff AS a_sog_diff,
  a.fowp_diff AS a_fowp_diff
FROM whl_games g
JOIN whl_teams ht ON ht.hockeytech_id = g.home_team_id
JOIN whl_teams at ON at.hockeytech_id = g.away_team_id
JOIN whl_rolling_averages h
  ON h.game_id = g.game_id
 AND h.whl_team_id = ht.id
 AND h.home_away = 1
JOIN whl_rolling_averages a
  ON a.game_id = g.game_id
 AND a.whl_team_id = at.id
 AND a.k_value = h.k_value
 AND a.home_away = 0
WHERE g.game_date IS NOT NULL
  AND g.home_goal_count IS NOT NULL
  AND g.away_goal_count IS NOT NULL
  AND h.k_value IN (5, 10, 15)
ORDER BY g.game_date, g.game_id, h.k_value
"""


@dataclass(frozen=True)
class DbConfig:
    dbname: str = os.getenv("PGDATABASE", "sportpredictor_development")
    host: str = os.getenv("PGHOST", "localhost")
    port: str = os.getenv("PGPORT", "5432")
    user: str = os.getenv("PGUSER", "postgres")
    password: str = os.getenv("PGPASSWORD", "qqqq")


def _run_psql_copy(sql: str, db: DbConfig) -> str:
    command = [
        "psql",
        "-h",
        db.host,
        "-p",
        str(db.port),
        "-U",
        db.user,
        "-d",
        db.dbname,
        "-c",
        f"COPY ({sql}) TO STDOUT WITH CSV HEADER",
    ]

    env = os.environ.copy()
    env["PGPASSWORD"] = db.password

    return subprocess.check_output(command, env=env, text=True)


def _row_metric_map(row: pd.Series, side: str) -> Dict[str, float]:
    prefix = "h_" if side == "home" else "a_"

    return {
        "goals_for_avg": float(row[f"{prefix}goals_for_avg"]),
        "goals_against_avg": float(row[f"{prefix}goals_against_avg"]),
        "shots_for_avg": float(row[f"{prefix}shots_for_avg"]),
        "shots_against_avg": float(row[f"{prefix}shots_against_avg"]),
        "power_play_percentage_avg": float(row[f"{prefix}power_play_percentage_avg"]),
        "power_play_percentage_against_avg": float(
            row[f"{prefix}power_play_percentage_against_avg"]
        ),
        "faceoff_win_percentage_avg": float(row[f"{prefix}faceoff_win_percentage_avg"]),
        "faceoff_win_percentage_against_avg": float(
            row[f"{prefix}faceoff_win_percentage_against_avg"]
        ),
        "goals_diff": float(row[f"{prefix}goals_diff"]),
        "ppp_diff": float(row[f"{prefix}ppp_diff"]),
        "sog_diff": float(row[f"{prefix}sog_diff"]),
        "fowp_diff": float(row[f"{prefix}fowp_diff"]),
    }


def load_canonical_dataset(db: Optional[DbConfig] = None) -> pd.DataFrame:
    """Load canonical game-level rows and append engineered features."""

    db_config = db or DbConfig()
    csv_text = _run_psql_copy(CANONICAL_SQL, db_config)
    frame = pd.read_csv(StringIO(csv_text), parse_dates=["game_date"])  # type: ignore[arg-type]

    if frame.empty:
        return frame

    engineered_rows = []
    for _, row in frame.iterrows():
        home_metrics = _row_metric_map(row, side="home")
        away_metrics = _row_metric_map(row, side="away")
        engineered_rows.append(build_pairwise_features(home_metrics, away_metrics))

    engineered_df = pd.DataFrame(engineered_rows)
    out = pd.concat([frame.reset_index(drop=True), engineered_df], axis=1)

    # Canonical ordering guarantees deterministic time splits.
    out = out.sort_values(["game_date", "game_id", "k_value"]).reset_index(drop=True)
    return out


def games_with_full_k_coverage(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return rows for games that contain all required k-values."""

    if dataset.empty:
        return dataset

    required = set(K_VALUES)
    counts = dataset.groupby("game_id")["k_value"].agg(set)
    keep_game_ids = counts[counts.apply(lambda ks: required.issubset(ks))].index
    return dataset[dataset["game_id"].isin(keep_game_ids)].copy()


def export_dataset_csv(path: Path, dataset: Optional[pd.DataFrame] = None) -> None:
    frame = dataset if dataset is not None else load_canonical_dataset()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def feature_columns() -> Iterable[str]:
    return tuple(FEATURE_COLUMNS)
