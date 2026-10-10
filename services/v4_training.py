"""Training and walk-forward evaluation for the V4 prototype."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from services.data_backend import CHL_LEAGUES, normalize_league_code
from services.training_dataset import DbConfig, _run_psql_copy
from services.v4_models import (
    V4_FEATURE_COLUMNS,
    V4ModelConfig,
    fit_v4_meta_model,
    replay_v4_features,
    v4_probability_from_context,
)


V4_SQL = """
SELECT
  g.game_id,
  g.game_date,
  g.season_id,
  g.status,
  g.home_goal_count,
  g.away_goal_count,
  g.home_team_id,
  g.away_team_id,
  g.period,
  g.scoring_breakdown,
  g.scheduled_time_utc
FROM chl_games g
WHERE g.league_id = {league_id}
  AND g.game_date IS NOT NULL
  AND g.home_team_id IS NOT NULL
  AND g.away_team_id IS NOT NULL
  AND g.home_goal_count IS NOT NULL
  AND g.away_goal_count IS NOT NULL
ORDER BY g.game_date, g.game_id
"""


def load_v4_games(db: DbConfig | None = None, league_code: str = "whl") -> pd.DataFrame:
    normalized = normalize_league_code(league_code)
    league = CHL_LEAGUES.get(normalized)
    if league is None:
        raise ValueError(f"Unsupported V4 league: {league_code}")
    csv_text = _run_psql_copy(V4_SQL.format(league_id=int(league["id"])), db or DbConfig())
    if not csv_text.strip():
        return pd.DataFrame()
    return pd.read_csv(StringIO(csv_text), parse_dates=["game_date", "scheduled_time_utc"])


def binary_metrics(y_true: Iterable[float], probabilities: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.clip(np.asarray(list(probabilities), dtype=float), 1e-6, 1.0 - 1e-6)
    if len(y) == 0:
        raise ValueError("Cannot score an empty forecast set")
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    return {
        "games": int(len(y)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": auc,
    }


def _score_frame(feature_frame: pd.DataFrame, probabilities: np.ndarray, start: int) -> dict[str, Any]:
    mask = feature_frame["home_win"].notna().to_numpy(dtype=bool)
    mask[:start] = False
    if not mask.any():
        return {"games": 0}
    y = feature_frame.loc[mask, "home_win"].to_numpy(dtype=int)
    return binary_metrics(y, probabilities[mask])


def walk_forward_v4(
    games: pd.DataFrame,
    config: V4ModelConfig | None = None,
    train_fraction: float = 0.8,
    league_code: str | None = None,
    baseline_probabilities: Iterable[float] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, Any]:
    """Replay states once and score only games after a chronological cutoff."""
    feature_frame = replay_v4_features(games, config=config, league_code=league_code)
    if feature_frame.empty:
        raise ValueError("No eligible completed games available for V4")
    train_end = int(len(feature_frame) * float(train_fraction))
    if train_end < 8 or len(feature_frame) - train_end < 1:
        raise ValueError("V4 walk-forward split is too small")

    aligned_baseline = None
    if baseline_probabilities is not None:
        aligned_baseline = np.asarray(list(baseline_probabilities), dtype=float)
        if len(aligned_baseline) != len(feature_frame):
            raise ValueError("baseline_probabilities must align with the eligible V4 frame")
    combiner, meta_columns = fit_v4_meta_model(
        feature_frame,
        train_end=train_end,
        baseline_probabilities=aligned_baseline,
    )
    probabilities = np.asarray(
        [
            v4_probability_from_context(
                {
                    **row,
                    **({"v4_baseline_prob": float(aligned_baseline[index])} if aligned_baseline is not None else {}),
                },
                combiner=combiner,
            )["home_prob"]
            for index, row in enumerate(feature_frame.to_dict(orient="records"))
        ],
        dtype=float,
    )
    holdout = _score_frame(feature_frame, probabilities, train_end)
    rating_holdout = _score_frame(
        feature_frame,
        feature_frame["v4_rating_home_prob"].to_numpy(dtype=float),
        train_end,
    )
    goal_holdout = _score_frame(
        feature_frame,
        feature_frame["v4_goal_home_prob"].to_numpy(dtype=float),
        train_end,
    )
    train_mask = feature_frame["home_win"].notna().to_numpy(dtype=bool)
    train_mask[train_end:] = False
    train_metrics = binary_metrics(
        feature_frame.loc[train_mask, "home_win"].to_numpy(dtype=int),
        probabilities[train_mask],
    )
    baseline_probability = float(feature_frame.loc[train_mask, "home_win"].mean())
    holdout_mask = feature_frame["home_win"].notna().to_numpy(dtype=bool)
    holdout_mask[:train_end] = False
    baseline = binary_metrics(
        feature_frame.loc[holdout_mask, "home_win"].to_numpy(dtype=int),
        np.full(int(holdout_mask.sum()), baseline_probability),
    )
    metrics = {
        "model_family": "whl_v4_dynamic_hierarchical",
        "dataset": {
            "raw_games": int(len(games)),
            "games_eligible": int(len(feature_frame)),
            "games_excluded": int(len(games) - len(feature_frame)),
            "date_min": str(feature_frame["game_date"].min()),
            "date_max": str(feature_frame["game_date"].max()),
        },
        "split": {
            "train_games": int(train_end),
            "holdout_games": int(len(feature_frame) - train_end),
            "train_fraction": float(train_fraction),
        },
        "train": train_metrics,
        "holdout": holdout,
        "component_holdout": {
            "dynamic_rating": rating_holdout,
            "hierarchical_goal_distribution": goal_holdout,
        },
        "home_rate_baseline_holdout": baseline,
        "meta_columns": meta_columns,
    }
    if aligned_baseline is not None:
        feature_frame = feature_frame.assign(v4_baseline_prob=aligned_baseline)
    return metrics, feature_frame.assign(v4_home_prob=probabilities), combiner


def save_v4_artifacts(
    output_root: Path,
    version: str,
    combiner: Any,
    metrics: dict[str, Any],
    promote: bool = False,
) -> Path:
    version_dir = output_root / version
    version_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_family": "whl_v4_dynamic_hierarchical",
        "model_version": version,
        "trained_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "v4_feature_columns": list(V4_FEATURE_COLUMNS),
        "v4_probability_columns": list(getattr(combiner, "columns", [])),
        "v4_meta_model": combiner,
    }
    bundle_path = version_dir / "model_bundle.joblib"
    joblib.dump(bundle, bundle_path)
    metrics_path = version_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    metadata = {
        "model_version": version,
        "model_family": bundle["model_family"],
        "trained_at_utc": bundle["trained_at_utc"],
        "bundle_path": str(bundle_path),
        "metrics_path": str(metrics_path),
        "gates_passed": False,
        "promotion_requires_explicit_review": True,
    }
    metadata_path = version_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    if promote:
        active_path = output_root / "active_model.json"
        active_tmp = output_root / "active_model.json.tmp"
        active_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        active_tmp.replace(active_path)
    return version_dir


def train_v4_model_package(
    output_root: Path,
    league_code: str = "whl",
    model_version: str | None = None,
    db: DbConfig | None = None,
    config: V4ModelConfig | None = None,
    promote: bool = False,
) -> dict[str, Any]:
    games = load_v4_games(db=db, league_code=league_code)
    metrics, predictions, combiner = walk_forward_v4(
        games,
        config=config,
        league_code=league_code,
    )
    version = model_version or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    version_dir = save_v4_artifacts(output_root, version, combiner, metrics, promote=promote)
    predictions_path = version_dir / "walk_forward_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    return {
        "version": version,
        "output_dir": str(version_dir),
        "promoted": bool(promote),
        "metrics": metrics,
        "predictions_path": str(predictions_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", default="whl")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--promote", action="store_true", help="Explicitly write the active pointer")
    args = parser.parse_args()
    result = train_v4_model_package(
        output_root=Path(args.output_root),
        league_code=args.league,
        model_version=args.model_version,
        promote=args.promote,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
