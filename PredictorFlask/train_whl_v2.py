"""Train, evaluate, and optionally promote WHL Predictor V2 model artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from whl_v2_dataset import DbConfig, export_dataset_csv, games_with_full_k_coverage, load_canonical_dataset
from whl_v2_features import FEATURE_COLUMNS, K_VALUES


@dataclass
class HoldoutMetrics:
    accuracy: float
    log_loss: float
    brier: float
    auc: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "accuracy": float(self.accuracy),
            "log_loss": float(self.log_loss),
            "brier": float(self.brier),
            "auc": float(self.auc),
        }


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> HoldoutMetrics:
    probs = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    preds = (probs >= 0.5).astype(int)

    auc_val: float
    try:
        auc_val = float(roc_auc_score(y_true, probs))
    except ValueError:
        auc_val = float("nan")

    return HoldoutMetrics(
        accuracy=float(accuracy_score(y_true, preds)),
        log_loss=float(log_loss(y_true, probs)),
        brier=float(brier_score_loss(y_true, probs)),
        auc=auc_val,
    )


def build_base_estimator() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def time_series_oof_probabilities(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    min_train_size: int,
) -> np.ndarray:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    oof = np.full(shape=(len(y),), fill_value=np.nan, dtype=float)

    for train_idx, valid_idx in splitter.split(X):
        if len(train_idx) < min_train_size:
            continue
        est = build_base_estimator()
        est.fit(X[train_idx], y[train_idx])
        oof[valid_idx] = est.predict_proba(X[valid_idx])[:, 1]

    return oof


def candidate_estimators(n_splits: int):
    time_cv = TimeSeriesSplit(n_splits=n_splits)

    yield "raw_logistic", LogisticRegression(max_iter=2000, random_state=42)
    yield (
        "platt_calibrated",
        CalibratedClassifierCV(
            estimator=LogisticRegression(max_iter=2000, random_state=42),
            method="sigmoid",
            cv=time_cv,
        ),
    )
    yield (
        "isotonic_calibrated",
        CalibratedClassifierCV(
            estimator=LogisticRegression(max_iter=2000, random_state=42),
            method="isotonic",
            cv=time_cv,
        ),
    )


def choose_best_model(metrics: Dict[str, HoldoutMetrics]) -> Tuple[str, HoldoutMetrics]:
    winner_name = max(
        metrics,
        key=lambda key: (
            metrics[key].accuracy,
            -metrics[key].log_loss,
            -metrics[key].brier,
        ),
    )
    return winner_name, metrics[winner_name]


def time_series_oof_for_estimator(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    min_train_size: int,
) -> np.ndarray:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    oof = np.full(shape=(len(y),), fill_value=np.nan, dtype=float)

    for train_idx, valid_idx in splitter.split(X):
        if len(train_idx) < min_train_size:
            continue

        model = clone(estimator)
        model.fit(X[train_idx], y[train_idx])
        oof[valid_idx] = model.predict_proba(X[valid_idx])[:, 1]

    return oof


def season_stability_report(
    seasons: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
    min_games: int,
) -> pd.DataFrame:
    preds = (probs >= 0.5).astype(int)
    frame = pd.DataFrame(
        {
            "season_id": seasons,
            "y_true": y_true,
            "pred": preds,
        }
    )

    summary = (
        frame.groupby("season_id")
        .apply(
            lambda g: pd.Series(
                {
                    "games": len(g),
                    "accuracy": float((g["y_true"] == g["pred"]).mean()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .sort_values("games", ascending=False)
    )

    return summary[summary["games"] >= min_games]


def prepare_stack_frames(dataset: pd.DataFrame):
    full = games_with_full_k_coverage(dataset)
    if full.empty:
        raise RuntimeError("No games have complete k=5/10/15 feature coverage.")

    anchor = (
        full[full["k_value"] == 15][["game_id", "game_date", "season_id", "home_win", "home_team_id", "away_team_id"]]
        .drop_duplicates(subset=["game_id"])
        .sort_values(["game_date", "game_id"])
        .reset_index(drop=True)
    )

    game_ids = anchor["game_id"].tolist()
    x_by_k: Dict[int, np.ndarray] = {}

    for k in K_VALUES:
        k_frame = (
            full[full["k_value"] == k][["game_id", *FEATURE_COLUMNS]]
            .drop_duplicates(subset=["game_id"])
            .set_index("game_id")
            .loc[game_ids]
        )
        x_by_k[k] = k_frame.to_numpy(dtype=float)

    y = anchor["home_win"].to_numpy(dtype=int)
    seasons = anchor["season_id"].to_numpy()

    return anchor, x_by_k, y, seasons


def save_model_artifacts(
    output_root: Path,
    version: str,
    bundle: Dict[str, object],
    metrics: Dict[str, object],
    should_promote: bool,
) -> Path:
    version_dir = output_root / version
    version_dir.mkdir(parents=True, exist_ok=True)

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
        "gates_passed": metrics["gates"]["all_passed"],
    }
    metadata_path = version_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    if should_promote:
        active_path = output_root / "active_model.json"
        active_tmp = output_root / "active_model.json.tmp"
        active_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        active_tmp.replace(active_path)

    return version_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output_root = str(Path(__file__).resolve().parent / "model_store" / "whl_v2")
    parser.add_argument("--output-root", default=default_output_root)
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--export-dataset", default=None)
    parser.add_argument("--min-accuracy", type=float, default=0.60)
    parser.add_argument("--min-season-accuracy", type=float, default=0.55)
    parser.add_argument("--min-season-games", type=int, default=100)
    parser.add_argument("--base-oof-splits", type=int, default=5)
    parser.add_argument("--meta-cv-splits", type=int, default=5)
    parser.add_argument("--min-train-size", type=int, default=600)
    parser.add_argument("--db-host", default=None)
    parser.add_argument("--db-port", default=None)
    parser.add_argument("--db-name", default=None)
    parser.add_argument("--db-user", default=None)
    parser.add_argument("--db-password", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    db = DbConfig(
        host=args.db_host or DbConfig.host,
        port=args.db_port or DbConfig.port,
        dbname=args.db_name or DbConfig.dbname,
        user=args.db_user or DbConfig.user,
        password=args.db_password or DbConfig.password,
    )

    dataset = load_canonical_dataset(db)
    if dataset.empty:
        raise RuntimeError("Canonical dataset is empty. Cannot train model.")

    if args.export_dataset:
        export_dataset_csv(Path(args.export_dataset), dataset)

    anchor, x_by_k, y, seasons = prepare_stack_frames(dataset)

    split_idx = int(len(anchor) * 0.8)
    if split_idx <= args.min_train_size:
        raise RuntimeError("Insufficient rows for train/holdout split.")

    y_train = y[:split_idx]
    y_holdout = y[split_idx:]
    seasons_holdout = seasons[split_idx:]

    base_holdout_metrics: Dict[str, Dict[str, float]] = {}
    challenger_holdout_metrics: Dict[str, Dict[str, float]] = {}

    oof_by_k: Dict[int, np.ndarray] = {}
    base_models: Dict[int, Pipeline] = {}
    holdout_probs_by_k: Dict[int, np.ndarray] = {}

    for k in K_VALUES:
        x_train = x_by_k[k][:split_idx]
        x_holdout = x_by_k[k][split_idx:]

        oof = time_series_oof_probabilities(
            X=x_train,
            y=y_train,
            n_splits=args.base_oof_splits,
            min_train_size=args.min_train_size,
        )
        oof_by_k[k] = oof

        base_model = build_base_estimator()
        base_model.fit(x_train, y_train)
        base_models[k] = base_model

        holdout_probs = base_model.predict_proba(x_holdout)[:, 1]
        holdout_probs_by_k[k] = holdout_probs

        base_holdout_metrics[str(k)] = compute_metrics(y_holdout, holdout_probs).as_dict()

        challenger = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=500,
            max_depth=6,
            min_samples_leaf=40,
            random_state=42,
        )
        challenger.fit(x_train, y_train)
        challenger_probs = challenger.predict_proba(x_holdout)[:, 1]
        challenger_holdout_metrics[str(k)] = compute_metrics(y_holdout, challenger_probs).as_dict()

    meta_train = np.column_stack([oof_by_k[k] for k in K_VALUES])
    valid_meta_mask = ~np.isnan(meta_train).any(axis=1)
    meta_train = meta_train[valid_meta_mask]
    y_meta_train = y_train[valid_meta_mask]

    if len(y_meta_train) < args.min_train_size:
        raise RuntimeError("Insufficient valid OOF rows for stacker training.")

    meta_holdout = np.column_stack([holdout_probs_by_k[k] for k in K_VALUES])

    stacker_models = {}
    stacker_holdout_metrics: Dict[str, HoldoutMetrics] = {}
    stacker_cv_metrics: Dict[str, HoldoutMetrics] = {}

    for model_name, model in candidate_estimators(args.meta_cv_splits):
        cv_oof = time_series_oof_for_estimator(
            estimator=model,
            X=meta_train,
            y=y_meta_train,
            n_splits=args.meta_cv_splits,
            min_train_size=args.min_train_size,
        )
        cv_mask = ~np.isnan(cv_oof)
        if not cv_mask.any():
            raise RuntimeError(f"No valid time-CV predictions produced for candidate '{model_name}'.")
        stacker_cv_metrics[model_name] = compute_metrics(y_meta_train[cv_mask], cv_oof[cv_mask])

        model.fit(meta_train, y_meta_train)
        probs = model.predict_proba(meta_holdout)[:, 1]
        stacker_models[model_name] = model
        stacker_holdout_metrics[model_name] = compute_metrics(y_holdout, probs)

    chosen_name, chosen_cv_metrics = choose_best_model(stacker_cv_metrics)
    chosen_metrics = stacker_holdout_metrics[chosen_name]
    chosen_model = stacker_models[chosen_name]
    chosen_holdout_probs = chosen_model.predict_proba(meta_holdout)[:, 1]

    home_baseline_prob = float(y_train.mean())
    home_baseline_probs = np.full(shape=len(y_holdout), fill_value=home_baseline_prob)
    baseline_metrics = compute_metrics(y_holdout, home_baseline_probs)

    season_report = season_stability_report(
        seasons=seasons_holdout,
        y_true=y_holdout,
        probs=chosen_holdout_probs,
        min_games=args.min_season_games,
    )

    season_floor = (
        float(season_report["accuracy"].min())
        if not season_report.empty
        else float(chosen_metrics.accuracy)
    )

    gates = {
        "accuracy_gate": {
            "passed": chosen_metrics.accuracy >= args.min_accuracy,
            "value": chosen_metrics.accuracy,
            "threshold": args.min_accuracy,
        },
        "season_floor_gate": {
            "passed": season_floor >= args.min_season_accuracy,
            "value": season_floor,
            "threshold": args.min_season_accuracy,
            "minimum_games_per_season": args.min_season_games,
        },
        "baseline_log_loss_gate": {
            "passed": chosen_metrics.log_loss < baseline_metrics.log_loss,
            "value": chosen_metrics.log_loss,
            "baseline": baseline_metrics.log_loss,
        },
    }
    gates["all_passed"] = all(gate["passed"] for gate in gates.values() if isinstance(gate, dict))

    now_utc = dt.datetime.now(dt.UTC)
    version = args.model_version or now_utc.strftime("%Y%m%dT%H%M%SZ")

    bundle = {
        "model_family": "whl_v2_hybrid_logistic_stacker",
        "model_version": version,
        "trained_at_utc": now_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "k_values": list(K_VALUES),
        "feature_columns": list(FEATURE_COLUMNS),
        "meta_input_columns": [f"p_k_{k}" for k in K_VALUES],
        "base_models": {str(k): model for k, model in base_models.items()},
        "meta_model": chosen_model,
        "meta_model_name": chosen_name,
    }

    metrics = {
        "dataset": {
            "rows_raw": int(len(dataset)),
            "rows_full_k_coverage": int(len(anchor) * len(K_VALUES)),
            "games_full_k_coverage": int(len(anchor)),
            "date_min": str(anchor["game_date"].min().date()),
            "date_max": str(anchor["game_date"].max().date()),
        },
        "splits": {
            "train_games": int(split_idx),
            "holdout_games": int(len(anchor) - split_idx),
            "train_home_win_rate": float(y_train.mean()),
            "holdout_home_win_rate": float(y_holdout.mean()),
        },
        "base_models_holdout": base_holdout_metrics,
        "challenger_holdout": challenger_holdout_metrics,
        "stacker_candidates_holdout": {
            name: model_metrics.as_dict() for name, model_metrics in stacker_holdout_metrics.items()
        },
        "stacker_candidates_time_cv": {
            name: model_metrics.as_dict() for name, model_metrics in stacker_cv_metrics.items()
        },
        "chosen_model": {
            "name": chosen_name,
            "selection_metric": "time_cv_accuracy_then_log_loss",
            "time_cv": chosen_cv_metrics.as_dict(),
            **chosen_metrics.as_dict(),
        },
        "baseline_home_rate_holdout": baseline_metrics.as_dict(),
        "season_stability_holdout": season_report.to_dict(orient="records"),
        "gates": gates,
    }

    output_root = Path(args.output_root)
    should_promote = (not args.no_promote) and gates["all_passed"]

    version_dir = save_model_artifacts(
        output_root=output_root,
        version=version,
        bundle=bundle,
        metrics=metrics,
        should_promote=should_promote,
    )

    print("Training complete")
    print(json.dumps({
        "version": version,
        "output_dir": str(version_dir),
        "promoted": should_promote,
        "gates": gates,
        "chosen_model": metrics["chosen_model"],
    }, indent=2))

    return 0 if gates["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
