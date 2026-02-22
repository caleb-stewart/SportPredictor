"""Train, evaluate, and optionally promote WHL Predictor V2 model artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from services.context_features import CONTEXT_FEATURE_COLUMNS
from services.training_dataset import (
    DbConfig,
    export_dataset_csv,
    games_with_full_k_coverage,
    load_canonical_dataset,
)
from services.training_features import (
    FEATURE_COLUMNS,
    GOALS_BRANCH_FEATURE_COLUMNS,
    K_VALUES,
    META_INPUT_COLUMNS_V3,
)

try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except Exception:  # noqa: BLE001
    CatBoostClassifier = None
    CATBOOST_AVAILABLE = False


@dataclass(frozen=True)
class HoldoutMetrics:
    accuracy: float
    log_loss: float
    brier: float
    auc: float

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": float(self.accuracy),
            "log_loss": float(self.log_loss),
            "brier": float(self.brier),
            "auc": float(self.auc),
        }


@dataclass(frozen=True)
class TrainerConfig:
    output_root: Path
    model_version: str | None
    no_promote: bool
    export_dataset_path: Path | None
    min_accuracy: float
    min_season_accuracy: float
    min_season_games: int
    base_oof_splits: int
    meta_cv_splits: int
    min_train_size: int
    db: DbConfig
    league_code: str = "whl"


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> HoldoutMetrics:
    probs = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    preds = (probs >= 0.5).astype(int)

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
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
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


def choose_best_model(metrics: dict[str, HoldoutMetrics]) -> tuple[str, HoldoutMetrics]:
    winner_name = max(
        metrics,
        key=lambda key: (
            -metrics[key].log_loss,
            metrics[key].accuracy,
            -metrics[key].brier,
        ),
    )
    return winner_name, metrics[winner_name]


def time_series_oof_for_estimator(
    estimator: Any,
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


def time_series_oof_regression(
    estimator: Any,
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
        oof[valid_idx] = model.predict(X[valid_idx])
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

    anchor_cols = [
        "game_id",
        "game_date",
        "season_id",
        "home_win",
        "home_team_id",
        "away_team_id",
        "home_goal_count",
        "away_goal_count",
        *CONTEXT_FEATURE_COLUMNS,
        "strength_adjusted_goals_diff",
        "strength_adjusted_sog_diff",
    ]
    present_anchor_cols = [col for col in anchor_cols if col in full.columns]
    anchor = (
        full[full["k_value"] == 15][present_anchor_cols]
        .drop_duplicates(subset=["game_id"])
        .sort_values(["game_date", "game_id"])
        .reset_index(drop=True)
    )

    game_ids = anchor["game_id"].tolist()
    x_by_k: dict[int, np.ndarray] = {}

    for k in K_VALUES:
        k_frame = (
            full[full["k_value"] == k][["game_id", *FEATURE_COLUMNS]]
            .drop_duplicates(subset=["game_id"])
            .set_index("game_id")
            .loc[game_ids]
        )
        x_by_k[k] = k_frame.fillna(0.0).to_numpy(dtype=float)

    y = anchor["home_win"].to_numpy(dtype=int)
    y_goal_diff = (
        anchor["home_goal_count"].to_numpy(dtype=float) - anchor["away_goal_count"].to_numpy(dtype=float)
        if {"home_goal_count", "away_goal_count"} <= set(anchor.columns)
        else np.zeros(shape=(len(anchor),), dtype=float)
    )
    context_cols_present = [col for col in CONTEXT_FEATURE_COLUMNS if col in anchor.columns]
    if context_cols_present:
        x_context = anchor[context_cols_present].fillna(0.0).to_numpy(dtype=float)
    else:
        x_context = np.zeros(shape=(len(anchor), 0), dtype=float)

    goals_cols_present = [col for col in GOALS_BRANCH_FEATURE_COLUMNS if col in full.columns]
    if goals_cols_present:
        goals_frame = (
            full[full["k_value"] == 15][["game_id", *goals_cols_present]]
            .drop_duplicates(subset=["game_id"])
            .set_index("game_id")
            .loc[game_ids]
        )
        x_goals = goals_frame.fillna(0.0).to_numpy(dtype=float)
    else:
        x_goals = np.zeros(shape=(len(anchor), 0), dtype=float)

    seasons = anchor["season_id"].to_numpy()
    return anchor, x_by_k, x_context, context_cols_present, x_goals, goals_cols_present, y, y_goal_diff, seasons


def save_model_artifacts(
    output_root: Path,
    version: str,
    bundle: dict[str, object],
    metrics: dict[str, object],
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


def train_model_package(config: TrainerConfig) -> tuple[dict[str, Any], int]:
    dataset = load_canonical_dataset(config.db, league_code=config.league_code)
    if dataset.empty:
        raise RuntimeError("Canonical dataset is empty. Cannot train model.")

    if config.export_dataset_path:
        export_dataset_csv(config.export_dataset_path, dataset)

    (
        anchor,
        x_by_k,
        x_context,
        context_feature_columns,
        x_goals,
        goals_feature_columns,
        y,
        y_goal_diff,
        seasons,
    ) = prepare_stack_frames(dataset)

    split_idx = int(len(anchor) * 0.8)
    if split_idx <= config.min_train_size:
        raise RuntimeError("Insufficient rows for train/holdout split.")

    y_train = y[:split_idx]
    y_holdout = y[split_idx:]
    seasons_holdout = seasons[split_idx:]
    y_goal_train = y_goal_diff[:split_idx]

    base_holdout_metrics: dict[str, dict[str, float]] = {}
    challenger_holdout_metrics: dict[str, dict[str, float]] = {}
    catboost_holdout_metrics: dict[str, dict[str, float] | str] = {}
    oof_by_k: dict[int, np.ndarray] = {}
    base_models: dict[int, Pipeline] = {}
    holdout_probs_by_k: dict[int, np.ndarray] = {}

    for k in K_VALUES:
        x_train = x_by_k[k][:split_idx]
        x_holdout = x_by_k[k][split_idx:]

        oof = time_series_oof_probabilities(
            X=x_train,
            y=y_train,
            n_splits=config.base_oof_splits,
            min_train_size=config.min_train_size,
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

        if CATBOOST_AVAILABLE and CatBoostClassifier is not None:
            cat = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="Logloss",
                depth=6,
                learning_rate=0.05,
                iterations=500,
                random_seed=42,
                verbose=False,
            )
            cat.fit(x_train, y_train)
            cat_probs = cat.predict_proba(x_holdout)[:, 1]
            catboost_holdout_metrics[str(k)] = compute_metrics(y_holdout, cat_probs).as_dict()
        else:
            catboost_holdout_metrics[str(k)] = "catboost unavailable"

    rating_model: Pipeline | None = None
    rating_oof: np.ndarray | None = None
    rating_holdout_probs: np.ndarray | None = None
    rating_holdout_metrics: dict[str, float] | None = None
    if x_context.shape[1] > 0:
        x_context_train = x_context[:split_idx]
        x_context_holdout = x_context[split_idx:]
        rating_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        )
        rating_oof = time_series_oof_probabilities(
            X=x_context_train,
            y=y_train,
            n_splits=config.base_oof_splits,
            min_train_size=config.min_train_size,
        )
        rating_model.fit(x_context_train, y_train)
        rating_holdout_probs = rating_model.predict_proba(x_context_holdout)[:, 1]
        rating_holdout_metrics = compute_metrics(y_holdout, rating_holdout_probs).as_dict()

    goals_regressor: HistGradientBoostingRegressor | None = None
    goals_calibrator: LogisticRegression | None = None
    goals_oof_prob: np.ndarray | None = None
    goals_holdout_prob: np.ndarray | None = None
    goals_holdout_metrics: dict[str, float] | None = None
    if x_goals.shape[1] > 0:
        x_goals_train = x_goals[:split_idx]
        x_goals_holdout = x_goals[split_idx:]

        goals_regressor = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=400,
            max_depth=6,
            min_samples_leaf=30,
            random_state=42,
        )
        oof_goal_diff = time_series_oof_regression(
            estimator=goals_regressor,
            X=x_goals_train,
            y=y_goal_train,
            n_splits=config.base_oof_splits,
            min_train_size=config.min_train_size,
        )
        valid_goal_mask = ~np.isnan(oof_goal_diff)
        if valid_goal_mask.any():
            goals_calibrator = LogisticRegression(max_iter=1000, random_state=42)
            goals_calibrator.fit(oof_goal_diff[valid_goal_mask].reshape(-1, 1), y_train[valid_goal_mask])
            goals_oof_prob = np.full(shape=(len(y_train),), fill_value=np.nan, dtype=float)
            goals_oof_prob[valid_goal_mask] = goals_calibrator.predict_proba(
                oof_goal_diff[valid_goal_mask].reshape(-1, 1)
            )[:, 1]

            goals_regressor.fit(x_goals_train, y_goal_train)
            holdout_goal_diff = goals_regressor.predict(x_goals_holdout).reshape(-1, 1)
            goals_holdout_prob = goals_calibrator.predict_proba(holdout_goal_diff)[:, 1]
            goals_holdout_metrics = compute_metrics(y_holdout, goals_holdout_prob).as_dict()

    meta_train_columns: list[np.ndarray] = [oof_by_k[k] for k in K_VALUES]
    meta_holdout_columns: list[np.ndarray] = [holdout_probs_by_k[k] for k in K_VALUES]
    meta_input_columns: list[str] = [f"p_k_{k}" for k in K_VALUES]

    if rating_oof is not None and rating_holdout_probs is not None:
        meta_train_columns.append(rating_oof)
        meta_holdout_columns.append(rating_holdout_probs)
        meta_input_columns.append("p_rating")

    if goals_oof_prob is not None and goals_holdout_prob is not None:
        meta_train_columns.append(goals_oof_prob)
        meta_holdout_columns.append(goals_holdout_prob)
        meta_input_columns.append("p_goals")

    meta_train = np.column_stack(meta_train_columns)
    valid_meta_mask = ~np.isnan(meta_train).any(axis=1)
    meta_train = meta_train[valid_meta_mask]
    y_meta_train = y_train[valid_meta_mask]

    if len(y_meta_train) < config.min_train_size:
        raise RuntimeError("Insufficient valid OOF rows for stacker training.")

    meta_holdout = np.column_stack(meta_holdout_columns)

    stacker_models: dict[str, Any] = {}
    stacker_holdout_metrics: dict[str, HoldoutMetrics] = {}
    stacker_cv_metrics: dict[str, HoldoutMetrics] = {}

    for model_name, model in candidate_estimators(config.meta_cv_splits):
        cv_oof = time_series_oof_for_estimator(
            estimator=model,
            X=meta_train,
            y=y_meta_train,
            n_splits=config.meta_cv_splits,
            min_train_size=config.min_train_size,
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
        min_games=config.min_season_games,
    )

    season_floor = (
        float(season_report["accuracy"].min())
        if not season_report.empty
        else float(chosen_metrics.accuracy)
    )

    gates = {
        "accuracy_gate": {
            "passed": chosen_metrics.accuracy >= config.min_accuracy,
            "value": chosen_metrics.accuracy,
            "threshold": config.min_accuracy,
        },
        "season_floor_gate": {
            "passed": season_floor >= config.min_season_accuracy,
            "value": season_floor,
            "threshold": config.min_season_accuracy,
            "minimum_games_per_season": config.min_season_games,
        },
        "baseline_log_loss_gate": {
            "passed": chosen_metrics.log_loss < baseline_metrics.log_loss,
            "value": chosen_metrics.log_loss,
            "baseline": baseline_metrics.log_loss,
        },
        "baseline_brier_gate": {
            "passed": chosen_metrics.brier < baseline_metrics.brier,
            "value": chosen_metrics.brier,
            "baseline": baseline_metrics.brier,
        },
    }
    gates["all_passed"] = all(gate["passed"] for gate in gates.values() if isinstance(gate, dict))

    now_utc = dt.datetime.now(dt.UTC)
    version = config.model_version or now_utc.strftime("%Y%m%dT%H%M%SZ")

    bundle = {
        "model_family": "whl_v3_hybrid_branch_stacker",
        "model_version": version,
        "trained_at_utc": now_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "k_values": list(K_VALUES),
        "feature_columns": list(FEATURE_COLUMNS),
        "meta_input_columns": list(meta_input_columns),
        "base_models": {str(k): model for k, model in base_models.items()},
        "rating_model": rating_model,
        "rating_feature_columns": list(context_feature_columns),
        "goals_regressor": goals_regressor,
        "goals_calibrator": goals_calibrator,
        "goals_feature_columns": list(goals_feature_columns),
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
        "catboost_challenger_holdout": catboost_holdout_metrics,
        "rating_branch_holdout": rating_holdout_metrics,
        "goals_branch_holdout": goals_holdout_metrics,
        "stacker_candidates_holdout": {
            name: model_metrics.as_dict() for name, model_metrics in stacker_holdout_metrics.items()
        },
        "stacker_candidates_time_cv": {
            name: model_metrics.as_dict() for name, model_metrics in stacker_cv_metrics.items()
        },
        "chosen_model": {
            "name": chosen_name,
            "selection_metric": "time_cv_log_loss_then_accuracy",
            "time_cv": chosen_cv_metrics.as_dict(),
            **chosen_metrics.as_dict(),
        },
        "baseline_home_rate_holdout": baseline_metrics.as_dict(),
        "season_stability_holdout": season_report.to_dict(orient="records"),
        "gates": gates,
    }

    should_promote = (not config.no_promote) and gates["all_passed"]
    version_dir = save_model_artifacts(
        output_root=config.output_root,
        version=version,
        bundle=bundle,
        metrics=metrics,
        should_promote=should_promote,
    )

    details: dict[str, Any] = {
        "version": version,
        "output_dir": str(version_dir),
        "promoted": should_promote,
        "model_family": bundle["model_family"],
        "gates": gates,
        "chosen_model": metrics["chosen_model"],
    }
    return details, (0 if gates["all_passed"] else 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output_root = str(Path(__file__).resolve().parents[1] / "model_store" / "whl_v2")
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
    parser.add_argument("--league-code", default="whl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_db = DbConfig()
    db = DbConfig(
        host=args.db_host or default_db.host,
        port=str(args.db_port or default_db.port),
        dbname=args.db_name or default_db.dbname,
        user=args.db_user or default_db.user,
        password=args.db_password or default_db.password,
    )

    config = TrainerConfig(
        output_root=Path(args.output_root),
        model_version=args.model_version,
        no_promote=bool(args.no_promote),
        export_dataset_path=Path(args.export_dataset) if args.export_dataset else None,
        min_accuracy=float(args.min_accuracy),
        min_season_accuracy=float(args.min_season_accuracy),
        min_season_games=int(args.min_season_games),
        base_oof_splits=int(args.base_oof_splits),
        meta_cv_splits=int(args.meta_cv_splits),
        min_train_size=int(args.min_train_size),
        db=db,
        league_code=str(args.league_code or "whl"),
    )

    details, code = train_model_package(config)
    print("Training complete")
    print(json.dumps(details, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
