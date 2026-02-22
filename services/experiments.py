from __future__ import annotations

import datetime as dt
import json
import re
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import ChlExperiment
from services.context_features import CONTEXT_FEATURE_COLUMNS
from services.data_backend import (
    DataBackendError,
    primary_store,
    require_supported_league_code,
    resolve_league_id_for_store,
)
from services.training_dataset import DbConfig, load_canonical_dataset
from services.training_features import FEATURE_COLUMNS


class ExperimentError(RuntimeError):
    pass


class ExperimentValidationError(ExperimentError):
    pass


class ExperimentNotFoundError(ExperimentError):
    pass


VALID_OPS = {"add", "subtract", "multiply", "divide"}
FEATURE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
BASE_EXPERIMENT_COLUMNS = [
    *FEATURE_COLUMNS,
    *CONTEXT_FEATURE_COLUMNS,
    "strength_adjusted_goals_diff",
    "strength_adjusted_sog_diff",
]


def _compute_metric_bundle(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    probs = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    pred = (probs >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, probs)),
        "brier": float(brier_score_loss(y_true, probs)),
        "auc": auc,
    }


def _model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )


def _apply_operation(left: np.ndarray, right: np.ndarray, op: str) -> np.ndarray:
    if op == "add":
        return left + right
    if op == "subtract":
        return left - right
    if op == "multiply":
        return left * right
    if op == "divide":
        return left / np.where(np.abs(right) < 1e-6, 1.0, right)
    raise ExperimentValidationError(f"Unsupported op: {op}")


def _write_report_file(experiment_id: uuid.UUID, payload: dict[str, Any]) -> str:
    root = Path(__file__).resolve().parents[1] / "reports" / "experiments"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{experiment_id}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def run_feature_proposal_experiment(
    db: Session,
    *,
    proposal_name: str,
    proposals: list[dict[str, Any]],
    seed: int = 42,
    league_code: str | None = None,
) -> dict[str, Any]:
    try:
        normalized_league = require_supported_league_code(league_code)
    except DataBackendError as exc:
        raise ExperimentValidationError(str(exc)) from exc
    store = primary_store()
    league_id = resolve_league_id_for_store(db, store, normalized_league)
    if league_id is None:
        raise ExperimentValidationError(f"Unable to resolve league scope for league_code={normalized_league}")

    if not proposals:
        raise ExperimentValidationError("At least one feature proposal is required.")

    for item in proposals:
        name = str(item.get("name") or "")
        op = str(item.get("op") or "")
        left = str(item.get("left_feature") or "")
        right = str(item.get("right_feature") or "")
        if not FEATURE_NAME_RE.fullmatch(name):
            raise ExperimentValidationError(f"Invalid feature name: {name!r}")
        if op not in VALID_OPS:
            raise ExperimentValidationError(f"Invalid op={op!r}. Allowed: {sorted(VALID_OPS)}")
        if not left or not right:
            raise ExperimentValidationError("left_feature and right_feature are required.")

    exp = ChlExperiment(
        id=uuid.uuid4(),
        league_id=league_id,
        status="running",
        experiment_type="feature_proposal",
        created_at=dt.datetime.now(dt.UTC),
        proposal_json={
            "proposal_name": proposal_name,
            "proposals": proposals,
            "seed": seed,
            "no_leakage_rule": "source_game_date < target_game_date",
        },
    )
    db.add(exp)
    db.commit()

    try:
        dataset = load_canonical_dataset(DbConfig(), league_code=normalized_league)
        if dataset.empty:
            raise ExperimentValidationError("Canonical dataset is empty.")

        anchor = (
            dataset[dataset["k_value"] == 15]
            .drop_duplicates(subset=["game_id"])
            .sort_values(["game_date", "game_id"])
            .reset_index(drop=True)
        )

        usable_base_cols = [col for col in BASE_EXPERIMENT_COLUMNS if col in anchor.columns]
        if not usable_base_cols:
            raise ExperimentValidationError("No baseline feature columns available for experiment.")

        base_frame = anchor[usable_base_cols].astype(float).fillna(0.0).copy()
        candidate_frame = base_frame.copy()
        derived_features: list[str] = []

        for item in proposals:
            name = str(item["name"])
            left = str(item["left_feature"])
            right = str(item["right_feature"])
            op = str(item["op"])
            if left not in candidate_frame.columns:
                raise ExperimentValidationError(f"Unknown left_feature: {left}")
            if right not in candidate_frame.columns:
                raise ExperimentValidationError(f"Unknown right_feature: {right}")
            candidate_frame[name] = _apply_operation(
                candidate_frame[left].to_numpy(dtype=float),
                candidate_frame[right].to_numpy(dtype=float),
                op=op,
            )
            derived_features.append(name)

        y = anchor["home_win"].to_numpy(dtype=int)
        split = int(len(anchor) * 0.8)
        if split < 300:
            raise ExperimentValidationError("Not enough rows to run experiment split.")

        rng = np.random.default_rng(seed)
        # Deterministic tiny jitter prevents degenerate duplicate columns in some proposals.
        jitter = rng.normal(loc=0.0, scale=1e-10, size=candidate_frame.shape)
        x_base = base_frame.to_numpy(dtype=float)
        x_candidate = (candidate_frame.to_numpy(dtype=float) + jitter).astype(float)

        base_model = _model()
        base_model.fit(x_base[:split], y[:split])
        base_probs = base_model.predict_proba(x_base[split:])[:, 1]
        base_metrics = _compute_metric_bundle(y[split:], base_probs)

        candidate_model = _model()
        candidate_model.fit(x_candidate[:split], y[:split])
        candidate_probs = candidate_model.predict_proba(x_candidate[split:])[:, 1]
        candidate_metrics = _compute_metric_bundle(y[split:], candidate_probs)

        verdict = {
            "accuracy_improved": bool(candidate_metrics["accuracy"] > base_metrics["accuracy"]),
            "log_loss_improved": bool(candidate_metrics["log_loss"] < base_metrics["log_loss"]),
            "brier_improved": bool(candidate_metrics["brier"] < base_metrics["brier"]),
        }
        verdict["accepted"] = all(verdict.values())

        result = {
            "proposal_name": proposal_name,
            "rows": int(len(anchor)),
            "split_train": int(split),
            "split_holdout": int(len(anchor) - split),
            "base_feature_count": int(x_base.shape[1]),
            "candidate_feature_count": int(x_candidate.shape[1]),
            "derived_features": derived_features,
            "baseline_metrics": base_metrics,
            "candidate_metrics": candidate_metrics,
            "delta_new_minus_old": {
                "accuracy": float(candidate_metrics["accuracy"] - base_metrics["accuracy"]),
                "log_loss": float(candidate_metrics["log_loss"] - base_metrics["log_loss"]),
                "brier": float(candidate_metrics["brier"] - base_metrics["brier"]),
            },
            "verdict": verdict,
        }
        report_path = _write_report_file(exp.id, result)
        result["report_path"] = report_path

        exp.status = "completed"
        exp.completed_at = dt.datetime.now(dt.UTC)
        exp.result_json = result
        db.commit()
        db.refresh(exp)

        return {
            "experiment_id": str(exp.id),
            "league_code": normalized_league,
            "status": exp.status,
            "experiment_type": exp.experiment_type,
            "result": result,
        }
    except Exception as exc:  # noqa: BLE001
        exp.status = "failed"
        exp.completed_at = dt.datetime.now(dt.UTC)
        exp.error_text = str(exc)
        db.commit()
        raise


def get_experiment(db: Session, experiment_id: str, league_code: str | None = None) -> dict[str, Any]:
    try:
        normalized_league = require_supported_league_code(league_code)
    except DataBackendError as exc:
        raise ExperimentValidationError(str(exc)) from exc
    store = primary_store()
    league_id = resolve_league_id_for_store(db, store, normalized_league)
    if league_id is None:
        raise ExperimentValidationError(f"Unable to resolve league scope for league_code={normalized_league}")

    try:
        parsed = uuid.UUID(experiment_id)
    except ValueError as exc:
        raise ExperimentValidationError("experiment_id must be a valid UUID.") from exc

    exp = db.scalar(select(ChlExperiment).where(ChlExperiment.id == parsed, ChlExperiment.league_id == league_id))
    if exp is None:
        raise ExperimentNotFoundError(f"Experiment not found: {experiment_id}")

    return {
        "experiment_id": str(exp.id),
        "league_code": normalized_league,
        "status": exp.status,
        "experiment_type": exp.experiment_type,
        "created_at": exp.created_at,
        "completed_at": exp.completed_at,
        "proposal": exp.proposal_json,
        "result": exp.result_json,
        "error_text": exp.error_text,
    }
