from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from core.config import settings
from services.data_backend import normalize_league_code
from services.training_features import (
    FeatureContractError,
    GOALS_BRANCH_FEATURE_COLUMNS,
    K_VALUES,
    build_pairwise_features as _shared_build_pairwise_features,
    feature_vector_from_engineered as _shared_feature_vector_from_engineered,
)
from services.v4_models import v4_probability_from_context


class PredictorError(RuntimeError):
    pass


class ModelNotAvailableError(PredictorError):
    pass


class PayloadContractError(PredictorError):
    pass


_bundle_lock = threading.Lock()
_cached_by_pointer: dict[str, tuple[float, dict[str, Any]]] = {}


def _model_root_for_league(league_code: str | None = None) -> Path:
    root = Path(settings.model_store_root).resolve()
    return (root / normalize_league_code(league_code or settings.default_league_code)).resolve()


def _resolve_active_pointer(league_code: str | None = None) -> tuple[Path, Path]:
    root = _model_root_for_league(league_code)
    pointer = root / settings.active_model_file
    if pointer.exists():
        return root, pointer

    # Backward compatibility for legacy single-league model roots.
    legacy_root = Path(settings.model_store_root).resolve()
    legacy_pointer = legacy_root / settings.active_model_file
    if normalize_league_code(league_code) == "whl" and legacy_pointer.exists():
        return legacy_root, legacy_pointer

    raise ModelNotAvailableError(f"No active model pointer found at: {pointer}")


def _read_active_pointer(league_code: str | None = None) -> tuple[Path, Path, dict[str, Any], str]:
    model_root, pointer_path = _resolve_active_pointer(league_code=league_code)
    raw = pointer_path.read_text(encoding="utf-8")
    active = json.loads(raw)
    return model_root, pointer_path, active, raw


def _resolve_bundle_path(model_root: Path, active: dict[str, Any]) -> Path:
    bundle_path = Path(active["bundle_path"])
    if not bundle_path.is_absolute():
        bundle_path = model_root / bundle_path
    return bundle_path


def get_active_model_pointer_metadata(league_code: str | None = None) -> dict[str, Any]:
    model_root, pointer_path, active, raw = _read_active_pointer(league_code=league_code)
    bundle_path = _resolve_bundle_path(model_root, active)
    return {
        "pointer_path": str(pointer_path),
        "pointer_mtime_ns": pointer_path.stat().st_mtime_ns,
        "pointer_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "model_version": active.get("model_version"),
        "model_family": active.get("model_family"),
        "bundle_path": str(bundle_path),
    }


def is_active_model_pointer_unchanged(frozen_pointer: dict[str, Any], league_code: str | None = None) -> bool:
    current = get_active_model_pointer_metadata(league_code=league_code)
    return (
        current.get("pointer_path") == frozen_pointer.get("pointer_path")
        and current.get("pointer_sha256") == frozen_pointer.get("pointer_sha256")
        and current.get("model_version") == frozen_pointer.get("model_version")
    )


def load_frozen_active_model(league_code: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    model_root, _, active, _ = _read_active_pointer(league_code=league_code)
    bundle_path = _resolve_bundle_path(model_root, active)
    if not bundle_path.exists():
        raise ModelNotAvailableError(f"Active bundle does not exist: {bundle_path}")

    bundle = joblib.load(bundle_path)
    bundle["_active_metadata"] = active
    return bundle, get_active_model_pointer_metadata(league_code=league_code)


def load_model_bundle_by_version(model_version: str, league_code: str | None = None) -> dict[str, Any]:
    metadata_path = _find_model_metadata_path(model_version, league_code=league_code)
    if metadata_path is None:
        searched = ", ".join(str(root / model_version / "metadata.json") for root in _model_version_roots(league_code))
        raise ModelNotAvailableError(f"Model metadata not found for version={model_version}; searched: {searched}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    bundle_path = _resolve_metadata_bundle_path(metadata_path, metadata)
    if not bundle_path.exists():
        raise ModelNotAvailableError(f"Model bundle not found for version={model_version}: {bundle_path}")

    bundle = joblib.load(bundle_path)
    bundle["_active_metadata"] = {
        "model_version": metadata.get("model_version") or model_version,
        "model_family": metadata.get("model_family"),
    }
    return bundle


def _model_version_roots(league_code: str | None = None) -> list[Path]:
    """Return scoped and legacy artifact roots in lookup order."""

    base_root = Path(settings.model_store_root).resolve()
    scoped_root = _model_root_for_league(league_code)
    roots = [scoped_root]
    if normalize_league_code(league_code) == "whl":
        roots.extend(
            [
                base_root,
                base_root / "whl_v3_baseline",
                base_root / "whl_v4",
                base_root.parent / "whl_v3_baseline",
                base_root.parent / "whl_v4",
            ]
        )

    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def _find_model_metadata_path(model_version: str, league_code: str | None = None) -> Path | None:
    for root in _model_version_roots(league_code):
        candidate = root / model_version / "metadata.json"
        if candidate.exists():
            return candidate
    return None


def _resolve_metadata_bundle_path(metadata_path: Path, metadata: dict[str, Any]) -> Path:
    raw_path = Path(str(metadata.get("bundle_path") or ""))
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path
    if raw_path.is_absolute():
        # Metadata may contain the path from the machine that trained it. In a
        # deployed model-store mount, the bundle is beside metadata.
        return metadata_path.parent / raw_path.name
    return metadata_path.parent / raw_path


def _json_atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def activate_model_version(model_version: str, league_code: str | None = None) -> dict[str, Any]:
    """Atomically make a verified model version the active pointer."""

    metadata_path = _find_model_metadata_path(model_version, league_code=league_code)
    if metadata_path is None:
        raise ModelNotAvailableError(f"Model metadata not found for version={model_version}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    bundle_path = _resolve_metadata_bundle_path(metadata_path, metadata)
    if not bundle_path.exists():
        raise ModelNotAvailableError(f"Model bundle not found for version={model_version}: {bundle_path}")
    if not bool(metadata.get("gates_passed", True)):
        raise ModelNotAvailableError(f"Model version={model_version} did not pass promotion gates")

    model_root, pointer_path, _, _ = _read_active_pointer(league_code=league_code)
    pointer_payload = dict(metadata)
    try:
        pointer_payload["bundle_path"] = str(bundle_path.relative_to(model_root))
    except ValueError:
        pointer_payload["bundle_path"] = str(bundle_path)

    metrics_path = metadata.get("metrics_path")
    if metrics_path:
        resolved_metrics = Path(str(metrics_path))
        if not resolved_metrics.exists() and resolved_metrics.is_absolute():
            resolved_metrics = metadata_path.parent / resolved_metrics.name
        if resolved_metrics.exists():
            try:
                pointer_payload["metrics_path"] = str(resolved_metrics.relative_to(model_root))
            except ValueError:
                pointer_payload["metrics_path"] = str(resolved_metrics)

    _json_atomic_write(pointer_path, pointer_payload)
    with _bundle_lock:
        _cached_by_pointer.pop(str(pointer_path), None)

    return {
        "active": True,
        "pointer_path": str(pointer_path),
        "model_version": pointer_payload.get("model_version") or model_version,
        "model_family": pointer_payload.get("model_family"),
        "bundle_path": str(bundle_path),
    }


def _coerce_float(value: Any, field_name: str) -> float:
    if value is None:
        raise PayloadContractError(f"Missing required metric '{field_name}'.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise PayloadContractError(f"Invalid numeric value for '{field_name}': {value!r}") from exc


def _build_pairwise_features(home_metrics: dict[str, Any], away_metrics: dict[str, Any]) -> dict[str, float]:
    try:
        return _shared_build_pairwise_features(home_metrics, away_metrics)
    except FeatureContractError as exc:
        raise PayloadContractError(str(exc)) from exc


def _feature_vector(engineered: dict[str, float]) -> list[float]:
    try:
        return _shared_feature_vector_from_engineered(engineered)
    except FeatureContractError as exc:
        raise PayloadContractError(str(exc)) from exc


def _context_vector(context_features: dict[str, Any], columns: list[str], field_prefix: str = "context") -> list[float]:
    vector: list[float] = []
    for col in columns:
        vector.append(_coerce_float(context_features.get(col), f"{field_prefix}.{col}"))
    return vector


def load_active_model(league_code: str | None = None) -> dict[str, Any]:
    model_root, pointer_path, active, _ = _read_active_pointer(league_code=league_code)
    current_mtime = pointer_path.stat().st_mtime
    cache_key = str(pointer_path)

    with _bundle_lock:
        cached = _cached_by_pointer.get(cache_key)
        if cached is not None and cached[0] == current_mtime:
            return cached[1]

        bundle_path = _resolve_bundle_path(model_root, active)

        if not bundle_path.exists():
            raise ModelNotAvailableError(f"Active bundle does not exist: {bundle_path}")

        bundle = joblib.load(bundle_path)
        bundle["_active_metadata"] = active
        _cached_by_pointer[cache_key] = (current_mtime, bundle)
        return bundle


def get_model_status(league_code: str | None = None) -> dict[str, Any]:
    try:
        bundle = load_active_model(league_code=league_code)
    except ModelNotAvailableError as exc:
        return {"active": False, "error": str(exc)}

    active_meta = bundle.get("_active_metadata") or {}
    return {
        "active": True,
        "model_version": active_meta.get("model_version") or bundle.get("model_version"),
        "model_family": active_meta.get("model_family") or bundle.get("model_family"),
        "k_values": bundle.get("k_values") or K_VALUES,
    }


def _build_goals_branch_features(
    payload: dict[str, Any],
    goals_feature_columns: list[str],
) -> list[float]:
    features_by_k = payload.get("features_by_k") or {}
    k15_data = features_by_k.get("15")
    if not isinstance(k15_data, dict):
        raise PayloadContractError("Missing k=15 features required for goals branch.")

    home = k15_data.get("home")
    away = k15_data.get("away")
    if not isinstance(home, dict) or not isinstance(away, dict):
        raise PayloadContractError("k=15 features must include 'home' and 'away' for goals branch.")

    engineered = _build_pairwise_features(home, away)
    context_features = payload.get("context_features") or {}
    if not isinstance(context_features, dict):
        raise PayloadContractError("Payload field 'context_features' must be an object when provided.")

    merged = {**engineered}
    for key, value in context_features.items():
        try:
            merged[key] = float(value)
        except (TypeError, ValueError):
            continue

    missing = [col for col in goals_feature_columns if col not in merged]
    if missing:
        raise PayloadContractError(f"Missing goals branch features: {missing}")
    return [float(merged[col]) for col in goals_feature_columns]


def _predict_with_bundle(payload: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    v4_meta_model = bundle.get("v4_meta_model")
    if v4_meta_model is not None or str(bundle.get("model_family", "")).startswith("whl_v4_"):
        context_features = payload.get("context_features")
        if not isinstance(context_features, dict):
            raise PayloadContractError("V4 model payload must include 'context_features'.")
        try:
            v4 = v4_probability_from_context(context_features, combiner=v4_meta_model)
        except KeyError as exc:
            raise PayloadContractError(f"Missing required V4 context feature '{exc.args[0]}'.") from exc

        home_team_id = payload.get("home_team_id")
        away_team_id = payload.get("away_team_id")
        final_home_prob = float(v4["home_prob"])
        predicted_winner_id = home_team_id if final_home_prob >= 0.5 else away_team_id
        active_meta = bundle.get("_active_metadata") or {}
        return {
            "home_team_prob": final_home_prob,
            "away_team_prob": float(v4["away_prob"]),
            "predicted_winner_id": predicted_winner_id,
            "model_version": active_meta.get("model_version") or bundle.get("model_version"),
            "model_family": active_meta.get("model_family") or bundle.get("model_family"),
            "k_components": {
                "v4_rating": {
                    "home_team_prob": v4["v4_rating_home_prob"],
                    "away_team_prob": 1.0 - v4["v4_rating_home_prob"],
                },
                "v4_goal_distribution": {
                    "home_team_prob": v4["v4_goal_home_prob"],
                    "away_team_prob": 1.0 - v4["v4_goal_home_prob"],
                },
            },
        }

    features_by_k = payload.get("features_by_k")
    if not isinstance(features_by_k, dict):
        raise PayloadContractError("Payload must include a 'features_by_k' object.")

    base_models = bundle.get("base_models")
    meta_model = bundle.get("meta_model")

    if not isinstance(base_models, dict) or meta_model is None:
        raise ModelNotAvailableError("Active model bundle is malformed.")

    component_probs: dict[str, dict[str, float]] = {}
    ordered_home_probs: list[float] = []
    meta_inputs: dict[str, float] = {}

    for k in K_VALUES:
        k_key = str(k)
        k_data = features_by_k.get(k_key)
        if not isinstance(k_data, dict):
            raise PayloadContractError(f"Missing features for k={k_key}.")

        home = k_data.get("home")
        away = k_data.get("away")
        if not isinstance(home, dict) or not isinstance(away, dict):
            raise PayloadContractError(f"features_by_k['{k_key}'] must include 'home' and 'away'.")

        engineered = _build_pairwise_features(home, away)
        vector = np.array([_feature_vector(engineered)], dtype=float)

        model = base_models.get(k_key)
        if model is None:
            raise ModelNotAvailableError(f"Base model missing for k={k_key}.")

        home_prob = float(model.predict_proba(vector)[:, 1][0])
        ordered_home_probs.append(home_prob)
        meta_inputs[f"p_k_{k}"] = home_prob
        component_probs[k_key] = {
            "home_team_prob": home_prob,
            "away_team_prob": 1.0 - home_prob,
        }

    rating_model = bundle.get("rating_model")
    if rating_model is not None:
        context_features = payload.get("context_features")
        if not isinstance(context_features, dict):
            raise PayloadContractError("Payload must include 'context_features' for this model.")
        rating_cols = bundle.get("rating_feature_columns") or []
        if not isinstance(rating_cols, list) or not rating_cols:
            raise ModelNotAvailableError("Model bundle has rating_model but no rating_feature_columns.")
        rating_vector = np.array([_context_vector(context_features, rating_cols)], dtype=float)
        meta_inputs["p_rating"] = float(rating_model.predict_proba(rating_vector)[:, 1][0])

    goals_regressor = bundle.get("goals_regressor")
    goals_calibrator = bundle.get("goals_calibrator")
    if goals_regressor is not None and goals_calibrator is not None:
        goals_cols = bundle.get("goals_feature_columns") or GOALS_BRANCH_FEATURE_COLUMNS
        if not isinstance(goals_cols, list) or not goals_cols:
            raise ModelNotAvailableError("Model bundle has goals branch but no goals_feature_columns.")
        goals_vector = np.array([_build_goals_branch_features(payload, goals_cols)], dtype=float)
        predicted_goal_diff = goals_regressor.predict(goals_vector).reshape(-1, 1)
        meta_inputs["p_goals"] = float(goals_calibrator.predict_proba(predicted_goal_diff)[:, 1][0])

    meta_input_columns = bundle.get("meta_input_columns") or [f"p_k_{k}" for k in K_VALUES]
    if not isinstance(meta_input_columns, list) or not meta_input_columns:
        raise ModelNotAvailableError("Model bundle meta_input_columns is malformed.")
    try:
        meta_ordered = [float(meta_inputs[col]) for col in meta_input_columns]
    except KeyError as exc:
        raise PayloadContractError(f"Missing required meta input '{exc.args[0]}' for this model.") from exc

    meta_vector = np.array([meta_ordered], dtype=float)
    final_home_prob = float(meta_model.predict_proba(meta_vector)[:, 1][0])
    final_away_prob = 1.0 - final_home_prob

    home_team_id = payload.get("home_team_id")
    away_team_id = payload.get("away_team_id")
    predicted_winner_id = home_team_id if final_home_prob >= final_away_prob else away_team_id

    active_meta = bundle.get("_active_metadata") or {}
    return {
        "home_team_prob": final_home_prob,
        "away_team_prob": final_away_prob,
        "predicted_winner_id": predicted_winner_id,
        "model_version": active_meta.get("model_version") or bundle.get("model_version"),
        "model_family": active_meta.get("model_family") or bundle.get("model_family"),
        "k_components": component_probs,
    }


def predict_from_payload(
    payload: dict[str, Any],
    bundle: dict[str, Any] | None = None,
    league_code: str | None = None,
) -> dict[str, Any]:
    if bundle is None:
        if league_code is None:
            bundle = load_active_model()
        else:
            bundle = load_active_model(league_code=league_code)
    return _predict_with_bundle(payload=payload, bundle=bundle)
