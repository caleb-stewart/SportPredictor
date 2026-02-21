from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from core.config import settings
from services.training_features import BASE_METRICS, FEATURE_COLUMNS, K_VALUES


class PredictorError(RuntimeError):
    pass


class ModelNotAvailableError(PredictorError):
    pass


class PayloadContractError(PredictorError):
    pass


_bundle_lock = threading.Lock()
_cached_bundle: dict[str, Any] | None = None
_cached_mtime: float | None = None


def _resolve_active_pointer() -> tuple[Path, Path]:
    root = Path(settings.model_store_root).resolve()
    pointer = root / settings.active_model_file
    if pointer.exists():
        return root, pointer
    raise ModelNotAvailableError(f"No active model pointer found at: {pointer}")


def _read_active_pointer() -> tuple[Path, Path, dict[str, Any], str]:
    model_root, pointer_path = _resolve_active_pointer()
    raw = pointer_path.read_text(encoding="utf-8")
    active = json.loads(raw)
    return model_root, pointer_path, active, raw


def _resolve_bundle_path(model_root: Path, active: dict[str, Any]) -> Path:
    bundle_path = Path(active["bundle_path"])
    if not bundle_path.is_absolute():
        bundle_path = model_root / bundle_path
    return bundle_path


def get_active_model_pointer_metadata() -> dict[str, Any]:
    model_root, pointer_path, active, raw = _read_active_pointer()
    bundle_path = _resolve_bundle_path(model_root, active)
    return {
        "pointer_path": str(pointer_path),
        "pointer_mtime_ns": pointer_path.stat().st_mtime_ns,
        "pointer_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "model_version": active.get("model_version"),
        "model_family": active.get("model_family"),
        "bundle_path": str(bundle_path),
    }


def is_active_model_pointer_unchanged(frozen_pointer: dict[str, Any]) -> bool:
    current = get_active_model_pointer_metadata()
    return (
        current.get("pointer_path") == frozen_pointer.get("pointer_path")
        and current.get("pointer_sha256") == frozen_pointer.get("pointer_sha256")
        and current.get("model_version") == frozen_pointer.get("model_version")
    )


def load_frozen_active_model() -> tuple[dict[str, Any], dict[str, Any]]:
    model_root, _, active, _ = _read_active_pointer()
    bundle_path = _resolve_bundle_path(model_root, active)
    if not bundle_path.exists():
        raise ModelNotAvailableError(f"Active bundle does not exist: {bundle_path}")

    bundle = joblib.load(bundle_path)
    bundle["_active_metadata"] = active
    return bundle, get_active_model_pointer_metadata()


def _coerce_float(value: Any, field_name: str) -> float:
    if value is None:
        raise PayloadContractError(f"Missing required metric '{field_name}'.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise PayloadContractError(f"Invalid numeric value for '{field_name}': {value!r}") from exc


def _build_pairwise_features(home_metrics: dict[str, Any], away_metrics: dict[str, Any]) -> dict[str, float]:
    engineered: dict[str, float] = {}
    for metric in BASE_METRICS:
        h_val = _coerce_float(home_metrics.get(metric), f"home.{metric}")
        a_val = _coerce_float(away_metrics.get(metric), f"away.{metric}")
        engineered[f"{metric}_diff"] = h_val - a_val
        engineered[f"{metric}_sum"] = (h_val + a_val) / 2.0
    return engineered


def _feature_vector(engineered: dict[str, float]) -> list[float]:
    values: list[float] = []
    for col in FEATURE_COLUMNS:
        if col not in engineered:
            raise PayloadContractError(f"Missing engineered feature '{col}'.")
        values.append(float(engineered[col]))
    return values


def load_active_model() -> dict[str, Any]:
    global _cached_bundle, _cached_mtime

    model_root, pointer_path, active, _ = _read_active_pointer()
    current_mtime = pointer_path.stat().st_mtime

    with _bundle_lock:
        if _cached_bundle is not None and _cached_mtime == current_mtime:
            return _cached_bundle

        bundle_path = _resolve_bundle_path(model_root, active)

        if not bundle_path.exists():
            raise ModelNotAvailableError(f"Active bundle does not exist: {bundle_path}")

        bundle = joblib.load(bundle_path)
        bundle["_active_metadata"] = active
        _cached_bundle = bundle
        _cached_mtime = current_mtime
        return bundle


def get_model_status() -> dict[str, Any]:
    try:
        bundle = load_active_model()
    except ModelNotAvailableError as exc:
        return {"active": False, "error": str(exc)}

    active_meta = bundle.get("_active_metadata") or {}
    return {
        "active": True,
        "model_version": active_meta.get("model_version") or bundle.get("model_version"),
        "model_family": active_meta.get("model_family") or bundle.get("model_family"),
        "k_values": bundle.get("k_values") or K_VALUES,
    }


def _predict_with_bundle(payload: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    features_by_k = payload.get("features_by_k")
    if not isinstance(features_by_k, dict):
        raise PayloadContractError("Payload must include a 'features_by_k' object.")

    base_models = bundle.get("base_models")
    meta_model = bundle.get("meta_model")

    if not isinstance(base_models, dict) or meta_model is None:
        raise ModelNotAvailableError("Active model bundle is malformed.")

    component_probs: dict[str, dict[str, float]] = {}
    ordered_home_probs: list[float] = []

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
        component_probs[k_key] = {
            "home_team_prob": home_prob,
            "away_team_prob": 1.0 - home_prob,
        }

    meta_vector = np.array([ordered_home_probs], dtype=float)
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


def predict_from_payload(payload: dict[str, Any], bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    return _predict_with_bundle(payload=payload, bundle=bundle or load_active_model())
