"""Model loading and inference helpers for WHL Predictor V2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np

from whl_v2_features import K_VALUES, build_pairwise_features, feature_vector_from_engineered


class ModelNotAvailableError(RuntimeError):
    """Raised when no active model artifact exists."""


class PayloadContractError(ValueError):
    """Raised when v2 inference payload is invalid."""


DEFAULT_MODEL_ROOT = Path(__file__).resolve().parent / "model_store" / "whl_v2"


def _resolve_model_root(model_root: str | Path) -> Path:
    return Path(model_root).resolve()


def load_active_model(model_root: str | Path = DEFAULT_MODEL_ROOT) -> Dict[str, object]:
    root = _resolve_model_root(model_root)
    active_path = root / "active_model.json"

    if not active_path.exists():
        raise ModelNotAvailableError(
            f"No active model pointer found at {active_path}. Run train_whl_v2.py first."
        )

    active = json.loads(active_path.read_text(encoding="utf-8"))
    bundle_path = Path(active["bundle_path"])  # absolute in metadata

    if not bundle_path.exists():
        raise ModelNotAvailableError(
            f"Active bundle does not exist: {bundle_path}."
        )

    bundle = joblib.load(bundle_path)
    bundle["_active_metadata"] = active
    return bundle


def _normalize_features_by_k(payload: Dict[str, object]) -> Dict[str, Dict[str, Dict[str, object]]]:
    features_by_k = payload.get("features_by_k")
    if not isinstance(features_by_k, dict):
        raise PayloadContractError("Payload must include a 'features_by_k' object.")

    normalized: Dict[str, Dict[str, Dict[str, object]]] = {}

    for k in K_VALUES:
        key = str(k)
        raw = features_by_k.get(key)
        if not isinstance(raw, dict):
            raise PayloadContractError(f"Missing features for k={key}.")

        home = raw.get("home")
        away = raw.get("away")
        if not isinstance(home, dict) or not isinstance(away, dict):
            raise PayloadContractError(
                f"features_by_k['{key}'] must include 'home' and 'away' objects."
            )

        normalized[key] = {
            "home": home,
            "away": away,
        }

    return normalized


def _compute_component_probabilities(
    features_by_k: Dict[str, Dict[str, Dict[str, object]]],
    bundle: Dict[str, object],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    base_models = bundle["base_models"]
    feature_columns = list(bundle["feature_columns"])

    base_probs: Dict[str, float] = {}
    components: Dict[str, Dict[str, float]] = {}

    for key in [str(k) for k in K_VALUES]:
        engineered = build_pairwise_features(
            home_metrics=features_by_k[key]["home"],
            away_metrics=features_by_k[key]["away"],
        )

        vector = feature_vector_from_engineered(engineered)
        if len(vector) != len(feature_columns):
            raise PayloadContractError(
                f"Feature vector for k={key} has length {len(vector)}; expected {len(feature_columns)}."
            )

        model = base_models[key]
        home_prob = float(model.predict_proba(np.array([vector], dtype=float))[:, 1][0])

        base_probs[key] = home_prob
        components[key] = {
            "home_team_prob": home_prob,
            "away_team_prob": 1.0 - home_prob,
        }

    return base_probs, components


def predict_from_payload(
    payload: Dict[str, object],
    bundle: Dict[str, object],
) -> Dict[str, object]:
    features_by_k = _normalize_features_by_k(payload)

    base_probs, components = _compute_component_probabilities(features_by_k, bundle)
    meta_vector = np.array([[base_probs[str(k)] for k in K_VALUES]], dtype=float)

    meta_model = bundle["meta_model"]
    home_team_prob = float(meta_model.predict_proba(meta_vector)[:, 1][0])
    away_team_prob = 1.0 - home_team_prob

    home_team_id = payload.get("home_team_id")
    away_team_id = payload.get("away_team_id")
    predicted_winner_id = home_team_id if home_team_prob >= away_team_prob else away_team_id

    return {
        "home_team_prob": home_team_prob,
        "away_team_prob": away_team_prob,
        "predicted_winner_id": predicted_winner_id,
        "model_version": bundle.get("model_version"),
        "model_family": bundle.get("model_family"),
        "k_components": components,
    }
