"""Feature engineering helpers for WHL Predictor V2 training."""

from __future__ import annotations

from typing import Iterable

from services.context_features import CONTEXT_FEATURE_COLUMNS

BASE_METRICS: list[str] = [
    "goals_for_avg",
    "goals_against_avg",
    "shots_for_avg",
    "shots_against_avg",
    "power_play_percentage_avg",
    "power_play_percentage_against_avg",
    "faceoff_win_percentage_avg",
    "faceoff_win_percentage_against_avg",
    "goals_diff",
    "ppp_diff",
    "sog_diff",
    "fowp_diff",
]

K_VALUES: list[int] = [5, 10, 15]

FEATURE_COLUMNS: list[str] = [
    *(f"{metric}_diff" for metric in BASE_METRICS),
    *(f"{metric}_sum" for metric in BASE_METRICS),
]

GOALS_BRANCH_FEATURE_COLUMNS: list[str] = [
    *FEATURE_COLUMNS,
    *CONTEXT_FEATURE_COLUMNS,
    "strength_adjusted_goals_diff",
    "strength_adjusted_sog_diff",
]

META_INPUT_COLUMNS_V3: list[str] = [
    *(f"p_k_{k}" for k in K_VALUES),
    "p_rating",
    "p_goals",
]


class FeatureContractError(ValueError):
    """Raised when a required input feature is missing."""


def _coerce_float(value: object, field_name: str) -> float:
    if value is None:
        raise FeatureContractError(f"Missing required metric '{field_name}'.")

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise FeatureContractError(
            f"Invalid numeric value for '{field_name}': {value!r}"
        ) from exc


def build_pairwise_features(
    home_metrics: dict[str, object], away_metrics: dict[str, object]
) -> dict[str, float]:
    """Build model-ready diff/sum features for a single home-away matchup."""
    engineered: dict[str, float] = {}

    for metric in BASE_METRICS:
        home_value = _coerce_float(home_metrics.get(metric), f"home.{metric}")
        away_value = _coerce_float(away_metrics.get(metric), f"away.{metric}")

        engineered[f"{metric}_diff"] = home_value - away_value
        engineered[f"{metric}_sum"] = (home_value + away_value) / 2.0

    return engineered


def feature_vector_from_engineered(engineered: dict[str, float]) -> list[float]:
    """Convert engineered feature map into the model feature order."""
    values: list[float] = []
    for col in FEATURE_COLUMNS:
        if col not in engineered:
            raise FeatureContractError(f"Missing engineered feature '{col}'.")
        values.append(float(engineered[col]))
    return values


def required_metrics() -> Iterable[str]:
    return tuple(BASE_METRICS)
