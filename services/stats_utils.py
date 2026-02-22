from __future__ import annotations

import math
from typing import Any

import numpy as np


def binary_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    y = y_true.astype(int)
    p = np.clip(probs.astype(float), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)

    return {
        "accuracy": float(np.mean(pred == y)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
    }


def mcnemar_p_value(
    old_correct: np.ndarray,
    new_correct: np.ndarray,
    *,
    exact_binomial_threshold: int | None = None,
) -> dict[str, Any]:
    b = int(np.sum((old_correct == 1) & (new_correct == 0)))
    c = int(np.sum((old_correct == 0) & (new_correct == 1)))
    n = b + c

    if n == 0:
        return {"b": b, "c": c, "n": 0, "p_value": 1.0, "method": "degenerate"}

    if exact_binomial_threshold is not None and n <= exact_binomial_threshold:
        x = min(b, c)
        log_two = math.log(2.0)
        terms = [
            math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - (n * log_two)
            for i in range(x + 1)
        ]
        max_log = max(terms)
        cdf = math.exp(max_log) * sum(math.exp(t - max_log) for t in terms)
        return {"b": b, "c": c, "n": n, "p_value": min(1.0, 2.0 * cdf), "method": "exact_binomial"}

    chi2_cc = ((abs(b - c) - 1.0) ** 2) / float(n)
    p_value = float(math.erfc(math.sqrt(max(chi2_cc, 0.0) / 2.0)))
    return {"b": b, "c": c, "n": n, "p_value": min(1.0, p_value), "method": "chi_square_cc"}


def bootstrap_accuracy_delta_ci(
    old_correct: np.ndarray,
    new_correct: np.ndarray,
    *,
    samples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    n = len(old_correct)
    if n == 0:
        return {"samples": samples, "seed": seed, "ci95_low": None, "ci95_high": None, "delta_mean": 0.0}

    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=float)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        deltas[i] = float(np.mean(new_correct[idx]) - np.mean(old_correct[idx]))

    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975]).tolist()
    return {
        "samples": samples,
        "seed": seed,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "delta_mean": float(np.mean(deltas)),
    }
