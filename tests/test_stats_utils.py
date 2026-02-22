from __future__ import annotations

import numpy as np

from services.stats_utils import (
    binary_metrics,
    bootstrap_accuracy_delta_ci,
    mcnemar_p_value,
)


def test_binary_metrics_expected_values():
    y_true = np.array([1, 0, 1, 0], dtype=int)
    probs = np.array([0.9, 0.1, 0.6, 0.4], dtype=float)

    out = binary_metrics(y_true, probs)

    assert out["accuracy"] == 1.0
    assert out["log_loss"] > 0.0
    assert out["brier"] > 0.0


def test_mcnemar_supports_exact_and_chi_square_modes():
    old_correct = np.array([1, 1, 0, 0], dtype=int)
    new_correct = np.array([1, 0, 0, 1], dtype=int)

    exact = mcnemar_p_value(old_correct, new_correct, exact_binomial_threshold=400)
    chi_square = mcnemar_p_value(old_correct, new_correct)

    assert exact["method"] == "exact_binomial"
    assert chi_square["method"] == "chi_square_cc"
    assert exact["n"] == chi_square["n"] == 2


def test_bootstrap_accuracy_delta_ci_is_deterministic_for_seed():
    old_correct = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    new_correct = np.array([1, 1, 0, 1, 1, 1], dtype=int)

    out_a = bootstrap_accuracy_delta_ci(old_correct, new_correct, samples=200, seed=123)
    out_b = bootstrap_accuracy_delta_ci(old_correct, new_correct, samples=200, seed=123)

    assert out_a == out_b
    assert out_a["ci95_low"] is not None
    assert out_a["ci95_high"] is not None
