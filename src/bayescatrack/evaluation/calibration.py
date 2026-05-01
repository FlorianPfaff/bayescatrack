"""Calibration diagnostics for probabilistic pairwise association models."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = (
    "brier_score",
    "calibration_bin_table",
    "expected_calibration_error",
    "maximum_calibration_error",
    "score_binary_calibration",
)


def brier_score(probabilities: Any, labels: Any) -> float:
    """Return the binary Brier score."""

    probabilities_array, labels_array = _validate_inputs(probabilities, labels)
    return float(np.mean((probabilities_array - labels_array) ** 2))


def expected_calibration_error(probabilities: Any, labels: Any, *, n_bins: int = 10) -> float:
    """Return equal-width-bin expected calibration error."""

    bins = calibration_bin_table(probabilities, labels, n_bins=n_bins)
    total = sum(int(row["count"]) for row in bins)
    return float(sum(int(row["count"]) / total * float(row["absolute_error"]) for row in bins))


def maximum_calibration_error(probabilities: Any, labels: Any, *, n_bins: int = 10) -> float:
    """Return maximum equal-width-bin calibration error."""

    bins = calibration_bin_table(probabilities, labels, n_bins=n_bins)
    return float(max((float(row["absolute_error"]) for row in bins if int(row["count"]) > 0), default=0.0))


def calibration_bin_table(probabilities: Any, labels: Any, *, n_bins: int = 10) -> list[dict[str, float | int]]:
    """Return reliability-bin statistics."""

    probabilities_array, labels_array = _validate_inputs(probabilities, labels)
    n_bins = _validate_n_bins(n_bins)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.searchsorted(edges, probabilities_array, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    rows: list[dict[str, float | int]] = []
    for bin_index in range(n_bins):
        in_bin = bin_indices == bin_index
        count = int(np.sum(in_bin))
        if count:
            confidence = float(np.mean(probabilities_array[in_bin]))
            accuracy = float(np.mean(labels_array[in_bin]))
        else:
            confidence = 0.0
            accuracy = 0.0
        rows.append(
            {
                "bin_index": bin_index,
                "bin_lower": float(edges[bin_index]),
                "bin_upper": float(edges[bin_index + 1]),
                "count": count,
                "mean_probability": confidence,
                "positive_rate": accuracy,
                "absolute_error": abs(confidence - accuracy),
            }
        )
    return rows


def score_binary_calibration(probabilities: Any, labels: Any, *, n_bins: int = 10, prefix: str = "calibration") -> dict[str, float | int]:
    """Return scalar calibration diagnostics with benchmark-friendly names."""

    if not prefix:
        raise ValueError("prefix must not be empty")
    probabilities_array, labels_array = _validate_inputs(probabilities, labels)
    n_bins = _validate_n_bins(n_bins)
    positives = int(np.sum(labels_array))
    total = int(labels_array.shape[0])
    return {
        f"{prefix}_examples": total,
        f"{prefix}_positive_examples": positives,
        f"{prefix}_negative_examples": int(total - positives),
        f"{prefix}_brier_score": brier_score(probabilities_array, labels_array),
        f"{prefix}_ece": expected_calibration_error(probabilities_array, labels_array, n_bins=n_bins),
        f"{prefix}_mce": maximum_calibration_error(probabilities_array, labels_array, n_bins=n_bins),
        f"{prefix}_n_bins": n_bins,
    }


def _validate_inputs(probabilities: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    probabilities_array = np.asarray(probabilities, dtype=float).reshape(-1)
    labels_array = np.asarray(labels).reshape(-1)
    if probabilities_array.shape[0] != labels_array.shape[0]:
        raise ValueError("probabilities and labels must contain the same number of elements")
    if probabilities_array.shape[0] == 0:
        raise ValueError("At least one calibration example is required")
    if not np.all(np.isfinite(probabilities_array)):
        raise ValueError("probabilities must be finite")
    if np.any((probabilities_array < 0.0) | (probabilities_array > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.all(np.isin(labels_array, [0, 1, False, True])):
        raise ValueError("labels must be binary values")
    return probabilities_array, labels_array.astype(float)


def _validate_n_bins(n_bins: int) -> int:
    parsed = int(n_bins)
    if parsed <= 0:
        raise ValueError("n_bins must be a positive integer")
    return parsed
