"""Track-level scoring helpers for longitudinal ROI identity matrices."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

_MISSING_STRINGS = {"", "none", "nan", "null"}


def normalize_track_matrix(track_matrix: Any) -> np.ndarray:
    """Return an object matrix containing integer ROI indices or ``None``."""

    matrix = np.asarray(track_matrix, dtype=object)
    if matrix.ndim != 2:
        raise ValueError("track_matrix must have shape (n_tracks, n_sessions)")

    normalized = np.empty(matrix.shape, dtype=object)
    for index, value in np.ndenumerate(matrix):
        normalized[index] = _parse_optional_int(value)
    return normalized


def track_lengths(track_matrix: Any) -> np.ndarray:
    """Return the number of present sessions for each track row."""

    matrix = normalize_track_matrix(track_matrix)
    if matrix.shape[0] == 0:
        return np.zeros((0,), dtype=int)
    present = np.vectorize(lambda value: value is not None, otypes=[bool])(matrix)
    return np.sum(present, axis=1, dtype=int)


def complete_track_set(
    track_matrix: Any,
    *,
    session_indices: Sequence[int] | None = None,
) -> set[tuple[int, ...]]:
    """Return exact full-track tuples for tracks present in every selected session."""

    matrix = normalize_track_matrix(track_matrix)
    selected_sessions = _selected_sessions(matrix, session_indices)
    complete_tracks: set[tuple[int, ...]] = set()
    for row in matrix:
        values = [row[session_idx] for session_idx in selected_sessions]
        if all(value is not None for value in values):
            complete_tracks.add(tuple(int(value) for value in values))
    return complete_tracks


def pairwise_track_set(
    track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
) -> set[tuple[int, int, int, int]]:
    """Return pairwise identity links encoded as ``(session_a, session_b, roi_a, roi_b)``."""

    matrix = normalize_track_matrix(track_matrix)
    pairs = list(_adjacent_session_pairs(matrix) if session_pairs is None else session_pairs)
    normalized_pairs: set[tuple[int, int, int, int]] = set()
    for session_a, session_b in pairs:
        _validate_session_index(matrix, session_a)
        _validate_session_index(matrix, session_b)
        if session_a >= session_b:
            raise ValueError("session_pairs must point forward in time")
        for row in matrix:
            roi_a = row[session_a]
            roi_b = row[session_b]
            if roi_a is None or roi_b is None:
                continue
            normalized_pairs.add((int(session_a), int(session_b), int(roi_a), int(roi_b)))
    return normalized_pairs


def score_complete_tracks(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_indices: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Score exact complete-track recovery with precision, recall, and F1."""

    predicted = complete_track_set(predicted_track_matrix, session_indices=session_indices)
    reference = complete_track_set(reference_track_matrix, session_indices=session_indices)
    true_positives = len(predicted & reference)
    false_positives = len(predicted - reference)
    false_negatives = len(reference - predicted)
    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    return {
        "complete_track_true_positives": true_positives,
        "complete_track_false_positives": false_positives,
        "complete_track_false_negatives": false_negatives,
        "complete_track_precision": precision,
        "complete_track_recall": recall,
        "complete_track_f1": f1,
        "complete_tracks": len(predicted),
        "reference_complete_tracks": len(reference),
    }


def score_pairwise_tracks(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
) -> dict[str, float | int]:
    """Score pairwise links induced by track matrices."""

    predicted = pairwise_track_set(predicted_track_matrix, session_pairs=session_pairs)
    reference = pairwise_track_set(reference_track_matrix, session_pairs=session_pairs)
    true_positives = len(predicted & reference)
    false_positives = len(predicted - reference)
    false_negatives = len(reference - predicted)
    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    return {
        "pairwise_true_positives": true_positives,
        "pairwise_false_positives": false_positives,
        "pairwise_false_negatives": false_negatives,
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": f1,
        "pairwise_links": len(predicted),
        "reference_pairwise_links": len(reference),
    }


def summarize_tracks(track_matrix: Any) -> dict[str, float | int]:
    """Summarize the number and length of predicted tracks."""

    matrix = normalize_track_matrix(track_matrix)
    lengths = track_lengths(matrix)
    n_tracks = int(matrix.shape[0])
    mean_track_length = float(np.mean(lengths)) if lengths.size else 0.0
    return {
        "tracks": n_tracks,
        "mean_track_length": mean_track_length,
        "max_track_length": int(np.max(lengths)) if lengths.size else 0,
    }


def score_track_matrices(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
    complete_session_indices: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Return pairwise, complete-track, and length metrics for two track matrices."""

    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    if predicted.shape[1] != reference.shape[1]:
        raise ValueError("Predicted and reference matrices must have the same number of sessions")

    scores: dict[str, float | int] = {}
    scores.update(score_pairwise_tracks(predicted, reference, session_pairs=session_pairs))
    scores.update(score_complete_tracks(predicted, reference, session_indices=complete_session_indices))
    scores.update(summarize_tracks(predicted))
    return scores


def _parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in _MISSING_STRINGS:
            return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    try:
        integer_value = int(value)
    except (TypeError, ValueError):
        return None
    if integer_value < 0:
        return None
    return integer_value


def _selected_sessions(matrix: np.ndarray, session_indices: Sequence[int] | None) -> list[int]:
    if session_indices is None:
        return list(range(matrix.shape[1]))
    selected = [int(session_idx) for session_idx in session_indices]
    for session_idx in selected:
        _validate_session_index(matrix, session_idx)
    return selected


def _adjacent_session_pairs(matrix: np.ndarray) -> list[tuple[int, int]]:
    return [(session_idx, session_idx + 1) for session_idx in range(max(0, matrix.shape[1] - 1))]


def _validate_session_index(matrix: np.ndarray, session_idx: int) -> None:
    if session_idx < 0 or session_idx >= matrix.shape[1]:
        raise IndexError(f"session index {session_idx} out of bounds for {matrix.shape[1]} sessions")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0
    return float(numerator) / float(denominator)
