"""Identity-switch diagnostics for longitudinal ROI identity matrices."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .complete_track_scores import normalize_track_matrix

__all__ = ("identity_switch_events", "score_identity_switches")


def identity_switch_events(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_indices: Sequence[int] | None = None,
) -> list[dict[str, int]]:
    """Return reference-trajectory identity-switch events.

    An identity switch is counted when the same reference neuron is matched to
    one predicted track in an earlier selected session and to a different
    predicted track in a later selected session. Missed detections do not reset
    the diagnostic state; the next matched prediction is compared with the last
    matched prediction for that reference trajectory.
    """

    reference, selected_sessions, prediction_lookup = _identity_switch_inputs(
        predicted_track_matrix,
        reference_track_matrix,
        session_indices=session_indices,
    )
    return _identity_switch_events_from_lookup(reference, selected_sessions, prediction_lookup)


def score_identity_switches(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_indices: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Score identity-switch diagnostics induced by two track matrices."""

    reference, selected_sessions, prediction_lookup = _identity_switch_inputs(
        predicted_track_matrix,
        reference_track_matrix,
        session_indices=session_indices,
    )
    events = _identity_switch_events_from_lookup(reference, selected_sessions, prediction_lookup)
    switched_reference_tracks = {event["reference_track"] for event in events}
    matched_reference_tracks = _reference_tracks_with_predictions(reference, selected_sessions, prediction_lookup)
    reference_tracks = int(reference.shape[0])
    identity_switch_count = len(events)
    return {
        "identity_switches": identity_switch_count,
        "reference_tracks": reference_tracks,
        "reference_tracks_with_predictions": matched_reference_tracks,
        "reference_tracks_with_identity_switches": len(switched_reference_tracks),
        "identity_switches_per_reference_track": _zero_safe_ratio(identity_switch_count, reference_tracks),
        "identity_switches_per_matched_reference_track": _zero_safe_ratio(
            identity_switch_count,
            matched_reference_tracks,
        ),
    }


def _identity_switch_inputs(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_indices: Sequence[int] | None,
) -> tuple[np.ndarray, list[int], list[dict[int, int]]]:
    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    if predicted.shape[1] != reference.shape[1]:
        raise ValueError("Predicted and reference matrices must have the same number of sessions")
    selected_sessions = _selected_sessions(reference, session_indices)
    prediction_lookup = _session_roi_to_track_map(predicted, matrix_name="predicted")
    return reference, selected_sessions, prediction_lookup


def _selected_sessions(matrix: np.ndarray, session_indices: Sequence[int] | None) -> list[int]:
    if session_indices is None:
        return list(range(matrix.shape[1]))
    selected = [int(session_idx) for session_idx in session_indices]
    for session_idx in selected:
        if session_idx < 0 or session_idx >= matrix.shape[1]:
            raise IndexError(f"session index {session_idx} out of bounds for {matrix.shape[1]} sessions")
    return selected


def _identity_switch_events_from_lookup(
    reference: np.ndarray,
    selected_sessions: Sequence[int],
    prediction_lookup: Sequence[dict[int, int]],
) -> list[dict[str, int]]:
    events: list[dict[str, int]] = []
    for reference_track_idx, reference_row in enumerate(reference):
        previous_session: int | None = None
        previous_predicted_track: int | None = None
        previous_roi: int | None = None
        for session_idx in selected_sessions:
            reference_roi = reference_row[session_idx]
            if reference_roi is None:
                continue
            roi = int(reference_roi)
            predicted_track_idx = prediction_lookup[session_idx].get(roi)
            if predicted_track_idx is None:
                continue

            current_session = int(session_idx)
            current_predicted_track = int(predicted_track_idx)
            if previous_predicted_track is not None and previous_predicted_track != current_predicted_track:
                assert previous_session is not None
                assert previous_roi is not None
                events.append(
                    {
                        "reference_track": int(reference_track_idx),
                        "previous_session": previous_session,
                        "session": current_session,
                        "previous_predicted_track": previous_predicted_track,
                        "predicted_track": current_predicted_track,
                        "previous_roi": previous_roi,
                        "roi": roi,
                    }
                )
            previous_session = current_session
            previous_predicted_track = current_predicted_track
            previous_roi = roi
    return events


def _reference_tracks_with_predictions(
    reference: np.ndarray,
    selected_sessions: Sequence[int],
    prediction_lookup: Sequence[dict[int, int]],
) -> int:
    matched = 0
    for reference_row in reference:
        if any(
            reference_row[session_idx] is not None
            and int(reference_row[session_idx]) in prediction_lookup[session_idx]
            for session_idx in selected_sessions
        ):
            matched += 1
    return matched


def _session_roi_to_track_map(matrix: np.ndarray, *, matrix_name: str) -> list[dict[int, int]]:
    lookup: list[dict[int, int]] = [{} for _ in range(matrix.shape[1])]
    for track_idx, row in enumerate(matrix):
        for session_idx, roi in enumerate(row):
            if roi is None:
                continue
            roi_int = int(roi)
            if roi_int in lookup[session_idx]:
                raise ValueError(
                    f"{matrix_name} track matrix contains ROI {roi_int} "
                    f"more than once in session {session_idx}"
                )
            lookup[session_idx][roi_int] = int(track_idx)
    return lookup


def _zero_safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)
