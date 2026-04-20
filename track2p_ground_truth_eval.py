#!/usr/bin/env python3
"""Ground-truth evaluation utilities for Track2p-style longitudinal tracking.

This module is intentionally application-specific. It complements
``track2p_pyrecest_bridge.py`` by handling the benchmark-facing piece that the
bridge currently lacks: loading Track2p ground-truth CSV files and scoring
predicted tracks against them using the metrics described in the Track2p paper.

The core representation is :class:`TrackTable`, a wide matrix with one row per
track and one column per session. Each entry is the ROI index in that session,
or ``-1`` if the track is missing/terminated at that session.

Two input styles are supported when loading CSV files:

* **Wide format**: one row per track, one column per session, with an optional
  leading identifier column such as ``track_id``.
* **Long format**: one row per (track, session) pair, with columns describing
  the track identifier, the session name, and the ROI index.

The metrics implemented are:

* **Complete tracks (CT)**: F1 score on exact full-track reconstruction.
* **Proportion correct by horizon**: fraction of GT tracks whose prefixes are
  reconstructed correctly for the first ``h`` sessions.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

_MISSING_VALUE_STRINGS = {"", "na", "nan", "none", "null", "-"}
_TRACK_ID_HEADERS = {
    "track_id",
    "track",
    "id",
    "gt_id",
    "gt_track",
    "trackid",
}
_LONG_TRACK_HEADERS = _TRACK_ID_HEADERS | {"trajectory", "traj", "trajectory_id"}
_LONG_SESSION_HEADERS = {"session", "session_name", "day", "dataset", "recording"}
_LONG_ROI_HEADERS = {
    "roi",
    "roi_index",
    "roi_idx",
    "cell",
    "cell_id",
    "cell_index",
    "s2p_index",
    "s2p_idx",
    "index",
}


@dataclass(frozen=True)
class TrackTable:
    """Track table with one row per track and one column per session."""

    session_names: tuple[str, ...]
    tracks: np.ndarray

    def __post_init__(self) -> None:
        session_names = tuple(str(name) for name in self.session_names)
        tracks = np.asarray(self.tracks, dtype=int)
        if tracks.ndim != 2:
            raise ValueError("tracks must have shape (n_tracks, n_sessions)")
        if tracks.shape[1] != len(session_names):
            raise ValueError(
                "tracks second dimension must equal the number of session names"
            )
        if len(session_names) == 0:
            raise ValueError("session_names must not be empty")
        object.__setattr__(self, "session_names", session_names)
        object.__setattr__(self, "tracks", tracks)

    @property
    def n_tracks(self) -> int:
        return int(self.tracks.shape[0])

    @property
    def n_sessions(self) -> int:
        return int(self.tracks.shape[1])

    def aligned_to(self, session_names: Sequence[str]) -> "TrackTable":
        """Return a copy with columns reordered to ``session_names``."""
        session_names = tuple(str(name) for name in session_names)
        if set(session_names) != set(self.session_names):
            raise ValueError("session names must match exactly for alignment")
        if session_names == self.session_names:
            return self
        indices = [self.session_names.index(name) for name in session_names]
        return TrackTable(session_names=session_names, tracks=self.tracks[:, indices])

    def prefixes(self, horizon: int) -> np.ndarray:
        """Return the first ``horizon`` session columns."""
        if not 1 <= horizon <= self.n_sessions:
            raise ValueError("horizon must be between 1 and the number of sessions")
        return self.tracks[:, :horizon]

    def to_list(self) -> list[list[int]]:
        return self.tracks.tolist()


@dataclass(frozen=True)
class TrackEvaluation:
    """Summary of Track2p-style benchmark metrics."""

    complete_tracks: float
    proportion_correct_by_horizon: dict[int, float]
    n_ground_truth_tracks: int
    n_predicted_tracks: int
    n_exact_full_track_matches: int

    def to_json_dict(self) -> dict[str, object]:
        return {
            "complete_tracks": float(self.complete_tracks),
            "proportion_correct_by_horizon": {
                str(horizon): float(value)
                for horizon, value in self.proportion_correct_by_horizon.items()
            },
            "n_ground_truth_tracks": int(self.n_ground_truth_tracks),
            "n_predicted_tracks": int(self.n_predicted_tracks),
            "n_exact_full_track_matches": int(self.n_exact_full_track_matches),
        }


def _normalize_header(header: str) -> str:
    return header.strip().lower().replace(" ", "_")


def _parse_roi_value(value: str | int | float | None) -> int:
    if value is None:
        return -1
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return -1
        if float(value).is_integer():
            return int(value)
        raise ValueError(f"ROI index must be integer-like, got {value!r}")

    text = str(value).strip()
    if _normalize_header(text) in _MISSING_VALUE_STRINGS:
        return -1
    number = float(text)
    if np.isnan(number):
        return -1
    if not float(number).is_integer():
        raise ValueError(f"ROI index must be integer-like, got {value!r}")
    return int(number)


def _rows_from_csv(csv_path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {csv_path} has no header row")
        fieldnames = [str(name) for name in reader.fieldnames]
        rows = [{str(k): v for k, v in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"CSV file {csv_path} contains no data rows")
    return fieldnames, rows


def _looks_like_long_format(headers: Sequence[str]) -> bool:
    normalized = {_normalize_header(header) for header in headers}
    return (
        bool(normalized & _LONG_TRACK_HEADERS)
        and bool(normalized & _LONG_SESSION_HEADERS)
        and bool(normalized & _LONG_ROI_HEADERS)
    )


def _find_matching_header(
    headers: Sequence[str], candidates: set[str]
) -> str | None:
    for header in headers:
        if _normalize_header(header) in candidates:
            return header
    return None


def _load_long_format(
    headers: Sequence[str], rows: Sequence[Mapping[str, str]], session_names: Sequence[str] | None
) -> TrackTable:
    track_header = _find_matching_header(headers, _LONG_TRACK_HEADERS)
    session_header = _find_matching_header(headers, _LONG_SESSION_HEADERS)
    roi_header = _find_matching_header(headers, _LONG_ROI_HEADERS)
    if track_header is None or session_header is None or roi_header is None:
        raise ValueError("could not infer track/session/roi columns from long CSV")

    if session_names is None:
        ordered_sessions: list[str] = []
        seen_sessions: set[str] = set()
        for row in rows:
            session_name = str(row[session_header]).strip()
            if session_name not in seen_sessions:
                seen_sessions.add(session_name)
                ordered_sessions.append(session_name)
        session_names = ordered_sessions
    else:
        session_names = [str(name) for name in session_names]

    session_to_index = {name: idx for idx, name in enumerate(session_names)}
    grouped: dict[str, np.ndarray] = {}
    for row in rows:
        track_id = str(row[track_header]).strip()
        session_name = str(row[session_header]).strip()
        if session_name not in session_to_index:
            continue
        roi_index = _parse_roi_value(row[roi_header])
        if track_id not in grouped:
            grouped[track_id] = np.full((len(session_names),), -1, dtype=int)
        grouped[track_id][session_to_index[session_name]] = roi_index

    ordered_track_ids = sorted(grouped)
    tracks = np.vstack([grouped[track_id] for track_id in ordered_track_ids])
    return TrackTable(session_names=tuple(session_names), tracks=tracks)


def _load_wide_format(
    headers: Sequence[str], rows: Sequence[Mapping[str, str]], session_names: Sequence[str] | None
) -> TrackTable:
    if session_names is None:
        candidate_headers = []
        for header in headers:
            normalized = _normalize_header(header)
            if normalized in _TRACK_ID_HEADERS:
                continue
            candidate_headers.append(header)
        if not candidate_headers:
            raise ValueError("could not infer any session columns from wide CSV")
        session_names = [str(name) for name in candidate_headers]
    else:
        session_names = [str(name) for name in session_names]

    tracks = np.full((len(rows), len(session_names)), -1, dtype=int)
    for row_index, row in enumerate(rows):
        for session_index, session_name in enumerate(session_names):
            if session_name not in row:
                raise ValueError(
                    f"session column {session_name!r} not present in CSV row"
                )
            tracks[row_index, session_index] = _parse_roi_value(row[session_name])
    return TrackTable(session_names=tuple(session_names), tracks=tracks)


def load_track_table_csv(
    csv_path: str | Path, *, session_names: Sequence[str] | None = None
) -> TrackTable:
    """Load a track table from a wide or long CSV file."""
    headers, rows = _rows_from_csv(csv_path)
    if _looks_like_long_format(headers):
        return _load_long_format(headers, rows, session_names)
    return _load_wide_format(headers, rows, session_names)


def load_track2p_ground_truth_csv(
    csv_path: str | Path, *, session_names: Sequence[str] | None = None
) -> TrackTable:
    """Alias specialized for Track2p's ``ground_truth.csv`` files."""
    return load_track_table_csv(csv_path, session_names=session_names)


def tracks_from_consecutive_matches(
    session_names: Sequence[str],
    matches: Sequence[
        Mapping[int, int]
        | Sequence[tuple[int, int]]
        | np.ndarray
        | tuple[Sequence[int], Sequence[int]]
    ],
    *,
    start_roi_indices: Sequence[int] | None = None,
) -> TrackTable:
    """Reconstruct wide tracks from consecutive pairwise assignments.

    Parameters
    ----------
    session_names
        Ordered session names, one per session.
    matches
        Consecutive session-to-session matches. Each element can be one of:

        * ``dict[int, int]`` mapping ROI indices in session ``k`` to session ``k+1``.
        * an iterable of ``(ref_idx, meas_idx)`` pairs.
        * an integer array with shape ``(n_matches, 2)``.
        * a tuple ``(ref_indices, meas_indices)`` of equal-length sequences.

    start_roi_indices
        Optional ROI indices in the first session from which tracks should be
        reconstructed. If omitted, the keys of the first match set are used.
    """
    session_names = tuple(str(name) for name in session_names)
    if len(session_names) < 2:
        raise ValueError("at least two session names are required")
    if len(matches) != len(session_names) - 1:
        raise ValueError("matches must have length len(session_names) - 1")

    normalized_matches = [_normalize_match_mapping(match) for match in matches]

    if start_roi_indices is None:
        first_keys = sorted(normalized_matches[0])
        start_roi_indices = first_keys
    else:
        start_roi_indices = [int(index) for index in start_roi_indices]

    tracks = np.full((len(start_roi_indices), len(session_names)), -1, dtype=int)
    for row_index, start_roi in enumerate(start_roi_indices):
        tracks[row_index, 0] = int(start_roi)
        current_roi = int(start_roi)
        for match_index, mapping in enumerate(normalized_matches):
            next_roi = mapping.get(current_roi, -1)
            tracks[row_index, match_index + 1] = int(next_roi)
            if next_roi < 0:
                break
            current_roi = int(next_roi)
    return TrackTable(session_names=session_names, tracks=tracks)


def _normalize_match_mapping(
    match: Mapping[int, int]
    | Sequence[tuple[int, int]]
    | np.ndarray
    | tuple[Sequence[int], Sequence[int]],
) -> dict[int, int]:
    if isinstance(match, Mapping):
        return {int(k): int(v) for k, v in match.items()}

    if isinstance(match, tuple) and len(match) == 2:
        left, right = match
        left = [int(value) for value in left]
        right = [int(value) for value in right]
        if len(left) != len(right):
            raise ValueError("match tuple arrays must have equal length")
        return dict(zip(left, right, strict=True))

    array_match = np.asarray(match)
    if array_match.ndim == 2 and array_match.shape[1] == 2:
        return {int(ref): int(meas) for ref, meas in array_match.tolist()}

    try:
        return {int(ref): int(meas) for ref, meas in match}  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover - defensive branch
        raise TypeError("unsupported match representation") from exc


def _align_prediction_to_ground_truth(
    ground_truth: TrackTable, prediction: TrackTable
) -> TrackTable:
    if set(ground_truth.session_names) != set(prediction.session_names):
        raise ValueError(
            "ground truth and prediction must refer to the same set of sessions"
        )
    return prediction.aligned_to(ground_truth.session_names)


def complete_tracks_score(ground_truth: TrackTable, prediction: TrackTable) -> float:
    """Return the Track2p 'complete tracks' (CT) score.

    This is the F1 score where positives are *exact* full-track reconstructions.
    Predicted tracks that terminate early or contain any incorrect identity are
    counted as false positives, exactly because they do not match any full GT
    track.
    """
    prediction = _align_prediction_to_ground_truth(ground_truth, prediction)
    gt_set = {tuple(track) for track in ground_truth.to_list()}
    pred_set = {tuple(track) for track in prediction.to_list()}
    true_positives = len(gt_set & pred_set)
    false_positives = len(pred_set - gt_set)
    false_negatives = len(gt_set - pred_set)
    denominator = 2 * true_positives + false_positives + false_negatives
    if denominator == 0:
        return 0.0
    return (2.0 * true_positives) / denominator


def proportion_correct_by_horizon(
    ground_truth: TrackTable, prediction: TrackTable
) -> dict[int, float]:
    """Return the fraction of GT tracks reconstructed correctly up to each horizon."""
    prediction = _align_prediction_to_ground_truth(ground_truth, prediction)
    result: dict[int, float] = {}
    denominator = ground_truth.n_tracks
    if denominator == 0:
        return {horizon: 0.0 for horizon in range(2, ground_truth.n_sessions + 1)}

    for horizon in range(2, ground_truth.n_sessions + 1):
        gt_prefixes = {tuple(track) for track in ground_truth.prefixes(horizon).tolist()}
        pred_prefixes = {
            tuple(track[:horizon])
            for track in prediction.to_list()
            if all(value >= 0 for value in track[:horizon])
        }
        correctly_reconstructed = len(gt_prefixes & pred_prefixes)
        result[horizon] = correctly_reconstructed / denominator
    return result


def evaluate_track_table_prediction(
    ground_truth: TrackTable, prediction: TrackTable
) -> TrackEvaluation:
    """Compute the Track2p benchmark metrics for one prediction."""
    prediction = _align_prediction_to_ground_truth(ground_truth, prediction)
    gt_set = {tuple(track) for track in ground_truth.to_list()}
    pred_set = {tuple(track) for track in prediction.to_list()}
    exact_matches = len(gt_set & pred_set)
    return TrackEvaluation(
        complete_tracks=complete_tracks_score(ground_truth, prediction),
        proportion_correct_by_horizon=proportion_correct_by_horizon(
            ground_truth, prediction
        ),
        n_ground_truth_tracks=ground_truth.n_tracks,
        n_predicted_tracks=prediction.n_tracks,
        n_exact_full_track_matches=exact_matches,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted Track2p-style tracks against ground truth."
    )
    parser.add_argument("ground_truth_csv", help="Path to Track2p ground_truth.csv")
    parser.add_argument("prediction_csv", help="Path to predicted tracks CSV")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    ground_truth = load_track2p_ground_truth_csv(args.ground_truth_csv)
    prediction = load_track_table_csv(
        args.prediction_csv, session_names=ground_truth.session_names
    )
    evaluation = evaluate_track_table_prediction(ground_truth, prediction)
    print(json.dumps(evaluation.to_json_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
