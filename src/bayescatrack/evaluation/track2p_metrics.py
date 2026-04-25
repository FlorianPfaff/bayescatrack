"""Track2p benchmark metric helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from bayescatrack.reference import Track2pReference

from .complete_track_scores import (
    complete_track_set,
    normalize_track_matrix,
    pairwise_track_set,
    score_complete_tracks,
    score_pairwise_tracks,
    score_track_matrices,
    summarize_tracks,
    track_lengths,
)


def score_track_matrix_against_reference(
    predicted_track_matrix: Any,
    reference: Track2pReference,
    *,
    curated_only: bool = False,
) -> dict[str, float | int]:
    """Score a predicted Suite2p-index track matrix against a Track2p reference."""

    reference_matrix = normalize_track_matrix(reference.suite2p_indices)
    if curated_only:
        if reference.curated_mask is None:
            raise ValueError("curated_only=True requires a Track2p reference with a curated_mask")
        reference_matrix = reference_matrix[np.asarray(reference.curated_mask, dtype=bool)]
    return score_track_matrices(predicted_track_matrix, reference_matrix)


__all__ = [
    "complete_track_set",
    "normalize_track_matrix",
    "pairwise_track_set",
    "score_complete_tracks",
    "score_pairwise_tracks",
    "score_track_matrices",
    "score_track_matrix_against_reference",
    "summarize_tracks",
    "track_lengths",
]
