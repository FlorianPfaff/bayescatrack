"""Track2p-style global-assignment ablation runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bayescatrack.association.pyrecest_global_assignment import (
    AssociationCost,
    GlobalAssignmentRun,
    solve_global_assignment_for_sessions,
    tracks_to_suite2p_index_matrix,
)
from bayescatrack.core.bridge import load_track2p_subject
from bayescatrack.evaluation.track2p_metrics import score_track_matrix_against_reference
from bayescatrack.reference import Track2pReference, load_track2p_reference


@dataclass(frozen=True)
class Track2pGlobalAssignmentAblation:
    """Subject-level result for the global-assignment stitching ablation."""

    subject_dir: Path
    variant: str
    assignment: GlobalAssignmentRun
    predicted_track_matrix: np.ndarray
    scores: dict[str, float | int]
    reference_source: str


# pylint: disable=too-many-arguments
def run_track2p_global_assignment_ablation(
    subject_dir: str | Path,
    *,
    plane_name: str = "plane0",
    input_format: str = "auto",
    reference: Track2pReference | str | Path | None = None,
    curated_only: bool = False,
    cost: AssociationCost = "registered-iou",
    max_gap: int = 2,
    transform_type: str = "affine",
    start_cost: float = 5.0,
    end_cost: float = 5.0,
    gap_penalty: float = 1.0,
    cost_threshold: float | None = 6.0,
    include_behavior: bool = True,
    order: str = "xy",
    weighted_centroids: bool = False,
    velocity_variance: float = 25.0,
    regularization: float = 1.0e-6,
    pairwise_cost_kwargs: dict[str, Any] | None = None,
    **suite2p_kwargs: Any,
) -> Track2pGlobalAssignmentAblation:
    """Run Track2p registration plus PyRecEst global assignment for one subject."""

    subject_dir = Path(subject_dir)
    sessions = load_track2p_subject(
        subject_dir,
        plane_name=plane_name,
        input_format=input_format,
        include_behavior=include_behavior,
        **suite2p_kwargs,
    )
    assignment = solve_global_assignment_for_sessions(
        sessions,
        max_gap=max_gap,
        cost=cost,
        transform_type=transform_type,
        start_cost=start_cost,
        end_cost=end_cost,
        gap_penalty=gap_penalty,
        cost_threshold=cost_threshold,
        order=order,
        weighted_centroids=weighted_centroids,
        velocity_variance=velocity_variance,
        regularization=regularization,
        pairwise_cost_kwargs=pairwise_cost_kwargs,
    )
    predicted_track_matrix = tracks_to_suite2p_index_matrix(assignment.result.tracks, sessions)
    track2p_reference = _load_reference(subject_dir, reference=reference, plane_name=plane_name)
    scores = score_track_matrix_against_reference(
        predicted_track_matrix,
        track2p_reference,
        curated_only=curated_only,
    )
    variant = "Same costs + global assignment" if cost == "registered-iou" else "BayesCaTrack costs + global assignment"
    return Track2pGlobalAssignmentAblation(
        subject_dir=subject_dir,
        variant=variant,
        assignment=assignment,
        predicted_track_matrix=predicted_track_matrix,
        scores=scores,
        reference_source=track2p_reference.source,
    )


def _load_reference(subject_dir: Path, *, reference: Track2pReference | str | Path | None, plane_name: str) -> Track2pReference:
    if isinstance(reference, Track2pReference):
        return reference
    if reference is not None:
        return load_track2p_reference(reference, plane_name=plane_name)
    return load_track2p_reference(subject_dir / "track2p", plane_name=plane_name)
