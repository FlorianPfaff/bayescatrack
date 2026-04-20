#!/usr/bin/env python3
"""Standalone Track2p/Suite2p bridge for PyRecEst.

This script is intentionally external to the PyRecEst package. It focuses on the
calcium-imaging formats used by Track2p and turns them into PyRecEst-friendly state
representations without adding neuroscience-specific code to PyRecEst itself.

Supported inputs
----------------
* Suite2p folders (``suite2p/planeX``)
* Track2p raw NPY folders (``data_npy/planeX``)

Typical usage
-------------
Inspect a subject directory::

    python track2p_pyrecest_bridge.py summary /path/to/jm039 --plane plane0

Export per-session measurements and state moments::

    python track2p_pyrecest_bridge.py export /path/to/jm039 /tmp/jm039_plane0.npz

Use from Python::

    from track2p_pyrecest_bridge import load_track2p_subject

    sessions = load_track2p_subject("/path/to/jm039", plane_name="plane0")
    filters = sessions[0].plane_data.to_pyrecest_kalman_filters()

The exported states follow the constant-velocity layout ``[pos_1, vel_1, pos_2, vel_2]``
with coordinate order controlled by ``order``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import re
from typing import Any

import numpy as np

_SESSION_NAME_PATTERN = re.compile(r"^(?P<session_date>\d{4}-\d{2}-\d{2})(?:_.+)?$")


@dataclass(frozen=True)
class CalciumPlaneData:
    """ROI-level representation of one imaging plane from one session."""

    roi_masks: np.ndarray
    traces: np.ndarray | None = None
    fov: np.ndarray | None = None
    spike_traces: np.ndarray | None = None
    neuropil_traces: np.ndarray | None = None
    cell_probabilities: np.ndarray | None = None
    roi_indices: np.ndarray | None = None
    roi_features: dict[str, np.ndarray] = field(default_factory=dict)
    source: str = "unknown"
    plane_name: str | None = None
    ops: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        roi_masks = np.asarray(self.roi_masks)
        if roi_masks.ndim != 3:
            raise ValueError("roi_masks must have shape (n_roi, height, width)")
        object.__setattr__(self, "roi_masks", roi_masks)

        n_rois = roi_masks.shape[0]

        for field_name in ("traces", "spike_traces", "neuropil_traces"):
            value = getattr(self, field_name)
            if value is None:
                continue
            value = np.asarray(value)
            if value.ndim != 2:
                raise ValueError(f"{field_name} must have shape (n_roi, n_timepoints)")
            if value.shape[0] != n_rois:
                raise ValueError(
                    f"{field_name} must have first dimension equal to the number of ROIs"
                )
            object.__setattr__(self, field_name, value)

        if self.fov is not None:
            fov = np.asarray(self.fov)
            if fov.ndim != 2:
                raise ValueError("fov must have shape (height, width)")
            if fov.shape != roi_masks.shape[1:]:
                raise ValueError("fov spatial shape must match the mask spatial shape")
            object.__setattr__(self, "fov", fov)

        if self.cell_probabilities is not None:
            probabilities = np.asarray(self.cell_probabilities, dtype=float)
            if probabilities.shape != (n_rois,):
                raise ValueError("cell_probabilities must have shape (n_roi,)")
            object.__setattr__(self, "cell_probabilities", probabilities)

        if self.roi_indices is not None:
            roi_indices = np.asarray(self.roi_indices, dtype=int)
            if roi_indices.shape != (n_rois,):
                raise ValueError("roi_indices must have shape (n_roi,)")
            object.__setattr__(self, "roi_indices", roi_indices)

        sanitized_features: dict[str, np.ndarray] = {}
        for key, value in self.roi_features.items():
            array_value = np.asarray(value)
            if array_value.ndim == 0:
                raise ValueError(f"ROI feature '{key}' must be at least one-dimensional")
            if array_value.shape[0] != n_rois:
                raise ValueError(
                    f"ROI feature '{key}' must have first dimension equal to n_roi"
                )
            sanitized_features[key] = array_value
        object.__setattr__(self, "roi_features", sanitized_features)

    @property
    def n_rois(self) -> int:
        return int(self.roi_masks.shape[0])

    @property
    def image_shape(self) -> tuple[int, int]:
        return int(self.roi_masks.shape[1]), int(self.roi_masks.shape[2])

    def centroids(self, order: str = "xy", weighted: bool = False) -> np.ndarray:
        """Return ROI centroids as a ``(2, n_roi)`` measurement matrix."""

        order = _validate_coordinate_order(order)
        if self.n_rois == 0:
            return np.zeros((2, 0), dtype=float)

        coords = np.zeros((2, self.n_rois), dtype=float)
        for roi_index, mask in enumerate(self.roi_masks):
            row_coords, col_coords = np.nonzero(mask)
            if row_coords.size == 0:
                raise ValueError(f"ROI {roi_index} has an empty mask")

            if weighted:
                weights = np.asarray(mask[row_coords, col_coords], dtype=float)
            else:
                weights = np.ones(row_coords.shape[0], dtype=float)

            weight_sum = float(np.sum(weights))
            if weight_sum <= 0.0:
                raise ValueError(f"ROI {roi_index} has non-positive total mask weight")

            centroid_y = float(np.dot(row_coords, weights) / weight_sum)
            centroid_x = float(np.dot(col_coords, weights) / weight_sum)

            if order == "xy":
                coords[:, roi_index] = np.array([centroid_x, centroid_y])
            else:
                coords[:, roi_index] = np.array([centroid_y, centroid_x])

        return coords

    def position_covariances(
        self,
        order: str = "xy",
        weighted: bool = False,
        regularization: float = 1e-6,
    ) -> np.ndarray:
        """Return per-ROI spatial covariance matrices with shape ``(2, 2, n_roi)``."""

        order = _validate_coordinate_order(order)
        if regularization < 0.0:
            raise ValueError("regularization must be non-negative")
        if self.n_rois == 0:
            return np.zeros((2, 2, 0), dtype=float)

        covariances = np.zeros((2, 2, self.n_rois), dtype=float)
        centroids = self.centroids(order=order, weighted=weighted)

        for roi_index, mask in enumerate(self.roi_masks):
            row_coords, col_coords = np.nonzero(mask)
            if weighted:
                weights = np.asarray(mask[row_coords, col_coords], dtype=float)
            else:
                weights = np.ones(row_coords.shape[0], dtype=float)

            weight_sum = float(np.sum(weights))
            if weight_sum <= 0.0:
                raise ValueError(f"ROI {roi_index} has non-positive total mask weight")

            if order == "xy":
                samples = np.vstack((col_coords, row_coords)).astype(float)
            else:
                samples = np.vstack((row_coords, col_coords)).astype(float)

            centered = samples - centroids[:, roi_index][:, None]
            covariance = (centered * weights[None, :]) @ centered.T / weight_sum
            if regularization > 0.0:
                covariance = covariance + regularization * np.eye(2)
            covariances[:, :, roi_index] = covariance

        return covariances

    def to_measurement_matrix(self, order: str = "xy", weighted: bool = False) -> np.ndarray:
        return self.centroids(order=order, weighted=weighted)

    def to_constant_velocity_state_moments(
        self,
        order: str = "xy",
        weighted: bool = False,
        velocity_variance: float = 25.0,
        regularization: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Embed ROI positions into constant-velocity state moments.

        Returns
        -------
        means, covariances
            ``means`` has shape ``(4, n_roi)`` and ``covariances`` has shape
            ``(4, 4, n_roi)``.
        """

        if velocity_variance < 0.0:
            raise ValueError("velocity_variance must be non-negative")

        means_2d = self.centroids(order=order, weighted=weighted)
        covariances_2d = self.position_covariances(
            order=order,
            weighted=weighted,
            regularization=regularization,
        )

        means = np.zeros((4, self.n_rois), dtype=float)
        covariances = np.zeros((4, 4, self.n_rois), dtype=float)

        for roi_index in range(self.n_rois):
            means[:, roi_index] = np.array(
                [means_2d[0, roi_index], 0.0, means_2d[1, roi_index], 0.0],
                dtype=float,
            )
            covariances[:, :, roi_index] = np.array(
                [
                    [covariances_2d[0, 0, roi_index], 0.0, covariances_2d[0, 1, roi_index], 0.0],
                    [0.0, velocity_variance, 0.0, 0.0],
                    [covariances_2d[1, 0, roi_index], 0.0, covariances_2d[1, 1, roi_index], 0.0],
                    [0.0, 0.0, 0.0, velocity_variance],
                ],
                dtype=float,
            )

        return means, covariances

    def to_pyrecest_gaussian_distributions(
        self,
        order: str = "xy",
        weighted: bool = False,
        velocity_variance: float = 25.0,
        regularization: float = 1e-6,
    ) -> list[Any]:
        """Return one PyRecEst ``GaussianDistribution`` per ROI.

        Import is delayed so the script remains usable for inspection/export even when
        PyRecEst is not installed in the current environment.
        """

        try:
            from pyrecest.distributions import GaussianDistribution
        except ImportError as exc:  # pragma: no cover - exercised in real runtime only
            raise ImportError(
                "PyRecEst is required for to_pyrecest_gaussian_distributions()."
            ) from exc

        means, covariances = self.to_constant_velocity_state_moments(
            order=order,
            weighted=weighted,
            velocity_variance=velocity_variance,
            regularization=regularization,
        )
        return [
            GaussianDistribution(means[:, roi_index], covariances[:, :, roi_index])
            for roi_index in range(self.n_rois)
        ]

    def to_pyrecest_kalman_filters(
        self,
        order: str = "xy",
        weighted: bool = False,
        velocity_variance: float = 25.0,
        regularization: float = 1e-6,
    ) -> list[Any]:
        """Return one PyRecEst ``KalmanFilter`` per ROI."""

        try:
            from pyrecest.filters.kalman_filter import KalmanFilter
        except ImportError as exc:  # pragma: no cover - exercised in real runtime only
            raise ImportError(
                "PyRecEst is required for to_pyrecest_kalman_filters()."
            ) from exc

        gaussians = self.to_pyrecest_gaussian_distributions(
            order=order,
            weighted=weighted,
            velocity_variance=velocity_variance,
            regularization=regularization,
        )
        return [KalmanFilter(gaussian) for gaussian in gaussians]

    def to_export_dict(
        self,
        *,
        order: str = "xy",
        weighted: bool = False,
        velocity_variance: float = 25.0,
        regularization: float = 1e-6,
        include_masks: bool = False,
    ) -> dict[str, np.ndarray]:
        """Return plain NumPy arrays suitable for NPZ export."""

        means, covariances = self.to_constant_velocity_state_moments(
            order=order,
            weighted=weighted,
            velocity_variance=velocity_variance,
            regularization=regularization,
        )
        export = {
            "measurements": self.to_measurement_matrix(order=order, weighted=weighted),
            "state_means": means,
            "state_covariances": covariances,
            "roi_indices": np.asarray(self.roi_indices if self.roi_indices is not None else np.arange(self.n_rois), dtype=int),
        }
        if self.traces is not None:
            export["traces"] = self.traces
        if self.spike_traces is not None:
            export["spike_traces"] = self.spike_traces
        if self.neuropil_traces is not None:
            export["neuropil_traces"] = self.neuropil_traces
        if self.cell_probabilities is not None:
            export["cell_probabilities"] = self.cell_probabilities
        if self.fov is not None:
            export["fov"] = self.fov
        if include_masks:
            export["roi_masks"] = self.roi_masks
        for key, value in self.roi_features.items():
            export[f"feature__{key}"] = value
        return export


@dataclass(frozen=True)
class Track2pSession:
    """One recording session from a Track2p-style subject directory."""

    session_dir: Path
    session_name: str
    session_date: date | None
    plane_data: CalciumPlaneData
    motion_energy: np.ndarray | None = None


def load_suite2p_plane(
    plane_dir: str | Path,
    *,
    include_non_cells: bool = False,
    cell_probability_threshold: float = 0.5,
    weighted_masks: bool = False,
    exclude_overlapping_pixels: bool = True,
    load_traces: bool = True,
    load_spike_traces: bool = True,
    load_neuropil_traces: bool = False,
) -> CalciumPlaneData:
    """Load one Suite2p plane folder into a :class:`CalciumPlaneData` instance."""

    if not 0.0 <= cell_probability_threshold <= 1.0:
        raise ValueError("cell_probability_threshold must be between 0 and 1")

    plane_dir = Path(plane_dir)
    stat = np.load(plane_dir / "stat.npy", allow_pickle=True)
    if stat.ndim != 1:
        raise ValueError("Suite2p stat.npy must be a one-dimensional object array")

    iscell_path = plane_dir / "iscell.npy"
    iscell = np.load(iscell_path, allow_pickle=True) if iscell_path.exists() else None

    ops_path = plane_dir / "ops.npy"
    ops = None
    fov = None
    if ops_path.exists():
        ops = np.load(ops_path, allow_pickle=True).item()
        mean_image = ops.get("meanImg")
        if mean_image is not None:
            fov = np.asarray(mean_image)

    image_shape = _infer_image_shape(stat, ops)

    selected_indices: list[int] = []
    roi_masks: list[np.ndarray] = []
    cell_probabilities: list[float] = []
    feature_names = (
        "radius",
        "aspect_ratio",
        "compact",
        "footprint",
        "skew",
        "std",
        "npix",
        "npix_norm",
    )
    collected_features: dict[str, list[float]] = {name: [] for name in feature_names}

    for roi_index, roi_stat in enumerate(stat):
        keep_roi = True
        probability = np.nan
        if iscell is not None:
            probability = (
                float(iscell[roi_index, 1])
                if np.ndim(iscell) == 2 and iscell.shape[1] > 1
                else float(iscell[roi_index])
            )
            is_cell = bool(iscell[roi_index, 0]) if np.ndim(iscell) == 2 else bool(iscell[roi_index])
            if not include_non_cells:
                keep_roi = is_cell and probability >= cell_probability_threshold

        if not keep_roi:
            continue

        ypix = np.asarray(roi_stat["ypix"], dtype=int)
        xpix = np.asarray(roi_stat["xpix"], dtype=int)
        lam = np.asarray(roi_stat.get("lam", np.ones_like(ypix)), dtype=float)

        if exclude_overlapping_pixels and "overlap" in roi_stat:
            overlap = np.asarray(roi_stat["overlap"], dtype=bool)
            if overlap.shape == ypix.shape:
                valid = ~overlap
                ypix = ypix[valid]
                xpix = xpix[valid]
                lam = lam[valid]

        if ypix.size == 0:
            continue

        mask_dtype = float if weighted_masks else bool
        mask = np.zeros(image_shape, dtype=mask_dtype)
        if weighted_masks:
            mask[ypix, xpix] = lam
        else:
            mask[ypix, xpix] = True

        selected_indices.append(roi_index)
        roi_masks.append(mask)
        cell_probabilities.append(probability)
        for feature_name in feature_names:
            collected_features[feature_name].append(float(roi_stat.get(feature_name, np.nan)))

    roi_mask_array = _stack_or_empty_masks(roi_masks, image_shape, weighted_masks)
    selected_indices_array = np.asarray(selected_indices, dtype=int)
    probability_array = (
        np.asarray(cell_probabilities, dtype=float)
        if roi_masks
        else np.zeros((0,), dtype=float)
    )
    feature_arrays = {
        key: np.asarray(value, dtype=float)
        for key, value in collected_features.items()
        if value
    }

    traces = None
    if load_traces and (plane_dir / "F.npy").exists():
        traces = np.load(plane_dir / "F.npy")
        traces = traces[selected_indices_array]

    spike_traces = None
    if load_spike_traces and (plane_dir / "spks.npy").exists():
        spike_traces = np.load(plane_dir / "spks.npy")
        spike_traces = spike_traces[selected_indices_array]

    neuropil_traces = None
    if load_neuropil_traces and (plane_dir / "Fneu.npy").exists():
        neuropil_traces = np.load(plane_dir / "Fneu.npy")
        neuropil_traces = neuropil_traces[selected_indices_array]

    return CalciumPlaneData(
        roi_masks=roi_mask_array,
        traces=traces,
        fov=fov,
        spike_traces=spike_traces,
        neuropil_traces=neuropil_traces,
        cell_probabilities=probability_array,
        roi_indices=selected_indices_array,
        roi_features=feature_arrays,
        source="suite2p",
        plane_name=plane_dir.name,
        ops=ops,
    )


def load_raw_npy_plane(plane_dir: str | Path) -> CalciumPlaneData:
    """Load one Track2p ``data_npy/planeX`` folder."""

    plane_dir = Path(plane_dir)
    roi_masks = np.load(plane_dir / "rois.npy")
    traces = np.load(plane_dir / "F.npy")
    fov = np.load(plane_dir / "fov.npy")

    if roi_masks.ndim != 3:
        raise ValueError("rois.npy must have shape (n_roi, height, width)")
    if traces.ndim != 2:
        raise ValueError("F.npy must have shape (n_roi, n_timepoints)")
    if traces.shape[0] != roi_masks.shape[0]:
        raise ValueError("F.npy and rois.npy must contain the same number of ROIs")
    if fov.shape != roi_masks.shape[1:]:
        raise ValueError("fov.npy spatial shape must match rois.npy")

    return CalciumPlaneData(
        roi_masks=np.asarray(roi_masks),
        traces=np.asarray(traces),
        fov=np.asarray(fov),
        roi_indices=np.arange(roi_masks.shape[0], dtype=int),
        source="raw_npy",
        plane_name=plane_dir.name,
    )


def find_track2p_session_dirs(subject_dir: str | Path) -> list[Path]:
    """Return Track2p-style session folders sorted chronologically."""

    subject_dir = Path(subject_dir)
    candidate_dirs = [path for path in subject_dir.iterdir() if path.is_dir()]
    recognized_dirs: list[tuple[date | None, str, Path]] = []
    for candidate in candidate_dirs:
        match = _SESSION_NAME_PATTERN.match(candidate.name)
        session_date = (
            date.fromisoformat(match.group("session_date")) if match is not None else None
        )
        if session_date is None and not ((candidate / "suite2p").exists() or (candidate / "data_npy").exists()):
            continue
        recognized_dirs.append((session_date, candidate.name, candidate))

    recognized_dirs.sort(key=lambda item: (item[0] is None, item[0], item[1]))
    return [path for _, _, path in recognized_dirs]


def load_track2p_subject(
    subject_dir: str | Path,
    *,
    plane_name: str = "plane0",
    input_format: str = "auto",
    include_behavior: bool = True,
    **suite2p_kwargs: Any,
) -> list[Track2pSession]:
    """Load all sessions of one Track2p-style subject folder."""

    if input_format not in {"auto", "suite2p", "npy"}:
        raise ValueError("input_format must be 'auto', 'suite2p', or 'npy'")

    subject_dir = Path(subject_dir)
    sessions: list[Track2pSession] = []
    for session_dir in find_track2p_session_dirs(subject_dir):
        suite2p_plane_dir = session_dir / "suite2p" / plane_name
        npy_plane_dir = session_dir / "data_npy" / plane_name

        plane_data: CalciumPlaneData | None = None
        if input_format in {"auto", "suite2p"} and suite2p_plane_dir.exists():
            plane_data = load_suite2p_plane(suite2p_plane_dir, **suite2p_kwargs)
        elif input_format in {"auto", "npy"} and npy_plane_dir.exists():
            plane_data = load_raw_npy_plane(npy_plane_dir)

        if plane_data is None:
            if input_format == "auto":
                continue
            raise FileNotFoundError(
                f"Could not find {input_format} data for session '{session_dir.name}' and plane '{plane_name}'"
            )

        motion_energy = None
        if include_behavior:
            motion_energy_path = session_dir / "move_deve" / "motion_energy_glob.npy"
            if motion_energy_path.exists():
                motion_energy = np.load(motion_energy_path)

        match = _SESSION_NAME_PATTERN.match(session_dir.name)
        session_date = (
            date.fromisoformat(match.group("session_date")) if match is not None else None
        )
        sessions.append(
            Track2pSession(
                session_dir=session_dir,
                session_name=session_dir.name,
                session_date=session_date,
                plane_data=plane_data,
                motion_energy=motion_energy,
            )
        )

    return sessions


def export_subject_to_npz(
    subject_dir: str | Path,
    output_path: str | Path,
    *,
    plane_name: str = "plane0",
    input_format: str = "auto",
    include_behavior: bool = True,
    include_masks: bool = False,
    order: str = "xy",
    weighted: bool = False,
    velocity_variance: float = 25.0,
    regularization: float = 1e-6,
    validate_pyrecest: bool = False,
    **suite2p_kwargs: Any,
) -> dict[str, Any]:
    """Export one subject into a single NPZ archive.

    The archive contains one block of per-session arrays keyed as
    ``session_{index}__<name>`` plus summary metadata.
    """

    sessions = load_track2p_subject(
        subject_dir,
        plane_name=plane_name,
        input_format=input_format,
        include_behavior=include_behavior,
        **suite2p_kwargs,
    )

    payload: dict[str, np.ndarray] = {
        "session_names": np.asarray([session.session_name for session in sessions], dtype=object),
        "session_dates": np.asarray([
            session.session_date.isoformat() if session.session_date is not None else ""
            for session in sessions
        ], dtype=object),
        "plane_name": np.asarray(plane_name, dtype=object),
        "input_format": np.asarray(input_format, dtype=object),
    }

    summary_sessions: list[dict[str, Any]] = []
    for session_index, session in enumerate(sessions):
        plane_data = session.plane_data
        export = plane_data.to_export_dict(
            order=order,
            weighted=weighted,
            velocity_variance=velocity_variance,
            regularization=regularization,
            include_masks=include_masks,
        )
        for key, value in export.items():
            payload[f"session_{session_index}__{key}"] = value
        if session.motion_energy is not None:
            payload[f"session_{session_index}__motion_energy"] = session.motion_energy

        if validate_pyrecest:
            # Force lazy imports and object construction without storing fragile pickles.
            _ = plane_data.to_pyrecest_gaussian_distributions(
                order=order,
                weighted=weighted,
                velocity_variance=velocity_variance,
                regularization=regularization,
            )

        summary_sessions.append(
            {
                "session_name": session.session_name,
                "session_date": session.session_date.isoformat() if session.session_date else None,
                "source": plane_data.source,
                "n_rois": plane_data.n_rois,
                "image_shape": list(plane_data.image_shape),
                "has_traces": plane_data.traces is not None,
                "has_fov": plane_data.fov is not None,
                "has_motion_energy": session.motion_energy is not None,
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)

    return {
        "subject_dir": str(Path(subject_dir)),
        "output_path": str(output_path),
        "n_sessions": len(sessions),
        "plane_name": plane_name,
        "input_format": input_format,
        "sessions": summary_sessions,
    }


def summarize_subject(
    subject_dir: str | Path,
    *,
    plane_name: str = "plane0",
    input_format: str = "auto",
    include_behavior: bool = True,
    **suite2p_kwargs: Any,
) -> dict[str, Any]:
    """Return JSON-serializable summary for one subject."""

    sessions = load_track2p_subject(
        subject_dir,
        plane_name=plane_name,
        input_format=input_format,
        include_behavior=include_behavior,
        **suite2p_kwargs,
    )
    return {
        "subject_dir": str(Path(subject_dir)),
        "plane_name": plane_name,
        "input_format": input_format,
        "n_sessions": len(sessions),
        "sessions": [
            {
                "session_name": session.session_name,
                "session_date": session.session_date.isoformat() if session.session_date else None,
                "source": session.plane_data.source,
                "n_rois": session.plane_data.n_rois,
                "image_shape": list(session.plane_data.image_shape),
                "trace_shape": list(session.plane_data.traces.shape) if session.plane_data.traces is not None else None,
                "has_fov": session.plane_data.fov is not None,
                "has_motion_energy": session.motion_energy is not None,
            }
            for session in sessions
        ],
    }


def _stack_or_empty_masks(
    roi_masks: list[np.ndarray],
    image_shape: tuple[int, int],
    weighted_masks: bool,
) -> np.ndarray:
    if roi_masks:
        return np.stack(roi_masks, axis=0)
    mask_dtype = float if weighted_masks else bool
    return np.zeros((0, image_shape[0], image_shape[1]), dtype=mask_dtype)


def _infer_image_shape(stat: np.ndarray, ops: dict[str, Any] | None) -> tuple[int, int]:
    if ops is not None and "Ly" in ops and "Lx" in ops:
        return int(ops["Ly"]), int(ops["Lx"])
    if len(stat) == 0:
        raise ValueError("Cannot infer image shape from an empty stat.npy without ops.npy")
    max_y = 0
    max_x = 0
    for roi_stat in stat:
        max_y = max(max_y, int(np.max(roi_stat["ypix"])))
        max_x = max(max_x, int(np.max(roi_stat["xpix"])))
    return max_y + 1, max_x + 1


def _validate_coordinate_order(order: str) -> str:
    if order not in {"xy", "yx"}:
        raise ValueError("order must be either 'xy' or 'yx'")
    return order


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Track2p/Suite2p loader that exports PyRecEst-ready state moments."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("subject_dir", type=Path, help="Track2p-style subject directory")
    common.add_argument("--plane", dest="plane_name", default="plane0", help="Plane subdirectory to load")
    common.add_argument(
        "--input-format",
        default="auto",
        choices=("auto", "suite2p", "npy"),
        help="Input format to load",
    )
    common.add_argument(
        "--include-behavior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load motion_energy_glob.npy when present",
    )
    common.add_argument(
        "--include-non-cells",
        action="store_true",
        help="Keep Suite2p ROIs that fail iscell filtering",
    )
    common.add_argument(
        "--cell-probability-threshold",
        type=float,
        default=0.5,
        help="Suite2p iscell probability threshold",
    )
    common.add_argument(
        "--weighted-masks",
        action="store_true",
        help="Reconstruct Suite2p masks using lam weights",
    )
    common.add_argument(
        "--exclude-overlapping-pixels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop Suite2p overlap pixels when reconstructing masks",
    )

    summary_parser = subparsers.add_parser("summary", parents=[common], help="Print JSON summary")
    summary_parser.set_defaults(_handler=_handle_summary)

    export_parser = subparsers.add_parser("export", parents=[common], help="Export an NPZ bundle")
    export_parser.add_argument("output_path", type=Path, help="Destination .npz file")
    export_parser.add_argument(
        "--include-masks",
        action="store_true",
        help="Include ROI masks in the export archive",
    )
    export_parser.add_argument(
        "--order",
        default="xy",
        choices=("xy", "yx"),
        help="Coordinate order in exported measurement/state arrays",
    )
    export_parser.add_argument(
        "--weighted",
        action="store_true",
        help="Use weighted centroids/covariances when masks contain weights",
    )
    export_parser.add_argument(
        "--velocity-variance",
        type=float,
        default=25.0,
        help="Velocity variance for the constant-velocity embedding",
    )
    export_parser.add_argument(
        "--regularization",
        type=float,
        default=1e-6,
        help="Small diagonal regularization added to 2D ROI covariances",
    )
    export_parser.add_argument(
        "--validate-pyrecest",
        action="store_true",
        help="Instantiate PyRecEst GaussianDistribution objects during export",
    )
    export_parser.set_defaults(_handler=_handle_export)

    return parser


def _suite2p_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "include_non_cells": args.include_non_cells,
        "cell_probability_threshold": args.cell_probability_threshold,
        "weighted_masks": args.weighted_masks,
        "exclude_overlapping_pixels": args.exclude_overlapping_pixels,
    }


def _handle_summary(args: argparse.Namespace) -> int:
    summary = summarize_subject(
        args.subject_dir,
        plane_name=args.plane_name,
        input_format=args.input_format,
        include_behavior=args.include_behavior,
        **_suite2p_kwargs_from_args(args),
    )
    print(json.dumps(summary, indent=2))
    return 0


def _handle_export(args: argparse.Namespace) -> int:
    summary = export_subject_to_npz(
        args.subject_dir,
        args.output_path,
        plane_name=args.plane_name,
        input_format=args.input_format,
        include_behavior=args.include_behavior,
        include_masks=args.include_masks,
        order=args.order,
        weighted=args.weighted,
        velocity_variance=args.velocity_variance,
        regularization=args.regularization,
        validate_pyrecest=args.validate_pyrecest,
        **_suite2p_kwargs_from_args(args),
    )
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return int(args._handler(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
