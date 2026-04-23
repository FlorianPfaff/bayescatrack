import subprocess  # nosec B404
import sys
import types
from datetime import date
from pathlib import Path

import numpy as np
import numpy.testing as npt

SCRIPT_PATH = Path(__file__).with_name("track2p_multisession_tracking.py")
sys.path.insert(0, str(SCRIPT_PATH.parent))

from track2p_multisession_tracking import (  # noqa: E402
    LongitudinalTrackingResult,
    MultisessionTrackingConfig,
    build_multisession_pairwise_costs,
    save_tracking_result_npz,
    track_sessions_multisession,
)
from track2p_pyrecest_bridge import CalciumPlaneData, Track2pSession  # noqa: E402


def _make_session(
    session_name: str,
    roi_masks: np.ndarray,
    *,
    roi_indices: np.ndarray | None = None,
    session_date: date | None = None,
) -> Track2pSession:
    plane_data = CalciumPlaneData(
        roi_masks=np.asarray(roi_masks),
        roi_indices=roi_indices,
        cell_probabilities=np.ones((roi_masks.shape[0],), dtype=float),
        source="raw_npy",
        plane_name="plane0",
    )
    return Track2pSession(
        session_dir=Path(session_name),
        session_name=session_name,
        session_date=session_date,
        plane_data=plane_data,
    )


def _install_fake_multisession_solver(monkeypatch, captured_pairwise_keys=None):
    fake_pyrecest = types.ModuleType("pyrecest")
    fake_assignment = types.ModuleType("pyrecest.assignment")
    fake_multisession = types.ModuleType("pyrecest.assignment.multisession")

    def fake_solve_multisession_assignment(
        pairwise_costs,
        session_sizes=None,
        *,
        start_cost=0.0,
        end_cost=0.0,
        gap_penalty=0.0,
        cost_threshold=None,
    ):
        del start_cost, end_cost, gap_penalty, cost_threshold
        if captured_pairwise_keys is not None:
            captured_pairwise_keys.extend(sorted(pairwise_costs))
        if session_sizes is None:
            max_session_index = max(max(edge) for edge in pairwise_costs)
            session_sizes = [0] * (max_session_index + 1)
            for (source_session_index, target_session_index), cost_matrix in pairwise_costs.items():
                session_sizes[source_session_index] = max(
                    session_sizes[source_session_index], cost_matrix.shape[0]
                )
                session_sizes[target_session_index] = max(
                    session_sizes[target_session_index], cost_matrix.shape[1]
                )
        tracks = []
        if session_sizes and all(session_size > 0 for session_size in session_sizes):
            tracks.append(
                {session_index: 0 for session_index, session_size in enumerate(session_sizes) if session_size > 0}
            )
        return {"tracks": tracks, "total_cost": 0.0}

    fake_multisession.solve_multisession_assignment = fake_solve_multisession_assignment
    monkeypatch.setitem(sys.modules, "pyrecest", fake_pyrecest)
    monkeypatch.setitem(sys.modules, "pyrecest.assignment", fake_assignment)
    monkeypatch.setitem(sys.modules, "pyrecest.assignment.multisession", fake_multisession)


def test_build_multisession_pairwise_costs_builds_skip_edges_when_requested():
    roi_masks = np.zeros((1, 4, 4), dtype=bool)
    roi_masks[0, 1:3, 1:3] = True
    sessions = [
        _make_session("2024-05-01_a", roi_masks, roi_indices=np.array([10])),
        _make_session("2024-05-02_a", roi_masks, roi_indices=np.array([11])),
        _make_session("2024-05-03_a", roi_masks, roi_indices=np.array([12])),
    ]

    pairwise_costs, pairwise_bundles = build_multisession_pairwise_costs(
        sessions,
        config=MultisessionTrackingConfig(
            max_session_gap=2,
            pairwise_cost_kwargs={"max_centroid_distance": 5.0, "roi_feature_weight": 0.0},
        ),
    )

    assert set(pairwise_costs) == {(0, 1), (0, 2), (1, 2)}
    assert {(bundle.source_session_index, bundle.target_session_index) for bundle in pairwise_bundles} == {
        (0, 1),
        (0, 2),
        (1, 2),
    }


def test_track_sessions_multisession_recovers_track_matrix_and_original_roi_indices(monkeypatch):
    _install_fake_multisession_solver(monkeypatch)

    roi_masks = np.zeros((1, 4, 4), dtype=bool)
    roi_masks[0, 1:3, 1:3] = True
    sessions = [
        _make_session(
            "2024-05-01_a",
            roi_masks,
            roi_indices=np.array([42]),
            session_date=date(2024, 5, 1),
        ),
        _make_session(
            "2024-05-02_a",
            roi_masks,
            roi_indices=np.array([43]),
            session_date=date(2024, 5, 2),
        ),
        _make_session(
            "2024-05-03_a",
            roi_masks,
            roi_indices=np.array([44]),
            session_date=date(2024, 5, 3),
        ),
    ]

    result = track_sessions_multisession(
        sessions,
        config=MultisessionTrackingConfig(
            max_session_gap=1,
            pairwise_cost_kwargs={"max_centroid_distance": 5.0, "roi_feature_weight": 0.0},
        ),
    )

    npt.assert_array_equal(result.track_matrix, np.array([[0, 0, 0]]))
    npt.assert_array_equal(result.track_roi_index_matrix, np.array([[42, 43, 44]]))
    assert result.session_names == ("2024-05-01_a", "2024-05-02_a", "2024-05-03_a")
    assert result.session_dates == ("2024-05-01", "2024-05-02", "2024-05-03")
    assert result.summary()["n_complete_tracks"] == 1


def test_track_sessions_multisession_short_circuits_single_session_without_solver_import():
    roi_masks = np.zeros((2, 4, 4), dtype=bool)
    roi_masks[0, 0, 0] = True
    roi_masks[1, 3, 3] = True
    session = _make_session("2024-05-01_a", roi_masks, roi_indices=np.array([7, 9]))

    result = track_sessions_multisession([session])

    npt.assert_array_equal(result.track_matrix, np.array([[0], [1]]))
    npt.assert_array_equal(result.track_roi_index_matrix, np.array([[7], [9]]))
    assert result.total_cost == 0.0


def test_save_tracking_result_npz(tmp_path):
    result = LongitudinalTrackingResult(
        tracks=({0: 1, 1: 2},),
        track_matrix=np.array([[1, 2]]),
        track_roi_index_matrix=np.array([[11, 22]]),
        session_names=("a", "b"),
        session_dates=("2024-05-01", "2024-05-02"),
        pairwise_bundles=tuple(),
        total_cost=1.5,
    )
    output_path = tmp_path / "tracks.npz"
    summary = save_tracking_result_npz(result, output_path)
    exported = np.load(output_path, allow_pickle=True)

    assert output_path.exists()
    npt.assert_array_equal(exported["track_matrix"], np.array([[1, 2]]))
    npt.assert_array_equal(exported["track_roi_index_matrix"], np.array([[11, 22]]))
    assert summary["total_cost"] == 1.5


def test_cli_runs_on_single_session(tmp_path):
    subject_dir = tmp_path / "jm123"
    plane_dir = subject_dir / "2024-05-01_a" / "data_npy" / "plane0"
    plane_dir.mkdir(parents=True)
    roi_masks = np.zeros((1, 3, 3), dtype=bool)
    roi_masks[0, 1, 1] = True
    np.save(plane_dir / "rois.npy", roi_masks)
    np.save(plane_dir / "F.npy", np.array([[1.0, 2.0]], dtype=float))
    np.save(plane_dir / "fov.npy", np.ones((3, 3), dtype=float))

    output_path = tmp_path / "tracks.npz"
    proc = subprocess.run(  # nosec B603
        [
            sys.executable,
            str(SCRIPT_PATH),
            str(subject_dir),
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert output_path.exists()
    assert '"n_tracks": 1' in proc.stdout
