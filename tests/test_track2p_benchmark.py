from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig, format_benchmark_table, run_track2p_benchmark


def _write_subject(subject_dir, write_raw_npy_session):
    masks_a = np.zeros((2, 4, 4), dtype=bool)
    masks_a[0, 0:2, 0:2] = True
    masks_a[1, 2:4, 2:4] = True
    masks_b = masks_a.copy()
    masks_c = masks_a.copy()

    write_raw_npy_session(subject_dir, "2024-05-01_a", masks_a, offset=0.0)
    write_raw_npy_session(subject_dir, "2024-05-02_a", masks_b, offset=10.0)
    write_raw_npy_session(subject_dir, "2024-05-03_a", masks_c, offset=20.0)

    track2p_dir = subject_dir / "track2p"
    track2p_dir.mkdir()
    np.save(
        track2p_dir / "track_ops.npy",
        {
            "all_ds_path": np.array(
                [
                    str(subject_dir / "2024-05-01_a"),
                    str(subject_dir / "2024-05-02_a"),
                    str(subject_dir / "2024-05-03_a"),
                ],
                dtype=object,
            ),
            "vector_curation_plane_0": np.array([1.0, 1.0]),
        },
        allow_pickle=True,
    )
    np.save(track2p_dir / "plane0_suite2p_indices.npy", np.array([[0, 0, 0], [1, 1, 1]], dtype=object), allow_pickle=True)


def _install_fake_multisession_assignment(monkeypatch):
    fake_pyrecest = types.ModuleType("pyrecest")
    fake_utils = types.ModuleType("pyrecest.utils")
    fake_assignment = types.ModuleType("pyrecest.utils.multisession_assignment")

    class Result:
        def __init__(self):
            self.tracks = [{0: 0, 1: 0, 2: 0}, {0: 1, 2: 1}]
            self.matched_edges = []
            self.total_cost = 0.0

    def solve_multisession_assignment(pairwise_costs, **kwargs):
        assert (0, 1) in pairwise_costs
        assert (0, 2) in pairwise_costs
        assert kwargs["session_sizes"] == (2, 2, 2)
        assert kwargs["gap_penalty"] == pytest.approx(1.0)
        return Result()

    fake_assignment.solve_multisession_assignment = solve_multisession_assignment
    monkeypatch.setitem(sys.modules, "pyrecest", fake_pyrecest)
    monkeypatch.setitem(sys.modules, "pyrecest.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "pyrecest.utils.multisession_assignment", fake_assignment)


def test_track2p_baseline_benchmark_scores_track2p_output(tmp_path, write_raw_npy_session):
    subject_dir = tmp_path / "jm001"
    _write_subject(subject_dir, write_raw_npy_session)

    rows = run_track2p_benchmark(Track2pBenchmarkConfig(data=tmp_path, method="track2p-baseline"))

    assert len(rows) == 1
    result = rows[0].to_dict()
    assert result["variant"] == "Track2p default"
    assert result["pairwise_f1"] == pytest.approx(1.0)
    assert result["complete_track_f1"] == pytest.approx(1.0)
    assert result["complete_tracks"] == 2
    assert "Track2p default" in format_benchmark_table([result])


def test_global_assignment_benchmark_uses_skip_edges(tmp_path, monkeypatch, write_raw_npy_session):
    subject_dir = tmp_path / "jm002"
    _write_subject(subject_dir, write_raw_npy_session)
    _install_fake_multisession_assignment(monkeypatch)

    from bayescatrack.association import pyrecest_global_assignment as global_assignment

    monkeypatch.setattr(global_assignment, "register_plane_pair", lambda _reference, moving, **_kwargs: moving)

    rows = run_track2p_benchmark(
        Track2pBenchmarkConfig(
            data=subject_dir,
            method="global-assignment",
            cost="registered-iou",
            max_gap=2,
        )
    )

    assert len(rows) == 1
    result = rows[0].to_dict()
    assert result["variant"] == "Same costs + global assignment"
    assert result["pairwise_f1"] == pytest.approx(2 / 3)
    assert result["complete_track_f1"] == pytest.approx(2 / 3)
    assert result["complete_tracks"] == 1
