from __future__ import annotations

from pathlib import Path

import numpy as np
from bayescatrack.experiments.track2p_roi_index_audit import (
    ManualGtRoiIndexAuditConfig,
    format_audit_markdown,
    run_manual_gt_roi_index_audit,
)


def _write_ground_truth_csv(
    subject_dir: Path, session_names: tuple[str, ...], rows: tuple[tuple[int, ...], ...]
) -> Path:
    ground_truth_path = subject_dir / "ground_truth.csv"
    lines = ["track_id," + ",".join(session_names)]
    for track_id, row in enumerate(rows):
        lines.append(f"{track_id}," + ",".join(str(value) for value in row))
    ground_truth_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ground_truth_path


def _write_suite2p_session(subject_dir: Path, session_name: str, iscell: np.ndarray) -> None:
    plane_dir = subject_dir / session_name / "suite2p" / "plane0"
    plane_dir.mkdir(parents=True, exist_ok=True)
    stat = []
    for index in range(iscell.shape[0]):
        stat.append(
            {
                "ypix": np.array([index % 4, index % 4]),
                "xpix": np.array([0, 1]),
                "lam": np.ones(2),
                "overlap": np.zeros(2, dtype=bool),
            }
        )
    np.save(plane_dir / "stat.npy", np.asarray(stat, dtype=object), allow_pickle=True)
    np.save(plane_dir / "iscell.npy", iscell)
    np.save(
        plane_dir / "ops.npy",
        {"Ly": 8, "Lx": 8, "meanImg": np.zeros((8, 8), dtype=float)},
        allow_pickle=True,
    )
    np.save(plane_dir / "F.npy", np.zeros((iscell.shape[0], 2), dtype=float))


def test_roi_index_audit_reports_raw_stat_rows_resolved_by_include_non_cells(tmp_path):
    subject_dir = tmp_path / "jm046"
    sessions = ("2024-05-01_a", "2024-05-02_a")
    iscell = np.array([[1.0, 0.95], [0.0, 0.10], [1.0, 0.90]], dtype=float)
    for session_name in sessions:
        _write_suite2p_session(subject_dir, session_name, iscell)
    _write_ground_truth_csv(subject_dir, sessions, ((1, 1), (2, 2)))

    result = run_manual_gt_roi_index_audit(
        ManualGtRoiIndexAuditConfig(data=subject_dir, input_format="suite2p")
    )

    assert not result.compatible
    assert result.incompatible_subjects == ("jm046",)
    row = result.rows[0]
    assert row.n_stat_rows == 3
    assert row.n_loaded_rois == 2
    assert row.n_loaded_cells == 2
    assert row.n_loaded_rois_with_non_cells == 3
    assert row.n_gt_rois == 2
    assert row.max_gt_roi_index == 2
    assert row.n_gt_rois_missing_from_loaded_indices == 1
    assert row.n_gt_rois_missing_with_include_non_cells == 0
    assert row.missing_gt_roi_examples == (1,)
    assert row.include_non_cells_resolves_mismatch
    assert row.gt_fits_raw_stat_row_space
    assert not row.gt_fits_filtered_cell_ordinal_space
    assert row.gt_index_space == "raw_stat_rows_requires_include_non_cells"

    row_dict = row.to_dict()
    assert row_dict["n_stat_rows"] == 3
    assert row_dict["n_gt_rois_missing_from_loaded_indices"] == 1
    assert row_dict["missing_gt_roi_examples"] == "1"
    assert "raw_stat_rows_requires_include_non_cells" in format_audit_markdown(result)


def test_roi_index_audit_reports_outside_stat_row_space(tmp_path):
    subject_dir = tmp_path / "jm047"
    sessions = ("2024-05-01_a", "2024-05-02_a")
    iscell = np.ones((3, 2), dtype=float)
    for session_name in sessions:
        _write_suite2p_session(subject_dir, session_name, iscell)
    _write_ground_truth_csv(subject_dir, sessions, ((10, 10),))

    result = run_manual_gt_roi_index_audit(
        ManualGtRoiIndexAuditConfig(data=tmp_path, input_format="suite2p")
    )

    assert not result.compatible
    row = result.rows[0]
    assert row.n_stat_rows == 3
    assert row.max_gt_roi_index == 10
    assert row.n_gt_rois_missing_from_loaded_indices == 1
    assert row.n_gt_rois_missing_with_include_non_cells == 1
    assert not row.include_non_cells_resolves_mismatch
    assert row.gt_fits_raw_stat_row_space is False
    assert row.gt_index_space == "outside_stat_row_space"


def test_roi_index_audit_reports_compatible_raw_stat_rows(tmp_path):
    subject_dir = tmp_path / "jm038"
    sessions = ("2024-05-01_a", "2024-05-02_a")
    iscell = np.ones((3, 2), dtype=float)
    for session_name in sessions:
        _write_suite2p_session(subject_dir, session_name, iscell)
    _write_ground_truth_csv(subject_dir, sessions, ((0, 0), (2, 2)))

    result = run_manual_gt_roi_index_audit(
        ManualGtRoiIndexAuditConfig(data=subject_dir, input_format="suite2p")
    )

    assert result.compatible
    assert result.incompatible_subjects == ()
    assert {row.gt_index_space for row in result.rows} == {"raw_stat_rows_loaded"}
    assert {row.n_gt_rois_missing_from_loaded_indices for row in result.rows} == {0}
