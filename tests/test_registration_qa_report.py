from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from bayescatrack.experiments.registration_qa_report import (
    RegistrationQAConfig,
    format_registration_qa_table,
    run_registration_qa_report,
    summarize_registration_qa_links,
)


def _write_ground_truth_csv(
    subject_dir: Path,
    session_names: tuple[str, ...],
    rows: tuple[tuple[int, ...], ...],
) -> Path:
    ground_truth_path = subject_dir / "ground_truth.csv"
    lines = ["track_id," + ",".join(session_names)]
    for track_id, row in enumerate(rows):
        lines.append(f"{track_id}," + ",".join(str(value) for value in row))
    ground_truth_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ground_truth_path


def test_registration_qa_report_summarizes_manual_gt_links(
    tmp_path,
    write_raw_npy_session,
):
    subject_dir = tmp_path / "jm001"
    masks = np.zeros((2, 5, 5), dtype=bool)
    masks[0, 0:2, 0:2] = True
    masks[1, 3:5, 3:5] = True
    write_raw_npy_session(subject_dir, "2024-05-01_a", masks, offset=0.0)
    write_raw_npy_session(subject_dir, "2024-05-02_a", masks.copy(), offset=1.0)
    _write_ground_truth_csv(
        subject_dir,
        ("2024-05-01_a", "2024-05-02_a"),
        ((0, 0), (1, 1)),
    )

    rows = run_registration_qa_report(
        RegistrationQAConfig(
            data=subject_dir,
            reference_kind="manual-gt",
            input_format="npy",
            transform_type="none",
            max_gap=1,
            cost="registered-iou",
        )
    )

    assert len(rows) == 2
    first = rows[0]
    assert first["registration_backend"] == "none"
    assert first["registered_iou"] == pytest.approx(1.0)
    assert first["raw_iou"] == pytest.approx(1.0)
    assert first["registered_centroid_distance"] == pytest.approx(0.0)
    assert first["gt_rank"] == 1
    assert first["gt_is_top1"] is True
    assert first["gt_candidate_admissible"] is True

    summary = summarize_registration_qa_links(rows)
    assert len(summary) == 1
    assert summary[0]["n_gt_links"] == 2
    assert summary[0]["median_registered_iou"] == pytest.approx(1.0)
    assert summary[0]["median_registered_centroid_distance"] == pytest.approx(0.0)
    assert summary[0]["gt_top1_rate"] == pytest.approx(1.0)
    assert summary[0]["gt_admissible_rate"] == pytest.approx(1.0)

    table = format_registration_qa_table(summary)
    assert "median_registered_iou" in table
    assert "jm001" in table
