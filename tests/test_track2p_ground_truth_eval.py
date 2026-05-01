import numpy as np

from bayescatrack.ground_truth_eval import (
    TrackTable,
    evaluate_track_table_prediction,
    load_track2p_ground_truth_csv,
)


def test_evaluate_track_table_prediction_scores_exact_tracks():
    ground_truth = TrackTable(
        session_names=("s1", "s2"),
        tracks=np.array([[1, 2], [3, 4]], dtype=int),
    )
    prediction = TrackTable(
        session_names=("s1", "s2"),
        tracks=np.array([[1, 2], [3, 5]], dtype=int),
    )

    evaluation = evaluate_track_table_prediction(ground_truth, prediction)

    assert evaluation.n_exact_full_track_matches == 1
    assert evaluation.complete_tracks == 0.5
    assert evaluation.proportion_correct_by_horizon[2] == 0.5


def test_load_track2p_ground_truth_csv_supports_semicolon_encoded_rows(tmp_path):
    ground_truth_path = tmp_path / "ground_truth.csv"
    ground_truth_path.write_text(
        "track_id,track\n"
        "0,67;38;15;169;;;\n"
        "1,11;;13;14\n",
        encoding="utf-8",
    )

    table = load_track2p_ground_truth_csv(
        ground_truth_path,
        session_names=("s1", "s2", "s3", "s4"),
    )

    assert table.session_names == ("s1", "s2", "s3", "s4")
    np.testing.assert_array_equal(
        table.tracks,
        np.array([[67, 38, 15, 169], [11, -1, 13, 14]], dtype=int),
    )
