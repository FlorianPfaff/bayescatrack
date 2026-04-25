import numpy as np
import pytest

from bayescatrack.evaluation.complete_track_scores import (
    complete_track_set,
    pairwise_track_set,
    score_complete_tracks,
    score_pairwise_tracks,
    score_track_matrices,
    track_lengths,
)


def test_complete_track_and_pairwise_scoring():
    reference = np.array(
        [
            [0, 10, 20],
            [1, 11, 21],
            [2, None, 22],
        ],
        dtype=object,
    )
    predicted = np.array(
        [
            [0, 10, 20],
            [1, None, 21],
            [3, 13, 23],
        ],
        dtype=object,
    )

    assert complete_track_set(reference) == {(0, 10, 20), (1, 11, 21)}
    assert pairwise_track_set(predicted) == {(0, 1, 0, 10), (1, 2, 10, 20), (0, 1, 3, 13), (1, 2, 13, 23)}

    complete_scores = score_complete_tracks(predicted, reference)
    assert complete_scores["complete_track_true_positives"] == 1
    assert complete_scores["complete_track_false_positives"] == 1
    assert complete_scores["complete_track_false_negatives"] == 1
    assert complete_scores["complete_track_f1"] == pytest.approx(0.5)

    pairwise_scores = score_pairwise_tracks(predicted, reference)
    assert pairwise_scores["pairwise_true_positives"] == 2
    assert pairwise_scores["pairwise_false_positives"] == 2
    assert pairwise_scores["pairwise_false_negatives"] == 2
    assert pairwise_scores["pairwise_f1"] == pytest.approx(0.5)

    scores = score_track_matrices(predicted, reference)
    assert scores["complete_tracks"] == 2
    assert scores["mean_track_length"] == pytest.approx(8 / 3)
    np.testing.assert_array_equal(track_lengths(predicted), np.array([3, 2, 3]))


def test_score_track_matrices_requires_same_number_of_sessions():
    with pytest.raises(ValueError, match="same number of sessions"):
        score_track_matrices(np.zeros((1, 2)), np.zeros((1, 3)))
