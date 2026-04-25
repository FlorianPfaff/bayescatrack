from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from bayescatrack.association.calibrated_costs import (
    CalibratedAssociationModel,
    fit_logistic_association_model,
    label_matrix_from_reference,
    pairwise_feature_tensor,
)
from bayescatrack.core.bridge import load_track2p_subject
from bayescatrack.reference import Track2pReference


def _write_subject(subject_dir, write_raw_npy_session):
    masks_a = np.zeros((2, 4, 4), dtype=bool)
    masks_a[0, 0:2, 0:2] = True
    masks_a[1, 2:4, 2:4] = True
    masks_b = masks_a.copy()
    masks_c = masks_a.copy()

    write_raw_npy_session(subject_dir, "2024-05-01_a", masks_a, offset=0.0)
    write_raw_npy_session(subject_dir, "2024-05-02_a", masks_b, offset=10.0)
    write_raw_npy_session(subject_dir, "2024-05-03_a", masks_c, offset=20.0)


def _install_fake_association_model(monkeypatch):
    fake_pyrecest = types.ModuleType("pyrecest")
    fake_utils = types.ModuleType("pyrecest.utils")
    fake_models = types.ModuleType("pyrecest.utils.association_models")

    class LogisticPairwiseAssociationModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.fit_args = None

        def fit(self, features, labels, sample_weight=None):
            self.fit_args = (np.asarray(features), np.asarray(labels), sample_weight)
            return self

        def pairwise_cost_matrix(self, features):
            return np.sum(np.asarray(features, dtype=float), axis=-1)

    fake_models.LogisticPairwiseAssociationModel = LogisticPairwiseAssociationModel
    monkeypatch.setitem(sys.modules, "pyrecest", fake_pyrecest)
    monkeypatch.setitem(sys.modules, "pyrecest.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "pyrecest.utils.association_models", fake_models)


def test_pairwise_features_and_reference_labels():
    components = {
        "centroid_distance": np.array([[0.0, 2.0], [3.0, 0.0]]),
        "iou": np.array([[1.0, 0.25], [0.0, 0.5]]),
        "mask_cosine_similarity": np.array([[1.0, 0.5], [0.2, 0.8]]),
    }
    features = pairwise_feature_tensor(
        components,
        feature_names=("centroid_distance", "one_minus_iou", "one_minus_mask_cosine"),
    )

    assert features.shape == (2, 2, 3)
    np.testing.assert_allclose(features[0, 1], np.array([2.0, 0.75, 0.5]))

    reference = Track2pReference(
        session_names=("day0", "day1"),
        suite2p_indices=np.array([[5, 7], [6, None]], dtype=object),
    )
    labels = label_matrix_from_reference(
        reference,
        0,
        1,
        reference_roi_indices=np.array([5, 6]),
        measurement_roi_indices=np.array([7, 8]),
    )

    np.testing.assert_array_equal(labels, np.array([[1, 0], [0, 0]]))


def test_fit_logistic_association_model_keeps_feature_schema(monkeypatch):
    _install_fake_association_model(monkeypatch)

    calibrated = fit_logistic_association_model(
        np.array([[0.0, 1.0], [2.0, 0.0]]),
        np.array([1, 0]),
        feature_names=("centroid_distance", "one_minus_iou"),
        model_kwargs={"class_weight": None},
    )
    costs = calibrated.pairwise_cost_matrix_from_components(
        {
            "centroid_distance": np.array([[0.0, 2.0]]),
            "iou": np.array([[1.0, 0.25]]),
        }
    )

    assert calibrated.feature_names == ("centroid_distance", "one_minus_iou")
    assert calibrated.model.kwargs == {"class_weight": None}
    np.testing.assert_allclose(costs, np.array([[0.0, 2.75]]))


def test_registered_pairwise_costs_accept_calibrated_model(tmp_path, monkeypatch, write_raw_npy_session):
    subject_dir = tmp_path / "jm003"
    _write_subject(subject_dir, write_raw_npy_session)

    class SumFeatureCostModel:
        def pairwise_cost_matrix(self, features):
            return np.sum(np.asarray(features, dtype=float), axis=-1)

    from bayescatrack.association import pyrecest_global_assignment as global_assignment

    monkeypatch.setattr(global_assignment, "register_plane_pair", lambda _reference, moving, **_kwargs: moving)
    sessions = load_track2p_subject(subject_dir, plane_name="plane0", input_format="auto")
    calibrated = CalibratedAssociationModel(SumFeatureCostModel(), feature_names=("one_minus_iou",))

    pairwise_costs = global_assignment.build_registered_pairwise_costs(
        sessions,
        max_gap=2,
        cost="calibrated",
        calibrated_model=calibrated,
    )

    assert set(pairwise_costs) == {(0, 1), (0, 2), (1, 2)}
    np.testing.assert_allclose(np.diag(pairwise_costs[(0, 2)]), np.zeros(2))
    assert pairwise_costs[(0, 2)][0, 1] > 0.0


def test_calibrated_cost_requires_model(tmp_path, monkeypatch, write_raw_npy_session):
    subject_dir = tmp_path / "jm004"
    _write_subject(subject_dir, write_raw_npy_session)

    from bayescatrack.association import pyrecest_global_assignment as global_assignment

    monkeypatch.setattr(global_assignment, "register_plane_pair", lambda _reference, moving, **_kwargs: moving)
    sessions = load_track2p_subject(subject_dir, plane_name="plane0", input_format="auto")

    with pytest.raises(ValueError, match="calibrated_model"):
        global_assignment.build_registered_pairwise_costs(sessions, cost="calibrated")
