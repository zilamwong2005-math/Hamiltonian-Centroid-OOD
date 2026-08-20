import numpy as np
import torch

from diagnose_imagenet1k_score_rules import (
    _derived_affinity_scores,
    _validation_masks,
    _zscore,
)


def test_affinity_score_rules_are_finite_and_include_backbone_class():
    affinity = torch.tensor([[2.0, 1.0, 0.5], [0.2, 0.4, 0.8]])
    prediction = torch.tensor([0, 2])
    scores = _derived_affinity_scores(
        affinity, prefix="exact", predicted_class=prediction
    )
    assert "exact/backbone_class_affinity" in scores
    assert torch.allclose(scores["exact/raw_max"], torch.tensor([2.0, 0.8]))
    assert all(torch.isfinite(value).all() for value in scores.values())


def test_validation_masks_partition_every_sample_and_stratify_id():
    labels = np.repeat(np.arange(3), 5)
    id_tune, id_holdout, ood_tune, ood_holdout = _validation_masks(labels, 11, 0)
    assert np.all(id_tune ^ id_holdout)
    assert np.all(ood_tune ^ ood_holdout)
    assert [id_tune[labels == label].sum() for label in range(3)] == [2, 2, 2]
    assert ood_tune.sum() == 5


def test_zscore_uses_reference_statistics_and_handles_constant_reference():
    values = np.array([1.0, 2.0, 3.0])
    standard = _zscore(values, values)
    assert abs(standard.mean()) < 1e-12
    assert abs(standard.std() - 1.0) < 1e-12
    constant = _zscore(values, np.ones(4))
    assert np.isfinite(constant).all()
