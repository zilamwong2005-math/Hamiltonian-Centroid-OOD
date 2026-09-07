from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from run_static_reduction_bridge import (
    GAUSSIAN_LIPSCHITZ,
    _average_ranks,
    _centroid_bound,
    _same_anchor_centroids,
    _spearman,
)


def test_average_ranks_handles_ties():
    result = _average_ranks(np.array([4.0, 1.0, 1.0, 3.0]))
    assert np.allclose(result, [4.0, 1.5, 1.5, 3.0])
    assert _spearman(result, result) == pytest.approx(1.0)


def test_radial_and_cosine_scores_have_identical_order_and_class():
    generator = torch.Generator().manual_seed(7)
    anchors = F.normalize(torch.randn(9, 5, 13, generator=generator), dim=-1)
    query = F.normalize(torch.randn(31, 13, generator=generator), dim=-1)
    _, centroid = _same_anchor_centroids(anchors)
    cosine = query @ centroid.T
    sigma = 0.47
    radial = torch.exp(-0.5 * (2.0 - 2.0 * cosine) / sigma**2)
    assert torch.equal(radial.argmax(1), cosine.argmax(1))
    assert _spearman(
        radial.max(1).values.numpy(), cosine.max(1).values.numpy()
    ) == pytest.approx(1.0)


def test_lemma_a2_gaussian_bound_holds_numerically():
    generator = torch.Generator().manual_seed(19)
    anchors = F.normalize(torch.randn(6, 5, 17, generator=generator), dim=-1)
    query = F.normalize(torch.randn(23, 17, generator=generator), dim=-1)
    sigma = 0.53
    epsilon, summary = _centroid_bound(anchors, sigma)
    mean, centroid = _same_anchor_centroids(anchors)
    similarity = torch.einsum("bd,ckd->bck", query, anchors)
    empirical = torch.exp(-0.5 * (2.0 - 2.0 * similarity) / sigma**2).mean(-1)
    radial = torch.exp(-0.5 * (2.0 - 2.0 * (query @ centroid.T)) / sigma**2)
    assert torch.all((empirical - radial).abs() <= epsilon[None, :] + 2e-6)
    assert summary["GaussianLipschitz"] == pytest.approx(GAUSSIAN_LIPSCHITZ)
    assert mean.shape == (6, 17)


def test_source_has_no_test_loader_access():
    source = (Path(__file__).resolve().parents[1] / "run_static_reduction_bridge.py").read_text(
        encoding="utf-8"
    )
    assert "eval_ood(" not in source
    assert 'dataloader_dict["ood"]["near"]' not in source
    assert 'dataloader_dict["ood"]["far"]' not in source
