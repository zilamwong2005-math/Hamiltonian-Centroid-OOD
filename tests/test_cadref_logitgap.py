import json

import pandas as pd
import pytest
import torch

from run_cadref_logitgap import (
    CADREF_COMMIT,
    CADREF_SOURCE_SHA256,
    LOGITGAP_COMMIT,
    LOGITGAP_SOURCE_SHA256,
    LOGITGAP_TOPN,
    METHODS,
    _cadref_confidence,
    _logitgap_confidence,
    _raw_class_means,
    rebuild_combined,
)
from run_ctm_baseline import TARGETS, _scheduled_runs


def test_protocol_matrix_has_22_method_runs():
    model_runs = _scheduled_runs(TARGETS, [0, 1, 2])
    assert len(model_runs) == 11
    assert len(model_runs) * len(METHODS) == 22
    assert LOGITGAP_TOPN == {10: 5, 100: 20, 200: 40, 1000: 200}


def test_pinned_official_provenance_is_complete():
    assert len(CADREF_COMMIT) == len(LOGITGAP_COMMIT) == 40
    assert len(CADREF_SOURCE_SHA256) == len(LOGITGAP_SOURCE_SHA256) == 64


def test_raw_class_means_are_not_normalised():
    sums = torch.tensor([[10.0, 2.0], [0.0, 6.0]], dtype=torch.float64)
    counts = torch.tensor([2, 3])
    means = _raw_class_means(sums, counts)
    assert torch.allclose(means, torch.tensor([[5.0, 1.0], [0.0, 2.0]]))
    assert not torch.allclose(means.norm(dim=1), torch.ones(2))


def test_raw_class_means_reject_missing_class():
    with pytest.raises(RuntimeError, match="missing classes"):
        _raw_class_means(torch.zeros(2, 3), torch.tensor([1, 0]))


def test_logitgap_matches_released_topn_definition():
    logits = torch.tensor([[5.0, 4.0, 1.0, -1.0], [3.0, 2.5, 2.0, 0.0]])
    score = _logitgap_confidence(logits, topn=2)
    expected = torch.tensor([(1.0 + 4.0) / 2, (0.5 + 1.0) / 2])
    assert torch.allclose(score, expected)


def test_logitgap_rejects_invalid_topn():
    with pytest.raises(ValueError, match="topn"):
        _logitgap_confidence(torch.ones(2, 3), topn=3)


def test_cadref_matches_released_energy_algebra():
    logits = torch.tensor([[2.0, 0.0]])
    features = torch.tensor([[3.0, 1.0]])
    means = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    weight = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    global_energy = torch.tensor(2.5)
    score = _cadref_confidence(logits, features, means, weight, global_energy)
    energy = torch.logsumexp(logits, dim=1)
    # dist=[2,-1], sign=[1,-1]: both components are positive-direction
    # errors, so ep_error=3/4 and en_error=0.
    expected = -(torch.tensor([0.75]) / energy)
    assert torch.allclose(score, expected)


def test_combined_table_reads_both_methods(tmp_path):
    result_dir = tmp_path / "full/cifar10/model/seed0"
    result_dir.mkdir(parents=True)
    metrics = pd.DataFrame(
        {
            "FPR@95": [1.0, 2.0],
            "AUROC": [99.0, 98.0],
            "AUPR_IN": [99.0, 98.0],
            "AUPR_OUT": [99.0, 98.0],
            "ACC": [95.0, 95.0],
        },
        index=["nearood", "farood"],
    )
    for method in METHODS:
        metrics.to_csv(result_dir / f"{method}.csv")
        (result_dir / f"{method}.json").write_text(
            json.dumps(
                {
                    "target": "cifar10",
                    "benchmark": "cifar10",
                    "seed": 0,
                    "method": method,
                    "model_tag": "model",
                    "model_source": "checkpoint",
                }
            ),
            encoding="utf-8",
        )
    output = rebuild_combined(tmp_path, "full")
    combined = pd.read_csv(output)
    assert len(combined) == 4
    assert set(combined["Method"]) == set(METHODS)
    assert set(combined["Dataset"]) == {"nearood", "farood"}
