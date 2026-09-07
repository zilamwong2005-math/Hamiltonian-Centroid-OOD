import json

import pandas as pd
import pytest
import torch

from run_ctm_baseline import (
    EXPECTED_METRIC_ROWS,
    EXPECTED_FORMAL_RUNS,
    TARGETS,
    _accumulate_class_sums,
    _ctm_confidence,
    _normalised_class_means,
    _scheduled_runs,
    rebuild_combined,
)


def test_formal_schedule_has_eleven_independent_models():
    runs = _scheduled_runs(TARGETS, [0, 1, 2])
    assert len(runs) == EXPECTED_FORMAL_RUNS == 11
    assert [run for run in runs if run[0] == "imagenet1k_resnet50"] == [
        ("imagenet1k_resnet50", 0)
    ]
    assert [run for run in runs if run[0] == "imagenet1k_densenet121"] == [
        ("imagenet1k_densenet121", 0)
    ]
    assert EXPECTED_METRIC_ROWS == {
        "cifar10": 8,
        "cifar100": 8,
        "imagenet200": 7,
        "imagenet1k": 7,
    }


def test_ctm_uses_raw_feature_mean_before_normalising():
    # For class zero, normalising each feature first would point along (1, 1),
    # whereas CTM's raw arithmetic mean points along (10, 1).
    features = torch.tensor([[10.0, 0.0], [0.0, 1.0], [0.0, 3.0]])
    labels = torch.tensor([0, 0, 1])
    sums = torch.zeros(2, 2, dtype=torch.float64)
    counts = torch.zeros(2, dtype=torch.long)
    _accumulate_class_sums(sums, counts, features[:2], labels[:2])
    _accumulate_class_sums(sums, counts, features[2:], labels[2:])
    means = _normalised_class_means(sums, counts)
    expected_zero = torch.nn.functional.normalize(
        torch.tensor([10.0, 1.0]), dim=0
    )
    assert torch.allclose(means[0], expected_zero, atol=1e-7)
    assert torch.allclose(means[1], torch.tensor([0.0, 1.0]))
    assert counts.tolist() == [2, 1]


def test_ctm_confidence_is_maximum_cosine():
    centroids = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    features = torch.tensor([[3.0, 4.0], [-2.0, 0.0]])
    scores = _ctm_confidence(features, centroids)
    assert torch.allclose(scores, torch.tensor([0.8, 0.0]), atol=1e-7)


def test_missing_class_is_rejected():
    with pytest.raises(RuntimeError, match="missing classes"):
        _normalised_class_means(
            torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64),
            torch.tensor([1, 0]),
        )


def test_combined_table_reads_audited_result_pair(tmp_path):
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
    metrics.to_csv(result_dir / "ctm.csv")
    (result_dir / "ctm.json").write_text(
        json.dumps(
            {
                "target": "cifar10",
                "benchmark": "cifar10",
                "seed": 0,
                "model_tag": "model",
                "model_source": "checkpoint",
            }
        ),
        encoding="utf-8",
    )
    output = rebuild_combined(tmp_path, "full")
    combined = pd.read_csv(output)
    assert len(combined) == 2
    assert set(combined["Dataset"]) == {"nearood", "farood"}
    assert set(combined["Method"]) == {"ctm"}
