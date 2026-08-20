from pathlib import Path

import numpy as np

from analyze_validation_fusion_mechanism import (
    ALPHAS,
    _analyse_scores,
    _correlation,
    _raw_path,
    _standardized_gap,
)


def test_correlation_handles_constant_and_linear_scores():
    assert _correlation(np.ones(4), np.arange(4)) == 0.0
    assert np.isclose(_correlation(np.arange(4), 2 * np.arange(4)), 1.0)
    assert _standardized_gap(np.ones(4), np.ones(4)) == 0.0
    assert _standardized_gap(np.array([2.0, 3.0]), np.array([0.0, 1.0])) > 0


def test_raw_path_is_target_and_seed_scoped():
    assert _raw_path(Path("out"), "smoke", "cifar10", 2) == Path(
        "out/smoke/cifar10/seed2/validation_scores.npz"
    )


def test_score_analysis_covers_all_alphas_and_splits(monkeypatch):
    def fake_metric(method, split, id_score, ood_score, prediction, labels):
        return {
            "Method": method,
            "Split": split,
            "FPR95": 1.0,
            "AUROC": float(np.mean(id_score) - np.mean(ood_score)),
            "AUPR_IN": 2.0,
            "AUPR_OUT": 3.0,
            "IDAccuracy": float(np.mean(prediction == labels) * 100),
            "IDCount": len(id_score),
            "OODCount": len(ood_score),
        }

    monkeypatch.setattr(
        "analyze_validation_fusion_mechanism._metric", fake_metric
    )
    arrays = {
        "id_msp": np.array([0.8, 0.7, 0.9, 0.6]),
        "ood_msp": np.array([0.3, 0.4, 0.2, 0.5]),
        "id_geometry": np.array([0.7, 0.5, 0.8, 0.6]),
        "ood_geometry": np.array([0.2, 0.3, 0.1, 0.4]),
        "id_prediction": np.array([0, 1, 0, 1]),
        "id_labels": np.array([0, 1, 0, 1]),
        "id_tune": np.array([True, True, False, False]),
        "ood_tune": np.array([True, False, True, False]),
    }

    rows, statistics = _analyse_scores("cifar10", 0, arrays)

    assert len(rows) == len(ALPHAS) * 3
    assert {row["Split"] for row in rows} == {"tune", "holdout", "full"}
    assert {row["Alpha"] for row in rows} == set(ALPHAS)
    assert statistics["IDCount"] == 4
    assert statistics["OODCount"] == 4
    assert statistics["MSP_MeanGap"] > 0
    assert statistics["Geometry_MeanGap"] > 0
    assert statistics["MSP_StandardizedGap"] > 0
    assert statistics["Geometry_StandardizedGap"] > 0
