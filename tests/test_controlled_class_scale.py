import numpy as np
import pandas as pd
import torch

import run_controlled_class_scale as module


def test_class_subsets_are_deterministic_nested_and_seeded():
    small = module._classes_for_k(0, 25)
    large = module._classes_for_k(0, 100)
    repeated = module._classes_for_k(0, 25)
    other = module._classes_for_k(1, 25)
    assert np.array_equal(small, repeated)
    assert set(small).issubset(set(large))
    assert not np.array_equal(small, other)


def test_local_labels_follow_selected_logit_columns():
    labels = np.array([0, 2, 4, 2, 1])
    selected, local = module._local_id_labels(labels, np.array([1, 2, 4]))
    assert selected.tolist() == [False, True, True, True, True]
    assert local.tolist() == [1, 2, 1, 0]


def test_analyse_configuration_covers_alpha_grid_and_splits(monkeypatch):
    def fake_metric(method, split, id_score, ood_score, prediction, labels):
        return {
            "Method": method,
            "Split": split,
            "FPR95": 10.0,
            "AUROC": 90.0,
            "AUPR_IN": 91.0,
            "AUPR_OUT": 89.0,
            "IDAccuracy": 80.0,
            "IDCount": len(id_score),
            "OODCount": len(ood_score),
        }

    monkeypatch.setattr(module, "_metric", fake_metric)
    id_logits = torch.tensor([
        [4.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 3.0, 0.0],
        [3.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 4.0, 0.0],
    ])
    ood_logits = torch.tensor([
        [1.0, 0.0, 1.0, 0.0],
        [0.5, 0.0, 0.7, 0.0],
    ])
    id_similarity = torch.tensor([
        [0.9, 0.0, 0.2, 0.0],
        [0.2, 0.0, 0.8, 0.0],
        [0.8, 0.0, 0.1, 0.0],
        [0.1, 0.0, 0.9, 0.0],
    ])
    ood_similarity = torch.tensor([
        [0.3, 0.0, 0.2, 0.0],
        [0.4, 0.0, 0.3, 0.0],
    ])
    monkeypatch.setattr(module, "EXPECTED_CLASSES", 4)
    rows, statistics = module._analyse_configuration(
        trial_seed=0,
        class_count=2,
        classes=np.array([0, 2]),
        id_logits=id_logits,
        ood_logits=ood_logits,
        id_labels=np.array([0, 2, 0, 2]),
        id_similarity=id_similarity,
        ood_similarity=ood_similarity,
        calibration_seed=0,
    )
    assert len(rows) == len(module.ALPHAS) * 3
    assert {row["Split"] for row in rows} == {"tune", "holdout", "full"}
    assert statistics["IDCount"] == 4
    assert statistics["OODCount"] == 2
    assert 0.0 <= statistics["OODFalseAcceptJaccard"] <= 1.0
    assert 0.0 <= statistics["OODFalseAcceptExclusiveRate"] <= 1.0


def test_trend_audit_recognises_decreasing_correlation():
    rows = []
    for seed in (0, 1, 2):
        for class_count, correlation in ((10, 0.9), (100, 0.7), (1000, 0.4)):
            rows.append({
                "TrialSeed": seed,
                "ClassCount": class_count,
                "OOD_MSP_Geometry_Correlation": correlation,
            })
    result = module._trend_audit(pd.DataFrame(rows))
    assert result["all_spearman_negative"]
    assert result["all_log_slopes_negative"]
