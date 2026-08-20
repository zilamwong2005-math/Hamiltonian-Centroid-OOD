import pandas as pd

from summarize_validation_fusion_mechanism import (
    _best_alpha,
    _endpoint_deltas,
)


def _summary_for_one_target():
    return pd.DataFrame([
        {
            "Target": "cifar10",
            "ClassCount": 10,
            "Alpha": alpha,
            "AUROC_mean": auroc,
            "FPR95_mean": fpr,
        }
        for alpha, auroc, fpr in (
            (0.0, 80.0, 50.0),
            (0.8, 90.0, 30.0),
            (1.0, 88.0, 35.0),
        )
    ])


def test_best_alpha_uses_auroc_then_fpr95(monkeypatch):
    monkeypatch.setattr(
        "summarize_validation_fusion_mechanism.TARGET_ORDER", ("cifar10",)
    )
    result = _best_alpha(_summary_for_one_target())
    assert result.loc[0, "BestAlphaByMeanAUROC"] == 0.8


def test_endpoint_delta_detects_strict_synergy(monkeypatch):
    monkeypatch.setattr(
        "summarize_validation_fusion_mechanism.TARGET_ORDER", ("cifar10",)
    )
    result = _endpoint_deltas(_summary_for_one_target())
    assert bool(result.loc[0, "LockedStrictlyBest"])
    assert result.loc[0, "LockedMinusMSP_AUROC"] == 10.0
    assert result.loc[0, "LockedMinusCentroid_FPR95"] == -5.0
