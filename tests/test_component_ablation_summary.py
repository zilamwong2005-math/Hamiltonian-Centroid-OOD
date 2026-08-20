import pandas as pd

from summarize_component_ablation import (
    DATASET_ORDER,
    METRICS,
    TARGET_ORDER,
    _deltas,
    _validate,
)


def _synthetic_runs():
    rows = []
    values = {
        "msp": (80.0, 50.0),
        "centroid_only": (85.0, 40.0),
        "locked_fusion": (90.0, 30.0),
    }
    for target in TARGET_ORDER:
        for method, (auroc, fpr) in values.items():
            seeds = (
                (0,)
                if method == "msp" and target.startswith("imagenet1k_")
                else (0, 1, 2)
            )
            for seed in seeds:
                for dataset in DATASET_ORDER:
                    row = {
                        "Target": target,
                        "Method": method,
                        "Seed": seed,
                        "Dataset": dataset,
                        "AUROC": auroc,
                        "FPR@95": fpr,
                    }
                    for metric in METRICS:
                        row.setdefault(metric, 75.0)
                    rows.append(row)
    return pd.DataFrame(rows)


def test_component_summary_validates_expected_seed_protocol():
    counts = _validate(_synthetic_runs())
    assert counts.sum() == 82


def test_synergy_requires_beating_both_components_on_both_metrics():
    deltas = _deltas(_synthetic_runs())
    assert len(deltas) == 10
    assert deltas["FusionStrictlyBest"].all()
    assert (deltas["Fusion_minus_MSP_AUROC"] == 10.0).all()
    assert (deltas["Fusion_minus_Centroid_FPR95"] == -10.0).all()
