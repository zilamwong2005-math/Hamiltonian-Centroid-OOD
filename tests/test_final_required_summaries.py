import pandas as pd

from benchmark_extended_baseline_efficiency import (
    _legacy_formal_config_is_auditable,
)
from summarize_all_method_comparison import (
    REQUIRED_COMPLETE_METHODS,
    summarize as summarize_methods,
)
from summarize_efficiency_comparison import summarize as summarize_efficiency


def _method_summary_frames():
    baseline_rows = []
    component_rows = []
    for benchmark in ("cifar10", "cifar100", "imagenet200", "imagenet1k"):
        for dataset in ("nearood", "farood"):
            for method in sorted(REQUIRED_COMPLETE_METHODS):
                row = {
                    "Benchmark": benchmark,
                    "Method": method,
                    "Dataset": dataset,
                }
                for metric in ("FPR@95", "AUROC", "AUPR_IN", "AUPR_OUT", "ACC"):
                    row[f"{metric}_mean"] = 30.0 if metric == "FPR@95" else 90.0
                    row[f"{metric}_std"] = 1.0
                    row[f"{metric}_count"] = 3
                if method in {"centroid_only", "locked_fusion"}:
                    component_rows.append(row)
                else:
                    baseline_rows.append(row)
    return pd.DataFrame(baseline_rows), pd.DataFrame(component_rows)


def test_unified_summary_requires_and_ranks_all_eight_tasks():
    baseline, component = _method_summary_frames()
    frame, ranks, _, _, _ = summarize_methods(baseline, component)
    assert frame["Task"].nunique() == 8
    complete = ranks[ranks["FullEightTaskCoverage"]]
    assert set(complete["Method"]) == REQUIRED_COMPLETE_METHODS
    assert complete["TaskCount"].eq(8).all()


def test_efficiency_summary_has_complete_batch_one_matrix():
    common = {
        "Seed": 0,
        "BatchSize": 1,
        "GPU": "test-gpu",
        "AuxiliaryStateMB": 1.0,
        "MeanMilliseconds": 2.0,
        "StdMilliseconds": 0.1,
        "MedianMilliseconds": 2.0,
        "P95Milliseconds": 2.2,
        "ImagesPerSecond": 500.0,
        "PeakAllocatedMB": 100.0,
        "IncrementalPeakMB": 2.0,
    }
    current_rows = []
    extended_rows = []
    for benchmark in ("cifar10", "cifar100", "imagenet200", "imagenet1k"):
        for mode in (
            "MSP-end-to-end", "Locked-centroid-MSP-end-to-end",
            "Centroid-detector-only",
        ):
            current_rows.append({**common, "Benchmark": benchmark, "Mode": mode})
        for method in ("ash", "dice", "she", "rmds", "rankfeat", "scale"):
            extended_rows.append({
                **common,
                "Benchmark": benchmark,
                "Method": method,
                "SetupSeconds": 0.0,
                "SetupProtocol": "test",
                "FullSVD": method == "rankfeat",
            })
    summary = summarize_efficiency(
        pd.DataFrame(current_rows), pd.DataFrame(extended_rows), 1
    )
    assert len(summary) == 36
    assert summary.groupby("Benchmark")["Method"].nunique().eq(9).all()
    assert summary.loc[
        summary["Method"].eq("MSP-end-to-end"), "LatencyRatioVsMSP"
    ].eq(1.0).all()


def test_only_complete_legacy_scale_config_is_accepted(tmp_path):
    result = tmp_path / "scale.csv"
    config = tmp_path / "scale.json"
    pd.DataFrame({
        "Dataset": ["nearood", "farood"],
        "AUROC": [80.0, 90.0],
    }).to_csv(result, index=False)
    metadata = {"protocol": "OpenOOD v1.5 local reproduction"}
    assert _legacy_formal_config_is_auditable(
        result, config, metadata, "scale"
    )
    assert not _legacy_formal_config_is_auditable(
        result, config, metadata, "ash"
    )
    metadata["max_eval_samples"] = 128
    assert not _legacy_formal_config_is_auditable(
        result, config, metadata, "scale"
    )
