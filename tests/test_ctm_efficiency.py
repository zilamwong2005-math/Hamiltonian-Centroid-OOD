from pathlib import Path

import pytest
import pandas as pd
import torch

from benchmark_inference_efficiency import (
    _ctm_target,
    discover_ctm_cache,
    load_ctm_cache,
)
from summarize_efficiency_comparison import paired_ctm_summary


def _write_cache(
    path: Path,
    *,
    benchmark: str = "cifar10",
    target: str = "cifar10",
    seed: int = 0,
    model_tag: str = "model",
    stage: str = "full",
    all_train: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = torch.full((10,), 5, dtype=torch.long)
    torch.save(
        {
            "metadata": {
                "method": "CTM",
                "stage": stage,
                "target": target,
                "benchmark": benchmark,
                "seed": seed,
                "model_tag": model_tag,
                "n_classes": 10,
                "setup_samples_per_class": 0,
            },
            "class_directions": torch.nn.functional.normalize(
                torch.randn(10, 8), dim=1
            ),
            "class_counts": counts,
            "audit": {
                "all_id_train_samples": all_train,
                "processed_setup_samples": 50,
                "train_dataset_size": 50,
                "elapsed_seconds": 1.25,
            },
        },
        path,
    )


def test_ctm_target_keeps_backbone_specific_imagenet_name():
    assert _ctm_target("cifar100") == "cifar100"
    assert _ctm_target("imagenet1k") == "imagenet1k_resnet50"


def test_discover_and_load_audited_full_ctm_cache(tmp_path: Path):
    path = (
        tmp_path
        / "centroids/full/cifar10/model"
        / "seed0_class_means.pt"
    )
    _write_cache(path)
    assert discover_ctm_cache(tmp_path, "cifar10", 0, "model") == path
    directions, metadata, audit = load_ctm_cache(
        path, benchmark="cifar10", seed=0, model_tag="model"
    )
    assert directions.shape == (10, 8)
    assert metadata["stage"] == "full"
    assert audit["processed_setup_samples"] == 50


def test_smoke_ctm_cache_is_rejected(tmp_path: Path):
    path = tmp_path / "smoke.pt"
    _write_cache(path, stage="smoke", all_train=False)
    with pytest.raises(RuntimeError, match="stage"):
        load_ctm_cache(
            path, benchmark="cifar10", seed=0, model_tag="model"
        )


def test_paired_summary_reports_latency_and_setup_ratios():
    rows = []
    for benchmark, classes in (
        ("cifar10", 10),
        ("cifar100", 100),
        ("imagenet200", 200),
        ("imagenet1k", 1000),
    ):
        common = {
            "Benchmark": benchmark,
            "StdMilliseconds": 0.1,
            "AuxiliaryStateMB": 1.0,
            "PeakAllocatedMB": 100.0,
            "SetupSeconds": 10.0,
        }
        rows.append({
            **common,
            "Method": "CTM-end-to-end",
            "MeanMilliseconds": 2.0,
            "SetupSamples": classes * 100,
            "CentroidEstimationSamples": classes * 100,
        })
        rows.append({
            **common,
            "Method": "Locked-centroid-MSP-end-to-end",
            "MeanMilliseconds": 2.2,
            "SetupSamples": classes * 12,
            "CentroidEstimationSamples": classes * 12,
        })
    result = paired_ctm_summary(pd.DataFrame(rows))
    assert len(result) == 4
    assert result["LockedLatencyRatioVsCTM"].round(6).eq(1.1).all()
    assert result["CTMSetupSampleRatioVsLocked"].round(6).eq(
        round(100 / 12, 6)
    ).all()
    assert result["CTMCentroidEstimationSampleRatioVsLocked"].round(6).eq(
        round(100 / 12, 6)
    ).all()
