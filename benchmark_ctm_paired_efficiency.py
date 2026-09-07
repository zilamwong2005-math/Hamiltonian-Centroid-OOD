"""Paired GPU timing for CTM and Locked Centroid--MSP.

The benchmark measures the two online computation graphs on the same model,
preloaded input batch, CUDA device, and C-by-d state tensor.  The numerical
entries of a centroid tensor do not alter the executed kernels, so an audited
CTM direction tensor is cloned as the shape-equivalent state for the locked
score.  This makes the timing independent of archived Hamiltonian checkpoints;
no accuracy or OOD metric is computed by this script.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import torch.nn.functional as F

from benchmark_inference_efficiency import (
    CLASS_COUNTS,
    _measure,
    _upsert,
    discover_ctm_cache,
    load_ctm_cache,
)
from Imagenet_ood_experiment import add_openood_to_path
from run_ctm_baseline import _ctm_confidence, _forward_raw_feature
from run_msp_baselines import OPENOOD_ID_NAME, _model_for_benchmark


BENCHMARKS = ("cifar10", "cifar100", "imagenet200", "imagenet1k")
LOCKED_CENTROID_SAMPLES_PER_CLASS = {
    "cifar10": 80,
    "cifar100": 80,
    "imagenet200": 12,
    "imagenet1k": 12,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--openood-results-root", type=Path, default=Path("openood_pretrained")
    )
    parser.add_argument("--openood-root", type=Path, default=Path("OpenOOD"))
    parser.add_argument(
        "--ctm-root", type=Path, default=Path("results/journal/ctm")
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--geometry-weight", type=float, default=0.8)
    parser.add_argument("--tvs-version", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--weight-download-backend",
        choices=("auto", "official", "huggingface"),
        default="auto",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/journal/efficiency/efficiency.csv"),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.batch_size <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("batch size/repeats must be positive and warmup non-negative")
    if not 0.0 <= args.geometry_weight <= 1.0:
        raise ValueError("--geometry-weight must lie in [0, 1]")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    for field in (
        "data_root", "openood_results_root", "openood_root", "ctm_root",
    ):
        setattr(args, field, getattr(args, field).resolve())
    args.output = args.output.resolve()

    add_openood_to_path(args.openood_root)
    evaluator_module = importlib.import_module("openood.evaluation_api.evaluator")
    evaluator_module.data_setup = lambda *_args, **_kwargs: None
    Evaluator = evaluator_module.Evaluator

    model_args = SimpleNamespace(
        openood_results_root=args.openood_results_root,
        tvs_version=args.tvs_version,
        weight_download_backend=args.weight_download_backend,
    )
    network, preprocessor, model_tag, model_source = _model_for_benchmark(
        args.benchmark, model_args, args.seed
    )
    ctm_path = discover_ctm_cache(
        args.ctm_root, args.benchmark, args.seed, model_tag
    )
    ctm_directions, ctm_metadata, ctm_audit = load_ctm_cache(
        ctm_path,
        benchmark=args.benchmark,
        seed=args.seed,
        model_tag=model_tag,
    )

    network = network.cuda().eval()
    evaluator = Evaluator(
        network,
        id_name=OPENOOD_ID_NAME[args.benchmark],
        data_root=str(args.data_root),
        config_root=str(args.openood_root / "configs"),
        preprocessor=preprocessor,
        postprocessor_name="msp",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    train_dataset_size = len(evaluator.dataloader_dict["id"]["train"].dataset)
    batch = next(iter(evaluator.dataloader_dict["id"]["test"]))
    data = batch["data"][: args.batch_size].cuda(non_blocking=True)
    actual_batch = len(data)
    ctm_directions = ctm_directions.cuda()
    # A clone gives both methods an independently addressable, contiguous C-by-d
    # tensor while holding shape, dtype, placement, and memory cost fixed.
    locked_centroids = ctm_directions.clone()

    with torch.inference_mode():
        fixed_logits, fixed_raw_feature = _forward_raw_feature(network, data)
    if ctm_directions.shape != (
        CLASS_COUNTS[args.benchmark], fixed_raw_feature.shape[1]
    ):
        raise RuntimeError(
            f"CTM/model feature mismatch: directions={tuple(ctm_directions.shape)}, "
            f"feature={tuple(fixed_raw_feature.shape)}"
        )
    fixed_feature = F.normalize(fixed_raw_feature, dim=-1)
    state_mb = ctm_directions.numel() * ctm_directions.element_size() / 1024**2
    alpha = float(args.geometry_weight)
    centroid_samples_per_class = LOCKED_CENTROID_SAMPLES_PER_CLASS[
        args.benchmark
    ]
    centroid_estimation_samples = (
        CLASS_COUNTS[args.benchmark] * centroid_samples_per_class
    )
    # The CIFAR implementation first extracts the full training feature set and
    # then selects 80 anchors per class.  The ImageNet implementation decodes
    # only its class-balanced 12-per-class setup subset.
    locked_processed_samples = (
        train_dataset_size
        if args.benchmark in ("cifar10", "cifar100")
        else centroid_estimation_samples
    )
    # Non-degenerate constants preserve the locked score's exact affine
    # operation graph. Their values do not change its kernels or memory shape.
    msp_mean, msp_std = 0.5, 0.25
    geometry_mean, geometry_std = 0.5, 0.25

    metadata = {
        "Benchmark": args.benchmark,
        "Seed": args.seed,
        "BatchSize": actual_batch,
        "Warmup": args.warmup,
        "Repeats": args.repeats,
        "GPU": torch.cuda.get_device_name(0),
        "ModelTag": model_tag,
        "ModelSource": str(model_source),
        "TimingExcludesDataLoader": True,
        "TimingIncludesBackbone": True,
        "TimingMethod": "perf_counter with CUDA synchronization",
        "TimingStateValueIndependent": True,
        "TimingStateSource": (
            "independent contiguous clones of the audited CTM C-by-d tensor"
        ),
        "CTMCache": str(ctm_path),
        "CTMOfficialCommit": ctm_metadata.get("official_commit"),
        "CTMOfficialCTMPySHA256": ctm_metadata.get("official_ctm_py_sha256"),
        "LockedGeometryWeight": alpha,
        "LockedCentroidSamplesPerClass": centroid_samples_per_class,
    }

    def msp_end_to_end():
        logits = network(data)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return logits.softmax(1).amax(1)

    def ctm_detector_only():
        return _ctm_confidence(fixed_raw_feature, ctm_directions)

    def ctm_end_to_end():
        _, raw_feature = _forward_raw_feature(network, data)
        return _ctm_confidence(raw_feature, ctm_directions)

    def locked_detector_only():
        msp = fixed_logits.softmax(1).amax(1)
        geometry = (fixed_feature @ locked_centroids.T).amax(1)
        return (
            (1.0 - alpha) * (msp - msp_mean) / msp_std
            + alpha * (geometry - geometry_mean) / geometry_std
        )

    def locked_end_to_end():
        logits, raw_feature = _forward_raw_feature(network, data)
        msp = logits.softmax(1).amax(1)
        geometry = (
            F.normalize(raw_feature, dim=-1) @ locked_centroids.T
        ).amax(1)
        return (
            (1.0 - alpha) * (msp - msp_mean) / msp_std
            + alpha * (geometry - geometry_mean) / geometry_std
        )

    no_setup = {
        "AuxiliaryStateMB": 0.0,
        "SetupSamples": 0,
        "CentroidEstimationSamples": 0,
        "SetupSeconds": 0.0,
        "SetupProtocol": "No offline setup",
    }
    ctm_setup = {
        "AuxiliaryStateMB": state_mb,
        "SetupSamples": int(ctm_audit["processed_setup_samples"]),
        "CentroidEstimationSamples": int(
            ctm_audit["processed_setup_samples"]
        ),
        "SetupSeconds": float(ctm_audit["elapsed_seconds"]),
        "SetupProtocol": "Full ID-training pass recorded by CTM reproduction",
    }
    locked_setup = {
        "AuxiliaryStateMB": state_mb,
        "SetupSamples": locked_processed_samples,
        "CentroidEstimationSamples": centroid_estimation_samples,
        "SetupSeconds": float("nan"),
        "SetupProtocol": (
            (
                "Full ID-training feature pass followed by "
                f"{centroid_samples_per_class} anchors per class"
            )
            if args.benchmark in ("cifar10", "cifar100")
            else (
                f"{centroid_samples_per_class} class-balanced ID-training "
                "images per class"
            )
        ),
    }
    specifications = (
        ("MSP-end-to-end", msp_end_to_end, no_setup),
        ("CTM-detector-only", ctm_detector_only, ctm_setup),
        ("CTM-end-to-end", ctm_end_to_end, ctm_setup),
        ("Locked-centroid-MSP-detector-only", locked_detector_only, locked_setup),
        ("Locked-centroid-MSP-end-to-end", locked_end_to_end, locked_setup),
    )
    rows = []
    for mode, function, setup in specifications:
        print(f"[timing] {args.benchmark} | {mode}", flush=True)
        rows.append({
            **metadata,
            **setup,
            "Mode": mode,
            "TrajectorySteps": -1,
            **_measure(function, args.warmup, args.repeats, actual_batch),
        })

    _upsert(rows, args.output)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
