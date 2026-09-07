"""Measure GPU latency/memory for MSP, Hamiltonian and locked centroid fusion.

Data loading is excluded: the benchmark reuses a fixed, preloaded GPU batch and
reports both detector-only cost and end-to-end network-plus-detector cost.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import torch.nn.functional as F

from Imagenet_ood_experiment import add_openood_to_path
from hamiltonian_detector import HamiltonianDetector, load_torch_checkpoint
from run_ctm_baseline import _ctm_confidence, _forward_raw_feature
from run_msp_baselines import OPENOOD_ID_NAME, _model_for_benchmark


BENCHMARKS = ("cifar10", "cifar100", "imagenet200", "imagenet1k")
CLASS_COUNTS = {"cifar10": 10, "cifar100": 100,
                "imagenet200": 200, "imagenet1k": 1000}
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
    parser.add_argument("--openood-results-root", type=Path,
                        default=Path("openood_pretrained"))
    parser.add_argument("--openood-root", type=Path, default=Path("OpenOOD"))
    parser.add_argument("--detector-checkpoint", type=Path)
    parser.add_argument("--detector-search-root", type=Path)
    parser.add_argument(
        "--locked-config",
        type=Path,
        help="locked_centroid_msp.json for this benchmark/seed",
    )
    parser.add_argument(
        "--locked-results-root",
        type=Path,
        default=Path("results/journal"),
        help="Journal root used to auto-discover locked fusion configs",
    )
    parser.add_argument(
        "--ctm-root",
        type=Path,
        help=(
            "Optional root produced by run_ctm_baseline.py. When supplied, "
            "the audited full-ID-training CTM endpoint is measured on the "
            "same fixed GPU batch as the proposed score."
        ),
    )
    parser.add_argument("--potential", default="gaussian")
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 1, 3, 10])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--tvs-version", type=int, choices=[1, 2], default=1)
    parser.add_argument("--weight-download-backend",
                        choices=["auto", "official", "huggingface"],
                        default="auto")
    parser.add_argument("--output", type=Path,
                        default=Path("results/journal/efficiency/efficiency.csv"))
    return parser


def _load_detector(path: Path) -> HamiltonianDetector:
    checkpoint = load_torch_checkpoint(path, map_location="cpu")
    return HamiltonianDetector.from_checkpoint(checkpoint)


def discover_locked_config(root: Path, benchmark: str, seed: int) -> Path:
    if benchmark == "imagenet1k":
        path = (
            root
            / "locked_imagenet1k_fusion"
            / "test"
            / f"seed{seed}"
            / "locked_centroid_msp.json"
        )
    else:
        path = (
            root
            / "locked_centroid_transfer"
            / "full"
            / benchmark
            / f"seed{seed}"
            / "locked_centroid_msp.json"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"Locked fusion config not found for {benchmark} seed {seed}: {path}"
        )
    return path


def _ctm_target(benchmark: str) -> str:
    return "imagenet1k_resnet50" if benchmark == "imagenet1k" else benchmark


def load_ctm_cache(
    path: Path,
    *,
    benchmark: str,
    seed: int,
    model_tag: str | None = None,
) -> tuple[torch.Tensor, dict, dict]:
    """Load and validate one audited full-training CTM class-direction cache."""

    payload = load_torch_checkpoint(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Malformed CTM cache: {path}")
    metadata = payload.get("metadata")
    audit = payload.get("audit")
    directions = payload.get("class_directions")
    counts = payload.get("class_counts")
    if not isinstance(metadata, dict) or not isinstance(audit, dict):
        raise RuntimeError(f"CTM cache lacks metadata/audit dictionaries: {path}")
    if not isinstance(directions, torch.Tensor) or not isinstance(counts, torch.Tensor):
        raise RuntimeError(f"CTM cache lacks class directions/counts: {path}")
    expected_target = _ctm_target(benchmark)
    checks = {
        "method": "CTM",
        "stage": "full",
        "target": expected_target,
        "benchmark": benchmark,
        "seed": int(seed),
        "n_classes": CLASS_COUNTS[benchmark],
        "setup_samples_per_class": 0,
    }
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"CTM cache metadata mismatch for {key}: expected {expected!r}, "
                f"found {metadata.get(key)!r} in {path}"
            )
    if model_tag is not None and metadata.get("model_tag") != model_tag:
        raise RuntimeError(
            f"CTM cache model mismatch: expected {model_tag!r}, "
            f"found {metadata.get('model_tag')!r} in {path}"
        )
    expected_classes = CLASS_COUNTS[benchmark]
    if directions.ndim != 2 or directions.shape[0] != expected_classes:
        raise RuntimeError(
            f"CTM directions have shape {tuple(directions.shape)}, expected "
            f"({expected_classes}, feature_dim)"
        )
    if counts.ndim != 1 or len(counts) != expected_classes or (counts <= 0).any():
        raise RuntimeError(f"Invalid CTM class counts in {path}")
    if not torch.isfinite(directions).all():
        raise RuntimeError(f"Non-finite CTM class direction in {path}")
    if not bool(audit.get("all_id_train_samples")):
        raise RuntimeError(f"Refusing non-full-training CTM cache: {path}")
    processed = int(audit.get("processed_setup_samples", -1))
    dataset_size = int(audit.get("train_dataset_size", -2))
    if processed <= 0 or processed != dataset_size or int(counts.sum()) != processed:
        raise RuntimeError(f"Incomplete CTM setup audit in {path}")
    return directions.float(), metadata, audit


def discover_ctm_cache(
    root: Path, benchmark: str, seed: int, model_tag: str | None = None
) -> Path:
    """Find the unique full-training CTM cache for a benchmark/model seed."""

    target = _ctm_target(benchmark)
    search_root = root / "centroids" / "full" / target
    candidates = sorted(search_root.rglob(f"seed{seed}_class_means.pt"))
    accepted = []
    errors = []
    for path in candidates:
        try:
            load_ctm_cache(
                path, benchmark=benchmark, seed=seed, model_tag=model_tag
            )
        except RuntimeError as error:
            errors.append(f"{path}: {error}")
        else:
            accepted.append(path)
    if len(accepted) != 1:
        detail = "\n".join(errors[:5])
        raise RuntimeError(
            f"Expected one audited full CTM cache for {benchmark} seed {seed} "
            f"under {search_root}; found {len(accepted)}."
            + (f"\nRejected candidates:\n{detail}" if detail else "")
        )
    return accepted[0]


def _state_megabytes(module: torch.nn.Module) -> float:
    tensors = list(module.parameters()) + list(module.buffers())
    return sum(t.numel() * t.element_size() for t in tensors) / 1024**2


def discover_detector(root: Path, benchmark: str, seed: int,
                      potential: str) -> Path:
    expected_classes = CLASS_COUNTS[benchmark]
    candidates = []
    for path in sorted(root.rglob("*.pt")):
        name = path.name.lower()
        if "detector" not in name or potential.lower() not in name:
            continue
        if "mass-uniform-none" not in name or "loss-static" not in name:
            continue
        if f"seed{seed}" not in name and f"_s{seed}_" not in name:
            continue
        try:
            detector = _load_detector(path)
        except Exception:
            continue
        if detector.n_classes == expected_classes and detector.potential == potential:
            candidates.append((path, detector))
    if not candidates:
        raise FileNotFoundError(
            f"No {benchmark} seed {seed} {potential} uniform/static detector "
            f"found under {root}. Pass --detector-checkpoint explicitly."
        )

    # Static bandwidth optimization is independent of trajectory length.  Use
    # the smallest-T checkpoint to minimize ambiguity, and verify that all
    # candidates have the same anchor shape before reporting the choice.
    candidates.sort(key=lambda item: (item[1].T, str(item[0])))
    chosen_path, chosen = candidates[0]
    same_shape = [
        path for path, detector in candidates
        if detector.anchors.shape == chosen.anchors.shape
    ]
    print(f"Detector candidates with matching shape: {len(same_shape)}")
    print(f"Selected detector checkpoint: {chosen_path}")
    return chosen_path


def _measure(function, warmup: int, repeats: int, batch_size: int) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            output = function()
            if not torch.isfinite(output).all():
                raise RuntimeError("Non-finite score encountered during warmup")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_memory = torch.cuda.memory_allocated()
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            output = function()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000.0)
        peak_memory = torch.cuda.max_memory_allocated()
    mean_ms = statistics.fmean(samples)
    std_ms = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "MeanMilliseconds": mean_ms,
        "StdMilliseconds": std_ms,
        "MedianMilliseconds": statistics.median(samples),
        "P95Milliseconds": sorted(samples)[max(0, int(0.95 * len(samples)) - 1)],
        "ImagesPerSecond": batch_size * 1000.0 / mean_ms,
        "PeakAllocatedMB": peak_memory / 1024**2,
        "IncrementalPeakMB": (peak_memory - baseline_memory) / 1024**2,
    }


def _upsert(rows: list[dict], path: Path) -> None:
    incoming = pd.DataFrame(rows)
    if path.is_file():
        current = pd.read_csv(path)
        incoming = pd.concat([current, incoming], ignore_index=True)
    identity = ["Benchmark", "Seed", "Mode", "TrajectorySteps", "BatchSize"]
    incoming = incoming.drop_duplicates(identity, keep="last").sort_values(identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming.to_csv(path, index=False, float_format="%.6f")


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    for field in (
        "data_root", "openood_results_root", "openood_root",
        "locked_results_root",
    ):
        setattr(args, field, getattr(args, field).resolve())
    if args.ctm_root is not None:
        args.ctm_root = args.ctm_root.resolve()
    args.output = args.output.resolve()
    if args.detector_checkpoint:
        detector_path = args.detector_checkpoint.resolve()
    else:
        if not args.detector_search_root:
            raise ValueError("Pass --detector-checkpoint or --detector-search-root")
        detector_path = discover_detector(
            args.detector_search_root.resolve(), args.benchmark,
            args.seed, args.potential,
        )
    locked_config_path = (
        args.locked_config.resolve()
        if args.locked_config is not None
        else discover_locked_config(
            args.locked_results_root, args.benchmark, args.seed
        )
    )
    locked_config = json.loads(locked_config_path.read_text(encoding="utf-8"))
    if locked_config.get("score_rule") != "centroid/max_cosine":
        raise RuntimeError(
            f"Unexpected locked score rule in {locked_config_path}: "
            f"{locked_config.get('score_rule')!r}"
        )
    alpha = float(locked_config["geometry_weight"])
    calibration = locked_config["calibration"]
    locked_detector_path = Path(locked_config["detector_path"])
    if not locked_detector_path.is_file():
        raise FileNotFoundError(
            f"Locked detector checkpoint is missing: {locked_detector_path}"
        )

    add_openood_to_path(args.openood_root)
    from openood_hamiltonian_postprocessor import _forward_with_feature

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
    ctm_path = None
    ctm_directions = None
    ctm_metadata = None
    ctm_audit = None
    if args.ctm_root is not None:
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
    detector = _load_detector(detector_path).cuda().eval()
    locked_detector = _load_detector(locked_detector_path)
    locked_centroids = locked_detector.centroids.cuda()
    locked_centroid_mb = (
        locked_centroids.numel() * locked_centroids.element_size() / 1024**2
    )
    del locked_detector
    with torch.inference_mode():
        fixed_logits, fixed_raw_feature = _forward_raw_feature(network, data)
        fixed_feature = F.normalize(fixed_raw_feature, dim=-1)
    if fixed_feature.shape[1] != detector.feat_dim:
        raise RuntimeError(
            f"Feature mismatch: model={fixed_feature.shape[1]}, "
            f"detector={detector.feat_dim}"
        )
    locked_centroid_samples = (
        CLASS_COUNTS[args.benchmark]
        * LOCKED_CENTROID_SAMPLES_PER_CLASS[args.benchmark]
    )
    if args.benchmark in {"cifar10", "cifar100"}:
        locked_setup_samples = train_dataset_size
        locked_setup_protocol = (
            "Full ID-training feature pass followed by 80 anchors per class"
        )
    else:
        locked_setup_samples = locked_centroid_samples
        locked_setup_protocol = "12 class-balanced ID-training images per class"
    ctm_setup = None
    if (
        ctm_directions is not None
        and ctm_path is not None
        and ctm_metadata is not None
        and ctm_audit is not None
    ):
        ctm_directions = ctm_directions.cuda()
        if ctm_directions.shape[1] != fixed_raw_feature.shape[1]:
            raise RuntimeError(
                f"CTM feature mismatch: cache={ctm_directions.shape[1]}, "
                f"model={fixed_raw_feature.shape[1]}"
            )
        ctm_state_mb = (
            ctm_directions.numel() * ctm_directions.element_size() / 1024**2
        )
        ctm_setup = {
            "CTMCache": str(ctm_path),
            "CTMOfficialCommit": ctm_metadata.get("official_commit"),
            "CTMOfficialCTMPySHA256": ctm_metadata.get("official_ctm_py_sha256"),
            "AuxiliaryStateMB": ctm_state_mb,
            "SetupSamples": int(ctm_audit["processed_setup_samples"]),
            "CentroidEstimationSamples": int(
                ctm_audit["processed_setup_samples"]
            ),
            "SetupSeconds": float(ctm_audit["elapsed_seconds"]),
            "SetupProtocol": "Full ID-training pass recorded by CTM reproduction",
        }

    metadata = {
        "Benchmark": args.benchmark,
        "Seed": args.seed,
        "BatchSize": actual_batch,
        "Warmup": args.warmup,
        "Repeats": args.repeats,
        "GPU": torch.cuda.get_device_name(0),
        "ModelTag": model_tag,
        "ModelSource": model_source,
        "DetectorCheckpoint": str(detector_path),
        "Potential": detector.potential,
        "AnchorsPerClass": detector.K,
        "CandidateK": detector.candidate_k,
        "TimingExcludesDataLoader": True,
        "LockedConfig": str(locked_config_path),
        "LockedDetectorCheckpoint": str(locked_detector_path),
        "LockedGeometryWeight": alpha,
        "TimingScope": "fixed preloaded GPU batch; data loading excluded",
        "TimingMethod": "perf_counter with CUDA synchronization",
    }
    rows = []

    def msp_end_to_end():
        logits = network(data)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return torch.softmax(logits, dim=1).amax(dim=1)

    rows.append({
        **metadata, "Mode": "MSP-end-to-end", "TrajectorySteps": -1,
        "AuxiliaryStateMB": 0.0, "SetupSamples": 0,
        "CentroidEstimationSamples": 0,
        "SetupSeconds": 0.0, "SetupProtocol": "No offline setup",
        **_measure(msp_end_to_end, args.warmup, args.repeats, actual_batch),
    })

    if ctm_directions is not None and ctm_setup is not None:
        def ctm_detector_only():
            return _ctm_confidence(fixed_raw_feature, ctm_directions)

        def ctm_end_to_end():
            _, raw_feature = _forward_raw_feature(network, data)
            return _ctm_confidence(raw_feature, ctm_directions)

        rows.append({
            **metadata, **ctm_setup, "Mode": "CTM-detector-only",
            "TrajectorySteps": -1,
            **_measure(ctm_detector_only, args.warmup, args.repeats, actual_batch),
        })
        rows.append({
            **metadata, **ctm_setup, "Mode": "CTM-end-to-end",
            "TrajectorySteps": -1,
            **_measure(ctm_end_to_end, args.warmup, args.repeats, actual_batch),
        })

    def centroid_only():
        return (fixed_feature @ locked_centroids.T).amax(dim=1)

    def locked_end_to_end():
        logits, feature = _forward_with_feature(network, data)
        msp = logits.softmax(1).amax(dim=1)
        geometry = (feature @ locked_centroids.T).amax(dim=1)
        return (
            (1.0 - alpha)
            * (msp - float(calibration["msp_mean"]))
            / max(float(calibration["msp_std"]), 1e-12)
            + alpha
            * (geometry - float(calibration["geometry_mean"]))
            / max(float(calibration["geometry_std"]), 1e-12)
        )

    rows.append({
        **metadata, "Mode": "Centroid-detector-only", "TrajectorySteps": -1,
        "AuxiliaryStateMB": locked_centroid_mb,
        "SetupSamples": locked_setup_samples,
        "CentroidEstimationSamples": locked_centroid_samples,
        "SetupSeconds": float("nan"),
        "SetupProtocol": locked_setup_protocol,
        **_measure(centroid_only, args.warmup, args.repeats, actual_batch),
    })
    rows.append({
        **metadata, "Mode": "Locked-centroid-MSP-end-to-end",
        "TrajectorySteps": -1,
        "AuxiliaryStateMB": locked_centroid_mb,
        "SetupSamples": locked_setup_samples,
        "CentroidEstimationSamples": locked_centroid_samples,
        "SetupSeconds": float("nan"),
        "SetupProtocol": locked_setup_protocol,
        **_measure(locked_end_to_end, args.warmup, args.repeats, actual_batch),
    })

    def locked_detector_only():
        msp = fixed_logits.softmax(1).amax(dim=1)
        geometry = (fixed_feature @ locked_centroids.T).amax(dim=1)
        return (
            (1.0 - alpha)
            * (msp - float(calibration["msp_mean"]))
            / max(float(calibration["msp_std"]), 1e-12)
            + alpha
            * (geometry - float(calibration["geometry_mean"]))
            / max(float(calibration["geometry_std"]), 1e-12)
        )

    rows.append({
        **metadata, "Mode": "Locked-centroid-MSP-detector-only",
        "TrajectorySteps": -1,
        "AuxiliaryStateMB": locked_centroid_mb,
        "SetupSamples": locked_setup_samples,
        "CentroidEstimationSamples": locked_centroid_samples,
        "SetupSeconds": float("nan"),
        "SetupProtocol": locked_setup_protocol,
        **_measure(locked_detector_only, args.warmup, args.repeats, actual_batch),
    })

    detector_state_mb = _state_megabytes(detector)

    for steps in args.steps:
        if steps < 0:
            raise ValueError("--steps values must be non-negative")
        detector.T = int(steps)

        def detector_only():
            return detector.score(fixed_feature)

        def end_to_end():
            _, feature = _forward_with_feature(network, data)
            return detector.score(feature)

        rows.append({
            **metadata, "Mode": "Hamiltonian-detector-only",
            "TrajectorySteps": steps,
            "AuxiliaryStateMB": detector_state_mb,
            **_measure(detector_only, args.warmup, args.repeats, actual_batch),
        })
        rows.append({
            **metadata, "Mode": "Hamiltonian-end-to-end",
            "TrajectorySteps": steps,
            "AuxiliaryStateMB": detector_state_mb,
            **_measure(end_to_end, args.warmup, args.repeats, actual_batch),
        })

    _upsert(rows, args.output)
    config_path = args.output.with_suffix(".json")
    config_path.write_text(json.dumps(vars(args), default=str, indent=2),
                           encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
