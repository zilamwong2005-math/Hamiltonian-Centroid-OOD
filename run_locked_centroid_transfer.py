"""Zero-shot transfer of the locked ImageNet-1K centroid/MSP score rule.

The geometry weight (0.8 in the accepted decision) is never re-tuned.  The
program applies the exact same rule to CIFAR-10, CIFAR-100 and ImageNet-200.
Only the mean and standard deviation of each component are calibrated from a
class-stratified half of the official ID validation split; no OOD validation or
test score is used for calibration.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import pandas as pd
import torch

from hamiltonian_detector import HamiltonianDetector, load_torch_checkpoint
from run_locked_imagenet1k_fusion import (
    _default_decision_path,
    _load_locked_decision,
    _make_locked_postprocessor,
)
from run_msp_baselines import (
    OPENOOD_ID_NAME,
    _batch_size_for,
    _model_for_benchmark,
)


BENCHMARKS = ("cifar10", "cifar100", "imagenet200")
EXPECTED_SHAPES = {
    "cifar10": (10, 512),
    "cifar100": (100, 512),
    "imagenet200": (200, 512),
}
IMAGE_SOURCE_EXPERIMENT = (
    "mass-uniform-none_loss-static-ts0_pred-backbone_eval-full"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS,
                        default=list(BENCHMARKS))
    parser.add_argument("--openood-root", type=Path, default=Path("OpenOOD"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--openood-results-root", type=Path,
                        default=Path("openood_pretrained"))
    parser.add_argument("--hamiltonian-root", type=Path,
                        default=Path("results_openood"))
    parser.add_argument("--journal-root", type=Path,
                        default=Path("results/journal"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("results/journal/locked_centroid_transfer"))
    parser.add_argument("--decision-path", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--batch-size-cifar", type=int, default=256)
    parser.add_argument("--batch-size-imagenet200", type=int, default=128)
    parser.add_argument("--batch-size-imagenet1k", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--tvs-version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--weight-download-backend",
                        choices=("auto", "official", "huggingface"),
                        default="auto")
    return parser


def _find_cifar_detector(journal_root: Path, benchmark: str, seed: int) -> Path:
    directory = journal_root / "cifar_trajectory" / "checkpoints"
    prefix = (
        f"{benchmark}_resnet18_openood_official_s{seed}_gaussian_"
        "mass-uniform-none_loss-static-ts0_"
    )
    candidates = sorted(directory.glob(f"{prefix}*_detector.pt"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one {benchmark} seed-{seed} T=0 detector under {directory}; "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def _find_imagenet200_detector(
    hamiltonian_root: Path, seed: int
) -> Path:
    directory = (
        hamiltonian_root
        / "imagenet200"
        / f"imagenet200_resnet18_s{seed}"
        / f"seed{seed}"
        / IMAGE_SOURCE_EXPERIMENT
    )
    candidates = [
        path for path in sorted(directory.glob("detector_*_gaussian_*.pt"))
        if "_mass-uniform-none_" in path.name
        and "_loss-static-ts0_" in path.name
        and "_t0_" in path.name
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one ImageNet-200 seed-{seed} T=0 detector under "
            f"{directory}; found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def _load_detector(args, benchmark: str, seed: int):
    path = (
        _find_cifar_detector(args.journal_root, benchmark, seed)
        if benchmark in ("cifar10", "cifar100")
        else _find_imagenet200_detector(args.hamiltonian_root, seed)
    )
    checkpoint = load_torch_checkpoint(path, map_location="cpu")
    detector = HamiltonianDetector.from_checkpoint(checkpoint).cuda().eval()
    expected_classes, expected_dim = EXPECTED_SHAPES[benchmark]
    if (detector.n_classes, detector.feat_dim) != (
        expected_classes, expected_dim
    ):
        raise RuntimeError(
            f"Unexpected {benchmark} detector shape: "
            f"{detector.n_classes} classes, {detector.feat_dim} dimensions"
        )
    return detector, path


def _normalise_metric_frame(metrics: pd.DataFrame, benchmark: str, seed: int):
    frame = metrics.reset_index()
    if "Dataset" not in frame:
        frame = frame.rename(columns={frame.columns[0]: "Dataset"})
    frame.insert(0, "Seed", seed)
    frame.insert(0, "Benchmark", benchmark)
    return frame


def _rebuild_combined(output_root: Path, mode: str) -> Path:
    frames = []
    for path in sorted((output_root / mode).rglob("locked_centroid_msp.csv")):
        relative = path.relative_to(output_root / mode)
        if len(relative.parts) < 3:
            continue
        benchmark = relative.parts[0]
        seed_part = relative.parts[1]
        if benchmark not in BENCHMARKS or not seed_part.startswith("seed"):
            continue
        seed = int(seed_part[4:])
        frame = pd.read_csv(path)
        unnamed = [name for name in frame if name.startswith("Unnamed:")]
        if unnamed:
            frame = frame.rename(columns={unnamed[0]: "Dataset"})
        elif "Dataset" not in frame:
            frame = frame.rename(columns={frame.columns[0]: "Dataset"})
        frame.insert(0, "Seed", seed)
        frame.insert(0, "Benchmark", benchmark)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No locked transfer results under {output_root / mode}")
    combined = pd.concat(frames, ignore_index=True)
    destination = output_root / f"locked_transfer_{mode}_all_runs.csv"
    combined.to_csv(destination, index=False, float_format="%.6f")
    return destination


def main() -> None:
    args = build_parser().parse_args()
    for field in (
        "openood_root", "data_root", "openood_results_root",
        "hamiltonian_root", "journal_root", "output_root",
    ):
        setattr(args, field, getattr(args, field).resolve())
    args.benchmarks = list(dict.fromkeys(args.benchmarks))
    if sorted(set(args.seeds)) != [0, 1, 2]:
        raise ValueError("The transfer protocol requires seeds 0, 1, and 2")
    args.seeds = [0, 1, 2]
    if args.max_eval_samples < 0:
        raise ValueError("max_eval_samples must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("Locked transfer evaluation requires CUDA")

    decision_path = (
        args.decision_path.resolve()
        if args.decision_path is not None
        else _default_decision_path(args.hamiltonian_root)
    )
    decision, decision_hash = _load_locked_decision(decision_path)
    gate_path = (
        args.journal_root
        / "locked_imagenet1k_fusion"
        / "validation_gate.json"
    )
    if not gate_path.is_file():
        raise FileNotFoundError(f"ImageNet-1K validation gate missing: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("decision_sha256") != decision_hash:
        raise RuntimeError("Validation gate and locked decision hashes differ")
    if not gate.get("authorised_for_one_locked_test_run", False):
        raise RuntimeError("Locked ImageNet-1K validation gate did not pass")

    from Imagenet_ood_experiment import (
        add_openood_to_path,
        limit_evaluator_for_smoke_test,
    )

    add_openood_to_path(args.openood_root)
    evaluator_module = importlib.import_module("openood.evaluation_api.evaluator")
    evaluator_module.data_setup = lambda *_args, **_kwargs: None
    Evaluator = evaluator_module.Evaluator
    BasePostprocessor = importlib.import_module(
        "openood.postprocessors.base_postprocessor"
    ).BasePostprocessor
    forward_with_feature = importlib.import_module(
        "openood_hamiltonian_postprocessor"
    )._forward_with_feature

    mode = "smoke" if args.max_eval_samples else "full"
    completion_path = args.output_root / f"locked_transfer_{mode}_completed.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("decision_sha256") != decision_hash:
            raise RuntimeError("Existing transfer used a different locked decision")
        print(f"Locked transfer already complete: {completion_path}")
        print(_rebuild_combined(args.output_root, mode))
        return

    alpha = float(decision["geometry_weight"])
    for benchmark in args.benchmarks:
        for seed in args.seeds:
            target_dir = args.output_root / mode / benchmark / f"seed{seed}"
            metric_path = target_dir / "locked_centroid_msp.csv"
            config_path = target_dir / "locked_centroid_msp.json"
            if metric_path.is_file() or config_path.is_file():
                if not (metric_path.is_file() and config_path.is_file()):
                    raise RuntimeError(f"Incomplete resumable result: {target_dir}")
                config = json.loads(config_path.read_text(encoding="utf-8"))
                if config.get("decision_sha256") != decision_hash:
                    raise RuntimeError(f"Decision mismatch in {config_path}")
                print(f"[resume] {benchmark} seed={seed}", flush=True)
                continue

            network, preprocessor, model_tag, model_source = _model_for_benchmark(
                benchmark, args, seed
            )
            network = network.cuda().eval()
            detector, detector_path = _load_detector(args, benchmark, seed)
            postprocessor = _make_locked_postprocessor(
                detector, alpha, args.calibration_seed,
                forward_with_feature, BasePostprocessor,
            )
            evaluator = Evaluator(
                network,
                id_name=OPENOOD_ID_NAME[benchmark],
                data_root=str(args.data_root),
                config_root=str(args.openood_root / "configs"),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                batch_size=_batch_size_for(benchmark, args),
                shuffle=False,
                num_workers=args.num_workers,
            )
            if args.max_eval_samples:
                limit_evaluator_for_smoke_test(evaluator, args.max_eval_samples)
            print(
                f"[locked transfer] {benchmark} seed={seed} mode={mode}",
                flush=True,
            )
            metrics = evaluator.eval_ood(fsood=False, progress=True)
            target_dir.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(metric_path, float_format="%.6f")
            config_path.write_text(json.dumps({
                "protocol": "zero-shot transfer of locked ImageNet-1K rule",
                "decision_sha256": decision_hash,
                "score_rule": decision["selected_geometry_method"],
                "geometry_weight": alpha,
                "calibration": postprocessor.calibration,
                "calibration_seed": args.calibration_seed,
                "benchmark": benchmark,
                "seed": seed,
                "model_tag": model_tag,
                "model_source": model_source,
                "detector_path": str(detector_path.resolve()),
                "max_eval_samples": args.max_eval_samples,
            }, indent=2), encoding="utf-8")
            print(f"[saved] {metric_path}", flush=True)
            del evaluator, postprocessor, detector, network
            torch.cuda.empty_cache()

    combined_path = _rebuild_combined(args.output_root, mode)
    completion_path.parent.mkdir(parents=True, exist_ok=True)
    completion_path.write_text(json.dumps({
        "decision_sha256": decision_hash,
        "benchmarks": args.benchmarks,
        "seeds": args.seeds,
        "max_eval_samples": args.max_eval_samples,
        "combined_csv": str(combined_path.resolve()),
        "completed": True,
    }, indent=2), encoding="utf-8")
    print(f"Saved locked transfer results: {combined_path}")


if __name__ == "__main__":
    main()
