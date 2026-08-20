"""Evaluate the centroid-only endpoint of the locked centroid/MSP rule.

This is a pre-specified component ablation, not a new hyperparameter search:
the geometry rule, centroid banks, ID-only calibration split, models and data
protocol are inherited unchanged, while the fixed fusion weight is set to the
component endpoint alpha=1.0.  MSP (alpha=0.0) and the locked alpha=0.8 results
already exist and are not re-run by this program.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import pandas as pd
import torch

from run_densenet121_generalization import (
    WEIGHT_SHA256,
    _build_model as _build_densenet,
    _download_pinned_weights,
    _make_locked_postprocessor as _make_densenet_postprocessor,
)
from run_locked_centroid_transfer import (
    _load_detector as _load_transfer_detector,
)
from run_locked_imagenet1k_fusion import (
    _default_decision_path,
    _load_detector as _load_imagenet1k_detector,
    _load_locked_decision,
    _make_locked_postprocessor,
)
from run_msp_baselines import (
    OPENOOD_ID_NAME,
    _batch_size_for,
    _model_for_benchmark,
)


TARGETS = (
    "cifar10",
    "cifar100",
    "imagenet200",
    "imagenet1k_resnet50",
    "imagenet1k_densenet121",
)
BASE_BENCHMARK = {
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "imagenet200": "imagenet200",
    "imagenet1k_resnet50": "imagenet1k",
    "imagenet1k_densenet121": "imagenet1k",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("smoke", "full"))
    parser.add_argument("--targets", nargs="+", choices=TARGETS,
                        default=list(TARGETS))
    parser.add_argument("--openood-root", type=Path, default=Path("OpenOOD"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--openood-results-root", type=Path,
                        default=Path("openood_pretrained"))
    parser.add_argument("--hamiltonian-root", type=Path,
                        default=Path("results_openood"))
    parser.add_argument("--journal-root", type=Path,
                        default=Path("results/journal"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("results/journal/component_centroid_only"))
    parser.add_argument("--densenet-cache-root", type=Path,
                        default=Path("cache/pretrained"))
    parser.add_argument("--decision-path", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--batch-size-cifar", type=int, default=256)
    parser.add_argument("--batch-size-imagenet200", type=int, default=128)
    parser.add_argument("--batch-size-imagenet1k", type=int, default=64)
    parser.add_argument("--setup-batch-size", type=int, default=128)
    parser.add_argument("--setup-samples-per-class", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=128)
    parser.add_argument("--tvs-version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--weight-download-backend",
                        choices=("auto", "official", "huggingface"),
                        default="auto")
    return parser


def _metric_path(output_root: Path, mode: str, target: str, seed: int) -> Path:
    return output_root / mode / target / f"seed{seed}" / "centroid_only.csv"


def _normalise_saved(path: Path, target: str, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    unnamed = [name for name in frame if name.startswith("Unnamed:")]
    if "Dataset" not in frame:
        if unnamed:
            frame = frame.rename(columns={unnamed[0]: "Dataset"})
        else:
            frame = frame.rename(columns={frame.columns[0]: "Dataset"})
    frame.insert(0, "Seed", seed)
    frame.insert(0, "Target", target)
    frame.insert(0, "Method", "centroid_only")
    return frame


def _rebuild_combined(output_root: Path, mode: str) -> Path:
    frames = []
    for target in TARGETS:
        for seed in (0, 1, 2):
            path = _metric_path(output_root, mode, target, seed)
            if path.is_file():
                frames.append(_normalise_saved(path, target, seed))
    if not frames:
        raise FileNotFoundError(f"No centroid-only results under {output_root / mode}")
    combined = pd.concat(frames, ignore_index=True)
    destination = output_root / f"centroid_only_{mode}_all_runs.csv"
    combined.to_csv(destination, index=False, float_format="%.6f")
    return destination


def main() -> None:
    args = build_parser().parse_args()
    for field in (
        "openood_root", "data_root", "openood_results_root",
        "hamiltonian_root", "journal_root", "output_root",
        "densenet_cache_root",
    ):
        setattr(args, field, getattr(args, field).resolve())
    args.targets = list(dict.fromkeys(args.targets))
    if sorted(set(args.seeds)) != [0, 1, 2]:
        raise ValueError("The component protocol requires seeds 0, 1, and 2")
    args.seeds = [0, 1, 2]
    if args.max_eval_samples <= 0 and args.stage == "smoke":
        raise ValueError("Smoke stage requires a positive max-eval-samples")
    if not torch.cuda.is_available():
        raise RuntimeError("Centroid-only evaluation requires CUDA")

    decision_path = (
        args.decision_path.resolve()
        if args.decision_path is not None
        else _default_decision_path(args.hamiltonian_root)
    )
    decision, decision_hash = _load_locked_decision(decision_path)
    gate_path = args.journal_root / "locked_imagenet1k_fusion/validation_gate.json"
    if not gate_path.is_file():
        raise FileNotFoundError(gate_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("decision_sha256") != decision_hash:
        raise RuntimeError("Validation gate and locked decision hashes differ")
    if not gate.get("authorised_for_one_locked_test_run", False):
        raise RuntimeError("The inherited ImageNet-1K validation gate did not pass")

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
    stratified_reservoir_indices = importlib.import_module(
        "openood_hamiltonian_postprocessor"
    ).stratified_reservoir_indices

    mode = args.stage
    evaluation_limit = args.max_eval_samples if mode == "smoke" else 0
    completion_path = args.output_root / f"centroid_only_{mode}_completed.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("parent_decision_sha256") != decision_hash:
            raise RuntimeError("Existing ablation used a different parent decision")
        print(f"Centroid-only {mode} already complete: {completion_path}")
        print(_rebuild_combined(args.output_root, mode))
        return

    dense_weights = args.densenet_cache_root / "densenet121_tv_in1k_pinned.bin"
    if "imagenet1k_densenet121" in args.targets:
        _download_pinned_weights(dense_weights)

    for target in args.targets:
        benchmark = BASE_BENCHMARK[target]
        for seed in args.seeds:
            metric_path = _metric_path(args.output_root, mode, target, seed)
            config_path = metric_path.with_suffix(".json")
            if metric_path.is_file() or config_path.is_file():
                if not (metric_path.is_file() and config_path.is_file()):
                    raise RuntimeError(f"Incomplete resumable result: {metric_path.parent}")
                saved = json.loads(config_path.read_text(encoding="utf-8"))
                if saved.get("parent_decision_sha256") != decision_hash:
                    raise RuntimeError(f"Decision mismatch in {config_path}")
                print(f"[resume] centroid-only {target} seed={seed}", flush=True)
                continue

            if target == "imagenet1k_densenet121":
                network, preprocessor, data_config = _build_densenet(dense_weights)
                network = network.cuda().eval()
                postprocessor = _make_densenet_postprocessor(
                    output_root=(args.journal_root / "backbone_densenet121"),
                    seed=seed,
                    setup_samples_per_class=args.setup_samples_per_class,
                    alpha=1.0,
                    calibration_seed=args.calibration_seed,
                    setup_batch_size=args.setup_batch_size,
                    num_workers=args.num_workers,
                    model_sha256=WEIGHT_SHA256,
                    forward_with_feature=forward_with_feature,
                    stratified_reservoir_indices=stratified_reservoir_indices,
                    BasePostprocessor=BasePostprocessor,
                )
                model_tag = "densenet121.tv_in1k"
                model_source = f"sha256:{WEIGHT_SHA256}"
                detector_path = (
                    args.journal_root
                    / "backbone_densenet121/centroids"
                    / f"densenet121_seed{seed}.pt"
                )
            else:
                network, preprocessor, model_tag, model_source = (
                    _model_for_benchmark(benchmark, args, seed)
                )
                network = network.cuda().eval()
                if target == "imagenet1k_resnet50":
                    detector, detector_path = _load_imagenet1k_detector(
                        args.hamiltonian_root, seed, torch.device("cuda")
                    )
                else:
                    detector, detector_path = _load_transfer_detector(
                        args, benchmark, seed
                    )
                postprocessor = _make_locked_postprocessor(
                    detector, 1.0, args.calibration_seed,
                    forward_with_feature, BasePostprocessor,
                )
                data_config = None

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
            if evaluation_limit:
                limit_evaluator_for_smoke_test(evaluator, evaluation_limit)
            print(
                f"[run] centroid-only {target} seed={seed} mode={mode}",
                flush=True,
            )
            metrics = evaluator.eval_ood(fsood=False, progress=True)
            metric_path.parent.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(metric_path, float_format="%.6f")
            config_path.write_text(json.dumps({
                "protocol": "pre-specified component endpoint; no retuning",
                "parent_decision_sha256": decision_hash,
                "parent_locked_weight": float(decision["geometry_weight"]),
                "component_weight": 1.0,
                "score_rule": decision["selected_geometry_method"],
                "calibration_seed": args.calibration_seed,
                "calibration": postprocessor.calibration,
                "target": target,
                "benchmark": benchmark,
                "seed": seed,
                "model_tag": model_tag,
                "model_source": model_source,
                "detector_or_centroid_path": str(detector_path.resolve()),
                "data_config": data_config,
                "max_eval_samples": evaluation_limit,
            }, indent=2), encoding="utf-8")
            print(f"[saved] {metric_path}", flush=True)

            del evaluator, postprocessor, network
            if target != "imagenet1k_densenet121":
                del detector
            torch.cuda.empty_cache()

    combined_path = _rebuild_combined(args.output_root, mode)
    completion_path.parent.mkdir(parents=True, exist_ok=True)
    completion_path.write_text(json.dumps({
        "protocol": "pre-specified centroid-only component endpoint",
        "parent_decision_sha256": decision_hash,
        "component_weight": 1.0,
        "targets": args.targets,
        "seeds": args.seeds,
        "max_eval_samples": evaluation_limit,
        "combined_csv": str(combined_path.resolve()),
        "completed": True,
    }, indent=2), encoding="utf-8")
    print(f"Saved centroid-only ablation: {combined_path}")


if __name__ == "__main__":
    main()
