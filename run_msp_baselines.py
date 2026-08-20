"""Reproduce MSP baselines with the exact local OpenOOD v1.5 protocol.

The script deliberately uses the same data root, model definitions, weights,
preprocessing and :class:`openood.evaluation_api.Evaluator` as the Hamiltonian
experiments.  This avoids comparing new runs with numbers copied from an older
OpenOOD paper/protocol.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch

from Imagenet_ood_experiment import (
    add_openood_to_path,
    build_network,
)
from ood_experiment import get_backbone, load_openood_encoder_direct
from openood_cifar import discover_cifar_checkpoint


BENCHMARKS = ("cifar10", "cifar100", "imagenet200", "imagenet1k")
NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
}
OPENOOD_ID_NAME = {
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "imagenet200": "imagenet200",
    "imagenet1k": "imagenet",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=BENCHMARKS,
        default=list(BENCHMARKS),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--openood-results-root",
        type=Path,
        default=Path("openood_pretrained"),
        help="Root containing official CIFAR/ImageNet-200 checkpoints",
    )
    parser.add_argument("--openood-root", type=Path, default=Path("OpenOOD"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("results_openood_baselines")
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--batch-size-cifar", type=int, default=256)
    parser.add_argument("--batch-size-imagenet200", type=int, default=128)
    parser.add_argument("--batch-size-imagenet1k", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--tvs-version", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--weight-download-backend",
        choices=["auto", "official", "huggingface"],
        default="auto",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _model_for_benchmark(benchmark: str, args: argparse.Namespace, seed: int):
    if benchmark in NUM_CLASSES:
        checkpoint = discover_cifar_checkpoint(
            args.openood_results_root, benchmark, seed
        )
        network = get_backbone("resnet18", NUM_CLASSES[benchmark])
        diagnostic = load_openood_encoder_direct(network, str(checkpoint))
        if diagnostic["head_missing"] or diagnostic["backbone_missing"]:
            raise RuntimeError(
                f"Official {benchmark} checkpoint did not load strictly: {checkpoint}"
            )
        return (
            network,
            None,
            f"{benchmark}_resnet18_openood_s{seed}",
            str(checkpoint.resolve()),
        )

    id_name = OPENOOD_ID_NAME[benchmark]
    network_kwargs = {
        "checkpoint": None,
        "results_root": args.openood_results_root,
        "seed": seed,
        "tvs_version": args.tvs_version,
    }
    # ``weight_download_backend`` was added after the first server bundle.
    # ImageNet-200 does not need it, and this signature check keeps the MSP
    # runner compatible with both already-uploaded and current experiment code.
    if "weight_download_backend" in inspect.signature(build_network).parameters:
        network_kwargs["weight_download_backend"] = args.weight_download_backend
    network, preprocessor, model_tag, model_source = build_network(
        id_name, **network_kwargs
    )
    return network, preprocessor, model_tag, model_source


def _target_path(
    output_root: Path, benchmark: str, model_tag: str, seed: int
) -> Path:
    # ImageNet-1K uses one deterministic torchvision classifier, so repeating
    # the same MSP inference for detector seeds 1 and 2 would add no evidence.
    if benchmark == "imagenet1k":
        return output_root / benchmark / model_tag / "msp.csv"
    return output_root / benchmark / model_tag / f"seed{seed}" / "msp.csv"


def _batch_size_for(benchmark: str, args: argparse.Namespace) -> int:
    if benchmark in NUM_CLASSES:
        return int(args.batch_size_cifar)
    if benchmark == "imagenet200":
        return int(args.batch_size_imagenet200)
    if benchmark == "imagenet1k":
        return int(args.batch_size_imagenet1k)
    raise ValueError(f"Unknown benchmark: {benchmark}")


def _normalise_saved_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    unnamed = [column for column in frame.columns if column.startswith("Unnamed:")]
    if unnamed:
        frame = frame.rename(columns={unnamed[0]: "Dataset"})
    elif "Dataset" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "Dataset"})
    return frame


def _infer_seed(path: Path) -> int:
    for part in path.parts:
        match = re.fullmatch(r"seed(\d+)", part.lower())
        if match:
            return int(match.group(1))
    return 0


def rebuild_long_form(output_root: Path) -> Path:
    """Collect every saved MSP table into one auditable long-form CSV."""

    rows = []
    for path in sorted(output_root.rglob("msp.csv")):
        relative = path.relative_to(output_root)
        if len(relative.parts) < 3:
            continue
        benchmark = relative.parts[0].lower()
        if benchmark not in BENCHMARKS:
            continue
        config_path = path.with_name("msp_config.json")
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.is_file()
            else {}
        )
        frame = _normalise_saved_metrics(path)
        for _, metric in frame.iterrows():
            row = {
                "Benchmark": benchmark,
                "Model": config.get("model_tag", relative.parts[1]),
                "Seed": int(config.get("seed", _infer_seed(path))),
                "Dataset": str(metric["Dataset"]),
                "ModelSource": config.get("model_source", "unknown"),
                "SourceFile": str(path.resolve()),
            }
            for column in ("FPR@95", "AUROC", "AUPR_IN", "AUPR_OUT", "ACC"):
                if column in metric:
                    row[column] = float(metric[column])
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No msp.csv files found under {output_root}")
    combined = pd.DataFrame(rows).sort_values(
        ["Benchmark", "Seed", "Dataset"]
    )
    destination = output_root / "msp_all_runs.csv"
    combined.to_csv(destination, index=False, float_format="%.6f")
    return destination


def main() -> None:
    args = build_parser().parse_args()
    args.data_root = args.data_root.resolve()
    args.openood_results_root = args.openood_results_root.resolve()
    args.openood_root = args.openood_root.resolve()
    args.output_root = args.output_root.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("OpenOOD Evaluator requires a CUDA GPU")
    if not args.openood_root.joinpath("openood", "evaluation_api").is_dir():
        raise FileNotFoundError(f"Invalid OpenOOD root: {args.openood_root}")

    add_openood_to_path(args.openood_root)
    evaluator_module = importlib.import_module("openood.evaluation_api.evaluator")
    # Data was already prepared and verified.  Suppress the API's legacy
    # Google-Drive downloader, which would also request unused CSID archives.
    evaluator_module.data_setup = lambda *_args, **_kwargs: None
    Evaluator = evaluator_module.Evaluator

    for benchmark in args.benchmarks:
        benchmark_seeds = [args.seeds[0]] if benchmark == "imagenet1k" else args.seeds
        if benchmark == "imagenet1k" and len(args.seeds) > 1:
            print(
                "[info] ImageNet-1K MSP is deterministic for the shared torchvision "
                f"classifier; evaluating seed {benchmark_seeds[0]} only."
            )
        for seed in benchmark_seeds:
            network, preprocessor, model_tag, model_source = _model_for_benchmark(
                benchmark, args, seed
            )
            target = _target_path(args.output_root, benchmark, model_tag, seed)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and not args.force:
                print(f"[skip] Existing MSP result: {target}")
                continue

            network = network.cuda().eval()
            batch_size = _batch_size_for(benchmark, args)
            print(
                f"[run] MSP | {benchmark} | seed={seed} | "
                f"batch_size={batch_size}",
                flush=True,
            )
            evaluator = Evaluator(
                network,
                id_name=OPENOOD_ID_NAME[benchmark],
                data_root=str(args.data_root),
                config_root=str(args.openood_root / "configs"),
                preprocessor=preprocessor,
                postprocessor_name="msp",
                batch_size=batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            metrics = evaluator.eval_ood(
                fsood=False, progress=not args.no_progress
            )
            metrics.to_csv(target, float_format="%.6f")
            target.with_name("msp_config.json").write_text(
                json.dumps(
                    {
                        "benchmark": benchmark,
                        "seed": seed,
                        "model_tag": model_tag,
                        "model_source": model_source,
                        "postprocessor": "msp",
                        "protocol": "OpenOOD v1.5",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[saved] {target}", flush=True)
            del evaluator, network
            torch.cuda.empty_cache()

    combined = rebuild_long_form(args.output_root)
    print(f"[saved] {combined}")


if __name__ == "__main__":
    main()
