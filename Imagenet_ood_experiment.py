"""Hamiltonian OOD evaluation on the standard OpenOOD v1.5 ImageNet splits.

Unlike the previous script, this version delegates dataset definitions,
preprocessing, inference, and metric aggregation to OpenOOD's official
``Evaluator``.  ImageNet-200 uses OpenOOD's ResNet-18 checkpoint; ImageNet-1K
uses the torchvision ResNet-50 V1 weights used by the OpenOOD baseline script.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pickle
import random
from urllib.parse import urlparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from hamiltonian_detector import (
    BANDWIDTH_LOSSES,
    MASS_MODES,
    MASS_NORMALIZATIONS,
    POTENTIAL_NAMES,
    load_torch_checkpoint,
)
from prepare_openood import prepare_benchmarks


TORCHVISION_RESNET50_MIRRORS = {
    # Exact byte-for-byte mirror of torchvision's IMAGENET1K_V1 checkpoint.
    # The full digest starts with the eight characters embedded in the
    # official filename (resnet50-0676ba61.pth).
    1: {
        "repo": "weleen/syncs",
        "filename": "resnet50-0676ba61.pth",
        "sha256": (
            "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
        ),
    }
}


ROOT = Path(__file__).resolve().parent
STANDARD_OOD_DATASETS = (
    "imagenet_1k",
    "ssb_hard",
    "ninco",
    "inaturalist",
    "texture",
    "openimage_o",
)


def canonical_id_name(value: str) -> Tuple[str, str]:
    normalized = value.lower().replace("-", "")
    if normalized in {"imagenet200", "in200"}:
        return "imagenet200", "imagenet200"
    if normalized in {"imagenet1k", "imagenet", "in1k"}:
        return "imagenet", "imagenet1k"
    raise ValueError("--id-data must be imagenet200 or imagenet1k")


def add_openood_to_path(openood_root: Path) -> None:
    root = str(openood_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _unwrap_state_dict(raw):
    if isinstance(raw, dict):
        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "backbone",
        ):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break
    if not isinstance(raw, dict):
        raise TypeError("Checkpoint does not contain a state dictionary")
    cleaned = {}
    for key, value in raw.items():
        for prefix in ("module.", "network.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def discover_imagenet200_checkpoint(results_root: Path, seed: int) -> Path:
    candidates = []
    for path in results_root.rglob("best.ckpt"):
        lowered = str(path).lower()
        if (
            "imagenet200_resnet18_224x224_base" in lowered
            and f"s{seed}" in {part.lower() for part in path.parts}
        ):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            "Could not find the OpenOOD ImageNet-200 ResNet-18 checkpoint under "
            f"{results_root}. Run prepare_openood.py or pass --checkpoint."
        )
    return sorted(candidates, key=lambda p: (len(p.parts), str(p)))[0]


def load_torchvision_resnet50_state_dict(
    weights, tvs_version: int, download_backend: str = "auto"
):
    """Load torchvision weights, falling back to a verified HF mirror.

    ``torch.hub`` downloads to a temporary file but cannot resume a truncated
    transfer.  AutoDL connections to download.pytorch.org can therefore waste
    a long download and finish with a hash error.  The fallback uses the
    project's resumable Hugging Face downloader and only installs the mirror
    after its exact official SHA-256 digest has been verified.
    """
    if download_backend not in {"auto", "official", "huggingface"}:
        raise ValueError(f"Unknown weight download backend: {download_backend}")

    primary_error = None
    if download_backend in {"auto", "official"}:
        try:
            return weights.get_state_dict(progress=True, check_hash=True)
        except Exception as error:
            primary_error = error
            if download_backend == "official":
                raise

    configured = TORCHVISION_RESNET50_MIRRORS.get(int(tvs_version))
    if configured is None:
        if primary_error is not None:
            raise primary_error
        raise RuntimeError(
            f"No verified Hugging Face mirror is configured for V{tvs_version}"
        )

    from prepare_openood import _download_huggingface, _verify_sha256

    mirror = dict(configured)
    official_filename = Path(urlparse(weights.url).path).name
    if official_filename != mirror["filename"]:
        raise RuntimeError(
            "Configured ResNet-50 mirror does not match torchvision URL: "
            f"{mirror['filename']} != {official_filename}"
        ) from primary_error

    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / official_filename
    if checkpoint.is_file():
        try:
            cached_digest = _verify_sha256(
                checkpoint, str(configured["sha256"])
            )
        except RuntimeError:
            print(
                f"[warning] Replacing invalid cached checkpoint: {checkpoint}",
                flush=True,
            )
        else:
            print(
                f"Using verified cached torchvision ResNet-50 V{tvs_version}: "
                f"{checkpoint} (sha256={cached_digest})",
                flush=True,
            )
            return weights.get_state_dict(progress=False, check_hash=True)

    if primary_error is not None:
        print(
            "[warning] Official torchvision download failed; switching to the "
            f"resumable verified Hugging Face mirror: {primary_error}",
            flush=True,
        )
    else:
        print(
            "Using the resumable verified Hugging Face mirror for "
            f"torchvision ResNet-50 V{tvs_version}.",
            flush=True,
        )
    try:
        _download_huggingface(mirror, checkpoint)
        digest = _verify_sha256(checkpoint, str(configured["sha256"]))
    except Exception as mirror_error:
        raise RuntimeError(
            "The verified Hugging Face mirror failed. The .part file is "
            "retained for the next resumable attempt."
        ) from mirror_error

    print(
        f"Verified torchvision ResNet-50 V{tvs_version}: "
        f"{checkpoint} (sha256={digest})",
        flush=True,
    )
    return weights.get_state_dict(progress=False, check_hash=True)


def build_network(
    id_name: str,
    checkpoint: Path | None,
    results_root: Path,
    seed: int,
    tvs_version: int,
    weight_download_backend: str = "auto",
):
    from openood.networks import ResNet18_224x224, ResNet50

    if id_name == "imagenet200":
        network = ResNet18_224x224(num_classes=200)
        custom_checkpoint = checkpoint is not None
        checkpoint = checkpoint or discover_imagenet200_checkpoint(results_root, seed)
        raw = load_torch_checkpoint(checkpoint, map_location="cpu")
        state_dict = _unwrap_state_dict(raw)
        missing, unexpected = network.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"ImageNet-200 checkpoint mismatch. Missing={missing}, unexpected={unexpected}"
            )
        preprocessor = None
        if custom_checkpoint:
            checkpoint_tag = "".join(
                character if character.isalnum() else "_"
                for character in checkpoint.stem
            )
            cache_tag = (
                f"imagenet200_resnet18_custom_{checkpoint_tag}_"
                f"{checkpoint.stat().st_size}"
            )
        else:
            cache_tag = f"imagenet200_resnet18_s{seed}"
        source = str(checkpoint.resolve())
    else:
        from torchvision.models import ResNet50_Weights

        network = ResNet50(num_classes=1000)
        if checkpoint is not None:
            raw = load_torch_checkpoint(checkpoint, map_location="cpu")
            state_dict = _unwrap_state_dict(raw)
            missing, unexpected = network.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"ImageNet-1K checkpoint mismatch. Missing={missing}, unexpected={unexpected}"
                )
            preprocessor = None
            checkpoint_tag = "".join(
                character if character.isalnum() else "_"
                for character in checkpoint.stem
            )
            cache_tag = (
                f"imagenet1k_resnet50_custom_{checkpoint_tag}_"
                f"{checkpoint.stat().st_size}"
            )
            source = str(checkpoint.resolve())
        else:
            weights = (
                ResNet50_Weights.IMAGENET1K_V1
                if tvs_version == 1
                else ResNet50_Weights.IMAGENET1K_V2
            )
            # Torchvision downloads into its normal cache when absent.
            network.load_state_dict(
                load_torchvision_resnet50_state_dict(
                    weights, tvs_version, weight_download_backend
                )
            )
            preprocessor = weights.transforms()
            cache_tag = f"imagenet1k_resnet50_tvsv{tvs_version}"
            source = weights.url
    return network, preprocessor, cache_tag, source


def append_long_form(
    frames: Dict[str, pd.DataFrame],
    output_path: Path,
    id_label: str,
    model_source: str,
    run_metadata: Dict[str, object],
) -> None:
    rows = []
    for potential, frame in frames.items():
        for dataset, values in frame.iterrows():
            row = {
                "ID": id_label,
                "Potential": potential,
                "Dataset": dataset,
                "ModelSource": model_source,
                **run_metadata,
            }
            row.update({column: float(values[column]) for column in frame.columns})
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False, float_format="%.4f")


def _limit_loader(loader: DataLoader, max_samples: int) -> DataLoader:
    # Smoke-test loaders are short lived.  Keeping a persistent worker pool for
    # every ID/OOD split leaves dozens of processes alive after eval_ood() has
    # already written its results and can make interpreter shutdown appear to
    # hang.  Let each iterator tear its workers down as soon as it is exhausted.
    count = min(int(max_samples), len(loader.dataset))
    return DataLoader(
        Subset(loader.dataset, range(count)),
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )


def limit_evaluator_for_smoke_test(evaluator, max_samples: int) -> None:
    """Limit evaluated splits only; keep the balanced ID setup set intact."""
    evaluator.dataloader_dict["id"]["test"] = _limit_loader(
        evaluator.dataloader_dict["id"]["test"], max_samples
    )
    for split in ("near", "far"):
        for name, loader in list(evaluator.dataloader_dict["ood"][split].items()):
            evaluator.dataloader_dict["ood"][split][name] = _limit_loader(
                loader, max_samples
            )
    for name, loader in list(evaluator.dataloader_dict["csid"].items()):
        evaluator.dataloader_dict["csid"][name] = _limit_loader(
            loader, max_samples
        )
    print(
        f"SMOKE TEST: evaluating at most {max_samples} samples per test dataset; "
        "do not report these metrics in the paper.",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id-data", default="imagenet200")
    parser.add_argument("--openood-root", type=Path, default=ROOT / "OpenOOD")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--openood-results-root", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--archive-dir",
        type=Path,
        help="Use <archive-dir>/<download-name>.zip before trying Google Drive",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "results_openood")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--tvs-version", type=int, default=1, choices=(1, 2))
    parser.add_argument(
        "--weight-download-backend",
        choices=("auto", "official", "huggingface"),
        default="auto",
        help=(
            "ImageNet-1K torchvision weight source. Hugging Face uses an "
            "exact SHA-256-verified mirror with resumable downloads."
        ),
    )

    parser.add_argument("--potential", choices=POTENTIAL_NAMES, default="gaussian")
    parser.add_argument("--potentials", nargs="+", choices=POTENTIAL_NAMES)
    parser.add_argument("--n-anchors", type=int, default=5)
    parser.add_argument("--setup-samples-per-class", type=int, default=12)
    parser.add_argument("--ham-epochs", type=int, default=10)
    parser.add_argument("--ham-lr", type=float, default=1e-3)
    parser.add_argument("--ham-batch-size", type=int, default=32)
    parser.add_argument(
        "--bandwidth-loss", choices=BANDWIDTH_LOSSES, default="static"
    )
    parser.add_argument(
        "--trajectory-train-steps",
        type=int,
        default=0,
        help="Differentiable training steps; 0 uses --n-steps",
    )
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--sim-batch", type=int, default=32)
    parser.add_argument("--sigma-init", type=float, default=0.5)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--sigma-max", type=float, default=4.0)
    parser.add_argument("--mass-mode", choices=MASS_MODES, default="uniform")
    parser.add_argument(
        "--mass-normalization", choices=MASS_NORMALIZATIONS, default="none"
    )
    parser.add_argument(
        "--mass-resolution",
        type=int,
        default=0,
        help="Effective-rank image size; 0 keeps the 224x224 matrix",
    )
    parser.add_argument(
        "--prediction-source",
        choices=("backbone", "hamiltonian"),
        default="backbone",
        help="Use backbone predictions for standard post-hoc OOD, or Hamiltonian predictions for classifier ablation",
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--setup-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=0,
        help="Smoke-test only: limit each evaluated dataset; 0 evaluates full splits",
    )
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--save-scores", action="store_true")
    parser.add_argument("--fsood", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    id_name, benchmark_name = canonical_id_name(args.id_data)
    args.openood_root = args.openood_root.resolve()
    args.data_root = args.data_root.resolve()
    args.openood_results_root = args.openood_results_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.max_eval_samples < 0:
        raise ValueError("--max-eval-samples must be non-negative")
    if args.archive_dir:
        args.archive_dir = args.archive_dir.resolve()
    if not args.openood_root.joinpath("openood", "evaluation_api").is_dir():
        raise FileNotFoundError(f"Invalid OpenOOD root: {args.openood_root}")

    if not args.skip_download:
        prepare_benchmarks(
            [benchmark_name],
            args.data_root,
            args.openood_results_root,
            download_datasets=True,
            # ImageNet-1K uses torchvision's standard V1 checkpoint by default.
            download_checkpoints=id_name == "imagenet200" and args.checkpoint is None,
            force=args.force_download,
            archive_dir=args.archive_dir,
            # Covariate-shifted ID sets (ImageNet-V2/C/R/ES) are only read
            # by OpenOOD when full-spectrum evaluation is explicitly enabled.
            dataset_names=None if args.fsood else STANDARD_OOD_DATASETS,
        )
    if args.download_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "OpenOOD v1.5 Evaluator moves batches with .cuda(); run this experiment on a CUDA server."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    add_openood_to_path(args.openood_root)
    # We have already prepared data with our resumable, verified downloader.
    # OpenOOD's Evaluator otherwise invokes its Google-Drive-only data_setup()
    # again, including unused CSID datasets even for ordinary OOD evaluation.
    evaluator_module = importlib.import_module("openood.evaluation_api.evaluator")
    evaluator_module.data_setup = lambda *_args, **_kwargs: None
    Evaluator = evaluator_module.Evaluator
    from openood_hamiltonian_postprocessor import HamiltonianPostprocessor

    network, preprocessor, cache_tag, model_source = build_network(
        id_name,
        args.checkpoint.resolve() if args.checkpoint else None,
        args.openood_results_root,
        args.seed,
        args.tvs_version,
        args.weight_download_backend,
    )
    network = network.cuda().eval()
    n_classes = 200 if id_name == "imagenet200" else 1000
    train_steps = (
        args.n_steps if args.trajectory_train_steps == 0 else args.trajectory_train_steps
    )
    experiment_tag = (
        f"mass-{args.mass_mode}-{args.mass_normalization}"
        f"_loss-{args.bandwidth_loss}-ts{train_steps}"
        f"_pred-{args.prediction_source}"
        f"_eval-{'full' if args.max_eval_samples == 0 else args.max_eval_samples}"
    )
    run_dir = (
        args.output_root
        / benchmark_name
        / cache_tag
        / f"seed{args.seed}"
        / experiment_tag
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    potentials = list(dict.fromkeys(args.potentials or [args.potential]))

    common_config = {
        **vars(args),
        "id_name_openood": id_name,
        "benchmark_name": benchmark_name,
        "model_source": model_source,
        "potentials": potentials,
    }
    serializable_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in common_config.items()
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(serializable_config, indent=2), encoding="utf-8"
    )

    all_metrics: Dict[str, pd.DataFrame] = {}
    for potential in potentials:
        print("\n" + "=" * 78)
        print(f"OpenOOD v1.5 | {benchmark_name} | potential={potential}")
        print("=" * 78, flush=True)
        postprocessor = HamiltonianPostprocessor(
            n_classes=n_classes,
            potential=potential,
            output_dir=run_dir,
            cache_tag=cache_tag,
            n_anchors=args.n_anchors,
            setup_samples_per_class=args.setup_samples_per_class,
            ham_epochs=args.ham_epochs,
            ham_lr=args.ham_lr,
            ham_batch_size=args.ham_batch_size,
            bandwidth_loss=args.bandwidth_loss,
            trajectory_train_steps=args.trajectory_train_steps,
            n_steps=args.n_steps,
            dt=args.dt,
            candidate_k=args.candidate_k,
            sim_batch=args.sim_batch,
            setup_batch_size=args.setup_batch_size,
            num_workers=args.num_workers,
            sigma_init=args.sigma_init,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            mass_mode=args.mass_mode,
            mass_normalization=args.mass_normalization,
            mass_resolution=args.mass_resolution,
            prediction_source=args.prediction_source,
            seed=args.seed,
            force_retrain=args.force_retrain,
        )
        evaluator = Evaluator(
            network,
            id_name=id_name,
            data_root=str(args.data_root),
            config_root=str(args.openood_root / "configs"),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if args.max_eval_samples:
            limit_evaluator_for_smoke_test(evaluator, args.max_eval_samples)
        metrics = evaluator.eval_ood(
            fsood=args.fsood, progress=not args.no_progress
        )
        metrics.to_csv(
            run_dir / f"openood_metrics_{potential}.csv", float_format="%.4f"
        )
        all_metrics[potential] = metrics
        if args.save_scores:
            with (run_dir / f"openood_scores_{potential}.pkl").open("wb") as stream:
                pickle.dump(evaluator.scores, stream, pickle.HIGHEST_PROTOCOL)

    append_long_form(
        all_metrics,
        run_dir / "openood_metrics_all_potentials.csv",
        benchmark_name,
        model_source,
        {
            "Seed": args.seed,
            "MassMode": args.mass_mode,
            "MassNormalization": args.mass_normalization,
            "BandwidthLoss": args.bandwidth_loss,
            "TrajectoryTrainSteps": (
                train_steps
            ),
            "PredictionSource": args.prediction_source,
        },
    )
    print(f"\nAll results saved under: {run_dir}")


if __name__ == "__main__":
    main()
