"""Reproduce CTM under the local OpenOOD v1.5 evaluation protocol.

CTM [Nguyen et al., 2023] forms one class direction from the arithmetic
mean of *raw* ID-training penultimate features.  Each class mean is normalised
once, each query feature is normalised once, and the OOD confidence is the
largest class-wise cosine similarity.  No OOD validation data, score
calibration, or fusion coefficient is used by this runner.

The formal matrix contains eleven independent model runs: three official
seeds for CIFAR-10, CIFAR-100, and ImageNet-200, plus one fixed torchvision
ResNet-50 and one fixed timm DenseNet-121 on ImageNet-1K.  In particular, the
fixed ImageNet-1K classifiers are not relabelled as three artificial seeds.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


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
NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenet200": 200,
    "imagenet1k_resnet50": 1000,
    "imagenet1k_densenet121": 1000,
}
FIXED_MODEL_TARGETS = {
    "imagenet1k_resnet50",
    "imagenet1k_densenet121",
}
EXPECTED_FORMAL_RUNS = 11
EXPECTED_METRIC_ROWS = {
    "cifar10": 8,
    "cifar100": 8,
    "imagenet200": 7,
    "imagenet1k": 7,
}
PROTOCOL_NAME = "CTM under the local OpenOOD v1.5 protocol"
# Pinned provenance for the implementation reproduced here.  The runner
# deliberately reimplements only CTM's class-mean cosine statistic inside the
# local OpenOOD evaluator, rather than mixing the original repository's data
# pipeline with ours.
CTM_REFERENCE = "Nguyen et al., A Cosine Similarity-based Method for OOD Detection (2023)"
CTM_OFFICIAL_REPOSITORY = "https://github.com/Fsoft-AIC/CTM-OOD"
CTM_OFFICIAL_COMMIT = "3587259bd6a69abd6b4103cb7311ffaa0857d60f"
CTM_OFFICIAL_CTM_PY_SHA256 = (
    "540f5edddd85669d04129247c0cbaab2b00aa2b94928fab76e4ee4f81651c49e"
)
CTM_OFFICIAL_UTILS_PY_SHA256 = (
    "114dbbf672e087099c09bdcf7d9da75aec3035eacecd8db0cbb797105c478f47"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("smoke", "full"))
    parser.add_argument(
        "--targets", "--benchmarks", dest="targets", nargs="+",
        choices=TARGETS, default=list(TARGETS),
        help=(
            "Experiment targets. ImageNet-1K target names include the backbone "
            "so that the two fixed classifiers remain distinct."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--openood-results-root", type=Path, default=Path("openood_pretrained")
    )
    parser.add_argument("--openood-root", type=Path, default=Path("OpenOOD"))
    # Accepted for a uniform journal-runner CLI.  CTM itself does not read
    # Hamiltonian checkpoints or the locked-decision files.
    parser.add_argument(
        "--hamiltonian-root", type=Path, default=Path("results_openood")
    )
    parser.add_argument(
        "--journal-root", type=Path, default=Path("results/journal")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/journal/ctm")
    )
    parser.add_argument(
        "--cache-root", type=Path, default=Path("cache/pretrained")
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--batch-size-cifar", type=int, default=256)
    parser.add_argument("--batch-size-imagenet200", type=int, default=128)
    parser.add_argument("--batch-size-imagenet1k", type=int, default=64)
    parser.add_argument("--setup-batch-size-cifar", type=int, default=512)
    parser.add_argument("--setup-batch-size-imagenet200", type=int, default=128)
    parser.add_argument("--setup-batch-size-imagenet1k", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--smoke-setup-samples-per-class", type=int, default=2,
        help="Class-balanced setup cap used only by the non-reportable smoke run.",
    )
    parser.add_argument(
        "--max-eval-samples", type=int, default=128,
        help="Per-dataset evaluation cap used only by --stage smoke.",
    )
    parser.add_argument("--tvs-version", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--weight-download-backend",
        choices=("auto", "official", "huggingface"), default="auto",
    )
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--force-centroids", action="store_true",
        help="Recompute the ID-training class means even when an audited cache exists.",
    )
    return parser


def _scheduled_runs(
    targets: Iterable[str], seeds: Iterable[int]
) -> list[tuple[str, int]]:
    """Return independent model runs without duplicating fixed ImageNet models."""

    unique_targets = list(dict.fromkeys(targets))
    unique_seeds = list(dict.fromkeys(int(seed) for seed in seeds))
    if not unique_seeds:
        raise ValueError("At least one seed is required")
    if any(seed < 0 for seed in unique_seeds):
        raise ValueError("Seeds must be non-negative")
    runs: list[tuple[str, int]] = []
    for target in unique_targets:
        if target in FIXED_MODEL_TARGETS:
            runs.append((target, 0))
        else:
            runs.extend((target, seed) for seed in unique_seeds)
    return runs


def _forward_raw_feature(net: Any, data: torch.Tensor):
    """Return logits and the unnormalised penultimate feature tensor."""

    try:
        output = net(data, return_feature=True)
    except TypeError as error:
        raise TypeError(
            "CTM requires a backbone implementing forward(x, return_feature=True)"
        ) from error
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise TypeError("Backbone did not return (logits, penultimate_feature)")
    logits, features = output[0], output[1]
    if features.ndim > 2:
        features = torch.flatten(features, 1)
    if features.ndim != 2:
        raise RuntimeError(
            f"Penultimate features must be a matrix; found {tuple(features.shape)}"
        )
    return logits, features


def _accumulate_class_sums(
    sums: torch.Tensor,
    counts: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    """Accumulate raw features using float64 CPU arithmetic."""

    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("features and labels have incompatible shapes")
    if features.shape[1] != sums.shape[1] or len(sums) != len(counts):
        raise ValueError("feature dimension or class count does not match accumulators")
    features_cpu = features.detach().to(device="cpu", dtype=torch.float64)
    labels_cpu = labels.detach().to(device="cpu", dtype=torch.long)
    if labels_cpu.numel() and (
        int(labels_cpu.min()) < 0 or int(labels_cpu.max()) >= len(counts)
    ):
        raise ValueError("training labels fall outside the configured class range")
    if not torch.isfinite(features_cpu).all():
        raise ValueError("non-finite penultimate feature encountered")
    sums.index_add_(0, labels_cpu, features_cpu)
    counts.index_add_(0, labels_cpu, torch.ones_like(labels_cpu))


def _normalised_class_means(
    sums: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    """Compute CTM's normalised directions from raw arithmetic class means."""

    if sums.ndim != 2 or counts.ndim != 1 or len(sums) != len(counts):
        raise ValueError("sums and counts have incompatible shapes")
    missing = torch.where(counts <= 0)[0]
    if len(missing):
        raise RuntimeError(
            f"CTM class means are missing classes: {missing[:20].tolist()}"
        )
    raw_means = sums / counts.to(dtype=sums.dtype)[:, None]
    norms = raw_means.norm(dim=1)
    degenerate = torch.where(~torch.isfinite(norms) | (norms <= 0))[0]
    if len(degenerate):
        raise RuntimeError(
            f"CTM class means are non-finite or zero for classes: "
            f"{degenerate[:20].tolist()}"
        )
    return F.normalize(raw_means.to(dtype=torch.float32), dim=-1)


def _ctm_confidence(
    features: torch.Tensor, class_directions: torch.Tensor
) -> torch.Tensor:
    """Maximum cosine similarity between a query and all class means."""

    if features.ndim != 2 or class_directions.ndim != 2:
        raise ValueError("features and class_directions must be matrices")
    if features.shape[1] != class_directions.shape[1]:
        raise ValueError("query and class-mean feature dimensions differ")
    return (F.normalize(features, dim=-1) @ class_directions.T).max(1).values


def _setup_batch_size(target: str, args: argparse.Namespace) -> int:
    if target in ("cifar10", "cifar100"):
        return int(args.setup_batch_size_cifar)
    if target == "imagenet200":
        return int(args.setup_batch_size_imagenet200)
    return int(args.setup_batch_size_imagenet1k)


def _eval_batch_size(target: str, args: argparse.Namespace) -> int:
    if target in ("cifar10", "cifar100"):
        return int(args.batch_size_cifar)
    if target == "imagenet200":
        return int(args.batch_size_imagenet200)
    return int(args.batch_size_imagenet1k)


def _metric_path(
    output_root: Path, stage: str, target: str, model_tag: str, seed: int
) -> Path:
    return output_root / stage / target / model_tag / f"seed{seed}" / "ctm.csv"


def _centroid_cache_path(
    output_root: Path, stage: str, target: str, model_tag: str, seed: int
) -> Path:
    return (
        output_root / "centroids" / stage / target / model_tag
        / f"seed{seed}_class_means.pt"
    )


def _normalise_saved(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    unnamed = [column for column in frame if column.startswith("Unnamed:")]
    if "Dataset" not in frame:
        frame = frame.rename(
            columns={unnamed[0] if unnamed else frame.columns[0]: "Dataset"}
        )
    config = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    frame.insert(0, "ModelSource", config["model_source"])
    frame.insert(0, "Model", config["model_tag"])
    frame.insert(0, "Seed", int(config["seed"]))
    frame.insert(0, "Benchmark", config["benchmark"])
    frame.insert(0, "Target", config["target"])
    frame.insert(0, "Method", "ctm")
    return frame


def rebuild_combined(output_root: Path, stage: str) -> Path:
    frames = []
    for path in sorted((output_root / stage).rglob("ctm.csv")):
        config_path = path.with_suffix(".json")
        if not config_path.is_file():
            raise RuntimeError(f"CTM result has no matching config: {path}")
        frames.append(_normalise_saved(path))
    if not frames:
        raise FileNotFoundError(f"No CTM results under {output_root / stage}")
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["Target", "Seed", "Dataset"]
    )
    destination = output_root / f"ctm_{stage}_all_runs.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False, float_format="%.6f")
    return destination


def _make_ctm_postprocessor(
    BasePostprocessor,
    *,
    target: str,
    seed: int,
    n_classes: int,
    cache_path: Path,
    cache_metadata: dict[str, Any],
    setup_batch_size: int,
    num_workers: int,
    setup_samples_per_class: int,
    progress_interval: int,
    force_centroids: bool,
    stratified_reservoir_indices,
    load_torch_checkpoint,
):
    class CTMPostprocessor(BasePostprocessor):
        def __init__(self):
            super().__init__(config=None)
            self.APS_mode = False
            self.hyperparam_search_done = True
            self.setup_flag = False
            self.class_directions: torch.Tensor | None = None
            self.audit: dict[str, Any] = {}

        def _load_cache(
            self, device: torch.device, train_dataset_size: int
        ) -> bool:
            if force_centroids or not cache_path.is_file():
                return False
            cached = load_torch_checkpoint(cache_path, map_location="cpu")
            expected_metadata = {
                **cache_metadata,
                "train_dataset_size": int(train_dataset_size),
            }
            if cached.get("metadata") != expected_metadata:
                print(f"[cache mismatch] recomputing CTM means: {cache_path}")
                return False
            directions = cached.get("class_directions")
            counts = cached.get("class_counts")
            if not isinstance(directions, torch.Tensor) or not isinstance(
                counts, torch.Tensor
            ):
                raise RuntimeError(f"Malformed CTM centroid cache: {cache_path}")
            if tuple(directions.shape)[0] != n_classes or len(counts) != n_classes:
                raise RuntimeError(f"Wrong class count in CTM cache: {cache_path}")
            if (counts <= 0).any() or not torch.isfinite(directions).all():
                raise RuntimeError(f"Invalid values in CTM cache: {cache_path}")
            self.class_directions = directions.to(device=device, dtype=torch.float32)
            self.audit = dict(cached.get("audit", {}))
            self.audit["cache_reused"] = True
            print(f"Loaded audited CTM class means: {cache_path}", flush=True)
            return True

        @torch.no_grad()
        def _build(self, net, train_loader) -> None:
            dataset = train_loader.dataset
            dataset_size = len(dataset)
            if setup_samples_per_class:
                if not hasattr(dataset, "imglist"):
                    raise TypeError("Smoke CTM setup requires ImglistDataset.imglist")
                indices = stratified_reservoir_indices(
                    dataset.imglist, n_classes, setup_samples_per_class, seed
                )
                setup_dataset = Subset(dataset, indices)
                expected_samples = len(indices)
                all_train = False
            else:
                setup_dataset = dataset
                expected_samples = dataset_size
                all_train = True

            loader = DataLoader(
                setup_dataset,
                batch_size=setup_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
            )
            device = next(net.parameters()).device
            sums: torch.Tensor | None = None
            counts = torch.zeros(n_classes, dtype=torch.long)
            processed = 0
            started = time.time()
            print(
                f"CTM setup ({target}, seed={seed}): streaming "
                f"{expected_samples:,} raw ID-training features...",
                flush=True,
            )
            net.eval()
            for batch_index, batch in enumerate(loader, 1):
                data = batch["data"].to(device, non_blocking=True)
                labels = batch["label"].long()
                _, features = _forward_raw_feature(net, data)
                if sums is None:
                    sums = torch.zeros(
                        n_classes, int(features.shape[1]), dtype=torch.float64
                    )
                _accumulate_class_sums(sums, counts, features, labels)
                processed += len(labels)
                if progress_interval and batch_index % progress_interval == 0:
                    elapsed = max(time.time() - started, 1e-9)
                    print(
                        f"  CTM setup batches {batch_index:,}/{len(loader):,}; "
                        f"images {processed:,}/{expected_samples:,}; "
                        f"{processed / elapsed:.1f} image/s",
                        flush=True,
                    )
            if sums is None or processed != expected_samples:
                raise RuntimeError(
                    f"Incomplete CTM setup pass: processed {processed}, "
                    f"expected {expected_samples}"
                )
            directions = _normalised_class_means(sums, counts)
            if int(counts.sum()) != expected_samples:
                raise RuntimeError("CTM per-class counts do not sum to setup size")
            self.class_directions = directions.to(device)
            self.audit = {
                "cache_reused": False,
                "feature_source": "raw penultimate feature before normalisation",
                "class_estimator": "arithmetic mean followed by L2 normalisation",
                "query_transform": "L2 normalisation",
                "confidence": "maximum class-mean cosine similarity",
                "ood_tuning": False,
                "all_id_train_samples": all_train,
                "train_dataset_size": dataset_size,
                "processed_setup_samples": processed,
                "class_count_min": int(counts.min()),
                "class_count_max": int(counts.max()),
                "feature_dimension": int(directions.shape[1]),
                "elapsed_seconds": time.time() - started,
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_name(cache_path.name + ".tmp")
            stored_metadata = {
                **cache_metadata,
                "train_dataset_size": int(dataset_size),
            }
            torch.save(
                {
                    "metadata": stored_metadata,
                    "class_directions": directions.cpu(),
                    "class_counts": counts,
                    "audit": self.audit,
                },
                temporary,
            )
            temporary.replace(cache_path)
            print(f"Saved CTM class means: {cache_path}", flush=True)

        def setup(self, net, id_loader_dict, ood_loader_dict) -> None:
            del ood_loader_dict
            device = next(net.parameters()).device
            train_loader = id_loader_dict["train"]
            if not self._load_cache(device, len(train_loader.dataset)):
                self._build(net, train_loader)
            self.setup_flag = True

        @torch.no_grad()
        def postprocess(self, net, data):
            if self.class_directions is None:
                raise RuntimeError("CTM setup has not completed")
            logits, features = _forward_raw_feature(net, data)
            prediction = logits.argmax(1)
            confidence = _ctm_confidence(features, self.class_directions)
            return prediction, confidence

    return CTMPostprocessor()


def _scope_is_complete(targets: list[str], seeds: list[int]) -> bool:
    return set(targets) == set(TARGETS) and set(seeds) == {0, 1, 2}


def main() -> None:
    args = build_parser().parse_args()
    for field in (
        "data_root", "openood_results_root", "openood_root",
        "hamiltonian_root", "journal_root", "output_root", "cache_root",
    ):
        setattr(args, field, getattr(args, field).resolve())
    args.targets = list(dict.fromkeys(args.targets))
    args.seeds = list(dict.fromkeys(args.seeds))
    runs = _scheduled_runs(args.targets, args.seeds)
    if args.stage == "smoke":
        if args.max_eval_samples <= 0:
            raise ValueError("Smoke stage requires --max-eval-samples > 0")
        if args.smoke_setup_samples_per_class <= 0:
            raise ValueError("Smoke stage requires a positive setup sample cap")
        evaluation_limit = int(args.max_eval_samples)
        setup_samples_per_class = int(args.smoke_setup_samples_per_class)
    else:
        evaluation_limit = 0
        setup_samples_per_class = 0
    if not torch.cuda.is_available():
        raise RuntimeError("CTM OpenOOD evaluation requires a CUDA GPU")
    if not args.openood_root.joinpath("openood", "evaluation_api").is_dir():
        raise FileNotFoundError(f"Invalid OpenOOD root: {args.openood_root}")

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
    helpers = importlib.import_module("openood_hamiltonian_postprocessor")
    stratified_reservoir_indices = helpers.stratified_reservoir_indices
    load_torch_checkpoint = importlib.import_module(
        "hamiltonian_detector"
    ).load_torch_checkpoint
    baseline_module = importlib.import_module("run_msp_baselines")
    model_for_benchmark = baseline_module._model_for_benchmark
    openood_id_name = baseline_module.OPENOOD_ID_NAME
    dense_module = importlib.import_module("run_densenet121_generalization")

    dense_weights = args.cache_root / "densenet121_tv_in1k_pinned.bin"
    if "imagenet1k_densenet121" in args.targets:
        dense_module._download_pinned_weights(dense_weights)

    for target, seed in runs:
        benchmark = BASE_BENCHMARK[target]
        if target == "imagenet1k_densenet121":
            network, preprocessor, data_config = dense_module._build_model(
                dense_weights
            )
            model_tag = "densenet121.tv_in1k"
            model_source = (
                f"{dense_weights};sha256:{dense_module.WEIGHT_SHA256}"
            )
        else:
            network, preprocessor, model_tag, model_source = model_for_benchmark(
                benchmark, args, seed
            )
            data_config = None

        metric_path = _metric_path(
            args.output_root, args.stage, target, model_tag, seed
        )
        config_path = metric_path.with_suffix(".json")
        if not args.force and (metric_path.is_file() or config_path.is_file()):
            if not (metric_path.is_file() and config_path.is_file()):
                raise RuntimeError(f"Incomplete resumable CTM result: {metric_path.parent}")
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            if saved.get("stage") != args.stage or saved.get(
                "all_id_train_samples"
            ) != (args.stage == "full"):
                raise RuntimeError(f"CTM protocol mismatch in {config_path}")
            print(f"[resume] CTM {target} seed={seed}", flush=True)
            del network
            continue

        cache_path = _centroid_cache_path(
            args.output_root, args.stage, target, model_tag, seed
        )
        train_dataset_size = None
        cache_metadata = {
            "format_version": 1,
            "method": "CTM",
            "protocol": PROTOCOL_NAME,
            "reference": CTM_REFERENCE,
            "official_repository": CTM_OFFICIAL_REPOSITORY,
            "official_commit": CTM_OFFICIAL_COMMIT,
            "official_ctm_py_sha256": CTM_OFFICIAL_CTM_PY_SHA256,
            "official_utils_py_sha256": CTM_OFFICIAL_UTILS_PY_SHA256,
            "stage": args.stage,
            "target": target,
            "benchmark": benchmark,
            "seed": seed,
            "model_tag": model_tag,
            "model_source": str(model_source),
            "n_classes": NUM_CLASSES[target],
            "setup_samples_per_class": setup_samples_per_class,
            "raw_feature_mean": True,
            "normalise_class_mean_after_averaging": True,
        }
        postprocessor = _make_ctm_postprocessor(
            BasePostprocessor,
            target=target,
            seed=seed,
            n_classes=NUM_CLASSES[target],
            cache_path=cache_path,
            cache_metadata=cache_metadata,
            setup_batch_size=_setup_batch_size(target, args),
            num_workers=args.num_workers,
            setup_samples_per_class=setup_samples_per_class,
            progress_interval=args.progress_interval,
            force_centroids=args.force_centroids,
            stratified_reservoir_indices=stratified_reservoir_indices,
            load_torch_checkpoint=load_torch_checkpoint,
        )
        network = network.cuda().eval()
        evaluator = Evaluator(
            network,
            id_name=openood_id_name[benchmark],
            data_root=str(args.data_root),
            config_root=str(args.openood_root / "configs"),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            batch_size=_eval_batch_size(target, args),
            shuffle=False,
            num_workers=args.num_workers,
        )
        train_dataset_size = len(evaluator.dataloader_dict["id"]["train"].dataset)
        if evaluation_limit:
            limit_evaluator_for_smoke_test(evaluator, evaluation_limit)
        print(
            f"[run] CTM | {target} | model-seed={seed} | stage={args.stage}",
            flush=True,
        )
        metrics = evaluator.eval_ood(
            fsood=False, progress=not args.no_progress
        )
        expected_metric_rows = EXPECTED_METRIC_ROWS[benchmark]
        if len(metrics) != expected_metric_rows:
            raise RuntimeError(
                f"Expected {expected_metric_rows} OpenOOD metric rows for "
                f"{benchmark}, found {len(metrics)}"
            )
        if set(("nearood", "farood")) - set(map(str, metrics.index)):
            raise RuntimeError("CTM metrics lack nearood/farood aggregate rows")
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(metric_path, float_format="%.6f")
        config_path.write_text(
            json.dumps(
                {
                    "protocol": PROTOCOL_NAME,
                    "stage": args.stage,
                    "method": "ctm",
                    "reference": CTM_REFERENCE,
                    "official_repository": CTM_OFFICIAL_REPOSITORY,
                    "official_commit": CTM_OFFICIAL_COMMIT,
                    "official_ctm_py_sha256": CTM_OFFICIAL_CTM_PY_SHA256,
                    "official_utils_py_sha256": CTM_OFFICIAL_UTILS_PY_SHA256,
                    "target": target,
                    "benchmark": benchmark,
                    "seed": seed,
                    "seed_semantics": (
                        "fixed_model_single_run"
                        if target in FIXED_MODEL_TARGETS
                        else "official_checkpoint_seed"
                    ),
                    "model_tag": model_tag,
                    "model_source": str(model_source),
                    "data_config": data_config,
                    "centroid_cache": str(cache_path.resolve()),
                    "feature_source": "raw penultimate feature",
                    "class_mean": "raw arithmetic mean then L2 normalisation",
                    "query_transform": "L2 normalisation",
                    "score": "maximum cosine similarity to class means",
                    "ood_tuning": False,
                    "all_id_train_samples": args.stage == "full",
                    "train_dataset_size": train_dataset_size,
                    "max_eval_samples": evaluation_limit,
                    "setup_audit": postprocessor.audit,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[saved] {metric_path}", flush=True)
        del evaluator, postprocessor, network
        torch.cuda.empty_cache()

    combined_path = rebuild_combined(args.output_root, args.stage)
    expected_runs = len(runs)
    result_configs = sorted((args.output_root / args.stage).rglob("ctm.json"))
    current_scope = []
    for config_path in result_configs:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        pair = (config.get("target"), int(config.get("seed", -1)))
        if pair in runs:
            current_scope.append(pair)
    missing = sorted(set(runs) - set(current_scope))
    if missing:
        raise RuntimeError(f"CTM run matrix incomplete: {missing}")

    complete_scope = _scope_is_complete(args.targets, args.seeds)
    completion_name = (
        f"ctm_{args.stage}_completed.json"
        if complete_scope
        else f"ctm_{args.stage}_partial_completed.json"
    )
    completion_path = args.output_root / completion_name
    completion_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "protocol": PROTOCOL_NAME,
                "reference": CTM_REFERENCE,
                "official_repository": CTM_OFFICIAL_REPOSITORY,
                "official_commit": CTM_OFFICIAL_COMMIT,
                "official_ctm_py_sha256": CTM_OFFICIAL_CTM_PY_SHA256,
                "official_utils_py_sha256": CTM_OFFICIAL_UTILS_PY_SHA256,
                "stage": args.stage,
                "targets": args.targets,
                "requested_seeds": args.seeds,
                "independent_runs": expected_runs,
                "expected_formal_runs": (
                    EXPECTED_FORMAL_RUNS if complete_scope else None
                ),
                "fixed_imagenet_models_repeated": False,
                "all_id_train_samples": args.stage == "full",
                "max_eval_samples": evaluation_limit,
                "combined_csv": str(combined_path.resolve()),
                "completed": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[saved] {combined_path}")
    print(f"[complete] {completion_path}")


if __name__ == "__main__":
    main()
