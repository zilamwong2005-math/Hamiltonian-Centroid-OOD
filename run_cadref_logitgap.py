"""Run CADRef and fixed LogitGap under the local OpenOOD v1.5 protocol.

This runner changes only the post-hoc score.  It deliberately reuses the
same ID/OOD splits, official checkpoints, preprocessing, model definitions,
and OpenOOD evaluator as the paper's other baselines.

CADRef follows the authors' released default Energy implementation: raw
penultimate class means and the global mean ID-training energy are estimated
from the complete ID training split in a formal run.  LogitGap is the fixed,
training-free variant; the number of comparison logits is 50% of the class
count for 10 classes and 20% for 100 or more classes.  No OOD sample is used
for setup or hyperparameter selection.
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
from torch.utils.data import DataLoader, Subset

from run_ctm_baseline import (
    BASE_BENCHMARK,
    EXPECTED_FORMAL_RUNS,
    EXPECTED_METRIC_ROWS,
    FIXED_MODEL_TARGETS,
    NUM_CLASSES,
    TARGETS,
    _accumulate_class_sums,
    _eval_batch_size,
    _forward_raw_feature,
    _scheduled_runs,
    _setup_batch_size,
)


METHODS = ("cadref", "logitgap")
PROTOCOL_NAME = "CADRef and fixed LogitGap under the local OpenOOD v1.5 protocol"

CADREF_REFERENCE = (
    "Ling et al., CADRef: Robust Out-of-Distribution Detection via "
    "Class-Aware Decoupled Relative Feature Leveraging (CVPR 2025)"
)
CADREF_REPOSITORY = "https://github.com/LingAndZero/CADRef"
CADREF_COMMIT = "121f74b47ebd71644a1c5a6d856880021268c7fa"
CADREF_SOURCE_SHA256 = (
    "02c578b29221a403968967aaa7eb199986e05f6164b2a7fa54af422525aaa1f1"
)

LOGITGAP_REFERENCE = (
    "Liang et al., Revisiting Logit Distributions for Reliable "
    "Out-of-Distribution Detection (NeurIPS 2025)"
)
LOGITGAP_REPOSITORY = "https://github.com/GIT-LJc/LogitGap"
LOGITGAP_COMMIT = "8492a8b9d85d7b1c873ad9ef592592401317857a"
LOGITGAP_SOURCE_SHA256 = (
    "502724a71330ee595c6cdd6af42b8f0b1b20a101ff26d3d66b35b9aa39d97e5b"
)

# This is the released fixed LogitGap convention.  ``topn`` is the number of
# non-maximum logits in the average in the authors' compute_utils.py.
LOGITGAP_TOPN = {10: 5, 100: 20, 200: 40, 1000: 200}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("smoke", "full"))
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS)
    )
    parser.add_argument(
        "--targets", "--benchmarks", dest="targets", nargs="+",
        choices=TARGETS, default=list(TARGETS),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--openood-results-root", type=Path, default=Path("openood_pretrained")
    )
    parser.add_argument("--openood-root", type=Path, default=Path("OpenOOD"))
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("results/journal/cadref_logitgap"),
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
        help="Class-balanced CADRef setup cap used only by the smoke run.",
    )
    parser.add_argument(
        "--max-eval-samples", type=int, default=128,
        help="Per-dataset cap used only by the non-reportable smoke run.",
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
        "--force-state", action="store_true",
        help="Recompute CADRef ID-training statistics even if a cache exists.",
    )
    return parser


def _classifier_weight(net: Any) -> torch.Tensor:
    if hasattr(net, "get_fc_layer"):
        layer = net.get_fc_layer()
    else:
        layer = getattr(net, "fc", None)
        if layer is None:
            layer = getattr(net, "classifier", None)
    if layer is None or not hasattr(layer, "weight"):
        raise TypeError("CADRef requires a linear classifier with a weight tensor")
    weight = layer.weight
    if weight.ndim != 2:
        raise TypeError("CADRef classifier weight must be a matrix")
    return weight


def _raw_class_means(sums: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    if sums.ndim != 2 or counts.ndim != 1 or len(sums) != len(counts):
        raise ValueError("sums and counts have incompatible shapes")
    missing = torch.where(counts <= 0)[0]
    if len(missing):
        raise RuntimeError(f"CADRef means are missing classes: {missing[:20].tolist()}")
    means = sums / counts.to(dtype=sums.dtype)[:, None]
    if not torch.isfinite(means).all():
        raise RuntimeError("CADRef class means contain non-finite values")
    return means.to(dtype=torch.float32)


def _cadref_confidence(
    logits: torch.Tensor,
    features: torch.Tensor,
    train_means: torch.Tensor,
    classifier_weight: torch.Tensor,
    global_mean_energy: torch.Tensor | float,
) -> torch.Tensor:
    """Released CADRef-Energy score, returned as ID confidence."""

    if logits.ndim != 2 or features.ndim != 2 or train_means.ndim != 2:
        raise ValueError("CADRef logits/features/means must be matrices")
    if len(logits) != len(features):
        raise ValueError("CADRef logits and features have different batch sizes")
    if features.shape[1] != train_means.shape[1]:
        raise ValueError("CADRef feature and class-mean dimensions differ")
    if classifier_weight.shape != train_means.shape:
        raise ValueError("CADRef classifier weight and class means differ")

    class_ids = logits.argmax(1)
    distance = features - train_means[class_ids]
    signs = classifier_weight[class_ids].sign()
    feature_l1 = features.norm(p=1, dim=1)
    if torch.any(feature_l1 == 0):
        raise RuntimeError("CADRef encountered a zero-L1 penultimate feature")
    positive_error = torch.clamp(distance * signs, min=0).norm(p=1, dim=1)
    negative_error = torch.clamp(distance * (-signs), min=0).norm(p=1, dim=1)
    positive_error = positive_error / feature_l1
    negative_error = negative_error / feature_l1

    energy = torch.logsumexp(logits, dim=1)
    global_energy = torch.as_tensor(
        global_mean_energy, device=logits.device, dtype=logits.dtype
    )
    if torch.any(energy == 0) or not torch.isfinite(global_energy) or global_energy == 0:
        raise RuntimeError("CADRef Energy denominator is zero or non-finite")
    confidence = -(positive_error / energy + negative_error / global_energy)
    if not torch.isfinite(confidence).all():
        raise RuntimeError("CADRef produced non-finite confidence values")
    return confidence


def _logitgap_confidence(logits: torch.Tensor, topn: int) -> torch.Tensor:
    """Mean gap between the maximum and the next ``topn`` logits."""

    if logits.ndim != 2:
        raise ValueError("LogitGap logits must be a matrix")
    if topn <= 0 or topn >= logits.shape[1]:
        raise ValueError("LogitGap topn must be between 1 and K-1")
    largest = torch.topk(logits, k=topn + 1, dim=1, largest=True).values
    score = (largest[:, :1] - largest[:, 1:]).mean(dim=1)
    if not torch.isfinite(score).all():
        raise RuntimeError("LogitGap produced non-finite confidence values")
    return score


def _metric_path(
    root: Path, stage: str, target: str, model_tag: str, seed: int, method: str
) -> Path:
    return root / stage / target / model_tag / f"seed{seed}" / f"{method}.csv"


def _state_path(
    root: Path, stage: str, target: str, model_tag: str, seed: int
) -> Path:
    return root / "state" / stage / target / model_tag / f"seed{seed}_cadref.pt"


def _normalise_saved(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    unnamed = [column for column in frame if column.startswith("Unnamed:")]
    if "Dataset" not in frame:
        frame = frame.rename(
            columns={unnamed[0] if unnamed else frame.columns[0]: "Dataset"}
        )
    config = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    for position, (name, value) in enumerate(
        (
            ("Method", config["method"]),
            ("Target", config["target"]),
            ("Benchmark", config["benchmark"]),
            ("Seed", int(config["seed"])),
            ("Model", config["model_tag"]),
            ("ModelSource", config["model_source"]),
        )
    ):
        frame.insert(position, name, value)
    return frame


def rebuild_combined(output_root: Path, stage: str) -> Path:
    frames = []
    for method in METHODS:
        for path in sorted((output_root / stage).rglob(f"{method}.csv")):
            if not path.with_suffix(".json").is_file():
                raise RuntimeError(f"Result has no matching config: {path}")
            frames.append(_normalise_saved(path))
    if not frames:
        raise FileNotFoundError(f"No CADRef/LogitGap results under {output_root / stage}")
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["Target", "Method", "Seed", "Dataset"]
    )
    destination = output_root / f"cadref_logitgap_{stage}_all_runs.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False, float_format="%.6f")
    return destination


def _make_logitgap_postprocessor(BasePostprocessor, *, topn: int):
    class LogitGapPostprocessor(BasePostprocessor):
        def __init__(self):
            super().__init__(config=None)
            self.APS_mode = False
            self.hyperparam_search_done = True
            self.setup_flag = True

        def setup(self, net, id_loader_dict, ood_loader_dict) -> None:
            del net, id_loader_dict, ood_loader_dict
            self.setup_flag = True

        @torch.no_grad()
        def postprocess(self, net, data):
            logits = net(data)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            return logits.argmax(1), _logitgap_confidence(logits, topn)

    return LogitGapPostprocessor()


def _make_cadref_postprocessor(
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
    force_state: bool,
    stratified_reservoir_indices,
    load_torch_checkpoint,
):
    class CADRefPostprocessor(BasePostprocessor):
        def __init__(self):
            super().__init__(config=None)
            self.APS_mode = False
            self.hyperparam_search_done = True
            self.setup_flag = False
            self.train_means: torch.Tensor | None = None
            self.global_mean_energy: torch.Tensor | None = None
            self.classifier_weight: torch.Tensor | None = None
            self.audit: dict[str, Any] = {}

        def _load(self, device: torch.device, train_size: int) -> bool:
            if force_state or not cache_path.is_file():
                return False
            cached = load_torch_checkpoint(cache_path, map_location="cpu")
            expected = {**cache_metadata, "train_dataset_size": int(train_size)}
            if cached.get("metadata") != expected:
                print(f"[cache mismatch] recomputing CADRef state: {cache_path}")
                return False
            means = cached.get("train_means")
            counts = cached.get("class_counts")
            energy = cached.get("global_mean_energy")
            if not isinstance(means, torch.Tensor) or not isinstance(
                counts, torch.Tensor
            ) or not isinstance(energy, torch.Tensor):
                raise RuntimeError(f"Malformed CADRef cache: {cache_path}")
            if means.shape[0] != n_classes or len(counts) != n_classes:
                raise RuntimeError(f"Wrong class count in CADRef cache: {cache_path}")
            if (counts <= 0).any() or not torch.isfinite(means).all():
                raise RuntimeError(f"Invalid CADRef cache values: {cache_path}")
            if energy.numel() != 1 or not torch.isfinite(energy) or energy == 0:
                raise RuntimeError(f"Invalid CADRef mean Energy: {cache_path}")
            self.train_means = means.to(device=device, dtype=torch.float32)
            self.global_mean_energy = energy.to(device=device, dtype=torch.float32)
            self.audit = dict(cached.get("audit", {}))
            self.audit["cache_reused"] = True
            print(f"Loaded audited CADRef state: {cache_path}", flush=True)
            return True

        @torch.no_grad()
        def _build(self, net, train_loader) -> None:
            dataset = train_loader.dataset
            dataset_size = len(dataset)
            if setup_samples_per_class:
                if not hasattr(dataset, "imglist"):
                    raise TypeError("Smoke CADRef setup requires ImglistDataset.imglist")
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
            energy_sum = torch.zeros((), dtype=torch.float64)
            processed = 0
            started = time.time()
            print(
                f"CADRef setup ({target}, seed={seed}): streaming "
                f"{expected_samples:,} ID-training samples...",
                flush=True,
            )
            net.eval()
            for batch_index, batch in enumerate(loader, 1):
                data = batch["data"].to(device, non_blocking=True)
                labels = batch["label"].long()
                logits, features = _forward_raw_feature(net, data)
                if sums is None:
                    sums = torch.zeros(
                        n_classes, int(features.shape[1]), dtype=torch.float64
                    )
                _accumulate_class_sums(sums, counts, features, labels)
                energies = torch.logsumexp(logits, dim=1).detach().cpu().double()
                if not torch.isfinite(energies).all():
                    raise RuntimeError("Non-finite ID-training Energy in CADRef setup")
                energy_sum += energies.sum()
                processed += len(labels)
                if progress_interval and batch_index % progress_interval == 0:
                    elapsed = max(time.time() - started, 1e-9)
                    print(
                        f"  CADRef setup batches {batch_index:,}/{len(loader):,}; "
                        f"images {processed:,}/{expected_samples:,}; "
                        f"{processed / elapsed:.1f} image/s",
                        flush=True,
                    )
            if sums is None or processed != expected_samples:
                raise RuntimeError(
                    f"Incomplete CADRef setup: {processed} != {expected_samples}"
                )
            means = _raw_class_means(sums, counts)
            global_energy = (energy_sum / processed).float()
            if not torch.isfinite(global_energy) or global_energy == 0:
                raise RuntimeError("CADRef global mean Energy is zero or non-finite")
            self.train_means = means.to(device)
            self.global_mean_energy = global_energy.to(device)
            self.audit = {
                "cache_reused": False,
                "feature_source": "raw penultimate feature",
                "class_estimator": "raw arithmetic mean",
                "logit_method": "Energy",
                "global_logit_statistic": "sample-weighted ID-training mean Energy",
                "ood_tuning": False,
                "all_id_train_samples": all_train,
                "train_dataset_size": dataset_size,
                "processed_setup_samples": processed,
                "class_count_min": int(counts.min()),
                "class_count_max": int(counts.max()),
                "feature_dimension": int(means.shape[1]),
                "global_mean_energy": float(global_energy),
                "elapsed_seconds": time.time() - started,
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_name(cache_path.name + ".tmp")
            torch.save(
                {
                    "metadata": {
                        **cache_metadata,
                        "train_dataset_size": int(dataset_size),
                    },
                    "train_means": means.cpu(),
                    "class_counts": counts,
                    "global_mean_energy": global_energy.cpu(),
                    "audit": self.audit,
                },
                temporary,
            )
            temporary.replace(cache_path)
            print(f"Saved CADRef state: {cache_path}", flush=True)

        def setup(self, net, id_loader_dict, ood_loader_dict) -> None:
            del ood_loader_dict
            device = next(net.parameters()).device
            train_loader = id_loader_dict["train"]
            if not self._load(device, len(train_loader.dataset)):
                self._build(net, train_loader)
            self.classifier_weight = _classifier_weight(net).detach()
            if self.classifier_weight.shape[0] != n_classes:
                raise RuntimeError("CADRef classifier class count mismatch")
            self.setup_flag = True

        @torch.no_grad()
        def postprocess(self, net, data):
            if (
                self.train_means is None
                or self.global_mean_energy is None
                or self.classifier_weight is None
            ):
                raise RuntimeError("CADRef setup has not completed")
            logits, features = _forward_raw_feature(net, data)
            confidence = _cadref_confidence(
                logits,
                features,
                self.train_means,
                self.classifier_weight,
                self.global_mean_energy,
            )
            return logits.argmax(1), confidence

    return CADRefPostprocessor()


def _scope_is_complete(targets: Iterable[str], seeds: Iterable[int]) -> bool:
    return set(targets) == set(TARGETS) and set(seeds) == {0, 1, 2}


def main() -> None:
    args = build_parser().parse_args()
    for field in (
        "data_root", "openood_results_root", "openood_root", "output_root",
        "cache_root",
    ):
        setattr(args, field, getattr(args, field).resolve())
    args.targets = list(dict.fromkeys(args.targets))
    args.methods = list(dict.fromkeys(args.methods))
    args.seeds = list(dict.fromkeys(args.seeds))
    runs = _scheduled_runs(args.targets, args.seeds)
    if args.stage == "smoke":
        if args.max_eval_samples <= 0 or args.smoke_setup_samples_per_class <= 0:
            raise ValueError("Smoke stage needs positive setup and evaluation caps")
        evaluation_limit = int(args.max_eval_samples)
        setup_samples_per_class = int(args.smoke_setup_samples_per_class)
    else:
        evaluation_limit = 0
        setup_samples_per_class = 0
    if not torch.cuda.is_available():
        raise RuntimeError("CADRef/LogitGap OpenOOD evaluation requires CUDA")
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
            model_source = f"{dense_weights};sha256:{dense_module.WEIGHT_SHA256}"
        else:
            network, preprocessor, model_tag, model_source = model_for_benchmark(
                benchmark, args, seed
            )
            data_config = None
        network = network.cuda().eval()

        for method in args.methods:
            metric_path = _metric_path(
                args.output_root, args.stage, target, model_tag, seed, method
            )
            config_path = metric_path.with_suffix(".json")
            if not args.force and (metric_path.is_file() or config_path.is_file()):
                if not (metric_path.is_file() and config_path.is_file()):
                    raise RuntimeError(f"Incomplete resumable result: {metric_path.parent}")
                saved = json.loads(config_path.read_text(encoding="utf-8"))
                if saved.get("stage") != args.stage or saved.get("method") != method:
                    raise RuntimeError(f"Protocol mismatch in {config_path}")
                print(f"[resume] {method} {target} seed={seed}", flush=True)
                continue

            if method == "logitgap":
                topn = LOGITGAP_TOPN[NUM_CLASSES[target]]
                postprocessor = _make_logitgap_postprocessor(
                    BasePostprocessor, topn=topn
                )
                setup_audit = {
                    "training_free": True,
                    "ood_tuning": False,
                    "number_of_classes": NUM_CLASSES[target],
                    "comparison_logits": topn,
                    "comparison_fraction": topn / NUM_CLASSES[target],
                }
                state_path = None
            else:
                topn = None
                state_path = _state_path(
                    args.output_root, args.stage, target, model_tag, seed
                )
                cache_metadata = {
                    "format_version": 1,
                    "method": "CADRef-Energy",
                    "protocol": PROTOCOL_NAME,
                    "official_repository": CADREF_REPOSITORY,
                    "official_commit": CADREF_COMMIT,
                    "official_source_sha256": CADREF_SOURCE_SHA256,
                    "stage": args.stage,
                    "target": target,
                    "benchmark": benchmark,
                    "seed": seed,
                    "model_tag": model_tag,
                    "model_source": str(model_source),
                    "n_classes": NUM_CLASSES[target],
                    "setup_samples_per_class": setup_samples_per_class,
                    "logit_method": "Energy",
                }
                postprocessor = _make_cadref_postprocessor(
                    BasePostprocessor,
                    target=target,
                    seed=seed,
                    n_classes=NUM_CLASSES[target],
                    cache_path=state_path,
                    cache_metadata=cache_metadata,
                    setup_batch_size=_setup_batch_size(target, args),
                    num_workers=args.num_workers,
                    setup_samples_per_class=setup_samples_per_class,
                    progress_interval=args.progress_interval,
                    force_state=args.force_state,
                    stratified_reservoir_indices=stratified_reservoir_indices,
                    load_torch_checkpoint=load_torch_checkpoint,
                )
                setup_audit = None

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
            train_size = len(evaluator.dataloader_dict["id"]["train"].dataset)
            if evaluation_limit:
                limit_evaluator_for_smoke_test(evaluator, evaluation_limit)
            print(
                f"[run] {method} | {target} | model-seed={seed} | "
                f"stage={args.stage}",
                flush=True,
            )
            metrics = evaluator.eval_ood(
                fsood=False, progress=not args.no_progress
            )
            if len(metrics) != EXPECTED_METRIC_ROWS[benchmark]:
                raise RuntimeError(
                    f"Expected {EXPECTED_METRIC_ROWS[benchmark]} rows for "
                    f"{benchmark}; found {len(metrics)}"
                )
            if {"nearood", "farood"} - set(map(str, metrics.index)):
                raise RuntimeError(f"{method} metrics lack nearood/farood rows")
            metric_path.parent.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(metric_path, float_format="%.6f")
            if method == "cadref":
                setup_audit = postprocessor.audit
                provenance = {
                    "reference": CADREF_REFERENCE,
                    "official_repository": CADREF_REPOSITORY,
                    "official_commit": CADREF_COMMIT,
                    "official_source_sha256": CADREF_SOURCE_SHA256,
                    "variant": "CADRef-Energy (released default)",
                    "state_cache": str(state_path.resolve()),
                }
            else:
                provenance = {
                    "reference": LOGITGAP_REFERENCE,
                    "official_repository": LOGITGAP_REPOSITORY,
                    "official_commit": LOGITGAP_COMMIT,
                    "official_source_sha256": LOGITGAP_SOURCE_SHA256,
                    "variant": "fixed LogitGap",
                    "comparison_logits": topn,
                }
            config_path.write_text(
                json.dumps(
                    {
                        "protocol": PROTOCOL_NAME,
                        "stage": args.stage,
                        "method": method,
                        **provenance,
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
                        "ood_tuning": False,
                        "all_id_train_samples": (
                            args.stage == "full" if method == "cadref" else None
                        ),
                        "train_dataset_size": train_size,
                        "max_eval_samples": evaluation_limit,
                        "setup_audit": setup_audit,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[saved] {metric_path}", flush=True)
            del evaluator, postprocessor
            torch.cuda.empty_cache()
        del network
        torch.cuda.empty_cache()

    combined_path = rebuild_combined(args.output_root, args.stage)
    expected = {
        (target, seed, method)
        for target, seed in runs
        for method in args.methods
    }
    found = set()
    for config_path in sorted((args.output_root / args.stage).rglob("*.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        key = (config.get("target"), int(config.get("seed", -1)), config.get("method"))
        if key in expected:
            found.add(key)
    missing = sorted(expected - found)
    if missing:
        raise RuntimeError(f"CADRef/LogitGap matrix incomplete: {missing}")

    complete_scope = (
        _scope_is_complete(args.targets, args.seeds)
        and set(args.methods) == set(METHODS)
    )
    completion_name = (
        f"cadref_logitgap_{args.stage}_completed.json"
        if complete_scope
        else f"cadref_logitgap_{args.stage}_partial_completed.json"
    )
    completion_path = args.output_root / completion_name
    completion_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "protocol": PROTOCOL_NAME,
                "stage": args.stage,
                "methods": args.methods,
                "targets": args.targets,
                "requested_seeds": args.seeds,
                "independent_model_runs_per_method": len(runs),
                "expected_formal_model_runs_per_method": (
                    EXPECTED_FORMAL_RUNS if complete_scope else None
                ),
                "method_runs": len(expected),
                "fixed_imagenet_models_repeated": False,
                "cadref_all_id_train_samples": args.stage == "full",
                "logitgap_training_free": True,
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
    if args.stage == "smoke":
        print("SMOKE ONLY: capped results must not be reported in the paper.")


if __name__ == "__main__":
    main()
