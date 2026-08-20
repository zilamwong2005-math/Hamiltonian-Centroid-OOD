"""Benchmark end-to-end latency/memory of the five extended OOD baselines.

The dataloader is excluded from timing. Setup-based postprocessors are cached
after their first exact full-train setup so interrupted runs are resumable.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from Imagenet_ood_experiment import add_openood_to_path
from run_msp_baselines import OPENOOD_ID_NAME, _model_for_benchmark
from run_openood_baselines import _selected_hyperparameters, _target


BENCHMARKS = ("cifar10", "cifar100", "imagenet200", "imagenet1k")
METHODS = ("ash", "dice", "she", "rmds", "rankfeat", "scale")
SETUP_ATTRIBUTES = {
    "dice": ("mean_act",),
    "she": ("activation_log",),
    "rmds": ("class_mean", "precision", "whole_mean", "whole_precision"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(METHODS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--openood-results-root", type=Path,
                        default=Path("openood_pretrained"))
    parser.add_argument("--openood-root", type=Path, default=Path("OpenOOD"))
    parser.add_argument("--baseline-root", type=Path,
                        default=Path("results_openood_baselines"))
    parser.add_argument("--setup-cache-root", type=Path,
                        default=Path("results/journal/efficiency/setup_cache"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/journal/efficiency/extended_posthoc_efficiency.csv"
    ))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--setup-batch-size", type=int, default=64,
        help="Batch size used only for full-train DICE/SHE/RMDS setup.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--tvs-version", type=int, choices=[1, 2], default=1)
    parser.add_argument("--weight-download-backend",
                        choices=["auto", "official", "huggingface"],
                        default="auto")
    parser.add_argument("--force-setup", action="store_true")
    return parser


def _measure(function, warmup: int, repeats: int, batch_size: int) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            output = function()
            if not torch.isfinite(output).all():
                raise RuntimeError("Non-finite score encountered during warmup")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_memory = torch.cuda.memory_allocated()
        elapsed = []
        for _ in range(repeats):
            started = time.perf_counter()
            output = function()
            torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - started) * 1000.0)
        peak_memory = torch.cuda.max_memory_allocated()
    mean_ms = statistics.fmean(elapsed)
    std_ms = statistics.stdev(elapsed) if len(elapsed) > 1 else 0.0
    return {
        "MeanMilliseconds": mean_ms,
        "StdMilliseconds": std_ms,
        "MedianMilliseconds": statistics.median(elapsed),
        "P95Milliseconds": sorted(elapsed)[
            max(0, int(0.95 * len(elapsed)) - 1)
        ],
        "ImagesPerSecond": batch_size * 1000.0 / mean_ms,
        "PeakAllocatedMB": peak_memory / 1024**2,
        "IncrementalPeakMB": (peak_memory - baseline_memory) / 1024**2,
    }


def _tensor_bytes(value) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    return 0


def _auxiliary_state_mb(method: str, postprocessor) -> float:
    attributes = {
        "ash": (),
        "scale": (),
        "rankfeat": (),
        "dice": ("mean_act", "masked_w"),
        "she": ("activation_log",),
        "rmds": (
            "class_mean", "precision", "whole_mean", "whole_precision"
        ),
    }[method]
    return sum(
        _tensor_bytes(getattr(postprocessor, name, None))
        for name in attributes
    ) / 1024**2


def _cache_paths(root: Path, benchmark: str, model_tag: str,
                 seed: int, method: str) -> tuple[Path, Path]:
    directory = root / benchmark / model_tag / f"seed{seed}"
    return directory / f"{method}.pt", directory / f"{method}.json"


def _save_setup_cache(method: str, postprocessor, cache: Path,
                      metadata: Path, setup_seconds: float) -> None:
    if method not in SETUP_ATTRIBUTES:
        return
    payload = {}
    for name in SETUP_ATTRIBUTES[method]:
        value = getattr(postprocessor, name)
        payload[name] = value.cpu() if torch.is_tensor(value) else value
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache)
    metadata.write_text(json.dumps({
        "method": method,
        "setup_seconds": setup_seconds,
        "implementation": getattr(
            postprocessor, "implementation_note", "OpenOOD v1.5"
        ),
    }, indent=2), encoding="utf-8")


def _load_setup_cache(method: str, postprocessor, cache: Path,
                      metadata: Path) -> float:
    payload = torch.load(cache, map_location="cpu")
    for name in SETUP_ATTRIBUTES[method]:
        value = payload[name]
        if method == "she" and torch.is_tensor(value):
            value = value.cuda()
        setattr(postprocessor, name, value)
    postprocessor.setup_flag = True
    if method == "dice":
        postprocessor.masked_w = None
    if method == "rmds":
        postprocessor._device_statistics = None
    details = json.loads(metadata.read_text(encoding="utf-8"))
    return float(details["setup_seconds"])


def _apply_saved_hyperparameters(postprocessor, config: dict) -> None:
    value = config.get("selected_hyperparameters")
    setter = getattr(postprocessor, "set_hyperparam", None)
    if value is not None and setter is not None:
        setter(value if isinstance(value, list) else [value])
    postprocessor.APS_mode = False
    postprocessor.hyperparam_search_done = True


def _legacy_formal_config_is_auditable(
    result_path: Path, config_path: Path, config: dict, method: str
) -> bool:
    """Accept only the known pre-metadata Scale result, never a smoke result."""
    if "max_eval_samples" in config:
        return False
    if method != "scale":
        return False
    if config.get("protocol") != "OpenOOD v1.5 local reproduction":
        return False
    if config_path.parent != result_path.parent or not result_path.is_file():
        return False
    frame = pd.read_csv(result_path)
    dataset_column = "Dataset" if "Dataset" in frame else frame.columns[0]
    datasets = set(frame[dataset_column].astype(str).str.lower())
    return {"nearood", "farood"}.issubset(datasets)


def _upsert(rows: list[dict], destination: Path) -> None:
    incoming = pd.DataFrame(rows)
    if destination.is_file():
        incoming = pd.concat(
            [pd.read_csv(destination), incoming], ignore_index=True
        )
    identity = ["Benchmark", "Seed", "Method", "BatchSize"]
    incoming = incoming.drop_duplicates(identity, keep="last").sort_values(
        identity
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    incoming.to_csv(destination, index=False, float_format="%.6f")


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    if (args.batch_size <= 0 or args.setup_batch_size <= 0
            or args.warmup < 0 or args.repeats <= 0):
        raise ValueError("batch size/repeats must be positive and warmup nonnegative")
    for field in (
        "data_root", "openood_results_root", "openood_root", "baseline_root",
        "setup_cache_root", "output",
    ):
        setattr(args, field, getattr(args, field).resolve())

    add_openood_to_path(args.openood_root)
    evaluator_module = importlib.import_module("openood.evaluation_api.evaluator")
    evaluator_module.data_setup = lambda *_args, **_kwargs: None
    Evaluator = evaluator_module.Evaluator
    get_postprocessor = importlib.import_module(
        "openood.evaluation_api.postprocessor"
    ).get_postprocessor
    ASHNet = importlib.import_module("openood.networks.ash_net").ASHNet
    ScaleNet = importlib.import_module("openood.networks.scale_net").ScaleNet

    model_args = SimpleNamespace(
        openood_results_root=args.openood_results_root,
        tvs_version=args.tvs_version,
        weight_download_backend=args.weight_download_backend,
    )
    network, preprocessor, model_tag, model_source = _model_for_benchmark(
        args.benchmark, model_args, args.seed
    )
    network = network.cuda().eval()
    rows = []

    for method in args.methods:
        result_path = _target(
            args.baseline_root, args.benchmark, model_tag, args.seed, method
        )
        config_path = result_path.with_suffix(".json")
        if not result_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(
                f"Formal {method} result/config is missing: {result_path}"
            )
        result_config = json.loads(config_path.read_text(encoding="utf-8"))
        legacy_formal = _legacy_formal_config_is_auditable(
            result_path, config_path, result_config, method
        )
        if (not legacy_formal
                and int(result_config.get("max_eval_samples", -1)) != 0):
            raise RuntimeError(f"Refusing smoke config: {config_path}")
        if method == "rankfeat" and result_config.get(
                "rankfeat_accelerate") is not False:
            raise RuntimeError(
                f"RankFeat formal result was not full-SVD: {config_path}"
            )

        recover_scale_aps = (
            method == "scale"
            and result_config.get("selected_hyperparameters") is None
        )
        postprocessor = get_postprocessor(
            str(args.openood_root / "configs"), method,
            OPENOOD_ID_NAME[args.benchmark],
        )
        wrappers = {"ash": ASHNet, "scale": ScaleNet}
        if not recover_scale_aps:
            _apply_saved_hyperparameters(postprocessor, result_config)
            eval_network = (
                wrappers[method](network) if method in wrappers else network
            )
        cache, cache_metadata = _cache_paths(
            args.setup_cache_root, args.benchmark, model_tag, args.seed, method
        )
        cache_loaded = (
            method in SETUP_ATTRIBUTES and cache.is_file()
            and cache_metadata.is_file() and not args.force_setup
        )
        if cache_loaded:
            setup_seconds = _load_setup_cache(
                method, postprocessor, cache, cache_metadata
            )
            print(f"[setup-cache] {method}: {cache}", flush=True)

        started = time.perf_counter()
        evaluator_kwargs = dict(
            net=network if recover_scale_aps else eval_network,
            id_name=OPENOOD_ID_NAME[args.benchmark],
            data_root=str(args.data_root),
            config_root=str(args.openood_root / "configs"),
            preprocessor=preprocessor,
            batch_size=args.setup_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if recover_scale_aps:
            # The old formal Scale JSON predates metadata capture. Re-run only
            # OpenOOD's deterministic validation APS; no Near/Far test loader
            # is iterated and the original result/config remains untouched.
            evaluator_kwargs["postprocessor_name"] = "scale"
        else:
            evaluator_kwargs["postprocessor"] = postprocessor
        evaluator = Evaluator(**evaluator_kwargs)
        if recover_scale_aps:
            setup_seconds = time.perf_counter() - started
            postprocessor = evaluator.postprocessor
            eval_network = evaluator.net
            setup_protocol = (
                "Validation-only OpenOOD APS recovered for legacy formal "
                "Scale config; no Near/Far test access"
            )
        elif method in SETUP_ATTRIBUTES and not cache_loaded:
            setup_seconds = time.perf_counter() - started
            _save_setup_cache(
                method, postprocessor, cache, cache_metadata, setup_seconds
            )
        elif method in {"ash", "scale"}:
            # The formal APS-selected percentile is reused.  Reporting loader
            # construction as method setup would be misleading, while rerunning
            # APS would no longer measure the exact locked formal endpoint.
            setup_seconds = np.nan
        elif method == "rankfeat":
            setup_seconds = 0.0
        if not recover_scale_aps:
            setup_protocol = ({
                "ash": (
                    "Formal OpenOOD APS percentile reused; calibration time "
                    "not remeasured"
                ),
                "dice": "Full ID-train streaming feature-mean setup",
                "she": "Full ID-train streaming correct-class mean setup",
                "rmds": "Full ID-train streaming covariance setup",
                "rankfeat": "No offline setup",
                "scale": (
                    "Formal OpenOOD APS percentile reused; calibration time "
                    "not remeasured"
                ),
            })[method]
        batch = next(iter(evaluator.dataloader_dict["id"]["test"]))
        data = batch["data"][:args.batch_size].cuda(non_blocking=True)
        actual_batch = int(data.shape[0])

        def score():
            _, confidence = postprocessor.postprocess(eval_network, data)
            return confidence

        timing = _measure(score, args.warmup, args.repeats, actual_batch)
        rows.append({
            "Benchmark": args.benchmark,
            "Seed": args.seed,
            "Method": method,
            "BatchSize": actual_batch,
            "SetupBatchSize": args.setup_batch_size,
            "Warmup": args.warmup,
            "Repeats": args.repeats,
            "GPU": torch.cuda.get_device_name(0),
            "ModelTag": model_tag,
            "ModelSource": model_source,
            "FormalResult": str(result_path),
            "FormalConfig": str(config_path),
            "LegacyFormalConfig": legacy_formal,
            "TimingExcludesDataLoader": True,
            "FormalHyperparametersReused": not recover_scale_aps,
            "ValidationHyperparametersRecovered": recover_scale_aps,
            "SelectedHyperparameters": json.dumps(
                _selected_hyperparameters(postprocessor)
            ),
            "FullSVD": method == "rankfeat",
            "SetupSeconds": setup_seconds,
            "SetupProtocol": setup_protocol,
            "SetupCacheLoaded": cache_loaded,
            "SetupCache": str(cache) if method in SETUP_ATTRIBUTES else "",
            "AuxiliaryStateMB": _auxiliary_state_mb(method, postprocessor),
            **timing,
        })
        print(pd.DataFrame([rows[-1]]).to_string(index=False), flush=True)
        del evaluator, postprocessor, eval_network, data
        torch.cuda.empty_cache()

    _upsert(rows, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(vars(args), default=str, indent=2), encoding="utf-8"
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
