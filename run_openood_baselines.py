"""Run multiple post-hoc baselines with the exact local OpenOOD protocol."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import traceback
from pathlib import Path

import pandas as pd
import torch

from Imagenet_ood_experiment import (
    add_openood_to_path,
    limit_evaluator_for_smoke_test,
)
from run_msp_baselines import (
    BENCHMARKS,
    OPENOOD_ID_NAME,
    _batch_size_for,
    _model_for_benchmark,
)


SUPPORTED_METHODS = (
    "msp", "ebo", "mls", "gen", "react", "scale", "knn", "vim",
    "ash", "dice", "she", "rmds", "rankfeat",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS,
                        default=list(BENCHMARKS))
    parser.add_argument("--methods", nargs="+", choices=SUPPORTED_METHODS,
                        default=["ebo", "mls", "gen", "react", "scale"])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--openood-results-root", type=Path,
                        default=Path("openood_pretrained"))
    parser.add_argument("--openood-root", type=Path, default=Path("OpenOOD"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("results_openood_baselines"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--batch-size-cifar", type=int, default=256)
    parser.add_argument("--batch-size-imagenet200", type=int, default=128)
    parser.add_argument("--batch-size-imagenet1k", type=int, default=64)
    parser.add_argument("--batch-size-rankfeat-cifar", type=int, default=64)
    parser.add_argument("--batch-size-rankfeat-imagenet200", type=int,
                        default=32)
    parser.add_argument("--batch-size-rankfeat-imagenet1k", type=int,
                        default=8)
    parser.add_argument(
        "--rankfeat-accelerate",
        action="store_true",
        help=("Use RankFeat power iteration. Omit for the faithful OpenOOD "
              "full-SVD reproduction."),
    )
    parser.add_argument(
        "--max-eval-samples", type=int, default=0,
        help=("Limit each test split after postprocessor setup. Values >0 are "
              "smoke tests and must not be reported."),
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--tvs-version", type=int, choices=[1, 2], default=1)
    parser.add_argument("--weight-download-backend",
                        choices=["auto", "official", "huggingface"],
                        default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def _target(output_root: Path, benchmark: str, model_tag: str,
            seed: int, method: str) -> Path:
    if benchmark == "imagenet1k":
        return output_root / benchmark / model_tag / f"{method}.csv"
    return output_root / benchmark / model_tag / f"seed{seed}" / f"{method}.csv"


def _seed_from_path(path: Path) -> int:
    for part in path.parts:
        match = re.fullmatch(r"seed(\d+)", part.lower())
        if match:
            return int(match.group(1))
    return 0


def _valid_vim_dimensions(feature_dim: int, candidates) -> list[int]:
    """Drop ViM null-space dimensions that empty the residual subspace."""
    valid = sorted(
        {int(value) for value in candidates if 0 < int(value) < feature_dim}
    )
    if valid:
        return valid
    return [max(1, min(256, feature_dim - 1))]


def _network_feature_dim(network) -> int:
    if hasattr(network, "get_fc_layer"):
        layer = network.get_fc_layer()
    else:
        layer = getattr(network, "fc", None)
    if layer is None or not hasattr(layer, "in_features"):
        raise TypeError("ViM requires a classifier layer with in_features")
    return int(layer.in_features)


def _method_batch_size(benchmark: str, method: str, args) -> int:
    """Use conservative SVD batches while retaining normal baseline batches."""
    if method != "rankfeat":
        return _batch_size_for(benchmark, args)
    suffix = "cifar" if benchmark.startswith("cifar") else benchmark
    return int(getattr(args, f"batch_size_rankfeat_{suffix}"))


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _selected_hyperparameters(postprocessor):
    getter = getattr(postprocessor, "get_hyperparam", None)
    if getter is None:
        return None
    try:
        return _json_safe(getter())
    except (AttributeError, NotImplementedError):
        return None


def _tuning_protocol(method: str) -> str:
    if method == "ash":
        return "OpenOOD APS using ID validation and OOD validation only"
    return "Fixed OpenOOD v1.5 configuration; no test-set tuning"


def rebuild_long_form(output_root: Path) -> Path:
    rows = []
    for method in SUPPORTED_METHODS:
        for path in sorted(output_root.rglob(f"{method}.csv")):
            relative = path.relative_to(output_root)
            if len(relative.parts) < 3:
                continue
            benchmark = relative.parts[0].lower()
            if benchmark not in BENCHMARKS:
                continue
            frame = pd.read_csv(path)
            unnamed = [c for c in frame.columns if c.startswith("Unnamed:")]
            if unnamed:
                frame = frame.rename(columns={unnamed[0]: "Dataset"})
            elif "Dataset" not in frame:
                frame = frame.rename(columns={frame.columns[0]: "Dataset"})
            config_path = path.with_suffix(".json")
            config = (json.loads(config_path.read_text(encoding="utf-8"))
                      if config_path.is_file() else {})
            for _, metric in frame.iterrows():
                row = {
                    "Benchmark": benchmark,
                    "Model": config.get("model_tag", relative.parts[1]),
                    "Seed": int(config.get("seed", _seed_from_path(path))),
                    "Method": method,
                    "Dataset": str(metric["Dataset"]),
                    "ModelSource": config.get("model_source", "unknown"),
                    "Protocol": config.get(
                        "protocol", "OpenOOD v1.5 local reproduction"
                    ),
                    "TuningProtocol": config.get("tuning_protocol", "unknown"),
                    "Implementation": config.get("implementation", "OpenOOD"),
                    "BatchSize": config.get("batch_size"),
                    "MaxEvalSamples": config.get("max_eval_samples", 0),
                    "SelectedHyperparameters": json.dumps(
                        config.get("selected_hyperparameters"),
                        ensure_ascii=False,
                    ),
                    "RankFeatAccelerate": config.get(
                        "rankfeat_accelerate"
                    ),
                    "SourceFile": str(path.resolve()),
                }
                for column in ("FPR@95", "AUROC", "AUPR_IN", "AUPR_OUT", "ACC"):
                    if column in metric:
                        row[column] = float(metric[column])
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No baseline CSVs found under {output_root}")
    combined = pd.DataFrame(rows).drop_duplicates(
        ["Benchmark", "Seed", "Method", "Dataset"], keep="last"
    ).sort_values(["Benchmark", "Method", "Seed", "Dataset"])
    destination = output_root / "baseline_all_runs.csv"
    combined.to_csv(destination, index=False, float_format="%.6f")
    return destination


def main() -> None:
    args = build_parser().parse_args()
    for field in ("data_root", "openood_results_root", "openood_root", "output_root"):
        setattr(args, field, getattr(args, field).resolve())
    if not torch.cuda.is_available():
        raise RuntimeError("OpenOOD Evaluator requires a CUDA GPU")
    add_openood_to_path(args.openood_root)
    evaluator_module = importlib.import_module("openood.evaluation_api.evaluator")
    evaluator_module.data_setup = lambda *_args, **_kwargs: None
    Evaluator = evaluator_module.Evaluator
    get_postprocessor = importlib.import_module(
        "openood.evaluation_api.postprocessor"
    ).get_postprocessor

    failures = []
    for benchmark in args.benchmarks:
        seeds = [args.seeds[0]] if benchmark == "imagenet1k" else args.seeds
        for seed in seeds:
            network, preprocessor, model_tag, model_source = _model_for_benchmark(
                benchmark, args, seed
            )
            network = network.cuda().eval()
            for method in args.methods:
                target = _target(args.output_root, benchmark, model_tag, seed, method)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_file() and not args.force:
                    print(f"[skip] {target}")
                    continue
                print(f"[run] {method} | {benchmark} | seed={seed}", flush=True)
                evaluator = None
                try:
                    evaluator_kwargs = dict(
                        net=network,
                        id_name=OPENOOD_ID_NAME[benchmark],
                        data_root=str(args.data_root),
                        config_root=str(args.openood_root / "configs"),
                        preprocessor=preprocessor,
                        batch_size=_method_batch_size(benchmark, method, args),
                        shuffle=False,
                        num_workers=args.num_workers,
                    )
                    if method in {"vim", "rankfeat"}:
                        postprocessor = get_postprocessor(
                            str(args.openood_root / "configs"),
                            method,
                            OPENOOD_ID_NAME[benchmark],
                        )
                        if method == "vim":
                            feature_dim = _network_feature_dim(network)
                            candidates = _valid_vim_dimensions(
                                feature_dim,
                                postprocessor.args_dict["dim_list"],
                            )
                            postprocessor.args_dict["dim_list"] = candidates
                            if not 0 < int(postprocessor.dim) < feature_dim:
                                postprocessor.dim = candidates[0]
                            print(
                                f"[vim] feature_dim={feature_dim}, "
                                f"valid_dim_list={candidates}",
                                flush=True,
                            )
                        else:
                            postprocessor.args.accelerate = bool(
                                args.rankfeat_accelerate
                            )
                            print(
                                "[rankfeat] "
                                f"accelerate={postprocessor.args.accelerate}, "
                                f"batch_size={evaluator_kwargs['batch_size']}",
                                flush=True,
                            )
                        evaluator_kwargs["postprocessor"] = postprocessor
                    else:
                        evaluator_kwargs["postprocessor_name"] = method
                    evaluator = Evaluator(**evaluator_kwargs)
                    if args.max_eval_samples > 0:
                        limit_evaluator_for_smoke_test(
                            evaluator, args.max_eval_samples
                        )
                    metrics = evaluator.eval_ood(
                        fsood=False, progress=not args.no_progress
                    )
                    metrics.to_csv(target, float_format="%.6f")
                    target.with_suffix(".json").write_text(
                        json.dumps({
                            "benchmark": benchmark,
                            "seed": seed,
                            "method": method,
                            "model_tag": model_tag,
                            "model_source": model_source,
                            "protocol": "OpenOOD v1.5 local reproduction",
                            "tuning_protocol": _tuning_protocol(method),
                            "selected_hyperparameters":
                                _selected_hyperparameters(
                                    evaluator.postprocessor
                                ),
                            "implementation": getattr(
                                evaluator.postprocessor,
                                "implementation_note",
                                "Unmodified OpenOOD v1.5 postprocessor",
                            ),
                            "batch_size": evaluator_kwargs["batch_size"],
                            "max_eval_samples": args.max_eval_samples,
                            "rankfeat_accelerate": (
                                bool(args.rankfeat_accelerate)
                                if method == "rankfeat" else None
                            ),
                        }, indent=2), encoding="utf-8"
                    )
                    # A successful resumable retry supersedes any traceback
                    # left by an earlier incompatible network adapter.
                    target.with_suffix(".failed.txt").unlink(missing_ok=True)
                    print(f"[saved] {target}", flush=True)
                except Exception as error:
                    failure = target.with_suffix(".failed.txt")
                    failure.write_text(traceback.format_exc(), encoding="utf-8")
                    failures.append((benchmark, seed, method, str(error)))
                    print(f"[failed] {benchmark} seed={seed} {method}: {error}",
                          flush=True)
                    if not args.continue_on_error:
                        raise
                finally:
                    if evaluator is not None:
                        del evaluator
                    torch.cuda.empty_cache()
            del network
            torch.cuda.empty_cache()

    combined = rebuild_long_form(args.output_root)
    print(f"[saved] {combined}")
    if failures:
        print("\nFailures:")
        for failure in failures:
            print(failure)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
