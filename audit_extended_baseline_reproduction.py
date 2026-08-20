"""Audit the five added OpenOOD baselines before paper reporting.

This is a protocol/configuration audit, not a claim that our local numbers
exactly reproduce the numbers printed in each method's original paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Imagenet_ood_experiment import add_openood_to_path


BENCHMARK_SEEDS = {
    "cifar10": (0, 1, 2),
    "cifar100": (0, 1, 2),
    "imagenet200": (0, 1, 2),
    "imagenet1k": (0,),
}
ID_TEST_COUNTS = {
    "cifar10": 9000,
    "cifar100": 9000,
    "imagenet200": 9000,
    "imagenet1k": 45000,
}
METHODS = ("ash", "dice", "she", "rmds", "rankfeat")
METRICS = ("FPR@95", "AUROC", "AUPR_IN", "AUPR_OUT", "ACC")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path,
                        default=Path("results_openood_baselines"))
    parser.add_argument("--openood-root", type=Path, default=Path("OpenOOD"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "results/journal/audit_extended_baselines"
    ))
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_metric_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    unnamed = [column for column in frame if column.startswith("Unnamed:")]
    if "Dataset" not in frame:
        frame = frame.rename(columns={
            unnamed[0] if unnamed else frame.columns[0]: "Dataset"
        })
    for metric in METRICS:
        if metric not in frame:
            raise RuntimeError(f"{path} lacks {metric}")
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    return frame


def _discover_configs(root: Path) -> dict[tuple[str, int, str], Path]:
    discovered = {}
    for method in METHODS:
        for path in root.rglob(f"{method}.json"):
            try:
                config = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            benchmark = str(config.get("benchmark", "")).lower()
            if benchmark not in BENCHMARK_SEEDS:
                continue
            key = (benchmark, int(config.get("seed", 0)), method)
            if int(config.get("max_eval_samples", -1)) == 0:
                discovered[key] = path
    return discovered


def _loop_relative_mahalanobis(features, class_mean, precision,
                               whole_mean, whole_precision):
    centered = features - whole_mean.view(1, -1)
    background = -torch.sum(
        torch.matmul(centered, whole_precision) * centered, dim=1
    )
    scores = []
    for mean in class_mean:
        delta = features - mean.view(1, -1)
        scores.append(
            -torch.sum(torch.matmul(delta, precision) * delta, dim=1)
            - background
        )
    return torch.stack(scores, dim=1)


def _rmds_equivalence(openood_root: Path) -> dict:
    add_openood_to_path(openood_root)
    from openood.postprocessors.rmds_postprocessor import (
        vectorized_relative_mahalanobis,
    )

    generator = torch.Generator().manual_seed(20260818)
    features = torch.randn(17, 9, generator=generator, dtype=torch.float64)
    class_mean = torch.randn(7, 9, generator=generator, dtype=torch.float64)
    whole_mean = torch.randn(9, generator=generator, dtype=torch.float64)
    matrix = torch.randn(9, 9, generator=generator, dtype=torch.float64)
    precision = matrix.t().matmul(matrix) + torch.eye(9, dtype=torch.float64)
    matrix = torch.randn(9, 9, generator=generator, dtype=torch.float64)
    whole_precision = (
        matrix.t().matmul(matrix) + torch.eye(9, dtype=torch.float64)
    )
    vectorized = vectorized_relative_mahalanobis(
        features, class_mean, precision, whole_mean, whole_precision
    )
    reference = _loop_relative_mahalanobis(
        features, class_mean, precision, whole_mean, whole_precision
    )
    maximum = float((vectorized - reference).abs().max())
    return {
        "test": "RMDS vectorized score versus direct class loop",
        "dtype": "float64",
        "max_abs_difference": maximum,
        "tolerance": 1e-10,
        "passed": maximum <= 1e-10,
    }


def _streaming_scatter_equivalence() -> dict:
    generator = torch.Generator().manual_seed(20260818)
    samples = torch.randn(97, 11, generator=generator, dtype=torch.float64)
    labels = torch.randint(0, 5, (97,), generator=generator)
    counts = torch.zeros(5, dtype=torch.float64)
    means = torch.zeros(5, 11, dtype=torch.float64)
    scatter = torch.zeros(11, 11, dtype=torch.float64)
    for start in range(0, len(samples), 13):
        feature = samples[start:start + 13]
        target = labels[start:start + 13]
        unique, inverse = torch.unique(target, sorted=True, return_inverse=True)
        group_count = torch.bincount(
            inverse, minlength=len(unique)
        ).to(torch.float64)
        group_sum = torch.zeros(len(unique), feature.shape[1], dtype=torch.float64)
        group_sum.index_add_(0, inverse, feature)
        group_mean = group_sum / group_count[:, None]
        residual = feature - group_mean[inverse]
        scatter.add_(residual.t().matmul(residual))
        old_count = counts[unique]
        old_mean = means[unique]
        merged_count = old_count + group_count
        delta = group_mean - old_mean
        coefficient = old_count * group_count / merged_count
        correction = delta * coefficient.clamp_min(0).sqrt()[:, None]
        scatter.add_(correction.t().matmul(correction))
        means[unique] = old_mean + delta * (group_count / merged_count)[:, None]
        counts[unique] = merged_count

    direct_scatter = torch.zeros_like(scatter)
    direct_means = torch.zeros_like(means)
    for label in range(5):
        subset = samples[labels.eq(label)]
        direct_means[label] = subset.mean(0)
        residual = subset - direct_means[label]
        direct_scatter.add_(residual.t().matmul(residual))
    maximum = max(
        float((scatter - direct_scatter).abs().max()),
        float((means - direct_means).abs().max()),
    )
    return {
        "test": "RMDS parallel streaming means/scatter versus direct tensors",
        "dtype": "float64",
        "max_abs_difference": maximum,
        "tolerance": 1e-10,
        "passed": maximum <= 1e-10,
    }


def audit(baseline_root: Path, openood_root: Path):
    configs = _discover_configs(baseline_root)
    expected = {
        (benchmark, seed, method)
        for benchmark, seeds in BENCHMARK_SEEDS.items()
        for seed in seeds for method in METHODS
    }
    missing = sorted(expected - set(configs))
    extra = sorted(set(configs) - expected)
    rows = []
    problems = []
    if missing:
        problems.append(f"Missing formal configs: {missing}")
    if extra:
        problems.append(f"Unexpected formal configs: {extra}")

    for key in sorted(expected & set(configs)):
        benchmark, seed, method = key
        config_path = configs[key]
        result_path = config_path.with_suffix(".csv")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        row = {
            "Benchmark": benchmark,
            "Seed": seed,
            "Method": method,
            "ResultFile": str(result_path),
            "ConfigFile": str(config_path),
            "ResultSHA256": _sha256(result_path) if result_path.is_file() else "",
            "ConfigSHA256": _sha256(config_path),
            "MaxEvalSamples": config.get("max_eval_samples"),
            "TuningProtocol": config.get("tuning_protocol"),
            "SelectedHyperparameters": json.dumps(
                config.get("selected_hyperparameters"), ensure_ascii=False
            ),
            "Implementation": config.get("implementation"),
            "RankFeatAccelerate": config.get("rankfeat_accelerate"),
        }
        passed = result_path.is_file()
        if not result_path.is_file():
            problems.append(f"Missing result for {key}: {result_path}")
        else:
            metrics = _read_metric_csv(result_path)
            required_groups = {"nearood", "farood"}
            datasets = set(metrics["Dataset"].astype(str).str.lower())
            finite = bool(np.isfinite(metrics[list(METRICS)].to_numpy()).all())
            passed &= required_groups.issubset(datasets) and finite
            row.update({
                "MetricRows": len(metrics),
                "HasNearFar": required_groups.issubset(datasets),
                "AllMetricsFinite": finite,
                "IDAccuracy": float(metrics["ACC"].iloc[0]),
            })
        if method == "rankfeat":
            passed &= config.get("rankfeat_accelerate") is False
        if method == "ash":
            passed &= config.get("selected_hyperparameters") is not None
            passed &= "validation" in str(
                config.get("tuning_protocol", "")
            ).lower()
        passed &= int(config.get("max_eval_samples", -1)) == 0
        row["Passed"] = passed
        if not passed:
            problems.append(f"Protocol/result audit failed for {key}")
        rows.append(row)

    frame = pd.DataFrame(rows)
    id_accuracy_variation = {}
    id_accuracy_tolerance = {}
    if not frame.empty:
        variation = frame.groupby(["Benchmark", "Seed"])["IDAccuracy"].agg(
            lambda values: float(values.max() - values.min())
        )
        excessive = {}
        for (benchmark, seed), difference in variation.items():
            label = f"{benchmark}/seed{seed}"
            # One sample can flip when wrappers follow a slightly different
            # CUDA arithmetic path.  Record it, but only fail when the spread
            # exceeds one test image (plus floating-point serialization slack).
            tolerance = 100.0 / ID_TEST_COUNTS[benchmark] + 1e-6
            id_accuracy_variation[label] = float(difference)
            id_accuracy_tolerance[label] = tolerance
            if difference > tolerance:
                excessive[label] = {
                    "difference_percentage_points": float(difference),
                    "one_sample_tolerance_percentage_points": tolerance,
                }
        if excessive:
            problems.append(
                "ID accuracy spread exceeds one test sample: "
                f"{excessive}"
            )

    rankfeat_config = openood_root / "configs/postprocessors/rankfeat.yml"
    source_files = {
        "rankfeat_config": rankfeat_config,
        "rankfeat_postprocessor": (
            openood_root / "openood/postprocessors/rankfeat_postprocessor.py"
        ),
        "rmds_postprocessor": (
            openood_root / "openood/postprocessors/rmds_postprocessor.py"
        ),
        "dice_postprocessor": (
            openood_root / "openood/postprocessors/dice_postprocessor.py"
        ),
        "she_postprocessor": (
            openood_root / "openood/postprocessors/she_postprocessor.py"
        ),
    }
    missing_sources = [name for name, path in source_files.items()
                       if not path.is_file()]
    if missing_sources:
        problems.append(f"Missing audited source files: {missing_sources}")
    rankfeat_text = (rankfeat_config.read_text(encoding="utf-8")
                     if rankfeat_config.is_file() else "")
    rankfeat_checks = {
        "APS_mode_false": "APS_mode: False" in rankfeat_text,
        "accelerate_false": "accelerate: False" in rankfeat_text,
        "temperature_one": "temperature: 1" in rankfeat_text,
    }
    if not all(rankfeat_checks.values()):
        problems.append(f"RankFeat config mismatch: {rankfeat_checks}")

    equivalence = [
        _rmds_equivalence(openood_root),
        _streaming_scatter_equivalence(),
    ]
    if not all(item["passed"] for item in equivalence):
        problems.append(f"RMDS equivalence failed: {equivalence}")
    report = {
        "audit_scope": (
            "Local OpenOOD protocol/configuration and engineering-equivalence "
            "audit; not an external published-number reproduction claim"
        ),
        "expected_formal_runs": len(expected),
        "audited_formal_runs": len(frame),
        "expected_matrix_complete": not missing and not extra,
        "no_smoke_results": bool(
            not frame.empty and frame["MaxEvalSamples"].eq(0).all()
        ),
        "rankfeat_full_svd_all_runs": bool(
            not frame.empty
            and frame.loc[frame["Method"].eq("rankfeat"),
                          "RankFeatAccelerate"].eq(False).all()
        ),
        "rankfeat_config_checks": rankfeat_checks,
        "id_accuracy_variation_percentage_points": id_accuracy_variation,
        "id_accuracy_one_sample_tolerance_percentage_points":
            id_accuracy_tolerance,
        "source_sha256": {
            name: _sha256(path) for name, path in source_files.items()
            if path.is_file()
        },
        "engineering_equivalence": equivalence,
        "problems": problems,
        "passed": not problems,
    }
    return frame, report


def main() -> None:
    args = build_parser().parse_args()
    baseline_root = args.baseline_root.resolve()
    openood_root = args.openood_root.resolve()
    output_dir = args.output_dir.resolve()
    frame, report = audit(baseline_root, openood_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "extended_baseline_file_audit.csv", index=False)
    (output_dir / "extended_baseline_reproduction_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("===== Extended baseline reproduction audit =====")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved under: {output_dir}")
    if not report["passed"]:
        raise RuntimeError("Extended baseline audit failed")


if __name__ == "__main__":
    main()
