"""Build paper-ready mean/std tables from CIFAR Hamiltonian runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CONFIG_COLUMNS = [
    "In-dist",
    "Model",
    "Protocol",
    "EncoderSource",
    "Potential",
    "MassMode",
    "MassNormalization",
    "MassResolution",
    "BandwidthLoss",
    "TrajectoryTrainSteps",
    "HamTrainSamplesPerClass",
    "Anchors",
    "Steps",
    "Dt",
    "CandidateK",
    "SigmaInit",
    "HamEpochs",
    "MaxEvalSamples",
]
METRICS = [
    "IDAccuracy",
    "AUROC",
    "AUPR_IN",
    "AUPR_OUT",
    "FPR95",
    "FPR95_IDTPR",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/cifar/ood_results_cifar_v3.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/cifar/summary"))
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--strict-seeds", action="store_true")
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include rows where MaxEvalSamples is non-zero",
    )
    return parser


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [
        column if isinstance(column, str) else "_".join(x for x in column if x)
        for column in frame.columns
    ]
    return frame


def _check_seeds(frame: pd.DataFrame, expected_seeds: set[int], strict: bool) -> None:
    missing = []
    identity = CONFIG_COLUMNS + ["OODGroup", "OOD"]
    for key, group in frame.groupby(identity, dropna=False):
        absent = sorted(expected_seeds - set(group["Seed"].astype(int)))
        if absent:
            missing.append(f"{key}: missing seeds {absent}")
    if missing:
        message = f"{len(missing)} configurations are incomplete:\n" + "\n".join(
            missing[:20]
        )
        if strict:
            raise RuntimeError(message)
        print(f"WARNING: {message}")


def main() -> None:
    args = build_parser().parse_args()
    frame = pd.read_csv(args.input)
    required = set(CONFIG_COLUMNS + ["Seed", "OODGroup", "OOD"] + METRICS)
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise RuntimeError(f"Input CSV lacks columns: {missing_columns}")

    if not args.include_smoke:
        frame = frame[frame["MaxEvalSamples"].astype(int) == 0].copy()
    if frame.empty:
        raise RuntimeError("No full-evaluation CIFAR rows were found")

    frame = frame.drop_duplicates(subset=["ExperimentID", "OOD"], keep="last")
    _check_seeds(frame, set(args.expected_seeds), args.strict_seeds)

    dataset_summary = (
        frame.groupby(CONFIG_COLUMNS + ["OODGroup", "OOD"], dropna=False)[METRICS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    dataset_summary = _flatten_columns(dataset_summary)

    # First make a macro average inside each seed; then aggregate those seed
    # means.  Pooling every image would overweight MNIST/Places365.
    per_seed_group = (
        frame.groupby(CONFIG_COLUMNS + ["Seed", "OODGroup"], dropna=False)[METRICS]
        .mean()
        .reset_index()
    )
    per_seed_all = (
        frame.groupby(CONFIG_COLUMNS + ["Seed"], dropna=False)[METRICS]
        .mean()
        .reset_index()
    )
    per_seed_all["OODGroup"] = "all"
    per_seed_group = pd.concat([per_seed_group, per_seed_all], ignore_index=True)
    group_summary = (
        per_seed_group.groupby(CONFIG_COLUMNS + ["OODGroup"], dropna=False)[METRICS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    group_summary = _flatten_columns(group_summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "cifar_per_dataset_mean_std.csv"
    group_path = args.output_dir / "cifar_group_mean_std.csv"
    dataset_summary.to_csv(dataset_path, index=False, float_format="%.6f")
    group_summary.to_csv(group_path, index=False, float_format="%.6f")
    print(f"Per-dataset summary: {dataset_path}")
    print(f"Near/Far/All summary: {group_path}")


if __name__ == "__main__":
    main()
