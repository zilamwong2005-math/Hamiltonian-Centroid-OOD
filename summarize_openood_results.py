"""Aggregate OpenOOD Hamiltonian runs across seeds into paper-ready CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


INPUT_NAME = "openood_metrics_all_potentials.csv"
GROUP_COLUMNS = [
    "ID",
    "Potential",
    "Dataset",
    "ModelSource",
    "MassMode",
    "MassNormalization",
    "BandwidthLoss",
    "TrajectoryTrainSteps",
    "PredictionSource",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--strict-seeds",
        action="store_true",
        help="Fail instead of warning when a configuration lacks an expected seed",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.output_root.resolve()
    paths = sorted(root.rglob(INPUT_NAME))
    if not paths:
        raise FileNotFoundError(f"No {INPUT_NAME} files found under {root}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["SourceFile"] = str(path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)

    required = set(GROUP_COLUMNS + ["Seed"])
    missing_columns = sorted(required - set(combined.columns))
    if missing_columns:
        raise RuntimeError(f"Input CSVs lack required columns: {missing_columns}")

    # A rerun writes the same configuration path, but de-duplicate defensively
    # in case users copied result directories before aggregating them.
    metric_columns = [
        column
        for column in combined.select_dtypes(include="number").columns
        if column not in {"Seed", "TrajectoryTrainSteps"}
    ]
    identity = GROUP_COLUMNS + ["Seed"]
    combined = combined.drop_duplicates(subset=identity, keep="last")

    expected = set(args.expected_seeds)
    missing_messages = []
    for key, group in combined.groupby(GROUP_COLUMNS, dropna=False):
        absent = sorted(expected - set(group["Seed"].astype(int)))
        if absent:
            missing_messages.append(f"{key}: missing seeds {absent}")
    if missing_messages:
        preview = "\n".join(missing_messages[:20])
        message = (
            f"{len(missing_messages)} configurations are incomplete:\n{preview}"
        )
        if args.strict_seeds:
            raise RuntimeError(message)
        print(f"WARNING: {message}")

    grouped = combined.groupby(GROUP_COLUMNS, dropna=False)[metric_columns]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in summary.columns
    ]

    combined_path = root / "openood_all_runs_combined.csv"
    summary_path = root / "openood_summary_mean_std.csv"
    combined.to_csv(combined_path, index=False, float_format="%.6f")
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    print(f"Combined rows: {len(combined):,} -> {combined_path}")
    print(f"Mean/std rows: {len(summary):,} -> {summary_path}")


if __name__ == "__main__":
    main()
