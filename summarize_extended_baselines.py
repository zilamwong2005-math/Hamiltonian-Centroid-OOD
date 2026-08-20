"""Validate and summarize ASH/DICE/SHE/RMDS/RankFeat reproductions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METHODS = ("ash", "dice", "she", "rmds", "rankfeat")
METHOD_LABELS = {
    "ash": "ASH",
    "dice": "DICE",
    "she": "SHE",
    "rmds": "RMDS",
    "rankfeat": "RankFeat",
}
BENCHMARK_LABELS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "imagenet200": "ImageNet-200",
    "imagenet1k": "ImageNet-1K",
}
EXPECTED_SEEDS = {
    "cifar10": {0, 1, 2},
    "cifar100": {0, 1, 2},
    "imagenet200": {0, 1, 2},
    # The torchvision ResNet-50 classifier is shared and deterministic.
    "imagenet1k": {0},
}
METRICS = ("FPR@95", "AUROC", "AUPR_IN", "AUPR_OUT", "ACC")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root", type=Path,
        default=Path("results_openood_baselines"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/journal/summary_extended_baselines"),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def _format_mean_std(mean: float, std: float, count: int) -> str:
    if count <= 1 or pd.isna(std):
        return f"{mean:.2f}"
    return f"{mean:.2f}±{std:.2f}"


def summarize(frame: pd.DataFrame, *, strict: bool = True):
    required = {
        "Benchmark", "Seed", "Method", "Dataset", "MaxEvalSamples", *METRICS
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"baseline_all_runs.csv lacks columns: {missing}")

    frame = frame.copy()
    for column in ("Benchmark", "Method", "Dataset"):
        frame[column] = frame[column].astype(str).str.lower()
    frame["Seed"] = pd.to_numeric(frame["Seed"]).astype(int)
    frame["MaxEvalSamples"] = pd.to_numeric(
        frame["MaxEvalSamples"], errors="coerce"
    ).fillna(0).astype(int)
    for metric in METRICS:
        frame[metric] = pd.to_numeric(frame[metric])
    frame = frame[
        frame["Method"].isin(METHODS)
        & frame["Dataset"].isin(["nearood", "farood"])
        & frame["MaxEvalSamples"].eq(0)
    ].copy()
    frame = frame.drop_duplicates(
        ["Benchmark", "Method", "Seed", "Dataset"], keep="last"
    )

    completeness_rows = []
    errors = []
    for benchmark, expected_seeds in EXPECTED_SEEDS.items():
        for method in METHODS:
            for dataset in ("nearood", "farood"):
                actual = set(frame.loc[
                    frame["Benchmark"].eq(benchmark)
                    & frame["Method"].eq(method)
                    & frame["Dataset"].eq(dataset),
                    "Seed",
                ])
                passed = actual == expected_seeds
                completeness_rows.append({
                    "Benchmark": benchmark,
                    "Method": method,
                    "Dataset": dataset,
                    "ExpectedSeeds": ",".join(map(str, sorted(expected_seeds))),
                    "ActualSeeds": ",".join(map(str, sorted(actual))),
                    "Passed": passed,
                })
                if not passed:
                    errors.append(
                        f"{benchmark}/{method}/{dataset}: expected "
                        f"{sorted(expected_seeds)}, found {sorted(actual)}"
                    )
    if strict and errors:
        raise RuntimeError(
            "Extended baseline matrix is incomplete:\n" + "\n".join(errors)
        )

    summary = frame.groupby(
        ["Benchmark", "Method", "Dataset"], sort=False
    )[list(METRICS)].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        column if isinstance(column, str)
        else "_".join(item for item in column if item)
        for column in summary.columns
    ]
    summary["BenchmarkLabel"] = summary["Benchmark"].map(BENCHMARK_LABELS)
    summary["MethodLabel"] = summary["Method"].map(METHOD_LABELS)
    summary["OODGroup"] = summary["Dataset"].map({
        "nearood": "Near-OOD", "farood": "Far-OOD"
    })

    paper = summary[["BenchmarkLabel", "MethodLabel", "OODGroup"]].copy()
    paper.columns = ["Benchmark", "Method", "OOD Group"]
    for metric in METRICS:
        paper[metric] = summary.apply(
            lambda row: _format_mean_std(
                row[f"{metric}_mean"], row[f"{metric}_std"],
                int(row[f"{metric}_count"]),
            ),
            axis=1,
        )
    return frame, summary, paper, pd.DataFrame(completeness_rows)


def main() -> None:
    args = build_parser().parse_args()
    args.baseline_root = args.baseline_root.resolve()
    args.output_dir = args.output_dir.resolve()
    source = args.baseline_root / "baseline_all_runs.csv"
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing {source}; rerun run_openood_baselines.py to rebuild it"
        )
    raw, summary, paper, completeness = summarize(
        pd.read_csv(source), strict=not args.allow_incomplete
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "extended_baselines_raw.csv", index=False)
    summary.to_csv(
        args.output_dir / "extended_baselines_mean_std.csv",
        index=False, float_format="%.6f",
    )
    completeness.to_csv(
        args.output_dir / "extended_baselines_completeness.csv", index=False
    )
    paper.to_csv(args.output_dir / "extended_baselines_paper_table.csv",
                 index=False)
    (args.output_dir / "extended_baselines_paper_table.tex").write_text(
        paper.to_latex(
            index=False,
            escape=True,
            caption=("OpenOOD v1.5 reproduction of additional post-hoc "
                     "OOD detection baselines."),
            label="tab:extended_baselines",
        ),
        encoding="utf-8",
    )
    print("===== Extended baseline paper table =====")
    print(paper.to_string(index=False))
    print("\n===== Completeness =====")
    print(completeness.groupby("Passed").size().to_string())
    print(f"\nSaved under: {args.output_dir}")


if __name__ == "__main__":
    main()
