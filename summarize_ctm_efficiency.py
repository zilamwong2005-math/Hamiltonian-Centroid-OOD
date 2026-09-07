"""Summarize paired batch-one CTM and locked-score efficiency measurements."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from summarize_efficiency_comparison import BENCHMARK_LABELS, paired_ctm_summary


REQUIRED_MODES = (
    "MSP-end-to-end",
    "CTM-end-to-end",
    "Locked-centroid-MSP-end-to-end",
    "CTM-detector-only",
    "Locked-centroid-MSP-detector-only",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/journal/efficiency/efficiency.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/journal/efficiency/summary_ctm"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def summarize_ctm_efficiency(
    raw: pd.DataFrame, batch_size: int = 1, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = {
        "Benchmark", "Seed", "BatchSize", "Mode", "GPU",
        "MeanMilliseconds", "StdMilliseconds", "AuxiliaryStateMB",
        "PeakAllocatedMB", "SetupSamples", "CentroidEstimationSamples",
        "SetupSeconds",
    }
    missing = sorted(required_columns - set(raw.columns))
    if missing:
        raise RuntimeError(f"Efficiency CSV lacks columns: {missing}")
    selected = raw[
        pd.to_numeric(raw["BatchSize"]).eq(batch_size)
        & pd.to_numeric(raw["Seed"]).eq(seed)
        & raw["Mode"].isin(REQUIRED_MODES)
    ].copy()
    selected = selected.drop_duplicates(
        ["Benchmark", "Seed", "BatchSize", "Mode"], keep="last"
    )
    errors = []
    for benchmark in BENCHMARK_LABELS:
        actual = set(selected.loc[selected["Benchmark"].eq(benchmark), "Mode"])
        if actual != set(REQUIRED_MODES):
            errors.append(
                f"{benchmark}: expected {sorted(REQUIRED_MODES)}, "
                f"found {sorted(actual)}"
            )
    if errors:
        raise RuntimeError("Paired CTM timing matrix incomplete:\n" + "\n".join(errors))
    gpus = sorted(set(selected["GPU"].dropna().astype(str)))
    if len(gpus) != 1:
        raise RuntimeError(f"Expected one shared GPU, found: {gpus}")

    paired_input = selected.rename(columns={"Mode": "Method"})
    paired = paired_ctm_summary(paired_input)

    def metric_for(mode: str, column: str, label: str) -> pd.DataFrame:
        return selected[selected["Mode"].eq(mode)][
            ["Benchmark", column]
        ].rename(columns={column: label})

    additions = [
        metric_for("MSP-end-to-end", "MeanMilliseconds", "MSPLatencyMilliseconds"),
        metric_for("CTM-detector-only", "MeanMilliseconds", "CTMScoreMilliseconds"),
        metric_for(
            "Locked-centroid-MSP-detector-only",
            "MeanMilliseconds",
            "LockedScoreMilliseconds",
        ),
    ]
    for addition in additions:
        paired = paired.merge(addition, on="Benchmark", how="left", validate="one_to_one")
    paired["CTMLatencyRatioVsMSP"] = (
        paired["CTMLatencyMilliseconds"] / paired["MSPLatencyMilliseconds"]
    )
    paired["LockedLatencyRatioVsMSP"] = (
        paired["LockedLatencyMilliseconds"] / paired["MSPLatencyMilliseconds"]
    )
    paired["LockedScoreRatioVsCTM"] = (
        paired["LockedScoreMilliseconds"] / paired["CTMScoreMilliseconds"]
    )
    paired["GPU"] = gpus[0]

    paper = paired[[
        "BenchmarkLabel", "MSPLatencyMilliseconds", "CTMLatencyMilliseconds",
        "LockedLatencyMilliseconds", "CTMLatencyRatioVsMSP",
        "LockedLatencyRatioVsMSP", "LockedLatencyRatioVsCTM",
        "CTMSetupSamples", "LockedSetupSamples", "CTMSetupSampleRatioVsLocked",
        "CTMCentroidEstimationSamples", "LockedCentroidEstimationSamples",
        "CTMCentroidEstimationSampleRatioVsLocked",
    ]].copy()
    paper.columns = [
        "Benchmark", "MSP (ms)", "CTM (ms)", "Ours (ms)", "CTM / MSP",
        "Ours / MSP", "Ours / CTM", "CTM setup samples",
        "Ours setup samples", "CTM / ours setup samples",
        "CTM centroid samples", "Ours centroid samples",
        "CTM / ours centroid samples",
    ]
    return paired, paper


def main() -> None:
    args = build_parser().parse_args()
    paired, paper = summarize_ctm_efficiency(
        pd.read_csv(args.input.resolve()), args.batch_size, args.seed
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(
        output_dir / "ctm_paired_efficiency.csv",
        index=False,
        float_format="%.6f",
    )
    paper.to_csv(
        output_dir / "ctm_efficiency_paper_table.csv",
        index=False,
        float_format="%.6f",
    )
    (output_dir / "ctm_efficiency_paper_table.tex").write_text(
        paper.to_latex(
            index=False,
            escape=True,
            float_format=lambda value: f"{value:.3f}",
            caption=(
                "Paired batch-one end-to-end latency and centroid setup budget. "
                "All online timings use one shared GPU and exclude data loading."
            ),
            label="tab:ctm_efficiency",
        ),
        encoding="utf-8",
    )
    print("===== CTM paired batch-one efficiency =====")
    print(paper.round(4).to_string(index=False))
    print(f"\nSaved under: {output_dir}")


if __name__ == "__main__":
    main()
