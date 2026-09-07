"""Create fair batch-1 efficiency tables, including paired CTM comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METHOD_LABELS = {
    "MSP-end-to-end": "MSP",
    "Locked-centroid-MSP-end-to-end": "Locked Centroid-MSP",
    "Centroid-detector-only": "Centroid score only",
    "CTM-end-to-end": "CTM",
    "ash": "ASH",
    "dice": "DICE",
    "she": "SHE",
    "rmds": "RMDS",
    "rankfeat": "RankFeat",
    "scale": "Scale",
}
BENCHMARK_LABELS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "imagenet200": "ImageNet-200",
    "imagenet1k": "ImageNet-1K",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, default=Path(
        "results/journal/efficiency/efficiency.csv"
    ))
    parser.add_argument("--extended", type=Path, default=Path(
        "results/journal/efficiency/extended_posthoc_efficiency.csv"
    ))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "results/journal/efficiency/summary_required"
    ))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def summarize(current: pd.DataFrame, extended: pd.DataFrame,
              batch_size: int, seed: int = 0) -> pd.DataFrame:
    common = [
        "Benchmark", "Seed", "BatchSize", "GPU", "AuxiliaryStateMB",
        "MeanMilliseconds", "StdMilliseconds", "MedianMilliseconds",
        "P95Milliseconds", "ImagesPerSecond", "PeakAllocatedMB",
        "IncrementalPeakMB",
    ]
    required_current = set(common + [
        "Mode", "SetupSamples", "CentroidEstimationSamples",
        "SetupSeconds", "SetupProtocol",
    ])
    required_extended = set(common + [
        "Method", "SetupSeconds", "SetupProtocol", "FullSVD"
    ])
    if missing := sorted(required_current - set(current.columns)):
        raise RuntimeError(f"Current efficiency CSV lacks: {missing}")
    if missing := sorted(required_extended - set(extended.columns)):
        raise RuntimeError(f"Extended efficiency CSV lacks: {missing}")

    current = current[
        pd.to_numeric(current["BatchSize"]).eq(batch_size)
        & pd.to_numeric(current["Seed"]).eq(seed)
        & current["Mode"].isin([
            "MSP-end-to-end", "Locked-centroid-MSP-end-to-end",
            "Centroid-detector-only", "CTM-end-to-end",
        ])
    ][common + [
        "Mode", "SetupSamples", "CentroidEstimationSamples",
        "SetupSeconds", "SetupProtocol",
    ]].copy()
    current = current.rename(columns={"Mode": "Method"})
    current["FullSVD"] = False
    current["Source"] = "Paired MSP/CTM/proposed-method efficiency benchmark"

    extended = extended[
        pd.to_numeric(extended["BatchSize"]).eq(batch_size)
        & pd.to_numeric(extended["Seed"]).eq(seed)
    ][common + [
        "Method", "SetupSeconds", "SetupProtocol", "FullSVD"
    ]].copy()
    extended["SetupSamples"] = float("nan")
    extended["CentroidEstimationSamples"] = float("nan")
    extended["Source"] = "Extended-baseline efficiency benchmark"
    frame = pd.concat([current, extended], ignore_index=True)
    frame = frame.drop_duplicates(
        ["Benchmark", "Seed", "Method", "BatchSize"], keep="last"
    )
    expected = set(METHOD_LABELS)
    errors = []
    for benchmark in BENCHMARK_LABELS:
        actual = set(frame.loc[frame["Benchmark"].eq(benchmark), "Method"])
        if actual != expected:
            errors.append(
                f"{benchmark}: expected {sorted(expected)}, found {sorted(actual)}"
            )
    if errors:
        raise RuntimeError("Efficiency matrix incomplete:\n" + "\n".join(errors))
    gpus = set(frame["GPU"].dropna().astype(str))
    if len(gpus) != 1:
        raise RuntimeError(
            f"Efficiency rows were not measured on one shared GPU: {sorted(gpus)}"
        )

    msp = frame[frame["Method"].eq("MSP-end-to-end")][
        ["Benchmark", "MeanMilliseconds", "PeakAllocatedMB"]
    ].rename(columns={
        "MeanMilliseconds": "MSPMeanMilliseconds",
        "PeakAllocatedMB": "MSPPeakAllocatedMB",
    })
    frame = frame.merge(msp, on="Benchmark", how="left")
    frame["LatencyRatioVsMSP"] = (
        frame["MeanMilliseconds"] / frame["MSPMeanMilliseconds"]
    )
    frame["PeakMemoryDeltaVsMSP_MB"] = (
        frame["PeakAllocatedMB"] - frame["MSPPeakAllocatedMB"]
    )
    frame["BenchmarkLabel"] = frame["Benchmark"].map(BENCHMARK_LABELS)
    frame["MethodLabel"] = frame["Method"].map(METHOD_LABELS)
    return frame.sort_values(["Benchmark", "LatencyRatioVsMSP", "Method"])


def paired_ctm_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Contrast CTM and the locked score without assuming either is faster."""

    methods = {
        "CTM-end-to-end": "CTM",
        "Locked-centroid-MSP-end-to-end": "Locked",
    }
    subset = frame[frame["Method"].isin(methods)].copy()
    subset["Endpoint"] = subset["Method"].map(methods)
    required = set(BENCHMARK_LABELS)
    counts = subset.groupby("Benchmark")["Endpoint"].nunique()
    incomplete = sorted(required - set(counts[counts.eq(2)].index))
    if incomplete:
        raise RuntimeError(f"Paired CTM efficiency rows are incomplete: {incomplete}")

    columns = [
        "MeanMilliseconds", "StdMilliseconds", "AuxiliaryStateMB",
        "PeakAllocatedMB", "SetupSamples", "CentroidEstimationSamples",
        "SetupSeconds",
    ]
    wide = subset.pivot(index="Benchmark", columns="Endpoint", values=columns)
    rows = []
    for benchmark in BENCHMARK_LABELS:
        ctm_latency = float(wide.loc[benchmark, ("MeanMilliseconds", "CTM")])
        locked_latency = float(wide.loc[benchmark, ("MeanMilliseconds", "Locked")])
        ctm_samples = float(wide.loc[benchmark, ("SetupSamples", "CTM")])
        locked_samples = float(wide.loc[benchmark, ("SetupSamples", "Locked")])
        ctm_estimation = float(
            wide.loc[benchmark, ("CentroidEstimationSamples", "CTM")]
        )
        locked_estimation = float(
            wide.loc[benchmark, ("CentroidEstimationSamples", "Locked")]
        )
        rows.append({
            "Benchmark": benchmark,
            "BenchmarkLabel": BENCHMARK_LABELS[benchmark],
            "CTMLatencyMilliseconds": ctm_latency,
            "CTMLatencyStdMilliseconds": float(
                wide.loc[benchmark, ("StdMilliseconds", "CTM")]
            ),
            "LockedLatencyMilliseconds": locked_latency,
            "LockedLatencyStdMilliseconds": float(
                wide.loc[benchmark, ("StdMilliseconds", "Locked")]
            ),
            "LockedLatencyRatioVsCTM": locked_latency / ctm_latency,
            "LockedLatencyDeltaVsCTMPercent": (
                locked_latency / ctm_latency - 1.0
            ) * 100.0,
            "CTMAuxiliaryStateMB": float(
                wide.loc[benchmark, ("AuxiliaryStateMB", "CTM")]
            ),
            "LockedAuxiliaryStateMB": float(
                wide.loc[benchmark, ("AuxiliaryStateMB", "Locked")]
            ),
            "CTMSetupSamples": int(ctm_samples),
            "LockedSetupSamples": int(locked_samples),
            "CTMSetupSampleRatioVsLocked": ctm_samples / locked_samples,
            "CTMCentroidEstimationSamples": int(ctm_estimation),
            "LockedCentroidEstimationSamples": int(locked_estimation),
            "CTMCentroidEstimationSampleRatioVsLocked": (
                ctm_estimation / locked_estimation
            ),
            "CTMRecordedSetupSeconds": float(
                wide.loc[benchmark, ("SetupSeconds", "CTM")]
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser().parse_args()
    frame = summarize(
        pd.read_csv(args.current.resolve()),
        pd.read_csv(args.extended.resolve()),
        args.batch_size,
        args.seed,
    )
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        args.output_dir / "efficiency_comparison_batch1.csv",
        index=False, float_format="%.6f",
    )
    paired = paired_ctm_summary(frame)
    paired.to_csv(
        args.output_dir / "ctm_paired_efficiency.csv",
        index=False, float_format="%.6f",
    )
    paper = frame[[
        "BenchmarkLabel", "MethodLabel", "MeanMilliseconds",
        "LatencyRatioVsMSP", "AuxiliaryStateMB", "PeakAllocatedMB",
        "SetupSamples", "CentroidEstimationSamples", "SetupSeconds",
        "SetupProtocol",
    ]].copy()
    paper.columns = [
        "Benchmark", "Method", "Latency (ms/image)", "Latency / MSP",
        "Aux. state (MB)", "Peak GPU (MB)", "Setup samples",
        "Centroid samples", "Offline setup (s)", "Offline setup protocol",
    ]
    paper.to_csv(
        args.output_dir / "efficiency_paper_table.csv",
        index=False, float_format="%.4f",
    )
    (args.output_dir / "efficiency_paper_table.tex").write_text(
        paper.to_latex(
            index=False, escape=True,
            float_format=lambda value: f"{value:.3f}",
            caption=("Batch-1 end-to-end inference efficiency on one shared "
                     "GPU; data loading is excluded."),
            label="tab:efficiency_comparison",
        ),
        encoding="utf-8",
    )
    print("===== Batch-1 efficiency comparison =====")
    print(paper.round(4).to_string(index=False))
    print("\n===== Paired CTM versus locked score =====")
    print(paired.round(4).to_string(index=False))
    print(f"\nSaved under: {args.output_dir}")


if __name__ == "__main__":
    main()
