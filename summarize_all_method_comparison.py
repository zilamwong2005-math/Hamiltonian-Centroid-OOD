"""Merge all local baselines with the locked method and compute fair ranks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ("FPR@95", "AUROC", "AUPR_IN", "AUPR_OUT", "ACC")
BENCHMARK_LABELS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "imagenet200": "ImageNet-200",
    "imagenet1k": "ImageNet-1K",
}
METHOD_LABELS = {
    "msp": "MSP", "ebo": "Energy", "mls": "MaxLogit", "gen": "GEN",
    "react": "ReAct", "scale": "Scale", "knn": "KNN", "vim": "ViM",
    "ash": "ASH", "dice": "DICE", "she": "SHE", "rmds": "RMDS",
    "rankfeat": "RankFeat", "centroid_only": "Centroid-only",
    "locked_fusion": "Locked Centroid-MSP",
}
MAIN_METHODS = (
    "msp", "ebo", "react", "scale", "knn", "vim", "ash", "rmds",
    "locked_fusion",
)
EXPECTED_TASKS = {
    f"{benchmark}/{group}"
    for benchmark in BENCHMARK_LABELS for group in ("nearood", "farood")
}
REQUIRED_COMPLETE_METHODS = {
    "msp", "ebo", "mls", "gen", "react", "scale", "ash", "dice",
    "she", "rmds", "rankfeat", "centroid_only", "locked_fusion",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path,
                        default=Path("results_openood_baselines"))
    parser.add_argument("--component-summary", type=Path, default=Path(
        "results/journal/summary_component_ablation/component_mean_std.csv"
    ))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "results/journal/summary_all_methods"
    ))
    return parser


def _baseline_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "Benchmark", "Method", "Seed", "Dataset", "MaxEvalSamples", *METRICS
    }
    if missing := sorted(required - set(frame.columns)):
        raise RuntimeError(f"Baseline CSV lacks: {missing}")
    for column in ("Benchmark", "Method", "Dataset"):
        frame[column] = frame[column].astype(str).str.lower()
    frame = frame[
        frame["Dataset"].isin(["nearood", "farood"])
        & pd.to_numeric(frame["MaxEvalSamples"], errors="coerce")
        .fillna(0).eq(0)
    ].copy()
    for metric in METRICS:
        frame[metric] = pd.to_numeric(frame[metric])
    frame = frame.drop_duplicates(
        ["Benchmark", "Method", "Seed", "Dataset"], keep="last"
    )
    summary = frame.groupby(
        ["Benchmark", "Method", "Dataset"]
    )[list(METRICS)].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        column if isinstance(column, str)
        else "_".join(item for item in column if item)
        for column in summary.columns
    ]
    return summary


def _component_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[
        frame["Target"].isin([
            "cifar10", "cifar100", "imagenet200", "imagenet1k_resnet50"
        ])
        & frame["Method"].isin(["centroid_only", "locked_fusion"])
    ].copy()
    frame["Benchmark"] = frame["Target"].replace({
        "imagenet1k_resnet50": "imagenet1k"
    })
    return frame[[
        "Benchmark", "Method", "Dataset",
        *[f"{metric}_{stat}" for metric in METRICS
          for stat in ("mean", "std", "count")],
    ]]


def _format(mean, std, count) -> str:
    if int(count) <= 1 or pd.isna(std):
        return f"{mean:.2f}"
    return f"{mean:.2f}±{std:.2f}"


def summarize(baseline: pd.DataFrame, component: pd.DataFrame):
    frame = pd.concat([baseline, component], ignore_index=True)
    frame = frame.drop_duplicates(
        ["Benchmark", "Method", "Dataset"], keep="last"
    )
    frame["Task"] = frame["Benchmark"] + "/" + frame["Dataset"]
    frame["AUROC_Rank"] = frame.groupby("Task")["AUROC_mean"].rank(
        ascending=False, method="min"
    )
    frame["FPR95_Rank"] = frame.groupby("Task")["FPR@95_mean"].rank(
        ascending=True, method="min"
    )
    frame["CompositeRank"] = (
        frame["AUROC_Rank"] + frame["FPR95_Rank"]
    ) / 2.0
    frame["BenchmarkLabel"] = frame["Benchmark"].map(BENCHMARK_LABELS)
    frame["MethodLabel"] = frame["Method"].map(METHOD_LABELS)
    frame["OODGroup"] = frame["Dataset"].map({
        "nearood": "Near-OOD", "farood": "Far-OOD"
    })

    completeness_errors = []
    for method in sorted(REQUIRED_COMPLETE_METHODS):
        tasks = set(frame.loc[frame["Method"].eq(method), "Task"])
        if tasks != EXPECTED_TASKS:
            completeness_errors.append(
                f"{method}: missing={sorted(EXPECTED_TASKS - tasks)}, "
                f"extra={sorted(tasks - EXPECTED_TASKS)}"
            )
    if completeness_errors:
        raise RuntimeError(
            "Unified comparison matrix is incomplete:\n"
            + "\n".join(completeness_errors)
        )

    ranks = frame.groupby("Method").agg(
        TaskCount=("Task", "nunique"),
        MeanAUROCRank=("AUROC_Rank", "mean"),
        MeanFPR95Rank=("FPR95_Rank", "mean"),
        MeanCompositeRank=("CompositeRank", "mean"),
        MeanAUROC=("AUROC_mean", "mean"),
        MeanFPR95=("FPR@95_mean", "mean"),
    ).reset_index()
    ranks["FullEightTaskCoverage"] = ranks["TaskCount"].eq(8)
    ranks["MethodLabel"] = ranks["Method"].map(METHOD_LABELS)
    ranks = ranks.sort_values(
        ["FullEightTaskCoverage", "MeanCompositeRank"],
        ascending=[False, True],
    )

    locked = frame[frame["Method"].eq("locked_fusion")][[
        "Task", "AUROC_mean", "FPR@95_mean"
    ]].rename(columns={
        "AUROC_mean": "Locked_AUROC", "FPR@95_mean": "Locked_FPR95"
    })
    pairwise = frame[~frame["Method"].eq("locked_fusion")].merge(
        locked, on="Task", how="inner"
    )
    pairwise["Delta_AUROC_Locked_minus_Method"] = (
        pairwise["Locked_AUROC"] - pairwise["AUROC_mean"]
    )
    pairwise["Delta_FPR95_Locked_minus_Method"] = (
        pairwise["Locked_FPR95"] - pairwise["FPR@95_mean"]
    )
    pairwise["LockedWinsAUROC"] = (
        pairwise["Delta_AUROC_Locked_minus_Method"] > 0
    )
    pairwise["LockedWinsFPR95"] = (
        pairwise["Delta_FPR95_Locked_minus_Method"] < 0
    )
    pairwise["LockedWinsBoth"] = (
        pairwise["LockedWinsAUROC"] & pairwise["LockedWinsFPR95"]
    )
    wins = pairwise.groupby("Method").agg(
        ComparedTasks=("Task", "nunique"),
        LockedAUROC_Wins=("LockedWinsAUROC", "sum"),
        LockedFPR95_Wins=("LockedWinsFPR95", "sum"),
        LockedBoth_Wins=("LockedWinsBoth", "sum"),
        MeanDeltaAUROC=("Delta_AUROC_Locked_minus_Method", "mean"),
        MeanDeltaFPR95=("Delta_FPR95_Locked_minus_Method", "mean"),
    ).reset_index()
    wins["MethodLabel"] = wins["Method"].map(METHOD_LABELS)

    main = frame[frame["Method"].isin(MAIN_METHODS)].copy()
    paper = main[["BenchmarkLabel", "OODGroup", "MethodLabel"]].copy()
    paper.columns = ["Benchmark", "OOD Group", "Method"]
    paper["AUROC"] = main.apply(
        lambda row: _format(
            row["AUROC_mean"], row["AUROC_std"], row["AUROC_count"]
        ), axis=1
    )
    paper["FPR95"] = main.apply(
        lambda row: _format(
            row["FPR@95_mean"], row["FPR@95_std"], row["FPR@95_count"]
        ), axis=1
    )
    method_order = {method: index for index, method in enumerate(MAIN_METHODS)}
    main_order = main["Method"].map(method_order)
    paper = paper.assign(_order=main_order.to_numpy()).sort_values(
        ["Benchmark", "OOD Group", "_order"]
    ).drop(columns="_order")
    return frame, ranks, pairwise, wins, paper


def main() -> None:
    args = build_parser().parse_args()
    baseline_path = args.baseline_root.resolve() / "baseline_all_runs.csv"
    component_path = args.component_summary.resolve()
    if not baseline_path.is_file() or not component_path.is_file():
        raise FileNotFoundError(
            f"Missing baseline/component inputs: {baseline_path}, {component_path}"
        )
    outputs = summarize(
        _baseline_summary(baseline_path), _component_summary(component_path)
    )
    frame, ranks, pairwise, wins, paper = outputs
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "all_methods_task_metrics.csv", index=False)
    ranks.to_csv(args.output_dir / "all_methods_ranking.csv", index=False)
    pairwise.to_csv(args.output_dir / "locked_pairwise_deltas.csv", index=False)
    wins.to_csv(args.output_dir / "locked_win_summary.csv", index=False)
    paper.to_csv(args.output_dir / "main_comparison_paper_table.csv", index=False)
    (args.output_dir / "main_comparison_paper_table.tex").write_text(
        paper.to_latex(
            index=False, escape=True,
            caption="Unified local OpenOOD v1.5 comparison.",
            label="tab:unified_comparison",
        ), encoding="utf-8"
    )
    print("===== Full-coverage average ranking =====")
    print(ranks[ranks["FullEightTaskCoverage"]].round(4).to_string(index=False))
    print("\n===== Locked-method win summary =====")
    print(wins.round(4).to_string(index=False))
    print(f"\nSaved under: {args.output_dir}")


if __name__ == "__main__":
    main()
