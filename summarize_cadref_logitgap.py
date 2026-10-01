"""Validate and summarise the same-protocol CADRef/LogitGap experiment matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = (
    "cifar10",
    "cifar100",
    "imagenet200",
    "imagenet1k_resnet50",
    "imagenet1k_densenet121",
)
TARGET_LABELS = {
    "cifar10": "CIFAR-10 / ResNet-18",
    "cifar100": "CIFAR-100 / ResNet-18",
    "imagenet200": "ImageNet-200 / ResNet-18",
    "imagenet1k_resnet50": "ImageNet-1K / ResNet-50",
    "imagenet1k_densenet121": "ImageNet-1K / DenseNet-121",
}
METHODS = ("cadref", "logitgap")
METHOD_LABELS = {"cadref": "CADRef", "logitgap": "LogitGap"}
GROUP_LABELS = {"nearood": "Near-OOD", "farood": "Far-OOD"}
METRICS = ("FPR@95", "AUROC", "AUPR_IN", "AUPR_OUT", "ACC")
EXPECTED_RUNS = {
    "cifar10": 3,
    "cifar100": 3,
    "imagenet200": 3,
    "imagenet1k_resnet50": 1,
    "imagenet1k_densenet121": 1,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "full"), default="full")
    parser.add_argument(
        "--input-root", type=Path,
        default=Path("results/journal/cadref_logitgap"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("results/journal/summary_cadref_logitgap"),
    )
    return parser


def load_and_validate(input_root: Path, stage: str) -> pd.DataFrame:
    path = input_root / f"cadref_logitgap_{stage}_all_runs.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"Target", "Method", "Seed", "Dataset", *METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Combined result is missing columns: {missing}")
    frame["Seed"] = pd.to_numeric(frame["Seed"], errors="raise").astype(int)
    for metric in METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors="raise")
        if not np.isfinite(frame[metric]).all():
            raise RuntimeError(f"Non-finite values in {metric}")
    if set(frame["Target"]) != set(TARGETS):
        raise RuntimeError(f"Unexpected targets: {sorted(set(frame['Target']))}")
    if set(frame["Method"]) != set(METHODS):
        raise RuntimeError(f"Unexpected methods: {sorted(set(frame['Method']))}")

    if stage == "full":
        for target in TARGETS:
            for method in METHODS:
                subset = frame[
                    (frame["Target"] == target) & (frame["Method"] == method)
                ]
                run_count = subset[["Target", "Seed"]].drop_duplicates().shape[0]
                if run_count != EXPECTED_RUNS[target]:
                    raise RuntimeError(
                        f"{target}/{method}: expected {EXPECTED_RUNS[target]} "
                        f"independent model runs, found {run_count}"
                    )
                per_seed = subset.groupby("Seed")["Dataset"].nunique()
                expected_rows = 8 if target in {"cifar10", "cifar100"} else 7
                if len(per_seed) != EXPECTED_RUNS[target] or not (
                    per_seed == expected_rows
                ).all():
                    raise RuntimeError(
                        f"{target}/{method}: incomplete per-dataset result rows"
                    )
    return frame


def summarise(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = frame[frame["Dataset"].isin(GROUP_LABELS)].copy()
    expected_raw = sum(EXPECTED_RUNS.values()) * len(METHODS) * 2
    if len(raw) != expected_raw:
        raise RuntimeError(
            f"Expected {expected_raw} Near/Far rows, found {len(raw)}"
        )
    records = []
    grouped = raw.groupby(
        ["Target", "Method", "Dataset"], sort=False, observed=True
    )
    for key, group in grouped:
        row = {
            "Target": key[0],
            "Method": key[1],
            "Dataset": key[2],
            "RunCount": int(len(group)),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = (
                float(group[metric].std(ddof=1)) if len(group) > 1 else np.nan
            )
        records.append(row)
    summary = pd.DataFrame(records)
    return raw, summary


def _display(mean: float, std: float) -> str:
    if pd.isna(std):
        return f"{mean:.2f}"
    return f"{mean:.2f}±{std:.2f}"


def paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        record = {
            "Target": TARGET_LABELS[row["Target"]],
            "OOD Group": GROUP_LABELS[row["Dataset"]],
            "Method": METHOD_LABELS[row["Method"]],
            "Runs": int(row["RunCount"]),
        }
        for metric in METRICS:
            record[metric] = _display(
                row[f"{metric}_mean"], row[f"{metric}_std"]
            )
        rows.append(record)
    return pd.DataFrame(rows)


def latex_table(summary: pd.DataFrame) -> str:
    order = summary.copy()
    order["target_order"] = order["Target"].map(
        {name: index for index, name in enumerate(TARGETS)}
    )
    order["group_order"] = order["Dataset"].map({"nearood": 0, "farood": 1})
    order["method_order"] = order["Method"].map({"cadref": 0, "logitgap": 1})
    order = order.sort_values(["target_order", "group_order", "method_order"])
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Same-protocol CADRef and fixed LogitGap results.}",
        r"\label{tab:cadref-logitgap}",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Target & OOD & Method & Runs & FPR95$\downarrow$ & AUROC$\uparrow$ & AUPR-IN$\uparrow$ \\",
        r"\midrule",
    ]
    for _, row in order.iterrows():
        fpr = _display(row["FPR@95_mean"], row["FPR@95_std"]).replace("±", r"$\pm$")
        auc = _display(row["AUROC_mean"], row["AUROC_std"]).replace("±", r"$\pm$")
        aupr = _display(row["AUPR_IN_mean"], row["AUPR_IN_std"]).replace("±", r"$\pm$")
        target = TARGET_LABELS[row["Target"]].replace("-", "--")
        lines.append(
            f"{target} & {GROUP_LABELS[row['Dataset']]} & "
            f"{METHOD_LABELS[row['Method']]} & {int(row['RunCount'])} & "
            f"{fpr} & {auc} & {aupr} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    frame = load_and_validate(input_root, args.stage)
    raw, summary = summarise(frame)
    paper = paper_table(summary)

    raw_path = output_root / f"cadref_logitgap_{args.stage}_near_far_raw.csv"
    summary_path = output_root / f"cadref_logitgap_{args.stage}_mean_std.csv"
    paper_path = output_root / f"cadref_logitgap_{args.stage}_paper_table.csv"
    latex_path = output_root / f"cadref_logitgap_{args.stage}_paper_table.tex"
    raw.to_csv(raw_path, index=False, float_format="%.6f")
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    paper.to_csv(paper_path, index=False)
    latex_path.write_text(latex_table(summary), encoding="utf-8")

    print("===== CADRef / LogitGap Near-Far summary =====")
    print(paper.to_string(index=False))
    print("\n===== Saved =====")
    for path in (raw_path, summary_path, paper_path, latex_path):
        print(path)


if __name__ == "__main__":
    main()
