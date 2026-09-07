"""Summarise same-protocol CTM, setup-bank centroid, and locked fusion results.

The output intentionally keeps CTM separate from the small-bank centroid
endpoint: CTM estimates each class direction from the complete ID training
split, whereas the paper endpoint uses the pre-specified 12-image setup bank.
"""

from __future__ import annotations

import argparse
import json
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
GROUP_LABELS = {"nearood": "Near-OOD", "farood": "Far-OOD"}
METRICS = ("FPR@95", "AUROC", "AUPR_IN", "AUPR_OUT", "ACC")
METHOD_LABELS = {
    "ctm": "CTM (full-train centroid)",
    "centroid_only": "Setup-bank centroid ($M=12$)",
    "locked_centroid_msp": "Locked Centroid--MSP",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "full"), default="full")
    parser.add_argument("--ctm-root", type=Path, default=Path("results/journal/ctm"))
    parser.add_argument("--journal-root", type=Path, default=Path("results/journal"))
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("results/journal/summary_ctm_comparison"),
    )
    return parser


def _dataset_column(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "Dataset" in frame:
        return frame
    unnamed = [column for column in frame if column.startswith("Unnamed:")]
    if unnamed:
        return frame.rename(columns={unnamed[0]: "Dataset"})
    return frame.rename(columns={frame.columns[0]: "Dataset"})


def _canonical(
    frame: pd.DataFrame, *, method: str, target: str | None = None
) -> pd.DataFrame:
    frame = _dataset_column(frame)
    if target is not None:
        frame["Target"] = target
    frame["Method"] = method
    required = {"Target", "Seed", "Dataset", *METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Missing result columns for {method}: {missing}")
    frame = frame[frame["Dataset"].isin(GROUP_LABELS)].copy()
    frame["Seed"] = frame["Seed"].astype(int)
    for metric in METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors="raise")
    return frame[["Target", "Method", "Seed", "Dataset", *METRICS]]


def _load_ctm(root: Path, stage: str) -> pd.DataFrame:
    path = root / f"ctm_{stage}_all_runs.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "Method" in frame:
        frame = frame.drop(columns="Method")
    result = _canonical(frame, method="ctm")
    if stage == "full":
        full = _dataset_column(frame)
        combinations = full[["Target", "Seed"]].drop_duplicates()
        counts = full.groupby(["Target", "Seed"])["Dataset"].nunique()
        expected = {
            ("cifar10", 0), ("cifar10", 1), ("cifar10", 2),
            ("cifar100", 0), ("cifar100", 1), ("cifar100", 2),
            ("imagenet200", 0), ("imagenet200", 1), ("imagenet200", 2),
            ("imagenet1k_resnet50", 0),
            ("imagenet1k_densenet121", 0),
        }
        found = set(map(tuple, combinations.itertuples(index=False, name=None)))
        expected_counts = {
            target_seed: (8 if target_seed[0] in {"cifar10", "cifar100"} else 7)
            for target_seed in expected
        }
        counts_dict = {tuple(key): int(value) for key, value in counts.items()}
        if (
            found != expected
            or len(full) != 83
            or counts_dict != expected_counts
        ):
            raise RuntimeError(
                "Formal CTM matrix must contain 11 independent runs: eight "
                "OpenOOD rows for each CIFAR run and seven for each ImageNet run"
            )
    return result


def _load_centroid(journal_root: Path, stage: str) -> pd.DataFrame:
    path = (
        journal_root / "component_centroid_only"
        / f"centroid_only_{stage}_all_runs.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "Method" in frame:
        frame = frame.drop(columns="Method")
    return _canonical(frame, method="centroid_only")


def _load_locked(journal_root: Path, stage: str) -> pd.DataFrame:
    mode = stage
    frames = []

    transfer_path = (
        journal_root / "locked_centroid_transfer"
        / f"locked_transfer_{mode}_all_runs.csv"
    )
    transfer = pd.read_csv(transfer_path)
    transfer["Target"] = transfer["Benchmark"].astype(str)
    frames.append(_canonical(transfer, method="locked_centroid_msp"))

    if stage == "full":
        resnet_path = (
            journal_root / "locked_imagenet1k_fusion"
            / "locked_test_all_seeds.csv"
        )
    else:
        # The locked ImageNet-1K smoke files are per seed; load those directly.
        smoke_paths = sorted(
            (journal_root / "locked_imagenet1k_fusion" / "smoke").rglob(
                "locked_centroid_msp.csv"
            )
        )
        if not smoke_paths:
            raise FileNotFoundError(
                journal_root / "locked_imagenet1k_fusion" / "smoke"
            )
        parts = []
        for index, path in enumerate(smoke_paths):
            part = _dataset_column(pd.read_csv(path))
            part.insert(0, "Seed", index)
            parts.append(part)
        resnet = pd.concat(parts, ignore_index=True)
        frames.append(
            _canonical(
                resnet, method="locked_centroid_msp",
                target="imagenet1k_resnet50",
            )
        )
        resnet_path = None
    if resnet_path is not None:
        resnet = pd.read_csv(resnet_path)
        frames.append(
            _canonical(
                resnet, method="locked_centroid_msp",
                target="imagenet1k_resnet50",
            )
        )

    dense_path = (
        journal_root / "backbone_densenet121"
        / f"densenet121_{mode}_all_runs.csv"
    )
    dense = pd.read_csv(dense_path)
    method_column = dense["Method"].astype(str).str.lower()
    dense = dense[method_column == "locked_centroid_msp"].copy()
    if "Method" in dense:
        dense = dense.drop(columns="Method")
    frames.append(
        _canonical(
            dense, method="locked_centroid_msp",
            target="imagenet1k_densenet121",
        )
    )
    return pd.concat(frames, ignore_index=True)


def _summarise(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = raw.groupby(
        ["Target", "Method", "Dataset"], sort=False, observed=True
    )
    records = []
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
    return pd.DataFrame(records)


def _display(mean: float, std: float) -> str:
    if pd.isna(std):
        return f"{mean:.2f}"
    return f"{mean:.2f}\\(\\pm\\){std:.2f}"


def _paper_table(summary: pd.DataFrame) -> pd.DataFrame:
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


def _latex(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Same-protocol comparison of CTM, the setup-bank centroid endpoint, and the locked fusion. Best AUROC and FPR@95 within each target and OOD group are bold.}",
        r"\label{tab:ctm-comparison}",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Target & OOD & Method & Runs & FPR95$\downarrow$ & AUROC$\uparrow$ & AUPR-IN$\uparrow$ \\",
        r"\midrule",
    ]
    ordered = summary.copy()
    ordered["target_order"] = ordered["Target"].map(
        {name: index for index, name in enumerate(TARGETS)}
    )
    ordered["group_order"] = ordered["Dataset"].map({"nearood": 0, "farood": 1})
    ordered["method_order"] = ordered["Method"].map(
        {name: index for index, name in enumerate(METHOD_LABELS)}
    )
    ordered = ordered.sort_values(["target_order", "group_order", "method_order"])
    for (target, dataset), group in ordered.groupby(
        ["Target", "Dataset"], sort=False
    ):
        best_auc = group["AUROC_mean"].max()
        best_fpr = group["FPR@95_mean"].min()
        for _, row in group.iterrows():
            fpr = _display(row["FPR@95_mean"], row["FPR@95_std"])
            auc = _display(row["AUROC_mean"], row["AUROC_std"])
            aupr = _display(row["AUPR_IN_mean"], row["AUPR_IN_std"])
            if np.isclose(row["FPR@95_mean"], best_fpr):
                fpr = rf"\textbf{{{fpr}}}"
            if np.isclose(row["AUROC_mean"], best_auc):
                auc = rf"\textbf{{{auc}}}"
            target_text = TARGET_LABELS[target].replace("-", "--")
            method_text = METHOD_LABELS[row["Method"]]
            lines.append(
                f"{target_text} & {GROUP_LABELS[dataset]} & {method_text} & "
                f"{int(row['RunCount'])} & {fpr} & {auc} & {aupr} \\\\"
            )
        lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def _deltas(summary: pd.DataFrame) -> pd.DataFrame:
    means = summary.pivot_table(
        index=["Target", "Dataset"], columns="Method",
        values=["AUROC_mean", "FPR@95_mean"], aggfunc="first",
    )
    rows = []
    for index, row in means.iterrows():
        for comparator in ("centroid_only", "locked_centroid_msp"):
            rows.append({
                "Target": index[0],
                "Dataset": index[1],
                "Comparison": f"CTM minus {comparator}",
                "DeltaAUROC": (
                    row[("AUROC_mean", "ctm")]
                    - row[("AUROC_mean", comparator)]
                ),
                "DeltaFPR95": (
                    row[("FPR@95_mean", "ctm")]
                    - row[("FPR@95_mean", comparator)]
                ),
                "Note": (
                    "mean-level descriptive comparison; fixed ImageNet-1K CTM "
                    "has one model run"
                ),
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser().parse_args()
    args.ctm_root = args.ctm_root.resolve()
    args.journal_root = args.journal_root.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    ctm = _load_ctm(args.ctm_root, args.stage)
    centroid = _load_centroid(args.journal_root, args.stage)
    locked = _load_locked(args.journal_root, args.stage)
    raw = pd.concat([ctm, centroid, locked], ignore_index=True)
    missing = set(TARGETS) - set(raw["Target"])
    if missing:
        raise RuntimeError(f"Comparison lacks targets: {sorted(missing)}")
    summary = _summarise(raw)
    paper = _paper_table(summary)
    deltas = _deltas(summary)

    raw_path = args.output_root / f"ctm_comparison_{args.stage}_raw.csv"
    summary_path = args.output_root / f"ctm_comparison_{args.stage}_mean_std.csv"
    paper_path = args.output_root / f"ctm_comparison_{args.stage}_paper.csv"
    latex_path = args.output_root / f"ctm_comparison_{args.stage}_paper.tex"
    delta_path = args.output_root / f"ctm_comparison_{args.stage}_deltas.csv"
    audit_path = args.output_root / f"ctm_comparison_{args.stage}_audit.json"
    raw.to_csv(raw_path, index=False, float_format="%.6f")
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    paper.to_csv(paper_path, index=False)
    latex_path.write_text(_latex(summary), encoding="utf-8")
    deltas.to_csv(delta_path, index=False, float_format="%.6f")
    audit_path.write_text(json.dumps({
        "protocol": "same local OpenOOD v1.5 data/model protocol",
        "stage": args.stage,
        "ctm_definition": (
            "full ID-train raw penultimate class mean, then L2-normalise; "
            "L2-normalised query; maximum cosine"
        ),
        "ctm_ood_tuning": False,
        "full_train_centroid_distinguished_from_setup_bank_centroid": True,
        "fixed_imagenet1k_ctm_models_repeated": False,
        "raw_rows": len(raw),
        "summary_rows": len(summary),
        "outputs": [str(path.resolve()) for path in (
            raw_path, summary_path, paper_path, latex_path, delta_path
        )],
        "completed": True,
    }, indent=2), encoding="utf-8")

    print("===== CTM same-protocol comparison =====")
    print(paper.to_string(index=False))
    print("\n===== Generated files =====")
    for path in (raw_path, summary_path, paper_path, latex_path, delta_path, audit_path):
        print(path)


if __name__ == "__main__":
    main()
