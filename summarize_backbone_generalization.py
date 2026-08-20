"""Summarise locked ImageNet-1K results across ResNet-50 and DenseNet-121."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ("ACC", "FPR@95", "AUROC", "AUPR_IN", "AUPR_OUT")
DATASETS = ("nearood", "farood")
METHOD_ORDER = ("locked_centroid_msp", "msp", "scale")
METHOD_LABELS = {
    "locked_centroid_msp": "Locked Centroid-MSP",
    "msp": "MSP",
    "scale": "Scale",
}
BACKBONE_LABELS = {
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
}
DATASET_LABELS = {"nearood": "Near-OOD", "farood": "Far-OOD"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--output-root", type=Path)
    return parser


def _read_metrics(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    unnamed = [name for name in frame.columns if name.startswith("Unnamed:")]
    if "Dataset" not in frame.columns:
        if unnamed:
            frame = frame.rename(columns={unnamed[0]: "Dataset"})
        else:
            frame = frame.rename(columns={frame.columns[0]: "Dataset"})
    frame = frame.drop(columns=[name for name in unnamed if name != "Dataset"],
                       errors="ignore")
    missing = set(("Dataset", *METRICS)) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def _load_runs(project_root: Path) -> pd.DataFrame:
    journal_root = project_root / "results/journal"
    baseline_root = project_root / "results_openood_baselines/imagenet1k"

    resnet_locked = _read_metrics(
        journal_root / "locked_imagenet1k_fusion/locked_test_all_seeds.csv"
    )
    if "Seed" not in resnet_locked:
        raise ValueError("ResNet-50 locked results have no Seed column")
    resnet_locked.insert(0, "Method", "locked_centroid_msp")
    resnet_locked.insert(0, "Backbone", "resnet50")

    resnet_parts = [resnet_locked]
    for method in ("msp", "scale"):
        frame = _read_metrics(
            baseline_root / "imagenet1k_resnet50_tvsv1" / f"{method}.csv"
        )
        frame.insert(0, "Seed", 0)
        frame.insert(0, "Method", method)
        frame.insert(0, "Backbone", "resnet50")
        resnet_parts.append(frame)

    dense = _read_metrics(
        journal_root
        / "backbone_densenet121/densenet121_full_all_runs.csv"
    )
    missing = {"Method", "Seed"} - set(dense.columns)
    if missing:
        raise ValueError(f"DenseNet results are missing: {sorted(missing)}")
    dense.insert(0, "Backbone", "densenet121")

    runs = pd.concat([*resnet_parts, dense], ignore_index=True)
    runs["Seed"] = runs["Seed"].astype(int)
    runs = runs[runs["Dataset"].isin(DATASETS)].copy()
    return runs


def _validate(runs: pd.DataFrame) -> None:
    expected = {}
    for backbone in BACKBONE_LABELS:
        for seed in (0, 1, 2):
            expected[(backbone, "locked_centroid_msp", seed)] = 2
        expected[(backbone, "msp", 0)] = 2
        expected[(backbone, "scale", 0)] = 2
    counts = runs.groupby(["Backbone", "Method", "Seed"]).size().to_dict()
    if counts != expected:
        raise RuntimeError(
            "Unexpected cross-backbone result completeness:\n"
            f"found={counts}\nexpected={expected}"
        )


def _numeric_summary(runs: pd.DataFrame) -> pd.DataFrame:
    result = (
        runs.groupby(["Backbone", "Method", "Dataset"])[list(METRICS)]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    result.columns = [
        "_".join(filter(None, map(str, column))).rstrip("_")
        if isinstance(column, tuple) else column
        for column in result.columns
    ]
    return result


def _paper_table(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for backbone in BACKBONE_LABELS:
        for method in METHOD_ORDER:
            for dataset in DATASETS:
                part = runs[
                    (runs["Backbone"] == backbone)
                    & (runs["Method"] == method)
                    & (runs["Dataset"] == dataset)
                ]
                if part.empty:
                    raise RuntimeError(
                        f"Missing {backbone}/{method}/{dataset} results"
                    )
                row = {
                    "Backbone": BACKBONE_LABELS[backbone],
                    "Method": METHOD_LABELS[method],
                    "OODGroup": DATASET_LABELS[dataset],
                }
                for metric in METRICS:
                    mean = float(part[metric].mean())
                    if len(part) > 1:
                        std = float(part[metric].std(ddof=1))
                        row[metric] = f"{mean:.2f}±{std:.2f}"
                    else:
                        row[metric] = f"{mean:.2f}"
                rows.append(row)
    return pd.DataFrame(rows)


def _deltas(runs: pd.DataFrame) -> pd.DataFrame:
    means = (
        runs.groupby(["Backbone", "Method", "Dataset"])[list(METRICS)]
        .mean()
    )
    rows = []
    for backbone in BACKBONE_LABELS:
        for dataset in DATASETS:
            locked = means.loc[(backbone, "locked_centroid_msp", dataset)]
            for baseline in ("msp", "scale"):
                reference = means.loc[(backbone, baseline, dataset)]
                rows.append({
                    "Backbone": BACKBONE_LABELS[backbone],
                    "OODGroup": DATASET_LABELS[dataset],
                    "Comparison": f"Locked minus {METHOD_LABELS[baseline]}",
                    "Delta_AUROC": locked["AUROC"] - reference["AUROC"],
                    "Delta_FPR95": locked["FPR@95"] - reference["FPR@95"],
                    "Delta_AUPR_IN": locked["AUPR_IN"] - reference["AUPR_IN"],
                    "Delta_AUPR_OUT": locked["AUPR_OUT"] - reference["AUPR_OUT"],
                })
    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else project_root / "results/journal/summary_backbone_generalization"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    runs = _load_runs(project_root)
    _validate(runs)
    summary = _numeric_summary(runs)
    paper = _paper_table(runs)
    deltas = _deltas(runs)

    runs.to_csv(output_root / "backbone_near_far_raw.csv", index=False)
    summary.to_csv(
        output_root / "backbone_near_far_mean_std.csv", index=False
    )
    paper.to_csv(output_root / "backbone_paper_table.csv", index=False)
    deltas.to_csv(output_root / "backbone_deltas.csv", index=False)

    latex = paper.copy()
    for metric in METRICS:
        latex[metric] = latex[metric].str.replace(
            "±", r"$\pm$", regex=False
        )
    (output_root / "backbone_paper_table.tex").write_text(
        latex.to_latex(
            index=False,
            escape=False,
            column_format="lllccccc",
            caption=(
                "ImageNet-1K backbone generalisation under the locked "
                "OpenOOD v1.5 protocol."
            ),
            label="tab:backbone_generalization",
        ),
        encoding="utf-8",
    )

    print("===== Cross-backbone paper table =====")
    print(paper.to_string(index=False))
    print("\n===== Locked method deltas =====")
    print(deltas.round(4).to_string(index=False))
    print("\n===== Generated files =====")
    for path in sorted(output_root.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
