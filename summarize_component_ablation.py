"""Build the alpha=0/0.8/1 component ablation across all locked targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TARGET_ORDER = (
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
METHOD_ORDER = ("msp", "locked_fusion", "centroid_only")
METHOD_LABELS = {
    "msp": "MSP ($\\alpha=0$)",
    "locked_fusion": "Locked fusion ($\\alpha=0.8$)",
    "centroid_only": "Centroid-only ($\\alpha=1$)",
}
DATASET_ORDER = ("nearood", "farood")
DATASET_LABELS = {"nearood": "Near-OOD", "farood": "Far-OOD"}
METRICS = ("ACC", "FPR@95", "AUROC", "AUPR_IN", "AUPR_OUT")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--output-root", type=Path)
    return parser


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    unnamed = [name for name in frame if name.startswith("Unnamed:")]
    if "Dataset" not in frame:
        if unnamed:
            frame = frame.rename(columns={unnamed[0]: "Dataset"})
        else:
            frame = frame.rename(columns={frame.columns[0]: "Dataset"})
    missing = set(("Dataset", *METRICS)) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    return frame


def _tag(
    frame: pd.DataFrame, target: str, method: str, default_seed: int | None = None
) -> pd.DataFrame:
    result = frame.copy()
    if "Seed" not in result:
        if default_seed is None:
            raise ValueError(f"{target}/{method} has no Seed column")
        result.insert(0, "Seed", default_seed)
    result["Seed"] = result["Seed"].astype(int)
    for column in ("Target", "Method"):
        if column in result:
            result = result.drop(columns=column)
    result.insert(0, "Method", method)
    result.insert(0, "Target", target)
    return result


def _load_all(project_root: Path) -> pd.DataFrame:
    journal = project_root / "results/journal"
    baselines = project_root / "results_openood_baselines"

    centroid = _read(
        journal
        / "component_centroid_only/centroid_only_full_all_runs.csv"
    )
    if not {"Target", "Seed"}.issubset(centroid.columns):
        raise ValueError("Centroid-only combined results lack Target/Seed")
    centroid["Method"] = "centroid_only"

    transfer = _read(
        journal
        / "locked_centroid_transfer/locked_transfer_full_all_runs.csv"
    )
    if not {"Benchmark", "Seed"}.issubset(transfer.columns):
        raise ValueError("Locked transfer results lack Benchmark/Seed")
    fusion_parts = []
    for target in ("cifar10", "cifar100", "imagenet200"):
        fusion_parts.append(_tag(
            transfer[transfer["Benchmark"] == target],
            target,
            "locked_fusion",
        ))

    resnet_fusion = _read(
        journal / "locked_imagenet1k_fusion/locked_test_all_seeds.csv"
    )
    fusion_parts.append(_tag(
        resnet_fusion, "imagenet1k_resnet50", "locked_fusion"
    ))

    dense_all = _read(
        journal
        / "backbone_densenet121/densenet121_full_all_runs.csv"
    )
    if not {"Method", "Seed"}.issubset(dense_all.columns):
        raise ValueError("DenseNet combined results lack Method/Seed")
    fusion_parts.append(_tag(
        dense_all[dense_all["Method"] == "locked_centroid_msp"],
        "imagenet1k_densenet121",
        "locked_fusion",
    ))

    msp_all = _read(baselines / "msp_all_runs.csv")
    if not {"Benchmark", "Seed"}.issubset(msp_all.columns):
        raise ValueError("MSP long-form results lack Benchmark/Seed")
    msp_parts = []
    for target in ("cifar10", "cifar100", "imagenet200"):
        msp_parts.append(_tag(
            msp_all[msp_all["Benchmark"] == target], target, "msp"
        ))
    msp_parts.append(_tag(
        msp_all[msp_all["Benchmark"] == "imagenet1k"],
        "imagenet1k_resnet50",
        "msp",
    ))
    msp_parts.append(_tag(
        dense_all[dense_all["Method"] == "msp"],
        "imagenet1k_densenet121",
        "msp",
    ))

    selected_columns = ["Target", "Method", "Seed", "Dataset", *METRICS]
    all_runs = pd.concat(
        [centroid[selected_columns],
         *[part[selected_columns] for part in fusion_parts],
         *[part[selected_columns] for part in msp_parts]],
        ignore_index=True,
    )
    return all_runs[all_runs["Dataset"].isin(DATASET_ORDER)].copy()


def _validate(all_runs: pd.DataFrame) -> pd.Series:
    counts = all_runs.groupby(["Target", "Method", "Seed"]).size()
    expected = {}
    for target in TARGET_ORDER:
        for seed in (0, 1, 2):
            expected[(target, "locked_fusion", seed)] = 2
            expected[(target, "centroid_only", seed)] = 2
        msp_seeds = (0,) if target.startswith("imagenet1k_") else (0, 1, 2)
        for seed in msp_seeds:
            expected[(target, "msp", seed)] = 2
    if counts.to_dict() != expected:
        raise RuntimeError(
            "Component result completeness mismatch:\n"
            f"found={counts.to_dict()}\nexpected={expected}"
        )
    return counts


def _numeric_summary(all_runs: pd.DataFrame) -> pd.DataFrame:
    result = (
        all_runs.groupby(["Target", "Method", "Dataset"])[list(METRICS)]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    result.columns = [
        "_".join(filter(None, map(str, column))).rstrip("_")
        if isinstance(column, tuple) else column
        for column in result.columns
    ]
    return result


def _paper_table(all_runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGET_ORDER:
        for dataset in DATASET_ORDER:
            for method in METHOD_ORDER:
                part = all_runs[
                    (all_runs["Target"] == target)
                    & (all_runs["Dataset"] == dataset)
                    & (all_runs["Method"] == method)
                ]
                if part.empty:
                    raise RuntimeError(f"Missing {target}/{dataset}/{method}")
                row = {
                    "Target": TARGET_LABELS[target],
                    "OODGroup": DATASET_LABELS[dataset],
                    "Method": METHOD_LABELS[method],
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


def _deltas(all_runs: pd.DataFrame) -> pd.DataFrame:
    means = all_runs.groupby(
        ["Target", "Dataset", "Method"]
    )[list(METRICS)].mean()
    rows = []
    for target in TARGET_ORDER:
        for dataset in DATASET_ORDER:
            fusion = means.loc[(target, dataset, "locked_fusion")]
            msp = means.loc[(target, dataset, "msp")]
            centroid = means.loc[(target, dataset, "centroid_only")]
            row = {
                "Target": TARGET_LABELS[target],
                "OODGroup": DATASET_LABELS[dataset],
                "Fusion_minus_MSP_AUROC": fusion["AUROC"] - msp["AUROC"],
                "Fusion_minus_MSP_FPR95": fusion["FPR@95"] - msp["FPR@95"],
                "Fusion_minus_Centroid_AUROC": (
                    fusion["AUROC"] - centroid["AUROC"]
                ),
                "Fusion_minus_Centroid_FPR95": (
                    fusion["FPR@95"] - centroid["FPR@95"]
                ),
                "Centroid_minus_MSP_AUROC": centroid["AUROC"] - msp["AUROC"],
                "Centroid_minus_MSP_FPR95": centroid["FPR@95"] - msp["FPR@95"],
            }
            row["FusionStrictlyBest"] = bool(
                row["Fusion_minus_MSP_AUROC"] > 0
                and row["Fusion_minus_MSP_FPR95"] < 0
                and row["Fusion_minus_Centroid_AUROC"] > 0
                and row["Fusion_minus_Centroid_FPR95"] < 0
            )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else project_root / "results/journal/summary_component_ablation"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    all_runs = _load_all(project_root)
    counts = _validate(all_runs)
    summary = _numeric_summary(all_runs)
    paper = _paper_table(all_runs)
    deltas = _deltas(all_runs)

    all_runs.to_csv(output_root / "component_near_far_raw.csv", index=False)
    summary.to_csv(output_root / "component_mean_std.csv", index=False)
    paper.to_csv(output_root / "component_paper_table.csv", index=False)
    deltas.to_csv(output_root / "component_deltas.csv", index=False)
    audit = {
        "target_group_count": int(len(deltas)),
        "fusion_strictly_best_count": int(deltas["FusionStrictlyBest"].sum()),
        "fusion_strictly_best_fraction": float(deltas["FusionStrictlyBest"].mean()),
        "definition": (
            "Fusion has higher AUROC and lower FPR95 than both MSP and "
            "centroid-only on the same target/group aggregate."
        ),
    }
    (output_root / "component_synergy_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    latex = paper.copy()
    for metric in METRICS:
        latex[metric] = latex[metric].str.replace(
            "±", r"$\pm$", regex=False
        )
    (output_root / "component_paper_table.tex").write_text(
        latex.to_latex(
            index=False,
            escape=False,
            column_format="lllccccc",
            caption=(
                "Component ablation of classifier confidence and class geometry."
            ),
            label="tab:component_ablation",
        ),
        encoding="utf-8",
    )

    print("===== Completeness =====")
    print(counts)
    print("\n===== Component paper table =====")
    print(paper.to_string(index=False))
    print("\n===== Component deltas =====")
    print(deltas.round(4).to_string(index=False))
    print("\n===== Synergy audit =====")
    print(json.dumps(audit, indent=2))
    print("\n===== Generated files =====")
    for path in sorted(output_root.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
