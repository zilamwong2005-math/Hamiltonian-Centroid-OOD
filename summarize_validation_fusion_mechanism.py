"""Summarise the validation-only MSP/centroid fusion mechanism experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_ORDER = (
    "cifar10",
    "cifar100",
    "imagenet200",
    "imagenet1k_resnet50",
    "imagenet1k_densenet121",
)
TARGET_LABEL = {
    "cifar10": "CIFAR-10 / ResNet-18",
    "cifar100": "CIFAR-100 / ResNet-18",
    "imagenet200": "ImageNet-200 / ResNet-18",
    "imagenet1k_resnet50": "ImageNet-1K / ResNet-50",
    "imagenet1k_densenet121": "ImageNet-1K / DenseNet-121",
}
ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
METRICS = ("AUROC", "FPR95", "AUPR_IN", "AUPR_OUT")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results/journal/validation_fusion_mechanism"),
    )
    parser.add_argument("--output-root", type=Path)
    return parser


def _validate(sweep: pd.DataFrame, statistics: pd.DataFrame) -> None:
    required_sweep = {
        "Target", "ClassCount", "Seed", "Alpha", "Split", *METRICS
    }
    required_statistics = {
        "Target", "ClassCount", "Seed",
        "MSP_StandardizedGap", "Geometry_StandardizedGap",
        "ID_MSP_Geometry_Correlation", "OOD_MSP_Geometry_Correlation",
    }
    if missing := required_sweep.difference(sweep.columns):
        raise ValueError(f"Sweep CSV is missing columns: {sorted(missing)}")
    if missing := required_statistics.difference(statistics.columns):
        raise ValueError(f"Statistics CSV is missing columns: {sorted(missing)}")
    if set(sweep["Target"]) != set(TARGET_ORDER):
        raise ValueError(f"Unexpected sweep targets: {sorted(sweep['Target'].unique())}")
    if set(statistics["Target"]) != set(TARGET_ORDER):
        raise ValueError(
            f"Unexpected statistics targets: {sorted(statistics['Target'].unique())}"
        )
    if set(sweep["Seed"].astype(int)) != {0, 1, 2}:
        raise ValueError("Sweep must contain seeds 0, 1, and 2")
    if set(statistics["Seed"].astype(int)) != {0, 1, 2}:
        raise ValueError("Statistics must contain seeds 0, 1, and 2")
    if set(np.round(sweep["Alpha"].astype(float), 8)) != set(ALPHAS):
        raise ValueError("Sweep alpha grid is incomplete")
    if set(sweep["Split"]) != {"tune", "holdout", "full"}:
        raise ValueError("Sweep split set is incomplete")
    expected = len(TARGET_ORDER) * 3 * len(ALPHAS) * 3
    if len(sweep) != expected or len(statistics) != len(TARGET_ORDER) * 3:
        raise ValueError(
            f"Incomplete inputs: sweep={len(sweep)}/{expected}, "
            f"statistics={len(statistics)}/{len(TARGET_ORDER) * 3}"
        )
    counts = sweep.groupby(["Target", "Seed", "Alpha", "Split"]).size()
    if not (counts == 1).all():
        raise ValueError("Duplicate or missing target/seed/alpha/split rows")


def _mean_std(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["Target", "ClassCount", "Alpha"], sort=False)
    result = grouped[list(METRICS)].agg(["mean", "std"]).reset_index()
    result.columns = [
        "_".join(str(item) for item in column if str(item))
        if isinstance(column, tuple) else str(column)
        for column in result.columns
    ]
    return result


def _endpoint_deltas(holdout_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGET_ORDER:
        selected = holdout_summary[holdout_summary["Target"].eq(target)].set_index(
            "Alpha"
        )
        locked, msp, centroid = selected.loc[0.8], selected.loc[0.0], selected.loc[1.0]
        rows.append({
            "Target": target,
            "TargetLabel": TARGET_LABEL[target],
            "ClassCount": int(locked["ClassCount"]),
            "LockedMinusMSP_AUROC": locked["AUROC_mean"] - msp["AUROC_mean"],
            "LockedMinusMSP_FPR95": locked["FPR95_mean"] - msp["FPR95_mean"],
            "LockedMinusCentroid_AUROC": (
                locked["AUROC_mean"] - centroid["AUROC_mean"]
            ),
            "LockedMinusCentroid_FPR95": (
                locked["FPR95_mean"] - centroid["FPR95_mean"]
            ),
            "LockedStrictlyBest": bool(
                locked["AUROC_mean"] > max(msp["AUROC_mean"], centroid["AUROC_mean"])
                and locked["FPR95_mean"] < min(
                    msp["FPR95_mean"], centroid["FPR95_mean"]
                )
            ),
        })
    return pd.DataFrame(rows)


def _best_alpha(holdout_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGET_ORDER:
        selected = holdout_summary[holdout_summary["Target"].eq(target)].copy()
        best = selected.sort_values(
            ["AUROC_mean", "FPR95_mean", "Alpha"],
            ascending=[False, True, True],
        ).iloc[0]
        rows.append({
            "Target": target,
            "TargetLabel": TARGET_LABEL[target],
            "ClassCount": int(best["ClassCount"]),
            "BestAlphaByMeanAUROC": float(best["Alpha"]),
            "BestAUROC": float(best["AUROC_mean"]),
            "BestFPR95": float(best["FPR95_mean"]),
        })
    return pd.DataFrame(rows)


def _correlation_summary(statistics: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "ID_MSP_Geometry_Correlation",
        "OOD_MSP_Geometry_Correlation",
        "MSP_StandardizedGap",
        "Geometry_StandardizedGap",
    )
    result = (
        statistics.groupby(["Target", "ClassCount"], sort=False)[list(columns)]
        .agg(["mean", "std"])
        .reset_index()
    )
    result.columns = [
        "_".join(str(item) for item in column if str(item))
        if isinstance(column, tuple) else str(column)
        for column in result.columns
    ]
    result.insert(1, "TargetLabel", result["Target"].map(TARGET_LABEL))
    return result


def _paper_table(
    holdout_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    best_alpha: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    correlations = correlations.set_index("Target")
    best_alpha = best_alpha.set_index("Target")
    for target in TARGET_ORDER:
        values = holdout_summary[holdout_summary["Target"].eq(target)].set_index(
            "Alpha"
        )
        corr = correlations.loc[target]
        rows.append({
            "Target": TARGET_LABEL[target],
            "Classes": int(corr["ClassCount"]),
            "OOD score correlation": (
                f"{corr['OOD_MSP_Geometry_Correlation_mean']:.3f}"
                f"$\\pm${corr['OOD_MSP_Geometry_Correlation_std']:.3f}"
            ),
            "Best alpha": f"{best_alpha.loc[target, 'BestAlphaByMeanAUROC']:.1f}",
            "MSP AUROC/FPR95": (
                f"{values.loc[0.0, 'AUROC_mean']:.2f}/"
                f"{values.loc[0.0, 'FPR95_mean']:.2f}"
            ),
            "Locked AUROC/FPR95": (
                f"{values.loc[0.8, 'AUROC_mean']:.2f}/"
                f"{values.loc[0.8, 'FPR95_mean']:.2f}"
            ),
            "Centroid AUROC/FPR95": (
                f"{values.loc[1.0, 'AUROC_mean']:.2f}/"
                f"{values.loc[1.0, 'FPR95_mean']:.2f}"
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser().parse_args()
    input_root = args.input_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else input_root / "summary"
    )
    sweep = pd.read_csv(input_root / "validation_alpha_sweep_full.csv")
    statistics = pd.read_csv(input_root / "validation_score_statistics_full.csv")
    _validate(sweep, statistics)

    holdout = sweep[sweep["Split"].eq("holdout")].copy()
    holdout_summary = _mean_std(holdout)
    deltas = _endpoint_deltas(holdout_summary)
    best_alpha = _best_alpha(holdout_summary)
    correlations = _correlation_summary(statistics)
    paper = _paper_table(holdout_summary, correlations, best_alpha)

    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "alpha_sweep": output_root / "validation_alpha_holdout_mean_std.csv",
        "best_alpha": output_root / "validation_best_alpha_by_target.csv",
        "deltas": output_root / "validation_locked_component_deltas.csv",
        "correlations": output_root / "validation_correlation_summary.csv",
        "paper_csv": output_root / "validation_mechanism_paper_table.csv",
        "paper_tex": output_root / "validation_mechanism_paper_table.tex",
    }
    holdout_summary.to_csv(outputs["alpha_sweep"], index=False, float_format="%.6f")
    best_alpha.to_csv(outputs["best_alpha"], index=False, float_format="%.6f")
    deltas.to_csv(outputs["deltas"], index=False, float_format="%.6f")
    correlations.to_csv(outputs["correlations"], index=False, float_format="%.6f")
    paper.to_csv(outputs["paper_csv"], index=False)
    outputs["paper_tex"].write_text(
        paper.to_latex(index=False, escape=False, column_format="llcllll"),
        encoding="utf-8",
    )
    audit = {
        "protocol": "validation-only holdout; no near/far test scores",
        "targets": list(TARGET_ORDER),
        "seeds": [0, 1, 2],
        "locked_alpha": 0.8,
        "locked_strictly_best_targets": deltas.loc[
            deltas["LockedStrictlyBest"], "Target"
        ].tolist(),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    audit_path = output_root / "validation_mechanism_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("===== Validation holdout best alpha =====")
    print(best_alpha.round(4).to_string(index=False))
    print("\n===== Locked alpha=0.8 component deltas =====")
    print(deltas.round(4).to_string(index=False))
    print("\n===== Score correlation and separation =====")
    print(correlations.round(4).to_string(index=False))
    print("\n===== Paper table =====")
    print(paper.to_string(index=False))
    print("\n===== Generated files =====")
    for path in (*outputs.values(), audit_path):
        print(path)


if __name__ == "__main__":
    main()
