"""ID-only calibration-budget sensitivity for the locked ImageNet-1K rule.

One validation image per class is reserved as a fixed evaluation set.  From
the remaining four validation images per class, nested calibration sets of
size 1--4 are used only to estimate MSP and geometry mean/std.  OOD validation
scores are never used for calibration, and no near/far test file is opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_imagenet1k_score_rules import _metric, _zscore
from run_locked_imagenet1k_fusion import (
    _default_decision_path,
    _load_locked_decision,
)


BUDGETS = (1, 2, 3, 4)
SEEDS = (0, 1, 2)
CLASS_COUNT = 1000
IMAGES_PER_CLASS = 5
SPLIT_OFFSET = 20260818
FORMAT_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("smoke", "full"))
    parser.add_argument(
        "--mechanism-root",
        type=Path,
        default=Path("results/journal/validation_fusion_mechanism"),
    )
    parser.add_argument(
        "--hamiltonian-root", type=Path, default=Path("results_openood")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/journal/calibration_budget_sensitivity"),
    )
    parser.add_argument("--decision-path", type=Path)
    return parser


def _score_path(mechanism_root: Path, seed: int) -> Path:
    return (
        mechanism_root
        / "full"
        / "imagenet1k_resnet50"
        / f"seed{seed}"
        / "validation_scores.npz"
    )


def _fixed_calibration_splits(
    labels: np.ndarray, split_seed: int = 0
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    labels = np.asarray(labels, dtype=int)
    if len(labels) != CLASS_COUNT * IMAGES_PER_CLASS:
        raise ValueError(f"Expected 5,000 ID validation labels, found {len(labels)}")
    evaluation = np.zeros(len(labels), dtype=bool)
    calibration = {
        budget: np.zeros(len(labels), dtype=bool) for budget in BUDGETS
    }
    for label in range(CLASS_COUNT):
        rows = np.flatnonzero(labels == label)
        if len(rows) != IMAGES_PER_CLASS:
            raise ValueError(f"Class {label} has {len(rows)} validation images")
        generator = np.random.default_rng(
            SPLIT_OFFSET + split_seed * CLASS_COUNT + label
        )
        rows = generator.permutation(rows)
        evaluation[rows[-1]] = True
        for budget in BUDGETS:
            calibration[budget][rows[:budget]] = True
    if evaluation.sum() != CLASS_COUNT:
        raise RuntimeError("Fixed ID evaluation split is incomplete")
    for budget in BUDGETS:
        if calibration[budget].sum() != budget * CLASS_COUNT:
            raise RuntimeError(f"Calibration budget {budget} is incomplete")
        if np.any(evaluation & calibration[budget]):
            raise RuntimeError("Calibration and fixed evaluation ID sets overlap")
    for smaller, larger in zip(BUDGETS[:-1], BUDGETS[1:]):
        if not np.all(calibration[smaller] <= calibration[larger]):
            raise RuntimeError("Calibration budgets are not nested")
    return evaluation, calibration


def _load_scores(path: Path) -> dict[str, np.ndarray]:
    required = {
        "id_msp", "ood_msp", "id_geometry", "ood_geometry",
        "id_prediction", "id_labels",
    }
    if not path.is_file():
        raise FileNotFoundError(f"Validation score cache missing: {path}")
    with np.load(path) as saved:
        if missing := required.difference(saved.files):
            raise ValueError(f"Score cache is missing: {sorted(missing)}")
        arrays = {name: saved[name] for name in required}
    if len(arrays["id_msp"]) != 5000 or len(arrays["ood_msp"]) != 1763:
        raise ValueError("Calibration experiment requires full validation scores")
    for name in ("id_msp", "ood_msp", "id_geometry", "ood_geometry"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Non-finite values in {path}: {name}")
    return arrays


def _analyse_seed(
    seed: int,
    arrays: dict[str, np.ndarray],
    alpha: float,
    budgets: tuple[int, ...],
) -> tuple[list[dict], list[dict]]:
    labels = arrays["id_labels"].astype(int)
    prediction = arrays["id_prediction"].astype(int)
    evaluation, calibration = _fixed_calibration_splits(labels)
    rows, statistics = [], []
    for budget in budgets:
        reference = calibration[budget]
        id_msp_z = _zscore(arrays["id_msp"], arrays["id_msp"][reference])
        ood_msp_z = _zscore(arrays["ood_msp"], arrays["id_msp"][reference])
        id_geometry_z = _zscore(
            arrays["id_geometry"], arrays["id_geometry"][reference]
        )
        ood_geometry_z = _zscore(
            arrays["ood_geometry"], arrays["id_geometry"][reference]
        )
        scores = {
            "MSP": (id_msp_z, ood_msp_z),
            "Centroid": (id_geometry_z, ood_geometry_z),
            "Locked fusion": (
                (1.0 - alpha) * id_msp_z + alpha * id_geometry_z,
                (1.0 - alpha) * ood_msp_z + alpha * ood_geometry_z,
            ),
        }
        for method, (id_score, ood_score) in scores.items():
            row = _metric(
                method,
                "fixed_id_validation_eval",
                id_score[evaluation],
                ood_score,
                prediction[evaluation],
                labels[evaluation],
            )
            row.update({
                "Seed": seed,
                "CalibrationImagesPerClass": budget,
                "CalibrationCount": int(reference.sum()),
                "GeometryWeight": alpha if method == "Locked fusion" else np.nan,
            })
            rows.append(row)
        statistics.append({
            "Seed": seed,
            "CalibrationImagesPerClass": budget,
            "CalibrationCount": int(reference.sum()),
            "EvaluationIDCount": int(evaluation.sum()),
            "OODCount": len(arrays["ood_msp"]),
            "MSPMean": float(arrays["id_msp"][reference].mean()),
            "MSPStd": float(arrays["id_msp"][reference].std()),
            "GeometryMean": float(arrays["id_geometry"][reference].mean()),
            "GeometryStd": float(arrays["id_geometry"][reference].std()),
        })
    return rows, statistics


def _flatten(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [
        "_".join(str(item) for item in column if str(item))
        if isinstance(column, tuple) else str(column)
        for column in result.columns
    ]
    return result


def main() -> None:
    args = build_parser().parse_args()
    args.mechanism_root = args.mechanism_root.resolve()
    args.hamiltonian_root = args.hamiltonian_root.resolve()
    args.output_root = args.output_root.resolve()
    decision_path = (
        args.decision_path.resolve()
        if args.decision_path is not None
        else _default_decision_path(args.hamiltonian_root)
    )
    decision, decision_sha256 = _load_locked_decision(decision_path)
    alpha = float(decision["geometry_weight"])
    if abs(alpha - 0.8) > 1e-12:
        raise RuntimeError(f"Expected locked alpha=0.8, found {alpha}")
    mechanism_completion = json.loads(
        (args.mechanism_root / "validation_mechanism_full_completed.json")
        .read_text(encoding="utf-8")
    )
    if mechanism_completion.get("decision_sha256") != decision_sha256:
        raise RuntimeError("Mechanism scores and locked decision hashes differ")

    mode = args.stage
    seeds = SEEDS if mode == "full" else (0,)
    budgets = BUDGETS if mode == "full" else (1, 2)
    rows, statistics = [], []
    for seed in seeds:
        arrays = _load_scores(_score_path(args.mechanism_root, seed))
        seed_rows, seed_statistics = _analyse_seed(
            seed, arrays, alpha, budgets
        )
        rows.extend(seed_rows)
        statistics.extend(seed_statistics)
    result = pd.DataFrame(rows)
    statistic_frame = pd.DataFrame(statistics)
    expected_rows = len(seeds) * len(budgets) * 3
    if len(result) != expected_rows or len(statistic_frame) != len(seeds) * len(budgets):
        raise RuntimeError("Calibration-budget output is incomplete")
    summary = _flatten(
        result.groupby(["CalibrationImagesPerClass", "Method"], sort=True)[
            ["AUROC", "FPR95", "AUPR_IN", "AUPR_OUT", "IDAccuracy"]
        ].agg(["mean", "std"]).reset_index()
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "raw": args.output_root / f"calibration_budget_{mode}_raw.csv",
        "statistics": args.output_root / f"calibration_budget_{mode}_statistics.csv",
        "summary": args.output_root / f"calibration_budget_{mode}_mean_std.csv",
    }
    result.to_csv(outputs["raw"], index=False, float_format="%.6f")
    statistic_frame.to_csv(outputs["statistics"], index=False, float_format="%.9f")
    summary.to_csv(outputs["summary"], index=False, float_format="%.6f")
    completion_path = args.output_root / f"calibration_budget_{mode}_completed.json"
    completion_path.write_text(json.dumps({
        "format_version": FORMAT_VERSION,
        "protocol": "ID-only nested calibration budgets; fixed ID validation evaluation",
        "near_far_test_access": False,
        "ood_validation_used_for_calibration": False,
        "decision_sha256": decision_sha256,
        "geometry_weight": alpha,
        "seeds": list(seeds),
        "budgets": list(budgets),
        "fixed_evaluation_images_per_class": 1,
        "raw_rows": len(result),
        "statistics_rows": len(statistic_frame),
        "outputs": {key: str(path) for key, path in outputs.items()},
        "completed": True,
    }, indent=2), encoding="utf-8")
    print("===== Calibration-budget sensitivity =====")
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved under: {args.output_root}")
    print("OOD validation scores were not used for calibration.")
    print("No near/far OOD test loader or score file was opened.")


if __name__ == "__main__":
    main()
