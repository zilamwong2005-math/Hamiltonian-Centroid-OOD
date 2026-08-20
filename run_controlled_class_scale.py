"""Controlled ImageNet-1K class-count scaling on validation data only.

The backbone, image features, preprocessing and OpenImage-O validation set are
held fixed.  Within each pre-specified nested class permutation, only the
number of active ImageNet classes changes.  Near/far test loaders and test
score files are never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from analyze_validation_fusion_mechanism import (
    _correlation,
    _standardized_gap,
)
from diagnose_imagenet1k_score_rules import _metric, _validation_masks, _zscore
from hamiltonian_detector import load_torch_checkpoint
from run_locked_imagenet1k_fusion import (
    _load_detector,
    _validation_feature_path,
)


FORMAL_CLASS_COUNTS = (10, 25, 50, 100, 200, 500, 1000)
SMOKE_CLASS_COUNTS = (10, 200, 1000)
ALPHAS = tuple(round(value / 10, 1) for value in range(11))
FORMAL_TRIAL_SEEDS = (0, 1, 2)
CLASS_PERMUTATION_OFFSET = 20260818
EXPECTED_ID_COUNT = 5000
EXPECTED_OOD_COUNT = 1763
EXPECTED_CLASSES = 1000
EXPECTED_FEATURE_DIM = 2048
FORMAT_VERSION = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("smoke", "full"))
    parser.add_argument(
        "--hamiltonian-root", type=Path, default=Path("results_openood")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/journal/controlled_class_scale"),
    )
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--smoke-ood-samples", type=int, default=128)
    parser.add_argument("--smoke-id-samples-per-class", type=int, default=2)
    parser.add_argument(
        "--force", action="store_true", help="Recompute and overwrite this stage"
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _class_permutation(seed: int) -> np.ndarray:
    return np.random.default_rng(
        CLASS_PERMUTATION_OFFSET + int(seed)
    ).permutation(EXPECTED_CLASSES)


def _classes_for_k(seed: int, class_count: int) -> np.ndarray:
    if class_count <= 1 or class_count > EXPECTED_CLASSES:
        raise ValueError(f"Invalid class count: {class_count}")
    # Sorting changes only column order, not membership.  The first K elements
    # of one fixed permutation guarantee nested class sets within each trial.
    return np.sort(_class_permutation(seed)[:class_count])


def _limit_id_per_class(labels: torch.Tensor, samples_per_class: int) -> torch.Tensor:
    keep = []
    labels_np = labels.numpy()
    for label in range(EXPECTED_CLASSES):
        rows = np.flatnonzero(labels_np == label)
        keep.extend(rows[:samples_per_class].tolist())
    return torch.as_tensor(sorted(keep), dtype=torch.long)


def _validate_feature_cache(cache: dict) -> None:
    required = {
        "id_features", "id_logits", "id_labels", "ood_features", "ood_logits"
    }
    if missing := required.difference(cache):
        raise ValueError(f"Validation feature cache is missing: {sorted(missing)}")
    id_features = cache["id_features"]
    id_logits = cache["id_logits"]
    id_labels = cache["id_labels"]
    ood_features = cache["ood_features"]
    ood_logits = cache["ood_logits"]
    expected_shapes = {
        "id_features": (EXPECTED_ID_COUNT, EXPECTED_FEATURE_DIM),
        "id_logits": (EXPECTED_ID_COUNT, EXPECTED_CLASSES),
        "id_labels": (EXPECTED_ID_COUNT,),
        "ood_features": (EXPECTED_OOD_COUNT, EXPECTED_FEATURE_DIM),
        "ood_logits": (EXPECTED_OOD_COUNT, EXPECTED_CLASSES),
    }
    actual = {
        "id_features": tuple(id_features.shape),
        "id_logits": tuple(id_logits.shape),
        "id_labels": tuple(id_labels.shape),
        "ood_features": tuple(ood_features.shape),
        "ood_logits": tuple(ood_logits.shape),
    }
    if actual != expected_shapes:
        raise ValueError(
            f"Unexpected validation cache tensor shapes: {actual}; "
            f"expected {expected_shapes}"
        )
    labels = id_labels.long().numpy()
    if labels.min() != 0 or labels.max() != EXPECTED_CLASSES - 1:
        raise ValueError("ID validation labels do not cover classes 0 through 999")
    counts = np.bincount(labels, minlength=EXPECTED_CLASSES)
    if not np.all(counts == EXPECTED_ID_COUNT // EXPECTED_CLASSES):
        raise ValueError(
            "Expected exactly five ImageNet validation images per class"
        )
    for name in ("id_features", "id_logits", "ood_features", "ood_logits"):
        if not torch.isfinite(cache[name]).all():
            raise ValueError(f"Non-finite values in validation cache tensor {name}")


@torch.inference_mode()
def _cosine_matrix(
    features: torch.Tensor,
    centroids: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    centroids = centroids.to(device)
    parts = []
    for start in tqdm(
        range(0, len(features), batch_size), desc="Full centroid similarity"
    ):
        query = F.normalize(
            features[start : start + batch_size].to(device, non_blocking=True),
            dim=-1,
        )
        parts.append((query @ centroids.T).cpu())
    return torch.cat(parts)


def _local_id_labels(labels: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = np.isin(labels, classes)
    lookup = np.full(EXPECTED_CLASSES, -1, dtype=np.int64)
    lookup[classes] = np.arange(len(classes))
    local = lookup[labels[selected]]
    if (local < 0).any():
        raise RuntimeError("Failed to map selected ImageNet labels")
    return selected, local


def _analyse_configuration(
    *,
    trial_seed: int,
    class_count: int,
    classes: np.ndarray,
    id_logits: torch.Tensor,
    ood_logits: torch.Tensor,
    id_labels: np.ndarray,
    id_similarity: torch.Tensor,
    ood_similarity: torch.Tensor,
    calibration_seed: int,
) -> tuple[list[dict], dict]:
    id_selected, local_labels = _local_id_labels(id_labels, classes)
    id_columns = torch.as_tensor(classes, dtype=torch.long)
    id_rows = torch.from_numpy(np.flatnonzero(id_selected))
    selected_id_logits = id_logits.index_select(0, id_rows).index_select(1, id_columns)
    selected_ood_logits = ood_logits.index_select(1, id_columns)
    selected_id_similarity = id_similarity.index_select(0, id_rows).index_select(
        1, id_columns
    )
    selected_ood_similarity = ood_similarity.index_select(1, id_columns)

    id_msp = selected_id_logits.softmax(1).max(1).values.numpy()
    ood_msp = selected_ood_logits.softmax(1).max(1).values.numpy()
    id_geometry = selected_id_similarity.max(1).values.numpy()
    ood_geometry = selected_ood_similarity.max(1).values.numpy()
    predictions = selected_id_logits.argmax(1).numpy()
    id_tune, id_holdout, ood_tune, ood_holdout = _validation_masks(
        local_labels, len(ood_msp), calibration_seed
    )
    splits = {
        "tune": (id_tune, ood_tune),
        "holdout": (id_holdout, ood_holdout),
        "full": (
            np.ones(len(id_msp), dtype=bool),
            np.ones(len(ood_msp), dtype=bool),
        ),
    }
    id_msp_z = _zscore(id_msp, id_msp[id_tune])
    ood_msp_z = _zscore(ood_msp, id_msp[id_tune])
    id_geometry_z = _zscore(id_geometry, id_geometry[id_tune])
    ood_geometry_z = _zscore(ood_geometry, id_geometry[id_tune])

    # Tail-error complementarity is more directly related to FPR95 than a
    # global score correlation.  Thresholds are fixed using ID-tune only and
    # then audited on the untouched ID/OOD holdouts.
    msp_threshold = float(np.quantile(id_msp_z[id_tune], 0.05))
    geometry_threshold = float(np.quantile(id_geometry_z[id_tune], 0.05))
    msp_false_accept = ood_msp_z[ood_holdout] >= msp_threshold
    geometry_false_accept = ood_geometry_z[ood_holdout] >= geometry_threshold
    false_accept_intersection = msp_false_accept & geometry_false_accept
    false_accept_union = msp_false_accept | geometry_false_accept
    false_accept_exclusive = msp_false_accept ^ geometry_false_accept
    union_count = int(false_accept_union.sum())
    hard_tail_correlation = (
        _correlation(
            ood_msp_z[ood_holdout][false_accept_union],
            ood_geometry_z[ood_holdout][false_accept_union],
        )
        if union_count >= 2 else 0.0
    )

    rows = []
    for alpha in ALPHAS:
        id_score = (1.0 - alpha) * id_msp_z + alpha * id_geometry_z
        ood_score = (1.0 - alpha) * ood_msp_z + alpha * ood_geometry_z
        for split, (id_mask, ood_mask) in splits.items():
            row = _metric(
                f"fusion_alpha_{alpha:g}",
                split,
                id_score[id_mask],
                ood_score[ood_mask],
                predictions[id_mask],
                local_labels[id_mask],
            )
            row.update({
                "TrialSeed": trial_seed,
                "ClassCount": class_count,
                "Alpha": alpha,
                "ClassSubsetSHA256": hashlib.sha256(
                    classes.astype("<i4").tobytes()
                ).hexdigest(),
            })
            rows.append(row)
    statistics = {
        "TrialSeed": trial_seed,
        "ClassCount": class_count,
        "ClassSubsetSHA256": rows[0]["ClassSubsetSHA256"],
        "IDCount": len(id_msp),
        "OODCount": len(ood_msp),
        "ID_MSP_Geometry_Correlation": _correlation(id_msp, id_geometry),
        "OOD_MSP_Geometry_Correlation": _correlation(ood_msp, ood_geometry),
        "OOD_Holdout_MSP_Geometry_Correlation": _correlation(
            ood_msp[ood_holdout], ood_geometry[ood_holdout]
        ),
        "OOD_HardTail_MSP_Geometry_Correlation": hard_tail_correlation,
        "MSP_StandardizedGap": _standardized_gap(id_msp, ood_msp),
        "Geometry_StandardizedGap": _standardized_gap(
            id_geometry, ood_geometry
        ),
        "CalibrationIDCount": int(id_tune.sum()),
        "TuneOODCount": int(ood_tune.sum()),
        "MSP_IDHoldoutAcceptRate": float(
            (id_msp_z[id_holdout] >= msp_threshold).mean()
        ),
        "Geometry_IDHoldoutAcceptRate": float(
            (id_geometry_z[id_holdout] >= geometry_threshold).mean()
        ),
        "MSP_OODHoldoutFalseAcceptRate": float(msp_false_accept.mean()),
        "Geometry_OODHoldoutFalseAcceptRate": float(
            geometry_false_accept.mean()
        ),
        "OODFalseAcceptIntersectionRate": float(
            false_accept_intersection.mean()
        ),
        "OODFalseAcceptUnionRate": float(false_accept_union.mean()),
        "OODFalseAcceptExclusiveRate": float(false_accept_exclusive.mean()),
        "OODFalseAcceptJaccard": (
            float(false_accept_intersection.sum() / union_count)
            if union_count else 0.0
        ),
    }
    return rows, statistics


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [
        "_".join(str(item) for item in column if str(item))
        if isinstance(column, tuple) else str(column)
        for column in result.columns
    ]
    return result


def _summaries(rows: pd.DataFrame, statistics: pd.DataFrame):
    holdout = rows[rows["Split"].eq("holdout")]
    summary = _flatten_columns(
        holdout.groupby(["ClassCount", "Alpha"], sort=True)[
            ["AUROC", "FPR95", "AUPR_IN", "AUPR_OUT"]
        ].agg(["mean", "std"]).reset_index()
    )
    best_rows, delta_rows = [], []
    for class_count in sorted(summary["ClassCount"].unique()):
        selected = summary[summary["ClassCount"].eq(class_count)].set_index("Alpha")
        best = selected.sort_values(
            ["AUROC_mean", "FPR95_mean"], ascending=[False, True]
        ).iloc[0]
        best_alpha = float(best.name)
        best_rows.append({
            "ClassCount": int(class_count),
            "BestAlphaByMeanAUROC": best_alpha,
            "BestAUROC": float(best["AUROC_mean"]),
            "BestFPR95": float(best["FPR95_mean"]),
        })
        locked, msp, centroid = selected.loc[0.8], selected.loc[0.0], selected.loc[1.0]
        delta_rows.append({
            "ClassCount": int(class_count),
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
    statistic_summary = _flatten_columns(
        statistics.groupby("ClassCount", sort=True)[[
            "ID_MSP_Geometry_Correlation",
            "OOD_MSP_Geometry_Correlation",
            "OOD_Holdout_MSP_Geometry_Correlation",
            "OOD_HardTail_MSP_Geometry_Correlation",
            "MSP_StandardizedGap",
            "Geometry_StandardizedGap",
            "MSP_OODHoldoutFalseAcceptRate",
            "Geometry_OODHoldoutFalseAcceptRate",
            "OODFalseAcceptIntersectionRate",
            "OODFalseAcceptUnionRate",
            "OODFalseAcceptExclusiveRate",
            "OODFalseAcceptJaccard",
        ]].agg(["mean", "std"]).reset_index()
    )
    return summary, pd.DataFrame(best_rows), pd.DataFrame(delta_rows), statistic_summary


def _trend_audit(statistics: pd.DataFrame) -> dict:
    per_seed = []
    for seed, selected in statistics.groupby("TrialSeed"):
        selected = selected.sort_values("ClassCount")
        log_k = np.log10(selected["ClassCount"].to_numpy(dtype=float))
        correlation = selected["OOD_MSP_Geometry_Correlation"].to_numpy(dtype=float)
        spearman = float(pd.Series(log_k).rank().corr(pd.Series(correlation).rank()))
        slope = float(np.polyfit(log_k, correlation, 1)[0])
        per_seed.append({
            "TrialSeed": int(seed),
            "SpearmanClassCountVsOODScoreCorrelation": spearman,
            "LinearSlopeVsLog10ClassCount": slope,
        })
    return {
        "interpretation": (
            "Descriptive validation-only trend; not a causal or independent-sample "
            "significance test"
        ),
        "per_trial": per_seed,
        "all_spearman_negative": all(
            item["SpearmanClassCountVsOODScoreCorrelation"] < 0
            for item in per_seed
        ),
        "all_log_slopes_negative": all(
            item["LinearSlopeVsLog10ClassCount"] < 0 for item in per_seed
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    args.hamiltonian_root = args.hamiltonian_root.resolve()
    args.output_root = args.output_root.resolve()
    feature_path = (
        args.feature_cache.resolve()
        if args.feature_cache is not None
        else _validation_feature_path(args.hamiltonian_root)
    )
    if not feature_path.is_file():
        raise FileNotFoundError(f"Validation-only feature cache missing: {feature_path}")
    if "validation" not in str(feature_path).lower():
        raise RuntimeError("Refusing a feature cache path without 'validation' in its name")

    cache = load_torch_checkpoint(feature_path, map_location="cpu")
    _validate_feature_cache(cache)
    feature_sha256 = _sha256(feature_path)
    mode = args.stage
    class_counts = FORMAL_CLASS_COUNTS if mode == "full" else SMOKE_CLASS_COUNTS
    trial_seeds = FORMAL_TRIAL_SEEDS if mode == "full" else (0,)
    if mode == "smoke":
        id_rows = _limit_id_per_class(
            cache["id_labels"], args.smoke_id_samples_per_class
        )
        for name in ("id_features", "id_logits", "id_labels"):
            cache[name] = cache[name].index_select(0, id_rows)
        for name in ("ood_features", "ood_logits"):
            cache[name] = cache[name][: args.smoke_ood_samples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_root = args.output_root / mode
    mode_root.mkdir(parents=True, exist_ok=True)
    all_rows, all_statistics = [], []
    for trial_seed in trial_seeds:
        run_path = mode_root / f"trial_seed{trial_seed}.csv"
        statistics_path = mode_root / f"trial_seed{trial_seed}_statistics.csv"
        metadata_path = mode_root / f"trial_seed{trial_seed}.json"
        existing = [path.is_file() for path in (run_path, statistics_path, metadata_path)]
        if any(existing) and not all(existing) and not args.force:
            raise RuntimeError(
                f"Incomplete trial-{trial_seed} output; restore or remove its three files"
            )
        if all(existing) and not args.force:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("feature_sha256") != feature_sha256:
                raise RuntimeError(f"Stale feature cache for trial {trial_seed}")
            if metadata.get("format_version") != FORMAT_VERSION:
                raise RuntimeError(
                    f"Trial {trial_seed} uses an older output format; rerun with --force"
                )
            print(f"[resume] controlled class-scale trial {trial_seed}", flush=True)
            all_rows.append(pd.read_csv(run_path))
            all_statistics.append(pd.read_csv(statistics_path))
            continue

        detector, detector_path = _load_detector(
            args.hamiltonian_root, trial_seed, device
        )
        print(f"[trial {trial_seed}] detector: {detector_path}", flush=True)
        id_similarity = _cosine_matrix(
            cache["id_features"], detector.centroids,
            args.batch_size, device,
        )
        ood_similarity = _cosine_matrix(
            cache["ood_features"], detector.centroids,
            args.batch_size, device,
        )
        rows, statistics = [], []
        id_labels = cache["id_labels"].long().numpy()
        for class_count in class_counts:
            classes = _classes_for_k(trial_seed, class_count)
            print(
                f"[trial {trial_seed}] K={class_count} "
                f"classes_sha256={hashlib.sha256(classes.astype('<i4').tobytes()).hexdigest()[:12]}",
                flush=True,
            )
            result_rows, result_statistics = _analyse_configuration(
                trial_seed=trial_seed,
                class_count=class_count,
                classes=classes,
                id_logits=cache["id_logits"],
                ood_logits=cache["ood_logits"],
                id_labels=id_labels,
                id_similarity=id_similarity,
                ood_similarity=ood_similarity,
                calibration_seed=args.calibration_seed,
            )
            rows.extend(result_rows)
            statistics.append(result_statistics)
        run = pd.DataFrame(rows)
        statistic_frame = pd.DataFrame(statistics)
        expected_rows = len(class_counts) * len(ALPHAS) * 3
        if len(run) != expected_rows or len(statistic_frame) != len(class_counts):
            raise RuntimeError(f"Incomplete trial {trial_seed}")
        run.to_csv(run_path, index=False, float_format="%.6f")
        statistic_frame.to_csv(statistics_path, index=False, float_format="%.9f")
        metadata_path.write_text(json.dumps({
            "format_version": FORMAT_VERSION,
            "protocol": "controlled ImageNet-1K validation-only class-count scaling",
            "near_far_test_access": False,
            "feature_cache": str(feature_path),
            "feature_sha256": feature_sha256,
            "trial_seed": trial_seed,
            "class_permutation_offset": CLASS_PERMUTATION_OFFSET,
            "class_counts": list(class_counts),
            "alphas": list(ALPHAS),
            "detector": str(detector_path),
            "mode": mode,
        }, indent=2), encoding="utf-8")
        all_rows.append(run)
        all_statistics.append(statistic_frame)
        del detector, id_similarity, ood_similarity
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = pd.concat(all_rows, ignore_index=True)
    statistics = pd.concat(all_statistics, ignore_index=True)
    expected_rows = len(trial_seeds) * len(class_counts) * len(ALPHAS) * 3
    if len(rows) != expected_rows or len(statistics) != len(trial_seeds) * len(class_counts):
        raise RuntimeError("Controlled class-scale aggregate is incomplete")
    summary, best, deltas, statistic_summary = _summaries(rows, statistics)
    trend = _trend_audit(statistics)

    outputs = {
        "raw": args.output_root / f"controlled_class_scale_{mode}_raw.csv",
        "statistics": args.output_root / f"controlled_class_scale_{mode}_statistics.csv",
        "holdout": args.output_root / f"controlled_class_scale_{mode}_holdout_mean_std.csv",
        "best": args.output_root / f"controlled_class_scale_{mode}_best_alpha.csv",
        "deltas": args.output_root / f"controlled_class_scale_{mode}_locked_deltas.csv",
        "statistic_summary": args.output_root / f"controlled_class_scale_{mode}_statistic_summary.csv",
        "trend": args.output_root / f"controlled_class_scale_{mode}_trend_audit.json",
    }
    rows.to_csv(outputs["raw"], index=False, float_format="%.6f")
    statistics.to_csv(outputs["statistics"], index=False, float_format="%.9f")
    summary.to_csv(outputs["holdout"], index=False, float_format="%.6f")
    best.to_csv(outputs["best"], index=False, float_format="%.6f")
    deltas.to_csv(outputs["deltas"], index=False, float_format="%.6f")
    statistic_summary.to_csv(
        outputs["statistic_summary"], index=False, float_format="%.6f"
    )
    outputs["trend"].write_text(json.dumps(trend, indent=2), encoding="utf-8")
    completion_path = args.output_root / f"controlled_class_scale_{mode}_completed.json"
    completion_path.write_text(json.dumps({
        "format_version": FORMAT_VERSION,
        "protocol": "controlled ImageNet-1K validation-only class-count scaling",
        "near_far_test_access": False,
        "feature_cache": str(feature_path),
        "feature_sha256": feature_sha256,
        "trial_seeds": list(trial_seeds),
        "class_counts": list(class_counts),
        "alphas": list(ALPHAS),
        "raw_rows": len(rows),
        "statistics_rows": len(statistics),
        "outputs": {key: str(value) for key, value in outputs.items()},
        "completed": True,
    }, indent=2), encoding="utf-8")

    print("\n===== Controlled class-count best alpha (holdout) =====")
    print(best.round(4).to_string(index=False))
    print("\n===== Locked alpha=0.8 deltas (holdout) =====")
    print(deltas.round(4).to_string(index=False))
    print("\n===== Correlation/separation trend =====")
    print(statistic_summary.round(4).to_string(index=False))
    print("\n===== Descriptive trend audit =====")
    print(json.dumps(trend, indent=2))
    print(f"\nSaved under: {args.output_root}")
    print("No near/far OOD test loader or score file was opened.")


if __name__ == "__main__":
    main()
