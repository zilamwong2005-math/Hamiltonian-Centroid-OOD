"""Validation-only sensitivity to ID setup samples per ImageNet class.

The experiment reuses the audited ImageNet-1K ResNet-50 validation feature
cache and the three class-balanced setup caches created before test scoring.
It never opens a near/far test loader or score file.
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
from run_controlled_class_scale import (
    EXPECTED_CLASSES,
    EXPECTED_FEATURE_DIM,
    _limit_id_per_class,
    _validate_feature_cache,
)
from run_locked_imagenet1k_fusion import (
    CACHE_TAG,
    _load_detector,
    _source_directory,
    _validation_feature_path,
)


SAMPLE_COUNTS = (1, 2, 4, 6, 8, 12)
ALPHAS = tuple(round(value / 10, 1) for value in range(11))
SEEDS = (0, 1, 2)
SETUP_SAMPLES = 12
SUBSET_OFFSET = 20260818
FORMAT_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("smoke", "full"))
    parser.add_argument(
        "--hamiltonian-root", type=Path, default=Path("results_openood")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/journal/centroid_sample_efficiency"),
    )
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--smoke-ood-samples", type=int, default=128)
    parser.add_argument("--smoke-id-samples-per-class", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _setup_feature_path(root: Path, seed: int) -> Path:
    filename = f"setup_features_{CACHE_TAG}_seed{seed}_m{SETUP_SAMPLES}.pt"
    preferred = _source_directory(root, seed) / filename
    if preferred.is_file():
        return preferred
    seed_root = root / "imagenet1k" / CACHE_TAG / f"seed{seed}"
    candidates = sorted(seed_root.rglob(filename))
    if not candidates:
        raise FileNotFoundError(
            f"No setup feature cache named {filename} under {seed_root}"
        )
    # Every matching cache should represent the same pre-specified sample set.
    hashes = {_sha256(path) for path in candidates}
    if len(hashes) != 1:
        raise RuntimeError(
            f"Conflicting setup feature caches for seed {seed}: {candidates}"
        )
    return candidates[0]


def _validate_setup_cache(cache: dict, seed: int) -> None:
    if missing := {"features", "labels", "source_indices"}.difference(cache):
        raise ValueError(f"Setup cache is missing keys: {sorted(missing)}")
    features, labels, source_indices = (
        cache["features"], cache["labels"].long(), cache["source_indices"]
    )
    expected_count = EXPECTED_CLASSES * SETUP_SAMPLES
    if tuple(features.shape) != (expected_count, EXPECTED_FEATURE_DIM):
        raise ValueError(f"Unexpected setup feature shape: {tuple(features.shape)}")
    if tuple(labels.shape) != (expected_count,):
        raise ValueError(f"Unexpected setup label shape: {tuple(labels.shape)}")
    if tuple(source_indices.shape) != (expected_count,):
        raise ValueError(
            f"Unexpected setup source-index shape: {tuple(source_indices.shape)}"
        )
    counts = torch.bincount(labels, minlength=EXPECTED_CLASSES)
    if not torch.equal(counts, torch.full_like(counts, SETUP_SAMPLES)):
        raise ValueError("Setup cache is not exactly class balanced")
    metadata = cache.get("metadata")
    if metadata is not None:
        if int(metadata.get("seed", -1)) != seed:
            raise ValueError("Setup cache seed metadata mismatch")
        if int(metadata.get("samples_per_class", -1)) != SETUP_SAMPLES:
            raise ValueError("Setup cache sample-count metadata mismatch")
    if not torch.isfinite(features).all():
        raise ValueError("Non-finite setup features")


def _nested_setup_rows(labels: torch.Tensor, seed: int) -> torch.Tensor:
    rows = []
    for class_index in range(EXPECTED_CLASSES):
        class_rows = torch.where(labels == class_index)[0]
        generator = torch.Generator().manual_seed(
            SUBSET_OFFSET + seed * EXPECTED_CLASSES + class_index
        )
        rows.append(class_rows[torch.randperm(len(class_rows), generator=generator)])
    return torch.stack(rows)


def _centroids_for_m(
    features: torch.Tensor,
    nested_rows: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    if sample_count not in SAMPLE_COUNTS:
        raise ValueError(f"Unsupported setup sample count: {sample_count}")
    selected = features[nested_rows[:, :sample_count]]
    return F.normalize(selected.mean(1), dim=-1)


@torch.inference_mode()
def _max_centroid_score(
    features: torch.Tensor,
    centroids: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    centroids = centroids.to(device)
    parts = []
    for start in tqdm(
        range(0, len(features), batch_size), desc="Centroid sample-efficiency score"
    ):
        query = F.normalize(
            features[start : start + batch_size].to(device, non_blocking=True),
            dim=-1,
        )
        parts.append((query @ centroids.T).max(1).values.cpu())
    return torch.cat(parts).numpy()


def _analyse_scores(
    *,
    seed: int,
    sample_count: int,
    id_msp: np.ndarray,
    ood_msp: np.ndarray,
    id_geometry: np.ndarray,
    ood_geometry: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    calibration_seed: int,
) -> tuple[list[dict], dict]:
    id_tune, id_holdout, ood_tune, ood_holdout = _validation_masks(
        labels, len(ood_msp), calibration_seed
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
    rows = []
    for alpha in ALPHAS:
        id_score = (1.0 - alpha) * id_msp_z + alpha * id_geometry_z
        ood_score = (1.0 - alpha) * ood_msp_z + alpha * ood_geometry_z
        for split, (id_mask, ood_mask) in splits.items():
            row = _metric(
                f"fusion_alpha_{alpha:g}", split,
                id_score[id_mask], ood_score[ood_mask],
                predictions[id_mask], labels[id_mask],
            )
            row.update({"Seed": seed, "SetupSamplesPerClass": sample_count, "Alpha": alpha})
            rows.append(row)
    statistics = {
        "Seed": seed,
        "SetupSamplesPerClass": sample_count,
        "IDCount": len(id_msp),
        "OODCount": len(ood_msp),
        "ID_MSP_Geometry_Correlation": _correlation(id_msp, id_geometry),
        "OOD_MSP_Geometry_Correlation": _correlation(ood_msp, ood_geometry),
        "MSP_StandardizedGap": _standardized_gap(id_msp, ood_msp),
        "Geometry_StandardizedGap": _standardized_gap(id_geometry, ood_geometry),
    }
    return rows, statistics


def _flatten(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [
        "_".join(str(item) for item in column if str(item))
        if isinstance(column, tuple) else str(column)
        for column in result.columns
    ]
    return result


def _summaries(rows: pd.DataFrame):
    holdout = rows[rows["Split"].eq("holdout")]
    summary = _flatten(
        holdout.groupby(["SetupSamplesPerClass", "Alpha"], sort=True)[
            ["AUROC", "FPR95", "AUPR_IN", "AUPR_OUT"]
        ].agg(["mean", "std"]).reset_index()
    )
    best_rows = []
    for sample_count in sorted(summary["SetupSamplesPerClass"].unique()):
        selected = summary[
            summary["SetupSamplesPerClass"].eq(sample_count)
        ].set_index("Alpha")
        best = selected.sort_values(
            ["AUROC_mean", "FPR95_mean"], ascending=[False, True]
        ).iloc[0]
        best_rows.append({
            "SetupSamplesPerClass": sample_count,
            "BestAlphaByMeanAUROC": float(best.name),
            "BestAUROC": float(best["AUROC_mean"]),
            "BestFPR95": float(best["FPR95_mean"]),
        })
    locked = summary[summary["Alpha"].eq(0.8)].copy()
    reference = locked[locked["SetupSamplesPerClass"].eq(12)].iloc[0]
    locked["DeltaAUROCvsM12"] = locked["AUROC_mean"] - reference["AUROC_mean"]
    locked["DeltaFPR95vsM12"] = locked["FPR95_mean"] - reference["FPR95_mean"]
    return summary, pd.DataFrame(best_rows), locked


def main() -> None:
    args = build_parser().parse_args()
    args.hamiltonian_root = args.hamiltonian_root.resolve()
    args.output_root = args.output_root.resolve()
    validation_path = (
        args.feature_cache.resolve()
        if args.feature_cache is not None
        else _validation_feature_path(args.hamiltonian_root)
    )
    if not validation_path.is_file():
        raise FileNotFoundError(f"Validation feature cache missing: {validation_path}")
    if "validation" not in str(validation_path).lower():
        raise RuntimeError("Refusing a non-validation feature cache path")
    validation = load_torch_checkpoint(validation_path, map_location="cpu")
    _validate_feature_cache(validation)
    validation_sha256 = _sha256(validation_path)
    mode = args.stage
    seeds = SEEDS if mode == "full" else (0,)
    sample_counts = SAMPLE_COUNTS if mode == "full" else (1, 12)
    if mode == "smoke":
        id_rows = _limit_id_per_class(
            validation["id_labels"], args.smoke_id_samples_per_class
        )
        for name in ("id_features", "id_logits", "id_labels"):
            validation[name] = validation[name].index_select(0, id_rows)
        for name in ("ood_features", "ood_logits"):
            validation[name] = validation[name][: args.smoke_ood_samples]

    id_msp = validation["id_logits"].softmax(1).max(1).values.numpy()
    ood_msp = validation["ood_logits"].softmax(1).max(1).values.numpy()
    predictions = validation["id_logits"].argmax(1).numpy()
    labels = validation["id_labels"].long().numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_root = args.output_root / mode
    mode_root.mkdir(parents=True, exist_ok=True)
    all_rows, all_statistics, audits = [], [], []
    for seed in seeds:
        result_path = mode_root / f"seed{seed}.csv"
        statistics_path = mode_root / f"seed{seed}_statistics.csv"
        audit_path = mode_root / f"seed{seed}.json"
        existing = [path.is_file() for path in (result_path, statistics_path, audit_path)]
        if any(existing) and not all(existing) and not args.force:
            raise RuntimeError(f"Incomplete seed-{seed} output; rerun with --force")
        if all(existing) and not args.force:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if audit.get("format_version") != FORMAT_VERSION:
                raise RuntimeError(f"Old seed-{seed} format; rerun with --force")
            if audit.get("validation_sha256") != validation_sha256:
                raise RuntimeError(f"Stale validation cache for seed {seed}")
            all_rows.append(pd.read_csv(result_path))
            all_statistics.append(pd.read_csv(statistics_path))
            audits.append(audit)
            print(f"[resume] sample-efficiency seed {seed}", flush=True)
            continue

        setup_path = _setup_feature_path(args.hamiltonian_root, seed)
        setup = load_torch_checkpoint(setup_path, map_location="cpu")
        _validate_setup_cache(setup, seed)
        setup_sha256 = _sha256(setup_path)
        nested_rows = _nested_setup_rows(setup["labels"].long(), seed)
        detector, detector_path = _load_detector(
            args.hamiltonian_root, seed, device
        )
        m12_centroids = _centroids_for_m(setup["features"], nested_rows, 12)
        checkpoint_difference = float(
            (m12_centroids - detector.centroids.cpu()).abs().max()
        )
        if checkpoint_difference > 2e-5:
            raise RuntimeError(
                f"Seed-{seed} M=12 centroid mismatch: {checkpoint_difference}"
            )
        rows, statistic_rows = [], []
        for sample_count in sample_counts:
            print(f"[seed {seed}] setup samples/class={sample_count}", flush=True)
            centroids = _centroids_for_m(
                setup["features"], nested_rows, sample_count
            )
            id_geometry = _max_centroid_score(
                validation["id_features"], centroids, args.batch_size, device
            )
            ood_geometry = _max_centroid_score(
                validation["ood_features"], centroids, args.batch_size, device
            )
            result_rows, result_statistics = _analyse_scores(
                seed=seed,
                sample_count=sample_count,
                id_msp=id_msp,
                ood_msp=ood_msp,
                id_geometry=id_geometry,
                ood_geometry=ood_geometry,
                predictions=predictions,
                labels=labels,
                calibration_seed=args.calibration_seed,
            )
            rows.extend(result_rows)
            statistic_rows.append(result_statistics)
        frame, statistic_frame = pd.DataFrame(rows), pd.DataFrame(statistic_rows)
        frame.to_csv(result_path, index=False, float_format="%.6f")
        statistic_frame.to_csv(statistics_path, index=False, float_format="%.9f")
        audit = {
            "format_version": FORMAT_VERSION,
            "protocol": "ImageNet-1K validation-only centroid sample efficiency",
            "near_far_test_access": False,
            "seed": seed,
            "sample_counts": list(sample_counts),
            "validation_cache": str(validation_path),
            "validation_sha256": validation_sha256,
            "setup_cache": str(setup_path),
            "setup_sha256": setup_sha256,
            "detector": str(detector_path),
            "m12_checkpoint_max_abs_difference": checkpoint_difference,
        }
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        all_rows.append(frame)
        all_statistics.append(statistic_frame)
        audits.append(audit)
        del detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = pd.concat(all_rows, ignore_index=True)
    statistics = pd.concat(all_statistics, ignore_index=True)
    expected_rows = len(seeds) * len(sample_counts) * len(ALPHAS) * 3
    if len(rows) != expected_rows or len(statistics) != len(seeds) * len(sample_counts):
        raise RuntimeError("Sample-efficiency aggregate is incomplete")
    summary, best, locked = _summaries(rows)
    outputs = {
        "raw": args.output_root / f"sample_efficiency_{mode}_raw.csv",
        "statistics": args.output_root / f"sample_efficiency_{mode}_statistics.csv",
        "holdout": args.output_root / f"sample_efficiency_{mode}_holdout_mean_std.csv",
        "best": args.output_root / f"sample_efficiency_{mode}_best_alpha.csv",
        "locked": args.output_root / f"sample_efficiency_{mode}_locked_sensitivity.csv",
    }
    rows.to_csv(outputs["raw"], index=False, float_format="%.6f")
    statistics.to_csv(outputs["statistics"], index=False, float_format="%.9f")
    summary.to_csv(outputs["holdout"], index=False, float_format="%.6f")
    best.to_csv(outputs["best"], index=False, float_format="%.6f")
    locked.to_csv(outputs["locked"], index=False, float_format="%.6f")
    completion_path = args.output_root / f"sample_efficiency_{mode}_completed.json"
    completion_path.write_text(json.dumps({
        "format_version": FORMAT_VERSION,
        "protocol": "ImageNet-1K validation-only centroid sample efficiency",
        "near_far_test_access": False,
        "seeds": list(seeds),
        "sample_counts": list(sample_counts),
        "alphas": list(ALPHAS),
        "raw_rows": len(rows),
        "statistics_rows": len(statistics),
        "m12_checkpoint_max_abs_difference": max(
            audit["m12_checkpoint_max_abs_difference"] for audit in audits
        ),
        "outputs": {key: str(path) for key, path in outputs.items()},
        "completed": True,
    }, indent=2), encoding="utf-8")
    print("\n===== Best alpha by setup samples/class =====")
    print(best.round(4).to_string(index=False))
    print("\n===== Locked alpha=0.8 sample-efficiency sensitivity =====")
    print(locked.round(4).to_string(index=False))
    print("\n===== M=12 checkpoint audit =====")
    print(
        f"max abs centroid difference: "
        f"{max(audit['m12_checkpoint_max_abs_difference'] for audit in audits):.9g}"
    )
    print(f"\nSaved under: {args.output_root}")
    print("No near/far OOD test loader or score file was opened.")


if __name__ == "__main__":
    main()
