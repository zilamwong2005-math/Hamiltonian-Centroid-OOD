"""Verify the three post-manuscript experiments required for the revision.

The audit is intentionally strict.  It accepts only:

* an eleven-model formal CTM matrix (83 OpenOOD rows);
* the validation-only ImageNet-1K six-stage static-reduction bridge; and
* the complete Gaussian/IMQ ImageNet-1K T=10 trajectory matrix over seeds 0--2.

Run this only after the formal stages have finished.  A failed audit is useful:
it reports the missing or malformed component rather than silently producing a
partial paper table.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd


CTM_TARGET_ORDER = (
    "cifar10",
    "cifar100",
    "imagenet200",
    "imagenet1k_resnet50",
    "imagenet1k_densenet121",
)
CTM_TARGETS = {
    ("cifar10", 0), ("cifar10", 1), ("cifar10", 2),
    ("cifar100", 0), ("cifar100", 1), ("cifar100", 2),
    ("imagenet200", 0), ("imagenet200", 1), ("imagenet200", 2),
    ("imagenet1k_resnet50", 0), ("imagenet1k_densenet121", 0),
}
BRIDGE_STAGES = (
    "S1_exact_full_class_static",
    "S2_common_scale",
    "S3_radial_centroid",
    "S4_anchor_centroid_cosine",
    "S5_setup_centroid_cosine",
    "S6_locked_centroid_msp",
)
BRIDGE_METHODS = {"MSP_reference", *BRIDGE_STAGES}
T10_CACHE_TAG = "imagenet1k_resnet50_tvsv1"
T10_EXPERIMENT_TAG = "mass-uniform-none_loss-static-ts10_pred-backbone_eval-full"
T10_DATASETS = {
    "ssb_hard",
    "ninco",
    "nearood",
    "inaturalist",
    "textures",
    "openimage_o",
    "farood",
}
T10_CONFIG = {
    "id_data": "imagenet1k",
    "tvs_version": 1,
    "n_anchors": 5,
    "setup_samples_per_class": 12,
    "ham_epochs": 10,
    "ham_lr": 0.001,
    "ham_batch_size": 32,
    "bandwidth_loss": "static",
    "trajectory_train_steps": 0,
    "n_steps": 10,
    "dt": 0.05,
    "candidate_k": 20,
    "sim_batch": 32,
    "sigma_init": 0.5,
    "sigma_min": 0.05,
    "sigma_max": 4.0,
    "mass_mode": "uniform",
    "mass_normalization": "none",
    "mass_resolution": 0,
    "prediction_source": "backbone",
    "batch_size": 64,
    "setup_batch_size": 128,
    "num_workers": 8,
    "max_eval_samples": 0,
    "force_retrain": False,
    "skip_download": True,
    "force_download": False,
    "download_only": False,
    "save_scores": False,
    "fsood": False,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("results_openood"))
    parser.add_argument("--journal-root", type=Path, default=Path("results/journal"))
    parser.add_argument("--output", type=Path)
    return parser


def _load_completion(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing completion manifest: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("completed", False):
        raise RuntimeError(f"Completion manifest is not marked complete: {path}")
    return value


def _ctm_audit(journal_root: Path) -> dict:
    root = journal_root / "ctm"
    completion = _load_completion(root / "ctm_full_completed.json")
    expected_completion = {
        "stage": "full",
        "targets": list(CTM_TARGET_ORDER),
        "requested_seeds": [0, 1, 2],
        "independent_runs": 11,
        "expected_formal_runs": 11,
        "fixed_imagenet_models_repeated": False,
        "all_id_train_samples": True,
        "max_eval_samples": 0,
    }
    for key, expected in expected_completion.items():
        if completion.get(key) != expected:
            raise RuntimeError(
                f"CTM completion manifest has unexpected {key}: "
                f"{completion.get(key)!r}; expected {expected!r}"
            )
    path = root / "ctm_full_all_runs.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"Target", "Seed", "Dataset", "Method", "AUROC", "FPR@95"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"CTM combined CSV lacks {missing}")
    pairs = set(zip(frame["Target"], frame["Seed"].astype(int)))
    if pairs != CTM_TARGETS:
        raise RuntimeError(f"CTM independent-run matrix mismatch: {sorted(pairs)}")
    counts = frame.groupby(["Target", "Seed"])["Dataset"].nunique()
    expected_counts = {
        pair: (8 if pair[0] in {"cifar10", "cifar100"} else 7)
        for pair in CTM_TARGETS
    }
    actual_counts = {tuple(index): int(value) for index, value in counts.items()}
    if len(frame) != 83 or actual_counts != expected_counts:
        raise RuntimeError(
            "CTM must contain 83 rows: two CIFAR targets x 3 seeds x 8 rows, "
            "ImageNet-200 x 3 seeds x 7 rows, and two fixed ImageNet-1K "
            "backbones x 7 rows."
        )
    if not frame["Method"].astype(str).str.lower().eq("ctm").all():
        raise RuntimeError("CTM combined CSV contains a non-CTM method label")
    duplicated = frame.duplicated(["Target", "Seed", "Dataset"], keep=False)
    if duplicated.any():
        raise RuntimeError("CTM combined CSV contains duplicate target/seed/dataset rows")
    return {
        "completion": str((root / "ctm_full_completed.json").resolve()),
        "combined_csv": str(path.resolve()),
        "independent_runs": len(pairs),
        "rows": len(frame),
        "full_id_train_centroids": True,
    }


def _bridge_audit(journal_root: Path) -> dict:
    root = journal_root / "static_reduction_bridge"
    completion_path = root / "static_reduction_bridge_full_completed.json"
    completion = _load_completion(completion_path)
    if completion.get("stage") != "full":
        raise RuntimeError("Bridge completion must attest the formal full stage")
    if completion.get("seeds") != [0, 1, 2]:
        raise RuntimeError("Bridge completion must attest detector seeds [0, 1, 2]")
    if completion.get("stages") != list(BRIDGE_STAGES):
        raise RuntimeError("Bridge completion does not contain the exact six-stage protocol")
    inputs = completion.get("validation_inputs")
    if not isinstance(inputs, dict):
        raise RuntimeError("Bridge completion lacks validation-input provenance")
    if inputs.get("id_count") != 5_000 or inputs.get("ood_count") != 1_763:
        raise RuntimeError("Bridge must use all 5,000 ID and 1,763 OpenImage-O validation images")
    if inputs.get("cache_tag") != T10_CACHE_TAG:
        raise RuntimeError("Bridge validation cache has the wrong model cache tag")
    cache_hash = str(inputs.get("source_cache_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", cache_hash):
        raise RuntimeError("Bridge completion lacks a valid validation-cache SHA256")
    control = completion.get("static_field_control")
    if not isinstance(control, dict):
        raise RuntimeError("Bridge completion lacks exact-field control provenance")
    if control.get("stored_detector_candidate_k") != 20:
        raise RuntimeError("Bridge detector must retain the deployed candidate_k=20")
    if control.get("bridge_exact_field_candidate_k") != 0:
        raise RuntimeError("Bridge S1 must be the all-class candidate_k=0 control")
    for key in (
        "near_far_test_access",
        "radial_cosine_rank_equivalence_passed",
        "radial_cosine_class_equivalence_passed",
        "radial_centroid_bound_passed",
    ):
        expected = False if key == "near_far_test_access" else True
        if completion.get(key) is not expected:
            raise RuntimeError(f"Bridge manifest has unexpected {key}: {completion.get(key)!r}")
    path = root / "full" / "stage_metrics.csv"
    bound_path = root / "full" / "radial_centroid_bound_audit.csv"
    if not path.is_file() or not bound_path.is_file():
        raise FileNotFoundError(f"Bridge outputs missing under {root / 'full'}")
    metrics = pd.read_csv(path)
    expected_rows = 3 * 3 * len(BRIDGE_METHODS)
    if len(metrics) != expected_rows:
        raise RuntimeError(f"Bridge metrics rows={len(metrics)}, expected {expected_rows}")
    if set(metrics["Seed"].astype(int)) != {0, 1, 2}:
        raise RuntimeError("Bridge must contain detector seeds 0, 1, and 2")
    if set(metrics["Method"]) != BRIDGE_METHODS:
        raise RuntimeError("Bridge stage names are incomplete or unexpected")
    if set(metrics["Split"]) != {"tune", "holdout", "full"}:
        raise RuntimeError("Bridge must retain tune, holdout, and full validation splits")
    metric_counts = metrics.groupby(["Seed", "Method", "Split"]).size()
    expected_metric_index = pd.MultiIndex.from_product(
        [[0, 1, 2], sorted(BRIDGE_METHODS), ["tune", "holdout", "full"]],
        names=["Seed", "Method", "Split"],
    )
    metric_counts = metric_counts.reindex(expected_metric_index, fill_value=0)
    if not (metric_counts == 1).all():
        raise RuntimeError(
            "Bridge metrics must contain exactly one row for every "
            "seed/method/validation-split combination"
        )
    bounds = pd.read_csv(bound_path)
    bound_ok = bounds["BoundSatisfied"].map(
        lambda value: str(value).strip().lower() == "true"
    )
    if not bound_ok.all():
        raise RuntimeError("Bridge radial-centroid bound audit contains a violation")
    expected_records = {0, 1, 2}
    records = completion.get("detectors", [])
    if {int(record.get("seed", -1)) for record in records} != expected_records:
        raise RuntimeError("Bridge detector provenance does not cover all three seeds")
    if any(
        record.get("stored_candidate_k") != 20
        or record.get("bridge_exact_field_candidate_k") != 0
        for record in records
    ):
        raise RuntimeError("Bridge detector provenance has inconsistent candidate-k controls")
    return {
        "completion": str(completion_path.resolve()),
        "metrics_csv": str(path.resolve()),
        "rows": len(metrics),
        "validation_only": True,
        "bound_rows": len(bounds),
    }


def _config_value_matches(actual, expected) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return str(actual).lower() == str(expected).lower()


def _is_formal_t10_config(config: dict, path: Path) -> bool:
    """Accept only the fixed command issued by trajectory_imagenet1k_t10."""

    try:
        seed = int(config.get("seed"))
    except (TypeError, ValueError):
        return False
    if seed not in {0, 1, 2}:
        return False
    if any(
        not _config_value_matches(config.get(key), expected)
        for key, expected in T10_CONFIG.items()
    ):
        return False
    potentials = [str(value).lower() for value in config.get("potentials", [])]
    if potentials != ["gaussian", "imq"]:
        return False
    parts = path.resolve().parts
    return (
        T10_CACHE_TAG in parts
        and f"seed{seed}" in parts
        and path.parent.name == T10_EXPERIMENT_TAG
    )


def _t10_audit(output_root: Path) -> dict:
    paths = sorted(output_root.rglob("openood_metrics_all_potentials.csv"))
    selected_frames = []
    selected_configs: list[str] = []
    for path in paths:
        config_path = path.with_name("run_config.json")
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not _is_formal_t10_config(config, path):
            continue
        frame = pd.read_csv(path)
        required = {"ID", "Potential", "Dataset", "Seed", "MassMode", "MassNormalization", "BandwidthLoss", "TrajectoryTrainSteps", "PredictionSource"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise RuntimeError(f"T=10 metrics lacks {missing}: {path}")
        frame["SourceFile"] = str(path.resolve())
        frame["SourceConfig"] = str(config_path.resolve())
        selected_frames.append(frame)
        selected_configs.append(str(config_path.resolve()))
    if not selected_frames:
        raise FileNotFoundError(
            "No ImageNet-1K T=10 result has the exact formal run_config.json under "
            f"{output_root}"
        )
    frame = pd.concat(selected_frames, ignore_index=True)
    mask = (
        frame["ID"].astype(str).str.lower().eq("imagenet1k")
        & frame["Potential"].astype(str).str.lower().isin({"gaussian", "imq"})
        & frame["Dataset"].astype(str).str.lower().isin(T10_DATASETS)
        & frame["MassMode"].astype(str).str.lower().eq("uniform")
        & frame["MassNormalization"].astype(str).str.lower().eq("none")
        & frame["BandwidthLoss"].astype(str).str.lower().eq("static")
        & pd.to_numeric(frame["TrajectoryTrainSteps"], errors="raise").eq(10)
        & frame["PredictionSource"].astype(str).str.lower().eq("backbone")
    )
    selected = frame.loc[mask].copy()
    selected["Seed"] = pd.to_numeric(selected["Seed"], errors="raise").astype(int)
    selected["Potential"] = selected["Potential"].str.lower()
    selected["Dataset"] = selected["Dataset"].str.lower()
    duplicated = selected.duplicated(["Seed", "Potential", "Dataset"], keep=False)
    if duplicated.any():
        files = sorted(selected.loc[duplicated, "SourceFile"].unique().tolist())
        raise RuntimeError(
            "ImageNet-1K T=10 has duplicate seed/potential/dataset rows across "
            f"formal sources: {files}"
        )
    counts = selected.groupby(["Seed", "Potential"])["Dataset"].nunique()
    expected_index = pd.MultiIndex.from_product(
        [[0, 1, 2], ["gaussian", "imq"]], names=["Seed", "Potential"]
    )
    counts = counts.reindex(expected_index, fill_value=0)
    if not (counts == 7).all() or len(selected) != 42:
        raise RuntimeError(
            "ImageNet-1K T=10 must contain exactly 42 rows: "
            "three seeds x Gaussian/IMQ x seven named/aggregate datasets."
        )
    return {
        "rows": len(selected),
        "seeds": [0, 1, 2],
        "potentials": ["gaussian", "imq"],
        "source_files": sorted(selected["SourceFile"].unique().tolist()),
        "run_configs": sorted(set(selected_configs)),
    }


def main() -> None:
    args = build_parser().parse_args()
    output_root = args.output_root.resolve()
    journal_root = args.journal_root.resolve()
    destination = (args.output or journal_root / "required_additions_audit.json").resolve()
    audit = {
        "ctm": _ctm_audit(journal_root),
        "static_reduction_bridge": _bridge_audit(journal_root),
        "imagenet1k_t10": _t10_audit(output_root),
        "completed": True,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(f"Saved audit: {destination}")


if __name__ == "__main__":
    main()
