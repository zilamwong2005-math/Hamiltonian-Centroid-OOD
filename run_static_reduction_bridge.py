"""Validation-only audit of the six-stage static-reduction bridge.

This runner is deliberately restricted to cached features from the official
ImageNet-1K ID validation split (5,000 images) and the OpenImage-O validation
split (1,763 images).  It never constructs, iterates, or loads scores from a
Near-OOD or Far-OOD test loader.

For each of the three pre-specified Hamiltonian detector seeds it evaluates:

S1  the exact learned Gaussian zero-step field over all 1,000 classes;
S2  the same empirical anchors with uniform weights and one shared bandwidth;
S3  a radial Gaussian field centred at the same-anchor centroid;
S4  maximum cosine similarity to the same-anchor centroid;
S5  maximum cosine similarity to the 12-sample setup-bank centroid; and
S6  the locked alpha=0.8 ID-standardised fusion of S5 and MSP.

S1 intentionally evaluates the all-class field (the mathematical
``candidate_k=0`` endpoint), whereas the stored ImageNet-1K detector used
``candidate_k=20`` for scalable deployment.  Thus S1 is an exact theoretical
static-field control, not a re-evaluation of the deployed candidate-pruned
score.

The output is a mechanistic bridge audit, not a claim that performance must
improve monotonically from one stage to the next.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from diagnose_imagenet1k_score_rules import _metric, _validation_masks, _zscore
from run_locked_imagenet1k_fusion import CACHE_TAG, _load_detector


EXPECTED_ID_COUNT = 5_000
EXPECTED_OOD_COUNT = 1_763
EXPECTED_CLASS_COUNT = 1_000
EXPECTED_FEATURE_DIM = 2_048
LOCKED_ALPHA = 0.8
GAUSSIAN_LIPSCHITZ = math.exp(-0.5)
DEPLOYED_DETECTOR_CANDIDATE_K = 20
BRIDGE_EXACT_CANDIDATE_K = 0

STAGES = (
    "S1_exact_full_class_static",
    "S2_common_scale",
    "S3_radial_centroid",
    "S4_anchor_centroid_cosine",
    "S5_setup_centroid_cosine",
    "S6_locked_centroid_msp",
)
TRANSITIONS = tuple(zip(STAGES[:-1], STAGES[1:]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--openood-root", type=Path, default=Path("OpenOOD"))
    parser.add_argument(
        "--hamiltonian-root", type=Path, default=Path("results_openood")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/journal/static_reduction_bridge"),
    )
    parser.add_argument("--validation-cache", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--smoke-id-per-class", type=int, default=2)
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=128,
        help=(
            "Smoke only: maximum OpenImage-O validation examples. ID keeps "
            "--smoke-id-per-class examples for each of 1,000 classes so that "
            "the class-stratified tune/holdout split remains valid."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an already completed result for the requested stage.",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_validation_caches(root: Path) -> list[Path]:
    base = root / "imagenet1k" / CACHE_TAG / "seed0"
    return [
        base
        / "mass-uniform-none_loss-static-ts10_pred-backbone_eval-full"
        / "validation_scaling_diagnosis"
        / "validation_features.pt",
        base
        / "mass-uniform-none_loss-static-ts10_pred-backbone_eval-full"
        / "validation_score_rule_diagnosis_v1"
        / "validation_features.pt",
    ]


def _resolve_validation_cache(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Validation cache is missing: {path}")
        _validate_cache_source_path(path)
        return path
    candidates = [path for path in _candidate_validation_caches(root) if path.is_file()]
    if not candidates:
        expected = "\n".join(str(path) for path in _candidate_validation_caches(root))
        raise FileNotFoundError(
            "The validation-only feature cache is missing. Run the existing "
            "ImageNet-1K scaling diagnosis once, or pass --validation-cache. "
            "For safety this runner will not fall back to a general OOD "
            f"evaluator. Expected one of:\n{expected}"
        )
    path = candidates[0]
    _validate_cache_source_path(path)
    return path


def _validate_cache_source_path(path: Path) -> None:
    """Pin the bridge to the audited ResNet-50 validation-cache provenance."""

    source = path.resolve()
    allowed_parents = {
        "validation_scaling_diagnosis",
        "validation_score_rule_diagnosis_v1",
    }
    if (
        source.name != "validation_features.pt"
        or CACHE_TAG not in source.parts
        or source.parent.name not in allowed_parents
    ):
        raise RuntimeError(
            "Refusing a validation cache outside the audited ImageNet-1K "
            "ResNet-50 diagnostic locations: "
            f"{source}"
        )


def _validate_full_cache(cache: dict) -> None:
    required = {
        "id_features",
        "id_logits",
        "id_labels",
        "ood_features",
        "ood_logits",
    }
    missing = required.difference(cache)
    if missing:
        raise RuntimeError(f"Validation cache is missing keys: {sorted(missing)}")

    id_features = cache["id_features"]
    id_logits = cache["id_logits"]
    id_labels = cache["id_labels"]
    ood_features = cache["ood_features"]
    ood_logits = cache["ood_logits"]
    if len(id_features) != EXPECTED_ID_COUNT or len(id_labels) != EXPECTED_ID_COUNT:
        raise RuntimeError(
            "Refusing cache: the ImageNet-1K ID validation split must contain "
            f"exactly {EXPECTED_ID_COUNT} examples, found {len(id_features)}."
        )
    if len(ood_features) != EXPECTED_OOD_COUNT:
        raise RuntimeError(
            "Refusing cache: the OpenImage-O validation split must contain "
            f"exactly {EXPECTED_OOD_COUNT} examples, found {len(ood_features)}."
        )
    if tuple(id_features.shape[1:]) != (EXPECTED_FEATURE_DIM,) or tuple(
        ood_features.shape[1:]
    ) != (EXPECTED_FEATURE_DIM,):
        raise RuntimeError("Refusing cache: expected 2,048-dimensional ResNet-50 features")
    if tuple(id_logits.shape) != (EXPECTED_ID_COUNT, EXPECTED_CLASS_COUNT):
        raise RuntimeError("Refusing cache: unexpected ImageNet-1K ID logits shape")
    if tuple(ood_logits.shape) != (EXPECTED_OOD_COUNT, EXPECTED_CLASS_COUNT):
        raise RuntimeError("Refusing cache: unexpected OpenImage-O validation logits shape")
    labels = id_labels.long()
    counts = torch.bincount(labels, minlength=EXPECTED_CLASS_COUNT)
    if len(counts) != EXPECTED_CLASS_COUNT or not torch.all(counts == 5):
        raise RuntimeError(
            "Refusing cache: the official ID validation subset must have five "
            "images per each of 1,000 classes"
        )
    if not all(torch.isfinite(cache[key]).all() for key in required):
        raise RuntimeError("Refusing cache: non-finite feature, logit, or label values")


def _smoke_subset(
    cache: dict, id_per_class: int, ood_samples: int, seed: int
) -> dict:
    if not 2 <= id_per_class <= 5:
        raise ValueError("--smoke-id-per-class must be between 2 and 5")
    if not 2 <= ood_samples <= EXPECTED_OOD_COUNT:
        raise ValueError(
            f"--max-eval-samples must be between 2 and {EXPECTED_OOD_COUNT} "
            "for --stage smoke"
        )
    labels = cache["id_labels"].long().numpy()
    id_rows: list[int] = []
    for label in range(EXPECTED_CLASS_COUNT):
        rows = np.flatnonzero(labels == label)
        rows = np.random.default_rng(seed + label).permutation(rows)
        id_rows.extend(rows[:id_per_class].tolist())
    id_index = torch.as_tensor(sorted(id_rows), dtype=torch.long)
    ood_rows = np.random.default_rng(seed + 50_000).permutation(
        EXPECTED_OOD_COUNT
    )[:ood_samples]
    ood_index = torch.as_tensor(np.sort(ood_rows), dtype=torch.long)
    return {
        "id_features": cache["id_features"][id_index],
        "id_logits": cache["id_logits"][id_index],
        "id_labels": cache["id_labels"][id_index],
        "ood_features": cache["ood_features"][ood_index],
        "ood_logits": cache["ood_logits"][ood_index],
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks with deterministic tie handling, without SciPy."""

    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left.dot(right) / denominator) if denominator > 0.0 else float("nan")


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _same_anchor_centroids(anchors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = anchors.mean(dim=1)
    return mean, F.normalize(mean, dim=-1)


def _centroid_bound(
    anchors: torch.Tensor, shared_sigma: float
) -> tuple[torch.Tensor, dict[str, float]]:
    """Lemma A.2 bound for uniform weights, W=1, and a Gaussian kernel."""

    mean, _ = _same_anchor_centroids(anchors)
    dispersion = torch.linalg.vector_norm(anchors - mean[:, None, :], dim=-1).mean(1)
    norm_correction = (1.0 - torch.linalg.vector_norm(mean, dim=-1)).abs()
    epsilon = (GAUSSIAN_LIPSCHITZ / shared_sigma) * (
        dispersion + norm_correction
    )
    summary = {
        "GaussianLipschitz": GAUSSIAN_LIPSCHITZ,
        "SharedSigma": float(shared_sigma),
        "EpsilonMin": float(epsilon.min()),
        "EpsilonMean": float(epsilon.mean()),
        "EpsilonMedian": float(epsilon.median()),
        "EpsilonMax": float(epsilon.max()),
    }
    return epsilon, summary


class _ErrorAudit:
    def __init__(self) -> None:
        self.count = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.max_abs = 0.0
        self.max_bound_excess = -math.inf
        self.violations = 0
        self.score_max_abs = 0.0

    def update(
        self,
        class_difference: torch.Tensor,
        bound: torch.Tensor,
        score_difference: torch.Tensor,
        tolerance: float = 1e-5,
    ) -> None:
        difference = class_difference.detach().double().cpu()
        bound = bound.detach().double().cpu()[None, :]
        self.count += difference.numel()
        self.sum_abs += float(difference.sum())
        self.sum_sq += float(difference.square().sum())
        self.max_abs = max(self.max_abs, float(difference.max()))
        excess = difference - bound
        self.max_bound_excess = max(self.max_bound_excess, float(excess.max()))
        self.violations += int((excess > tolerance).sum())
        self.score_max_abs = max(
            self.score_max_abs, float(score_difference.detach().max().cpu())
        )

    def row(self, population: str, seed: int, bound_summary: dict) -> dict:
        return {
            "Seed": seed,
            "Population": population,
            **bound_summary,
            "ComparedClassPairs": self.count,
            "EmpiricalClassMAE": self.sum_abs / self.count,
            "EmpiricalClassRMSE": math.sqrt(self.sum_sq / self.count),
            "EmpiricalClassMaxAbs": self.max_abs,
            "EmpiricalScoreMaxAbs": self.score_max_abs,
            "MaxBoundExcess": self.max_bound_excess,
            "BoundViolationCount": self.violations,
            "BoundSatisfied": self.violations == 0,
        }


@torch.no_grad()
def _score_population(
    detector,
    features: torch.Tensor,
    logits: torch.Tensor,
    batch_size: int,
    epsilon: torch.Tensor,
    bound_summary: dict,
    population: str,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    device = detector.anchors.device
    anchors = F.normalize(detector.anchors, dim=-1)
    _, anchor_centroids = _same_anchor_centroids(anchors)
    setup_centroids = F.normalize(detector.centroids, dim=-1)
    shared_sigma = float(bound_summary["SharedSigma"])

    score_parts: dict[str, list[torch.Tensor]] = {name: [] for name in STAGES[:-1]}
    class_parts: dict[str, list[torch.Tensor]] = {
        name: [] for name in STAGES[:-1]
    }
    msp_parts: list[torch.Tensor] = []
    audit = _ErrorAudit()

    for start in tqdm(
        range(0, len(features), batch_size),
        desc=f"Bridge scores seed={seed} {population}",
    ):
        query = F.normalize(
            features[start : start + batch_size].to(device, non_blocking=True),
            dim=-1,
        )
        batch_logits = logits[start : start + batch_size].to(
            device, non_blocking=True
        )

        # S1: exact all-class static field, including learned per-anchor
        # bandwidths. Passing candidates=None deliberately selects the flat,
        # memory-efficient all-class implementation; semantically this is
        # candidate_k=0, not the deployed detector's candidate_k=20 score.
        # HamiltonianDetector stores the implementation-equivalent sum because
        # the omitted positive 1/K factor does not affect OOD rankings.  The
        # bridge, however, compares fields numerically, so restore Definition
        # (2)'s class-wise average here.
        exact_class = detector.affinity_per_class(query, candidates=None) / detector.K

        # S2: same anchors, but uniform pi=1/K and one shared ID-learned scale.
        similarity = torch.einsum("bd,ckd->bck", query, anchors)
        anchor_dist2 = (2.0 - 2.0 * similarity).clamp_min(0.0)
        common_class = torch.exp(
            -0.5 * anchor_dist2 / (shared_sigma * shared_sigma)
        ).mean(dim=-1)

        # S3 and S4 share the same-anchor centroid.  S3 is a strictly
        # increasing transformation of S4 for the common Gaussian scale.
        anchor_cosine = query @ anchor_centroids.T
        centroid_dist2 = (2.0 - 2.0 * anchor_cosine).clamp_min(0.0)
        radial_class = torch.exp(
            -0.5 * centroid_dist2 / (shared_sigma * shared_sigma)
        )
        setup_cosine = query @ setup_centroids.T

        stage_classes = {
            "S1_exact_full_class_static": exact_class.argmax(1),
            "S2_common_scale": common_class.argmax(1),
            "S3_radial_centroid": radial_class.argmax(1),
            "S4_anchor_centroid_cosine": anchor_cosine.argmax(1),
            "S5_setup_centroid_cosine": setup_cosine.argmax(1),
        }
        stage_scores = {
            "S1_exact_full_class_static": exact_class.max(1).values,
            "S2_common_scale": common_class.max(1).values,
            "S3_radial_centroid": radial_class.max(1).values,
            "S4_anchor_centroid_cosine": anchor_cosine.max(1).values,
            "S5_setup_centroid_cosine": setup_cosine.max(1).values,
        }
        for name, value in stage_scores.items():
            score_parts[name].append(value.cpu())
            class_parts[name].append(stage_classes[name].cpu())
        msp_parts.append(batch_logits.softmax(1).max(1).values.cpu())

        class_difference = (common_class - radial_class).abs()
        score_difference = (
            stage_scores["S2_common_scale"] - stage_scores["S3_radial_centroid"]
        ).abs()
        audit.update(class_difference, epsilon, score_difference)

    scores = {name: torch.cat(parts).numpy() for name, parts in score_parts.items()}
    classes = {name: torch.cat(parts).numpy() for name, parts in class_parts.items()}
    scores["MSP_reference"] = torch.cat(msp_parts).numpy()
    return scores, classes, audit.row(population, seed, bound_summary)


def _append_locked_fusion(
    id_scores: dict[str, np.ndarray],
    ood_scores: dict[str, np.ndarray],
    id_tune: np.ndarray,
) -> dict[str, float]:
    id_geometry = id_scores["S5_setup_centroid_cosine"]
    ood_geometry = ood_scores["S5_setup_centroid_cosine"]
    id_msp = id_scores["MSP_reference"]
    ood_msp = ood_scores["MSP_reference"]
    id_scores["S6_locked_centroid_msp"] = (
        LOCKED_ALPHA * _zscore(id_geometry, id_geometry[id_tune])
        + (1.0 - LOCKED_ALPHA) * _zscore(id_msp, id_msp[id_tune])
    )
    ood_scores["S6_locked_centroid_msp"] = (
        LOCKED_ALPHA * _zscore(ood_geometry, id_geometry[id_tune])
        + (1.0 - LOCKED_ALPHA) * _zscore(ood_msp, id_msp[id_tune])
    )
    return {
        "Alpha": LOCKED_ALPHA,
        "CalibrationCount": int(id_tune.sum()),
        "MSPMean": float(id_msp[id_tune].mean()),
        "MSPStd": float(id_msp[id_tune].std()),
        "GeometryMean": float(id_geometry[id_tune].mean()),
        "GeometryStd": float(id_geometry[id_tune].std()),
    }


def _metric_rows(
    seed: int,
    id_scores: dict[str, np.ndarray],
    ood_scores: dict[str, np.ndarray],
    predictions: np.ndarray,
    labels: np.ndarray,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[dict]:
    rows = []
    for order, name in enumerate(("MSP_reference", *STAGES)):
        for split, (id_mask, ood_mask) in splits.items():
            row = _metric(
                name,
                split,
                id_scores[name][id_mask],
                ood_scores[name][ood_mask],
                predictions[id_mask],
                labels[id_mask],
            )
            row.update({"Seed": seed, "StageOrder": order})
            rows.append(row)
    return rows


def _transition_rows(
    seed: int,
    id_scores: dict[str, np.ndarray],
    ood_scores: dict[str, np.ndarray],
    id_classes: dict[str, np.ndarray],
    ood_classes: dict[str, np.ndarray],
    id_tune: np.ndarray,
) -> list[dict]:
    rows = []
    populations = {
        "ID": lambda name: id_scores[name],
        "OOD": lambda name: ood_scores[name],
        "Combined": lambda name: np.concatenate((id_scores[name], ood_scores[name])),
    }
    class_populations = {
        "ID": lambda name: id_classes[name],
        "OOD": lambda name: ood_classes[name],
        "Combined": lambda name: np.concatenate((id_classes[name], ood_classes[name])),
    }
    for source, target in TRANSITIONS:
        source_id_z = _zscore(id_scores[source], id_scores[source][id_tune])
        target_id_z = _zscore(id_scores[target], id_scores[target][id_tune])
        source_ood_z = _zscore(ood_scores[source], id_scores[source][id_tune])
        target_ood_z = _zscore(ood_scores[target], id_scores[target][id_tune])
        standardised = {
            "ID": (source_id_z, target_id_z),
            "OOD": (source_ood_z, target_ood_z),
            "Combined": (
                np.concatenate((source_id_z, source_ood_z)),
                np.concatenate((target_id_z, target_ood_z)),
            ),
        }
        for population, getter in populations.items():
            left, right = getter(source), getter(target)
            difference = right - left
            left_z, right_z = standardised[population]
            z_difference = right_z - left_z
            class_agreement = float("nan")
            if source in id_classes and target in id_classes:
                class_getter = class_populations[population]
                class_agreement = float(
                    np.mean(class_getter(source) == class_getter(target))
                )
            rows.append(
                {
                    "Seed": seed,
                    "SourceStage": source,
                    "TargetStage": target,
                    "Population": population,
                    "Spearman": _spearman(left, right),
                    "Pearson": _pearson(left, right),
                    "RawMAE": float(np.mean(np.abs(difference))),
                    "RawRMSE": float(np.sqrt(np.mean(difference**2))),
                    "RawMaxAbs": float(np.max(np.abs(difference))),
                    "StandardisedMAE": float(np.mean(np.abs(z_difference))),
                    "StandardisedRMSE": float(np.sqrt(np.mean(z_difference**2))),
                    "ClassAgreement": class_agreement,
                }
            )
    return rows


def _metric_delta_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons: Iterable[tuple[str, str]] = (*TRANSITIONS, ("MSP_reference", STAGES[-1]))
    for seed in sorted(metrics["Seed"].unique()):
        seed_frame = metrics[metrics["Seed"].eq(seed)]
        for split in ("tune", "holdout", "full"):
            frame = seed_frame[seed_frame["Split"].eq(split)].set_index("Method")
            for source, target in comparisons:
                rows.append(
                    {
                        "Seed": seed,
                        "Split": split,
                        "SourceStage": source,
                        "TargetStage": target,
                        "DeltaAUROC": float(frame.loc[target, "AUROC"] - frame.loc[source, "AUROC"]),
                        "DeltaFPR95": float(frame.loc[target, "FPR95"] - frame.loc[source, "FPR95"]),
                        "DeltaAUPR_IN": float(frame.loc[target, "AUPR_IN"] - frame.loc[source, "AUPR_IN"]),
                        "DeltaAUPR_OUT": float(frame.loc[target, "AUPR_OUT"] - frame.loc[source, "AUPR_OUT"]),
                    }
                )
    return pd.DataFrame(rows)


def _mean_std(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in {"Seed", "StageOrder"}
    ]
    return frame.groupby(group_columns, sort=False)[numeric].agg(["mean", "std"]).reset_index()


def main() -> None:
    args = build_parser().parse_args()
    args.openood_root = args.openood_root.resolve()
    args.hamiltonian_root = args.hamiltonian_root.resolve()
    args.output_root = args.output_root.resolve()
    if sorted(set(args.seeds)) != [0, 1, 2]:
        raise ValueError("The bridge protocol requires exactly detector seeds 0, 1, and 2")
    args.seeds = [0, 1, 2]
    if args.calibration_seed != 0:
        raise ValueError("The locked journal protocol fixes --calibration-seed at 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    # This is the only data input.  The strict count/shape audit below rejects
    # caches made from OpenOOD test splits.
    cache_path = _resolve_validation_cache(args.hamiltonian_root, args.validation_cache)
    try:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        cache = torch.load(cache_path, map_location="cpu")
    _validate_full_cache(cache)
    full_cache_hash = _sha256(cache_path)
    if args.stage == "smoke":
        cache = _smoke_subset(
            cache,
            args.smoke_id_per_class,
            args.max_eval_samples,
            args.calibration_seed,
        )

    labels = cache["id_labels"].long().numpy()
    predictions = cache["id_logits"].argmax(1).numpy()
    id_tune, id_holdout, ood_tune, ood_holdout = _validation_masks(
        labels, len(cache["ood_features"]), args.calibration_seed
    )
    splits = {
        "tune": (id_tune, ood_tune),
        "holdout": (id_holdout, ood_holdout),
        "full": (np.ones(len(labels), bool), np.ones(len(cache["ood_features"]), bool)),
    }

    # _metric imports OpenOOD's official metric implementation lazily.
    from Imagenet_ood_experiment import add_openood_to_path

    add_openood_to_path(args.openood_root)
    output_dir = args.output_root / args.stage
    completion_path = args.output_root / f"static_reduction_bridge_{args.stage}_completed.json"
    if completion_path.is_file() and not args.force:
        raise RuntimeError(
            f"The {args.stage} bridge is already complete: {completion_path}. "
            "Use --force only after intentionally auditing the existing output."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metric_rows: list[dict] = []
    transition_rows: list[dict] = []
    bound_rows: list[dict] = []
    calibration_rows: list[dict] = []
    epsilon_rows: list[dict] = []
    detector_records: list[dict] = []
    for seed in args.seeds:
        detector, detector_path = _load_detector(args.hamiltonian_root, seed, device)
        if detector.potential != "gaussian":
            raise RuntimeError(f"Seed-{seed} detector is not Gaussian")
        if not torch.allclose(detector.masses, torch.ones_like(detector.masses)):
            raise RuntimeError(f"Seed-{seed} detector is not the uniform-mass control")
        if detector.candidate_k != DEPLOYED_DETECTOR_CANDIDATE_K:
            raise RuntimeError(
                f"Seed-{seed} detector must retain the deployed candidate_k="
                f"{DEPLOYED_DETECTOR_CANDIDATE_K}; found {detector.candidate_k}"
            )
        shared_sigma = float(detector.sigma.detach().float().quantile(0.5))
        epsilon, bound_summary = _centroid_bound(
            F.normalize(detector.anchors.detach(), dim=-1), shared_sigma
        )
        for class_index, value in enumerate(epsilon.detach().cpu().tolist()):
            epsilon_rows.append(
                {"Seed": seed, "Class": class_index, "Epsilon": value, **bound_summary}
            )
        print(
            f"[seed {seed}] shared sigma={shared_sigma:.8f}; detector={detector_path}",
            flush=True,
        )
        id_scores, id_classes, id_bound = _score_population(
            detector,
            cache["id_features"],
            cache["id_logits"],
            args.batch_size,
            epsilon,
            bound_summary,
            "ID",
            seed,
        )
        ood_scores, ood_classes, ood_bound = _score_population(
            detector,
            cache["ood_features"],
            cache["ood_logits"],
            args.batch_size,
            epsilon,
            bound_summary,
            "OOD",
            seed,
        )
        bound_rows.extend((id_bound, ood_bound))
        calibration = _append_locked_fusion(id_scores, ood_scores, id_tune)
        calibration.update({"Seed": seed, "SharedSigma": shared_sigma})
        calibration_rows.append(calibration)
        metric_rows.extend(
            _metric_rows(
                seed,
                id_scores,
                ood_scores,
                predictions,
                labels,
                splits,
            )
        )
        transition_rows.extend(
            _transition_rows(
                seed,
                id_scores,
                ood_scores,
                id_classes,
                ood_classes,
                id_tune,
            )
        )
        detector_records.append(
            {
                "seed": seed,
                "path": str(detector_path.resolve()),
                "sha256": _sha256(detector_path),
                "stored_candidate_k": int(detector.candidate_k),
                "bridge_exact_field_candidate_k": BRIDGE_EXACT_CANDIDATE_K,
                "shared_sigma_rule": "median of all ID-trained per-anchor sigmas",
                "shared_sigma": shared_sigma,
            }
        )
        del detector
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = pd.DataFrame(metric_rows)
    transitions = pd.DataFrame(transition_rows)
    bounds = pd.DataFrame(bound_rows)
    calibrations = pd.DataFrame(calibration_rows)
    epsilons = pd.DataFrame(epsilon_rows)
    deltas = _metric_delta_rows(metrics)
    metrics.to_csv(output_dir / "stage_metrics.csv", index=False, float_format="%.9f")
    transitions.to_csv(
        output_dir / "transition_diagnostics.csv", index=False, float_format="%.9f"
    )
    bounds.to_csv(output_dir / "radial_centroid_bound_audit.csv", index=False, float_format="%.9f")
    calibrations.to_csv(
        output_dir / "locked_calibration_statistics.csv", index=False, float_format="%.9f"
    )
    epsilons.to_csv(output_dir / "radial_centroid_class_bounds.csv", index=False, float_format="%.9f")
    deltas.to_csv(output_dir / "transition_metric_deltas.csv", index=False, float_format="%.9f")
    _mean_std(metrics, ["Method", "Split", "StageOrder"]).to_csv(
        output_dir / "stage_metrics_mean_std.csv", index=False, float_format="%.9f"
    )
    _mean_std(
        transitions, ["SourceStage", "TargetStage", "Population"]
    ).to_csv(
        output_dir / "transition_diagnostics_mean_std.csv",
        index=False,
        float_format="%.9f",
    )

    # The radial/cosine equivalence is a mathematical invariant of stages 3/4.
    radial_cosine = transitions[
        transitions["SourceStage"].eq("S3_radial_centroid")
        & transitions["TargetStage"].eq("S4_anchor_centroid_cosine")
    ]
    rank_equivalence_passed = bool(
        np.all(radial_cosine["Spearman"].to_numpy() >= 1.0 - 1e-7)
    )
    class_equivalence_passed = bool(
        np.all(radial_cosine["ClassAgreement"].to_numpy() >= 1.0 - 1e-12)
    )
    if not rank_equivalence_passed:
        raise RuntimeError("Stage 3 and Stage 4 did not preserve sample rankings")
    if not class_equivalence_passed:
        raise RuntimeError("Stage 3 and Stage 4 did not select identical classes")
    if not bounds["BoundSatisfied"].all():
        raise RuntimeError("The empirical radial-centroid error exceeded Lemma A.2")

    completion = {
        "format_version": 1,
        "protocol": "ImageNet-1K validation-only six-stage static-reduction bridge",
        "stage": args.stage,
        "near_far_test_access": False,
        "validation_inputs": {
            "id": "official ImageNet-1K validation subset",
            "ood": "OpenImage-O validation",
            "id_count": len(cache["id_features"]),
            "ood_count": len(cache["ood_features"]),
            "source_cache": str(cache_path),
            "source_cache_sha256": full_cache_hash,
            "cache_tag": CACHE_TAG,
        },
        "seeds": args.seeds,
        "stages": list(STAGES),
        "static_field_control": {
            "S1": "exact all-class zero-step Gaussian field",
            "stored_detector_candidate_k": DEPLOYED_DETECTOR_CANDIDATE_K,
            "bridge_exact_field_candidate_k": BRIDGE_EXACT_CANDIDATE_K,
            "relation_to_deployed_score": (
                "theoretical all-class control; not the deployed "
                "candidate-pruned score"
            ),
        },
        "locked_alpha": LOCKED_ALPHA,
        "calibration_seed": args.calibration_seed,
        "shared_sigma_rule": "per-seed median of all ID-trained Gaussian anchor sigmas",
        "detectors": detector_records,
        "performance_monotonicity_assumed": False,
        "radial_cosine_rank_equivalence_passed": rank_equivalence_passed,
        "radial_cosine_class_equivalence_passed": class_equivalence_passed,
        "radial_centroid_bound_passed": True,
        "outputs": {
            "metrics": str((output_dir / "stage_metrics.csv").resolve()),
            "transitions": str((output_dir / "transition_diagnostics.csv").resolve()),
            "bounds": str((output_dir / "radial_centroid_bound_audit.csv").resolve()),
            "metric_deltas": str((output_dir / "transition_metric_deltas.csv").resolve()),
        },
        "completed": True,
    }
    temporary = completion_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(completion, indent=2), encoding="utf-8")
    temporary.replace(completion_path)

    print("\n===== Six-stage validation bridge (full split) =====")
    print(
        metrics[metrics["Split"].eq("full")]
        .loc[:, ["Seed", "StageOrder", "Method", "AUROC", "FPR95", "AUPR_IN", "AUPR_OUT"]]
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\n===== Radial-centroid bound audit =====")
    print(bounds.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"\nSaved under: {output_dir}")
    print("No Near-OOD or Far-OOD test loader or score file was opened.")


if __name__ == "__main__":
    main()
