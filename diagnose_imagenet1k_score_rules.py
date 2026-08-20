"""Validation-only score-rule diagnosis for ImageNet-1K scaling.

This program never opens the near-OOD or far-OOD test loaders.  It compares
class-count-normalised potential scores and conservative MSP/potential fusion
using only the official 5,000-image ImageNet validation split and the 1,763
OpenImage-O validation images.  One half selects a rule; the other half is a
held-out validation check.  A test run is allowed only after this script has
written and accepted a locked rule.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


TEMPERATURES = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
FUSION_ALPHAS = tuple(round(value, 2) for value in np.linspace(0.0, 1.0, 21))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openood-root", type=Path, default=Path("OpenOOD"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--openood-results-root", type=Path, default=Path("openood_pretrained")
    )
    parser.add_argument("--output-root", type=Path, default=Path("results_openood"))
    parser.add_argument("--seed", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--candidate-k", type=int, default=20,
        help="Candidate-class approximation evaluated alongside exact scoring",
    )
    return parser


def _derived_affinity_scores(
    affinity: torch.Tensor,
    *,
    prefix: str,
    predicted_class: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return high-means-ID scores without using any OOD labels."""

    if affinity.ndim != 2 or affinity.shape[1] < 2:
        raise ValueError("affinity must have shape [batch, at least two classes]")
    eps = torch.finfo(affinity.dtype).eps
    top2 = affinity.topk(2, dim=1).values
    total = affinity.sum(1).clamp_min(eps)
    probability = affinity.clamp_min(0) / total[:, None]
    scores = {
        f"{prefix}/raw_max": top2[:, 0],
        f"{prefix}/raw_margin": top2[:, 0] - top2[:, 1],
        f"{prefix}/log_ratio": (top2[:, 0] + eps).log()
        - (top2[:, 1] + eps).log(),
        f"{prefix}/peak_share": top2[:, 0] / total,
        f"{prefix}/negative_entropy": (
            probability * (probability + eps).log()
        ).sum(1),
    }
    for temperature in TEMPERATURES:
        scores[f"{prefix}/softmax_max_t{temperature:g}"] = F.softmax(
            affinity / temperature, dim=1
        ).max(1).values
        powered = (affinity.clamp_min(eps).log() / temperature).softmax(1)
        scores[f"{prefix}/power_max_t{temperature:g}"] = powered.max(1).values
    if predicted_class is not None:
        if predicted_class.ndim != 1 or len(predicted_class) != len(affinity):
            raise ValueError("predicted_class must have shape [batch]")
        predicted = affinity.gather(1, predicted_class[:, None]).squeeze(1)
        scores[f"{prefix}/backbone_class_affinity"] = predicted
        scores[f"{prefix}/backbone_class_share"] = predicted / total
    return scores


def _validation_masks(
    labels: np.ndarray, ood_count: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Class-stratified ID and deterministic OOD tune/holdout masks."""

    labels = np.asarray(labels)
    id_tune = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        generator = np.random.default_rng(seed + 10_000 + int(label))
        rows = generator.permutation(rows)
        id_tune[rows[: max(1, len(rows) // 2)]] = True
    ood_rows = np.random.default_rng(seed + 20_000).permutation(ood_count)
    ood_tune = np.zeros(ood_count, dtype=bool)
    ood_tune[ood_rows[: ood_count // 2]] = True
    return id_tune, ~id_tune, ood_tune, ~ood_tune


def _zscore(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64)
    scale = float(reference.std())
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return (np.asarray(values, dtype=np.float64) - float(reference.mean())) / scale


def _metric(
    method: str,
    split: str,
    id_score: np.ndarray,
    ood_score: np.ndarray,
    id_prediction: np.ndarray,
    id_labels: np.ndarray,
) -> dict[str, float | int | str]:
    from openood.evaluators.metrics import compute_all_metrics

    if not np.isfinite(id_score).all() or not np.isfinite(ood_score).all():
        raise ValueError(f"Non-finite confidence in {method}/{split}")
    result = np.asarray(
        compute_all_metrics(
            np.concatenate((id_score, ood_score)),
            np.concatenate((id_labels, -np.ones(len(ood_score), dtype=int))),
            np.concatenate((id_prediction, np.zeros(len(ood_score), dtype=int))),
        )
    ) * 100.0
    return {
        "Method": method,
        "Split": split,
        "FPR95": float(result[0]),
        "AUROC": float(result[1]),
        "AUPR_IN": float(result[2]),
        "AUPR_OUT": float(result[3]),
        "IDAccuracy": float(result[4]),
        "IDCount": len(id_score),
        "OODCount": len(ood_score),
    }


@torch.no_grad()
def _extract_split(network, loader, forward_with_feature, description: str):
    features, logits, labels = [], [], []
    for batch in tqdm(loader, desc=description):
        data = batch["data"].cuda(non_blocking=True)
        output, feature = forward_with_feature(network, data)
        features.append(feature.cpu())
        logits.append(output.cpu())
        labels.append(batch["label"].long().cpu())
    return torch.cat(features), torch.cat(logits), torch.cat(labels)


@torch.no_grad()
def _score_features(
    detector,
    features: torch.Tensor,
    logits: torch.Tensor,
    batch_size: int,
    candidate_k: int,
) -> dict[str, np.ndarray]:
    parts: dict[str, list[torch.Tensor]] = {}
    for start in tqdm(
        range(0, len(features), batch_size), desc="Static score rules"
    ):
        query = features[start : start + batch_size].cuda(non_blocking=True)
        batch_logits = logits[start : start + batch_size].cuda(non_blocking=True)
        prediction = batch_logits.argmax(1)

        exact = detector.affinity_per_class(query, candidates=None)
        batch_scores = _derived_affinity_scores(
            exact, prefix="potential_exact", predicted_class=prediction
        )

        candidates = detector.select_candidates(query, candidate_k=candidate_k)
        candidate_affinity = detector.affinity_per_class(query, candidates=candidates)
        batch_scores.update(
            _derived_affinity_scores(
                candidate_affinity, prefix=f"potential_candidate{candidate_k}"
            )
        )

        centroid_similarity = F.normalize(query, dim=-1) @ detector.centroids.T
        centroid_top2 = centroid_similarity.topk(2, dim=1).values
        batch_scores["centroid/max_cosine"] = centroid_top2[:, 0]
        batch_scores["centroid/cosine_margin"] = (
            centroid_top2[:, 0] - centroid_top2[:, 1]
        )
        batch_scores["centroid/backbone_class_cosine"] = centroid_similarity.gather(
            1, prediction[:, None]
        ).squeeze(1)

        probability = batch_logits.softmax(1)
        logit_top2 = batch_logits.topk(2, dim=1).values
        batch_scores["logit/MSP"] = probability.max(1).values
        batch_scores["logit/MLS"] = logit_top2[:, 0]
        batch_scores["logit/margin"] = logit_top2[:, 0] - logit_top2[:, 1]
        batch_scores["logit/energy"] = torch.logsumexp(batch_logits, dim=1)

        for name, values in batch_scores.items():
            parts.setdefault(name, []).append(values.detach().cpu())
    return {name: torch.cat(values).numpy() for name, values in parts.items()}


def main() -> None:
    args = build_parser().parse_args()
    for name in ("openood_root", "data_root", "openood_results_root", "output_root"):
        setattr(args, name, getattr(args, name).resolve())
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnosis requires a CUDA GPU")

    from Imagenet_ood_experiment import add_openood_to_path, build_network

    add_openood_to_path(args.openood_root)
    evaluator_module = importlib.import_module("openood.evaluation_api.evaluator")
    evaluator_module.data_setup = lambda *_args, **_kwargs: None
    Evaluator = evaluator_module.Evaluator
    postprocessor_module = importlib.import_module("openood_hamiltonian_postprocessor")
    HamiltonianPostprocessor = postprocessor_module.HamiltonianPostprocessor
    forward_with_feature = postprocessor_module._forward_with_feature

    network, preprocessor, cache_tag, model_source = build_network(
        "imagenet", None, args.openood_results_root, args.seed, 1
    )
    network = network.cuda().eval()
    formal_dir = (
        args.output_root
        / "imagenet1k"
        / cache_tag
        / f"seed{args.seed}"
        / "mass-uniform-none_loss-static-ts10_pred-backbone_eval-full"
    )
    if not formal_dir.is_dir():
        raise FileNotFoundError(f"Formal ImageNet-1K directory missing: {formal_dir}")

    postprocessor = HamiltonianPostprocessor(
        n_classes=1000,
        potential="gaussian",
        output_dir=formal_dir,
        cache_tag=cache_tag,
        n_anchors=5,
        setup_samples_per_class=12,
        ham_epochs=10,
        ham_lr=1e-3,
        ham_batch_size=16,
        bandwidth_loss="static",
        n_steps=10,
        dt=0.05,
        candidate_k=20,
        sim_batch=32,
        setup_batch_size=128,
        num_workers=args.num_workers,
        sigma_init=0.5,
        sigma_min=0.05,
        sigma_max=4.0,
        mass_mode="uniform",
        mass_normalization="none",
        prediction_source="backbone",
        seed=args.seed,
    )
    evaluator = Evaluator(
        network,
        id_name="imagenet",
        data_root=str(args.data_root),
        config_root=str(args.openood_root / "configs"),
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    detector = postprocessor.detector
    if detector is None:
        raise RuntimeError("Gaussian detector was not loaded")
    detector.T = 0

    output_dir = formal_dir / "validation_score_rule_diagnosis_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    old_cache = formal_dir / "validation_scaling_diagnosis" / "validation_features.pt"
    cache_path = output_dir / "validation_features.pt"
    source_cache = old_cache if old_cache.is_file() else cache_path
    if source_cache.is_file():
        cache = torch.load(source_cache, map_location="cpu")
        id_features, id_logits, id_labels = (
            cache["id_features"], cache["id_logits"], cache["id_labels"]
        )
        ood_features, ood_logits = cache["ood_features"], cache["ood_logits"]
        print(f"Loaded validation feature cache: {source_cache}", flush=True)
    else:
        id_features, id_logits, id_labels = _extract_split(
            network, evaluator.dataloader_dict["id"]["val"],
            forward_with_feature, "ID validation features"
        )
        ood_features, ood_logits, _ = _extract_split(
            network, evaluator.dataloader_dict["ood"]["val"],
            forward_with_feature, "OOD validation features"
        )
        temporary = cache_path.with_suffix(".tmp")
        torch.save(
            {
                "model_source": model_source,
                "id_features": id_features,
                "id_logits": id_logits,
                "id_labels": id_labels,
                "ood_features": ood_features,
                "ood_logits": ood_logits,
            },
            temporary,
        )
        temporary.replace(cache_path)

    print("Scoring ID validation split...", flush=True)
    id_scores = _score_features(
        detector, id_features, id_logits, args.batch_size, args.candidate_k
    )
    print("Scoring OpenImage-O validation split...", flush=True)
    ood_scores = _score_features(
        detector, ood_features, ood_logits, args.batch_size, args.candidate_k
    )
    if set(id_scores) != set(ood_scores):
        raise RuntimeError("ID/OOD score rules do not match")

    labels = id_labels.numpy()
    predictions = id_logits.argmax(1).numpy()
    id_tune, id_holdout, ood_tune, ood_holdout = _validation_masks(
        labels, len(ood_features), args.seed
    )
    splits = {
        "tune": (id_tune, ood_tune),
        "holdout": (id_holdout, ood_holdout),
        "full": (np.ones(len(labels), bool), np.ones(len(ood_features), bool)),
    }

    base_rows = []
    for method in sorted(id_scores):
        for split, (id_mask, ood_mask) in splits.items():
            base_rows.append(
                _metric(
                    method, split,
                    id_scores[method][id_mask], ood_scores[method][ood_mask],
                    predictions[id_mask], labels[id_mask],
                )
            )
    base = pd.DataFrame(base_rows)
    base.to_csv(output_dir / "base_score_rule_metrics.csv", index=False,
                float_format="%.6f")

    msp_name = "logit/MSP"
    geometry_methods = [name for name in id_scores if not name.startswith("logit/")]
    id_msp_z = _zscore(id_scores[msp_name], id_scores[msp_name][id_tune])
    ood_msp_z = _zscore(ood_scores[msp_name], id_scores[msp_name][id_tune])
    fusion_rows = []
    for method in geometry_methods:
        id_geometry_z = _zscore(id_scores[method], id_scores[method][id_tune])
        ood_geometry_z = _zscore(ood_scores[method], id_scores[method][id_tune])
        for alpha in FUSION_ALPHAS:
            id_fused = (1.0 - alpha) * id_msp_z + alpha * id_geometry_z
            ood_fused = (1.0 - alpha) * ood_msp_z + alpha * ood_geometry_z
            for split, (id_mask, ood_mask) in splits.items():
                row = _metric(
                    f"fusion/MSP+{method}", split,
                    id_fused[id_mask], ood_fused[ood_mask],
                    predictions[id_mask], labels[id_mask],
                )
                row["GeometryMethod"] = method
                row["GeometryWeight"] = alpha
                fusion_rows.append(row)
    fusion = pd.DataFrame(fusion_rows)
    fusion.to_csv(output_dir / "fusion_sweep.csv", index=False,
                  float_format="%.6f")

    tune = fusion[fusion["Split"].eq("tune")].sort_values(
        ["AUROC", "FPR95", "GeometryWeight"],
        ascending=[False, True, True],
    )
    selected_tune = tune.iloc[0]
    selected = fusion[
        fusion["GeometryMethod"].eq(selected_tune["GeometryMethod"])
        & fusion["GeometryWeight"].eq(selected_tune["GeometryWeight"])
    ].copy()
    selected.to_csv(output_dir / "selected_rule_validation.csv", index=False,
                    float_format="%.6f")

    holdout_selected = selected[selected["Split"].eq("holdout")].iloc[0]
    holdout_msp = base[
        base["Method"].eq(msp_name) & base["Split"].eq("holdout")
    ].iloc[0]
    delta_auroc = float(holdout_selected["AUROC"] - holdout_msp["AUROC"])
    delta_fpr = float(holdout_selected["FPR95"] - holdout_msp["FPR95"])
    recommended = bool(
        float(selected_tune["GeometryWeight"]) > 0.0
        and (
            (delta_auroc >= 0.5 and delta_fpr <= 0.0)
            or (delta_auroc >= 1.0 and delta_fpr <= 2.0)
        )
    )
    decision = {
        "protocol": "validation-only tune/holdout; no near/far test access",
        "seed": args.seed,
        "selected_geometry_method": selected_tune["GeometryMethod"],
        "geometry_weight": float(selected_tune["GeometryWeight"]),
        "tune_AUROC": float(selected_tune["AUROC"]),
        "tune_FPR95": float(selected_tune["FPR95"]),
        "holdout_delta_AUROC_vs_MSP": delta_auroc,
        "holdout_delta_FPR95_vs_MSP": delta_fpr,
        "recommended_for_one_locked_test_run": recommended,
    }
    (output_dir / "locked_rule_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    print("\n===== Validation base score rules (full split, top 15 AUROC) =====")
    print(
        base[base["Split"].eq("full")]
        .sort_values("AUROC", ascending=False)
        .head(15)
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\n===== Locked fusion rule =====")
    print(selected.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(json.dumps(decision, indent=2))
    print(f"\nSaved validation-only diagnosis under: {output_dir}")
    print("No near/far OOD test loaders were read.")


if __name__ == "__main__":
    main()
