"""Validation-only diagnosis for Hamiltonian scaling on ImageNet-1K.

The sweep never reads the OpenOOD test OOD loaders.  It extracts features from
the 5,000-image ID validation split and the 1,763-image OpenImage-O validation
split, then isolates trajectory length and candidate-class approximation while
holding the already-trained Gaussian detector fixed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from Imagenet_ood_experiment import (
    ROOT,
    add_openood_to_path,
    build_network,
)

add_openood_to_path(ROOT / "OpenOOD")
from openood_hamiltonian_postprocessor import (
    HamiltonianPostprocessor,
    _forward_with_feature,
)


VALIDATION_SWEEP = (
    # Candidate sensitivity without trajectory integration.
    (0, 5),
    (0, 20),
    (0, 50),
    (0, 100),
    # Trajectory-length sensitivity at the formal candidate setting.
    (1, 20),
    (3, 20),
    (5, 20),
    (10, 20),
    # One wider-neighbourhood trajectory check.
    (10, 50),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openood-root", type=Path, default=ROOT / "OpenOOD")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--openood-results-root", type=Path, default=ROOT / "openood_pretrained"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "results_openood")
    parser.add_argument("--seed", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser


@torch.no_grad()
def extract_split(network, loader, description: str):
    features, logits, labels = [], [], []
    for batch in tqdm(loader, desc=description):
        data = batch["data"].cuda(non_blocking=True)
        output, feature = _forward_with_feature(network, data)
        features.append(feature.cpu())
        logits.append(output.cpu())
        labels.append(batch["label"].long().cpu())
    return torch.cat(features), torch.cat(logits), torch.cat(labels)


@torch.no_grad()
def detector_scores(detector, features: torch.Tensor, batch_size: int) -> np.ndarray:
    parts = []
    for start in tqdm(range(0, len(features), batch_size), desc="Hamiltonian scores"):
        batch = features[start : start + batch_size].cuda(non_blocking=True)
        parts.append(detector.score(batch).cpu())
    return torch.cat(parts).numpy()


def metric_row(
    name: str,
    id_conf: np.ndarray,
    ood_conf: np.ndarray,
    id_pred: np.ndarray,
    id_labels: np.ndarray,
    *,
    steps: int | None,
    candidate_k: int | None,
) -> dict[str, float | int | str | None]:
    from openood.evaluators.metrics import compute_all_metrics

    ood_labels = -np.ones(len(ood_conf), dtype=int)
    # OOD predictions do not affect accuracy because their labels are -1.
    ood_pred = np.zeros(len(ood_conf), dtype=int)
    metrics = np.asarray(
        compute_all_metrics(
            np.concatenate((id_conf, ood_conf)),
            np.concatenate((id_labels, ood_labels)),
            np.concatenate((id_pred, ood_pred)),
        )
    ) * 100.0
    return {
        "Method": name,
        "TrajectorySteps": steps,
        "CandidateK": candidate_k,
        "FPR95": metrics[0],
        "AUROC": metrics[1],
        "AUPR_IN": metrics[2],
        "AUPR_OUT": metrics[3],
        "IDAccuracy": metrics[4],
        "IDScoreMean": float(id_conf.mean()),
        "IDScoreStd": float(id_conf.std()),
        "OODScoreMean": float(ood_conf.mean()),
        "OODScoreStd": float(ood_conf.std()),
        "IDCount": len(id_conf),
        "OODCount": len(ood_conf),
    }


def main() -> None:
    args = build_parser().parse_args()
    for name in ("openood_root", "data_root", "openood_results_root", "output_root"):
        setattr(args, name, getattr(args, name).resolve())

    add_openood_to_path(args.openood_root)
    evaluator_module = importlib.import_module("openood.evaluation_api.evaluator")
    evaluator_module.data_setup = lambda *_args, **_kwargs: None
    Evaluator = evaluator_module.Evaluator

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
        raise FileNotFoundError(f"Formal ImageNet-1K directory is missing: {formal_dir}")

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
        raise RuntimeError("Formal Gaussian detector was not loaded")

    output_dir = formal_dir / "validation_scaling_diagnosis"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "validation_features.pt"
    if feature_path.is_file():
        cached = torch.load(feature_path, map_location="cpu")
        id_features = cached["id_features"]
        id_logits = cached["id_logits"]
        id_labels = cached["id_labels"]
        ood_features = cached["ood_features"]
        ood_logits = cached["ood_logits"]
        print(f"Loaded validation feature cache: {feature_path}", flush=True)
    else:
        id_features, id_logits, id_labels = extract_split(
            network, evaluator.dataloader_dict["id"]["val"], "ID validation features"
        )
        ood_features, ood_logits, _ = extract_split(
            network, evaluator.dataloader_dict["ood"]["val"], "OOD validation features"
        )
        temporary = feature_path.with_suffix(".tmp")
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
        temporary.replace(feature_path)
        print(f"Saved validation feature cache: {feature_path}", flush=True)

    id_pred = id_logits.argmax(1).numpy()
    labels_numpy = id_labels.numpy()
    rows = []

    id_msp = id_logits.softmax(1).max(1).values.numpy()
    ood_msp = ood_logits.softmax(1).max(1).values.numpy()
    rows.append(
        metric_row(
            "MSP",
            id_msp,
            ood_msp,
            id_pred,
            labels_numpy,
            steps=None,
            candidate_k=None,
        )
    )

    sigma = detector.sigma.detach().cpu().flatten()
    sigma_summary = {
        "Potential": detector.potential,
        "Count": len(sigma),
        "Min": float(sigma.min()),
        "Q1": float(torch.quantile(sigma, 0.25)),
        "Median": float(torch.quantile(sigma, 0.5)),
        "Q3": float(torch.quantile(sigma, 0.75)),
        "Max": float(sigma.max()),
        "AtMinPercent": float((sigma <= detector.sigma_min * 1.001).float().mean() * 100),
        "AtMaxPercent": float((sigma >= detector.sigma_max * 0.999).float().mean() * 100),
    }
    (output_dir / "gaussian_sigma_summary.json").write_text(
        json.dumps(sigma_summary, indent=2), encoding="utf-8"
    )
    print("Gaussian sigma summary:", sigma_summary, flush=True)

    original_steps = detector.T
    original_candidate_k = detector.candidate_k
    try:
        for steps, candidate_k in VALIDATION_SWEEP:
            detector.T = steps
            detector.candidate_k = candidate_k
            print(
                f"\nValidation sweep: steps={steps}, candidate_k={candidate_k}",
                flush=True,
            )
            id_conf = detector_scores(detector, id_features, args.batch_size)
            ood_conf = detector_scores(detector, ood_features, args.batch_size)
            rows.append(
                metric_row(
                    "Hamiltonian-Gaussian",
                    id_conf,
                    ood_conf,
                    id_pred,
                    labels_numpy,
                    steps=steps,
                    candidate_k=candidate_k,
                )
            )
    finally:
        detector.T = original_steps
        detector.candidate_k = original_candidate_k

    result = pd.DataFrame(rows)
    result_path = output_dir / "validation_trajectory_candidate_sweep.csv"
    result.to_csv(result_path, index=False, float_format="%.6f")
    print("\n===== Validation-only scaling diagnosis =====")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved: {result_path}")
    print("No near/far OOD test images were used in this sweep.")


if __name__ == "__main__":
    main()
