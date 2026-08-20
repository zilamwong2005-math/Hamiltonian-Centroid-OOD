"""Validate, then test, the locked ImageNet-1K centroid/MSP fusion rule.

The score rule and its geometry weight must come from the validation-only
diagnosis JSON.  ``--stage validate`` checks the already locked rule across
three independently sampled centroid banks without touching near/far test
loaders.  ``--stage test`` refuses to run unless that validation gate passed,
then evaluates exactly once with the locked configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from diagnose_imagenet1k_score_rules import _metric, _validation_masks, _zscore
from hamiltonian_detector import HamiltonianDetector, load_torch_checkpoint


CACHE_TAG = "imagenet1k_resnet50_tvsv1"
SOURCE_EXPERIMENT = "mass-uniform-none_loss-static-ts0_pred-backbone_eval-full"
EXPECTED_RULE = "centroid/max_cosine"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("validate", "test"))
    parser.add_argument("--openood-root", type=Path, default=Path("OpenOOD"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--openood-results-root", type=Path, default=Path("openood_pretrained")
    )
    parser.add_argument(
        "--hamiltonian-root", type=Path, default=Path("results_openood")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/journal/locked_imagenet1k_fusion"),
    )
    parser.add_argument("--decision-path", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--calibration-seed", type=int, default=0)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_locked_decision(path: Path) -> tuple[dict, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Locked validation decision is missing: {path}")
    decision = json.loads(path.read_text(encoding="utf-8"))
    if not decision.get("recommended_for_one_locked_test_run", False):
        raise RuntimeError("Validation diagnosis did not authorise a locked test run")
    if decision.get("selected_geometry_method") != EXPECTED_RULE:
        raise RuntimeError(
            "This runner only implements the audited centroid/max_cosine rule; "
            f"found {decision.get('selected_geometry_method')!r}"
        )
    alpha = float(decision.get("geometry_weight", -1.0))
    if not 0.0 < alpha <= 1.0:
        raise RuntimeError(f"Invalid locked geometry weight: {alpha}")
    protocol = str(decision.get("protocol", "")).lower()
    if "validation-only" not in protocol or "no near/far" not in protocol:
        raise RuntimeError("Decision JSON does not attest validation-only selection")
    return decision, _sha256(path)


def _source_directory(root: Path, seed: int) -> Path:
    return root / "imagenet1k" / CACHE_TAG / f"seed{seed}" / SOURCE_EXPERIMENT


def _detector_path(root: Path, seed: int) -> Path:
    directory = _source_directory(root, seed)
    candidates = sorted(directory.glob("detector_*_gaussian_*.pt"))
    candidates = [
        path for path in candidates
        if "_mass-uniform-none_" in path.name
        and "_loss-static-ts0_" in path.name
        and "_t0_" in path.name
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one seed-{seed} static Gaussian detector under {directory}; "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def _load_detector(root: Path, seed: int, device: torch.device):
    path = _detector_path(root, seed)
    checkpoint = load_torch_checkpoint(path, map_location="cpu")
    detector = HamiltonianDetector.from_checkpoint(checkpoint).to(device).eval()
    if detector.n_classes != 1000 or detector.feat_dim != 2048:
        raise RuntimeError(
            f"Unexpected detector shape: classes={detector.n_classes}, "
            f"feature_dim={detector.feat_dim}"
        )
    return detector, path


@torch.no_grad()
def _centroid_scores(
    detector: HamiltonianDetector, features: torch.Tensor, batch_size: int
) -> np.ndarray:
    parts = []
    for start in tqdm(
        range(0, len(features), batch_size), desc="Centroid confidence"
    ):
        query = F.normalize(
            features[start : start + batch_size].to(
                detector.centroids.device, non_blocking=True
            ),
            dim=-1,
        )
        parts.append((query @ detector.centroids.T).max(1).values.cpu())
    return torch.cat(parts).numpy()


def _fuse_scores(
    id_msp: np.ndarray,
    ood_msp: np.ndarray,
    id_geometry: np.ndarray,
    ood_geometry: np.ndarray,
    reference_mask: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    msp_reference = id_msp[reference_mask]
    geometry_reference = id_geometry[reference_mask]
    id_fused = (
        (1.0 - alpha) * _zscore(id_msp, msp_reference)
        + alpha * _zscore(id_geometry, geometry_reference)
    )
    ood_fused = (
        (1.0 - alpha) * _zscore(ood_msp, msp_reference)
        + alpha * _zscore(ood_geometry, geometry_reference)
    )
    statistics = {
        "msp_mean": float(msp_reference.mean()),
        "msp_std": float(msp_reference.std()),
        "geometry_mean": float(geometry_reference.mean()),
        "geometry_std": float(geometry_reference.std()),
        "calibration_count": int(reference_mask.sum()),
    }
    return id_fused, ood_fused, statistics


def _default_decision_path(hamiltonian_root: Path) -> Path:
    return (
        hamiltonian_root
        / "imagenet1k"
        / CACHE_TAG
        / "seed0"
        / "mass-uniform-none_loss-static-ts10_pred-backbone_eval-full"
        / "validation_score_rule_diagnosis_v1"
        / "locked_rule_decision.json"
    )


def _validation_feature_path(hamiltonian_root: Path) -> Path:
    return (
        hamiltonian_root
        / "imagenet1k"
        / CACHE_TAG
        / "seed0"
        / "mass-uniform-none_loss-static-ts10_pred-backbone_eval-full"
        / "validation_scaling_diagnosis"
        / "validation_features.pt"
    )


def run_validation(args, decision: dict, decision_hash: str) -> None:
    feature_path = _validation_feature_path(args.hamiltonian_root)
    if not feature_path.is_file():
        raise FileNotFoundError(f"Validation feature cache is missing: {feature_path}")
    cache = load_torch_checkpoint(feature_path, map_location="cpu")
    id_features = cache["id_features"]
    id_logits = cache["id_logits"]
    id_labels = cache["id_labels"].numpy()
    ood_features = cache["ood_features"]
    ood_logits = cache["ood_logits"]
    id_prediction = id_logits.argmax(1).numpy()
    id_msp = id_logits.softmax(1).max(1).values.numpy()
    ood_msp = ood_logits.softmax(1).max(1).values.numpy()
    alpha = float(decision["geometry_weight"])

    id_tune, id_holdout, ood_tune, ood_holdout = _validation_masks(
        id_labels, len(ood_features), args.calibration_seed
    )
    splits = {
        "tune": (id_tune, ood_tune),
        "holdout": (id_holdout, ood_holdout),
        "full": (np.ones(len(id_labels), bool), np.ones(len(ood_features), bool)),
    }
    rows = []
    calibrations = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in args.seeds:
        detector, detector_path = _load_detector(args.hamiltonian_root, seed, device)
        print(f"[seed {seed}] detector: {detector_path}", flush=True)
        id_geometry = _centroid_scores(detector, id_features, args.batch_size)
        ood_geometry = _centroid_scores(detector, ood_features, args.batch_size)
        id_fused, ood_fused, calibration = _fuse_scores(
            id_msp, ood_msp, id_geometry, ood_geometry, id_tune, alpha
        )
        calibration.update({"Seed": seed, "Detector": str(detector_path)})
        calibrations.append(calibration)
        for split, (id_mask, ood_mask) in splits.items():
            msp_row = _metric(
                "MSP", split,
                id_msp[id_mask], ood_msp[ood_mask],
                id_prediction[id_mask], id_labels[id_mask],
            )
            msp_row["Seed"] = seed
            rows.append(msp_row)
            fusion_row = _metric(
                "Locked-Centroid-MSP", split,
                id_fused[id_mask], ood_fused[ood_mask],
                id_prediction[id_mask], id_labels[id_mask],
            )
            fusion_row["Seed"] = seed
            rows.append(fusion_row)

    result = pd.DataFrame(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        args.output_root / "locked_validation_three_seeds.csv",
        index=False,
        float_format="%.6f",
    )
    pd.DataFrame(calibrations).to_csv(
        args.output_root / "locked_calibration_statistics.csv",
        index=False,
        float_format="%.9f",
    )

    holdout = result[result["Split"].eq("holdout")].pivot(
        index="Seed", columns="Method", values=["AUROC", "FPR95"]
    )
    per_seed = []
    passed = True
    for seed in args.seeds:
        delta_auroc = float(
            holdout.loc[seed, ("AUROC", "Locked-Centroid-MSP")]
            - holdout.loc[seed, ("AUROC", "MSP")]
        )
        delta_fpr = float(
            holdout.loc[seed, ("FPR95", "Locked-Centroid-MSP")]
            - holdout.loc[seed, ("FPR95", "MSP")]
        )
        seed_passed = bool(
            (delta_auroc >= 0.5 and delta_fpr <= 0.0)
            or (delta_auroc >= 1.0 and delta_fpr <= 2.0)
        )
        passed = passed and seed_passed
        per_seed.append({
            "Seed": seed,
            "Delta_AUROC_vs_MSP": delta_auroc,
            "Delta_FPR95_vs_MSP": delta_fpr,
            "Passed": seed_passed,
        })
    gate = {
        "decision_sha256": decision_hash,
        "score_rule": EXPECTED_RULE,
        "geometry_weight": alpha,
        "calibration_protocol": (
            "two ID-validation images per class selected with calibration_seed; "
            "no OOD or test score is used for normalisation"
        ),
        "calibration_seed": args.calibration_seed,
        "seeds": args.seeds,
        "per_seed_holdout": per_seed,
        "authorised_for_one_locked_test_run": passed,
    }
    (args.output_root / "validation_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    print("\n===== Fixed-rule validation across detector seeds =====")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n===== Test gate =====")
    print(json.dumps(gate, indent=2))
    print("No near/far OOD test loaders were read.")


def _make_locked_postprocessor(
    detector: HamiltonianDetector,
    alpha: float,
    calibration_seed: int,
    forward_with_feature,
    BasePostprocessor,
):
    class LockedCentroidMSPPostprocessor(BasePostprocessor):
        def __init__(self):
            super().__init__(config=None)
            self.APS_mode = False
            self.hyperparam_search_done = True
            self.setup_flag = False
            self.calibration = None

        @torch.no_grad()
        def setup(self, net, id_loader_dict, ood_loader_dict):
            del ood_loader_dict
            msp_parts, geometry_parts, label_parts = [], [], []
            device = next(net.parameters()).device
            for batch in tqdm(
                id_loader_dict["val"], desc="Locked ID-only calibration"
            ):
                data = batch["data"].to(device, non_blocking=True)
                logits, feature = forward_with_feature(net, data)
                msp_parts.append(logits.softmax(1).max(1).values.cpu())
                geometry_parts.append(
                    (F.normalize(feature, dim=-1) @ detector.centroids.T)
                    .max(1).values.cpu()
                )
                label_parts.append(batch["label"].long().cpu())
            msp = torch.cat(msp_parts).numpy()
            geometry = torch.cat(geometry_parts).numpy()
            labels = torch.cat(label_parts).numpy()
            tune, _, _, _ = _validation_masks(labels, 2, calibration_seed)
            self.msp_mean = float(msp[tune].mean())
            self.msp_std = max(float(msp[tune].std()), 1e-12)
            self.geometry_mean = float(geometry[tune].mean())
            self.geometry_std = max(float(geometry[tune].std()), 1e-12)
            self.calibration = {
                "msp_mean": self.msp_mean,
                "msp_std": self.msp_std,
                "geometry_mean": self.geometry_mean,
                "geometry_std": self.geometry_std,
                "calibration_count": int(tune.sum()),
            }
            self.setup_flag = True

        @torch.no_grad()
        def postprocess(self, net, data):
            logits, feature = forward_with_feature(net, data)
            prediction = logits.argmax(1)
            msp = logits.softmax(1).max(1).values
            geometry = (
                F.normalize(feature, dim=-1) @ detector.centroids.T
            ).max(1).values
            confidence = (
                (1.0 - alpha) * (msp - self.msp_mean) / self.msp_std
                + alpha
                * (geometry - self.geometry_mean)
                / self.geometry_std
            )
            return prediction, confidence

    return LockedCentroidMSPPostprocessor()


def run_test(args, decision: dict, decision_hash: str) -> None:
    gate_path = args.output_root / "validation_gate.json"
    if not gate_path.is_file():
        raise FileNotFoundError(
            f"Run --stage validate before the locked test: {gate_path}"
        )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("decision_sha256") != decision_hash:
        raise RuntimeError("Decision JSON changed after validation; refusing test run")
    if not gate.get("authorised_for_one_locked_test_run", False):
        raise RuntimeError("Three-seed validation gate did not authorise testing")
    completion_path = args.output_root / "locked_test_completed.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("decision_sha256") != decision_hash:
            raise RuntimeError("A locked test with a different decision already exists")
        raise RuntimeError(
            "The locked test is already complete; refusing repeated test access"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Locked ImageNet-1K test requires CUDA")

    from Imagenet_ood_experiment import add_openood_to_path, build_network

    add_openood_to_path(args.openood_root)
    evaluator_module = importlib.import_module("openood.evaluation_api.evaluator")
    evaluator_module.data_setup = lambda *_args, **_kwargs: None
    Evaluator = evaluator_module.Evaluator
    postprocessor_module = importlib.import_module("openood_hamiltonian_postprocessor")
    BasePostprocessor = importlib.import_module(
        "openood.postprocessors.base_postprocessor"
    ).BasePostprocessor
    forward_with_feature = postprocessor_module._forward_with_feature

    network, preprocessor, cache_tag, model_source = build_network(
        "imagenet", None, args.openood_results_root, 0, 1
    )
    if cache_tag != CACHE_TAG:
        raise RuntimeError(f"Expected {CACHE_TAG}, found {cache_tag}")
    network = network.cuda().eval()
    alpha = float(decision["geometry_weight"])
    frames = []
    for seed in args.seeds:
        seed_dir = args.output_root / "test" / f"seed{seed}"
        metric_path = seed_dir / "locked_centroid_msp.csv"
        config_path = seed_dir / "locked_centroid_msp.json"
        if metric_path.is_file() or config_path.is_file():
            if not (metric_path.is_file() and config_path.is_file()):
                raise RuntimeError(
                    f"Incomplete locked seed-{seed} result; inspect {seed_dir}"
                )
            saved_config = json.loads(config_path.read_text(encoding="utf-8"))
            if saved_config.get("decision_sha256") != decision_hash:
                raise RuntimeError(
                    f"Seed-{seed} result belongs to a different locked decision"
                )
            frame = pd.read_csv(metric_path)
            unnamed = [name for name in frame if name.startswith("Unnamed:")]
            if unnamed:
                frame = frame.rename(columns={unnamed[0]: "Dataset"})
            elif "Dataset" not in frame:
                frame = frame.rename(columns={frame.columns[0]: "Dataset"})
            frame.insert(0, "Seed", seed)
            frames.append(frame)
            print(f"[resume] seed={seed} already saved; not re-evaluating test", flush=True)
            continue
        detector, detector_path = _load_detector(
            args.hamiltonian_root, seed, torch.device("cuda")
        )
        postprocessor = _make_locked_postprocessor(
            detector, alpha, args.calibration_seed,
            forward_with_feature, BasePostprocessor,
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
        print(f"[locked test] seed={seed}", flush=True)
        metrics = evaluator.eval_ood(fsood=False, progress=True)
        seed_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(metric_path, float_format="%.6f")
        config = {
            "protocol": "OpenOOD v1.5 one locked test run",
            "decision_sha256": decision_hash,
            "validation_gate": str(gate_path.resolve()),
            "score_rule": EXPECTED_RULE,
            "geometry_weight": alpha,
            "calibration_seed": args.calibration_seed,
            "calibration": postprocessor.calibration,
            "detector_seed": seed,
            "detector_path": str(detector_path.resolve()),
            "model_source": model_source,
        }
        config_path.write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        frame = metrics.reset_index().rename(columns={"index": "Dataset"})
        frame.insert(0, "Seed", seed)
        frames.append(frame)
        del evaluator, postprocessor, detector
        torch.cuda.empty_cache()
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(
        args.output_root / "locked_test_all_seeds.csv",
        index=False,
        float_format="%.6f",
    )
    completion_path.write_text(
        json.dumps({
            "decision_sha256": decision_hash,
            "seeds": args.seeds,
            "completed": True,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Saved locked test results under: {args.output_root}")


def main() -> None:
    args = build_parser().parse_args()
    for name in (
        "openood_root", "data_root", "openood_results_root",
        "hamiltonian_root", "output_root",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if sorted(set(args.seeds)) != [0, 1, 2]:
        raise ValueError("The journal protocol requires exactly seeds 0, 1, and 2")
    args.seeds = [0, 1, 2]
    decision_path = (
        args.decision_path.resolve()
        if args.decision_path is not None
        else _default_decision_path(args.hamiltonian_root)
    )
    decision, decision_hash = _load_locked_decision(decision_path)
    print(f"Locked decision: {decision_path}")
    print(f"Decision SHA256: {decision_hash}")
    if args.stage == "validate":
        run_validation(args, decision, decision_hash)
    else:
        run_test(args, decision, decision_hash)


if __name__ == "__main__":
    main()
