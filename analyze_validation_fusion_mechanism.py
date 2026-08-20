"""Validation-only analysis of class geometry/MSP fusion across scales.

Only official ID-validation and OOD-validation loaders are iterated.  The
near/far test loaders are never evaluated.  The script measures score
separation, score correlation and an alpha sensitivity sweep without changing
the already locked alpha=0.8 test rule.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diagnose_imagenet1k_score_rules import (
    _extract_split,
    _metric,
    _validation_masks,
    _zscore,
)
from run_densenet121_generalization import (
    WEIGHT_SHA256,
    _build_model as _build_densenet,
    _download_pinned_weights,
    _make_locked_postprocessor as _make_densenet_postprocessor,
)
from run_locked_centroid_transfer import _load_detector as _load_transfer_detector
from run_locked_imagenet1k_fusion import (
    _centroid_scores,
    _default_decision_path,
    _load_detector as _load_imagenet1k_detector,
    _load_locked_decision,
    _make_locked_postprocessor,
)
from run_msp_baselines import (
    OPENOOD_ID_NAME,
    _batch_size_for,
    _model_for_benchmark,
)


TARGETS = (
    "cifar10",
    "cifar100",
    "imagenet200",
    "imagenet1k_resnet50",
    "imagenet1k_densenet121",
)
BASE_BENCHMARK = {
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "imagenet200": "imagenet200",
    "imagenet1k_resnet50": "imagenet1k",
    "imagenet1k_densenet121": "imagenet1k",
}
CLASS_COUNT = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenet200": 200,
    "imagenet1k_resnet50": 1000,
    "imagenet1k_densenet121": 1000,
}
ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("smoke", "full"))
    parser.add_argument("--targets", nargs="+", choices=TARGETS,
                        default=list(TARGETS))
    parser.add_argument("--openood-root", type=Path, default=Path("OpenOOD"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--openood-results-root", type=Path,
                        default=Path("openood_pretrained"))
    parser.add_argument("--hamiltonian-root", type=Path,
                        default=Path("results_openood"))
    parser.add_argument("--journal-root", type=Path,
                        default=Path("results/journal"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("results/journal/validation_fusion_mechanism"))
    parser.add_argument("--densenet-cache-root", type=Path,
                        default=Path("cache/pretrained"))
    parser.add_argument("--decision-path", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--batch-size-cifar", type=int, default=256)
    parser.add_argument("--batch-size-imagenet200", type=int, default=128)
    parser.add_argument("--batch-size-imagenet1k", type=int, default=64)
    parser.add_argument("--setup-batch-size", type=int, default=128)
    parser.add_argument("--setup-samples-per-class", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=128)
    parser.add_argument("--tvs-version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--weight-download-backend",
                        choices=("auto", "official", "huggingface"),
                        default="auto")
    return parser


def _raw_path(output_root: Path, mode: str, target: str, seed: int) -> Path:
    return output_root / mode / target / f"seed{seed}" / "validation_scores.npz"


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.std() < 1e-12 or right.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _standardized_gap(id_score: np.ndarray, ood_score: np.ndarray) -> float:
    """ID-minus-OOD mean gap in pooled-standard-deviation units."""

    id_score = np.asarray(id_score, dtype=np.float64)
    ood_score = np.asarray(ood_score, dtype=np.float64)
    pooled = float(np.sqrt((id_score.var() + ood_score.var()) / 2.0))
    if not np.isfinite(pooled) or pooled < 1e-12:
        return 0.0
    return float((id_score.mean() - ood_score.mean()) / pooled)


def _analyse_scores(
    target: str, seed: int, arrays: dict[str, np.ndarray]
) -> tuple[list[dict], dict]:
    id_msp = arrays["id_msp"]
    ood_msp = arrays["ood_msp"]
    id_geometry = arrays["id_geometry"]
    ood_geometry = arrays["ood_geometry"]
    id_prediction = arrays["id_prediction"].astype(int)
    id_labels = arrays["id_labels"].astype(int)
    id_tune = arrays["id_tune"].astype(bool)
    ood_tune = arrays["ood_tune"].astype(bool)

    id_msp_z = _zscore(id_msp, id_msp[id_tune])
    ood_msp_z = _zscore(ood_msp, id_msp[id_tune])
    id_geometry_z = _zscore(id_geometry, id_geometry[id_tune])
    ood_geometry_z = _zscore(ood_geometry, id_geometry[id_tune])
    splits = {
        "tune": (id_tune, ood_tune),
        "holdout": (~id_tune, ~ood_tune),
        "full": (
            np.ones(len(id_msp), dtype=bool),
            np.ones(len(ood_msp), dtype=bool),
        ),
    }
    rows = []
    for alpha in ALPHAS:
        id_score = (1.0 - alpha) * id_msp_z + alpha * id_geometry_z
        ood_score = (1.0 - alpha) * ood_msp_z + alpha * ood_geometry_z
        for split, (id_mask, ood_mask) in splits.items():
            row = _metric(
                f"fusion_alpha_{alpha:g}", split,
                id_score[id_mask], ood_score[ood_mask],
                id_prediction[id_mask], id_labels[id_mask],
            )
            row.update({
                "Target": target,
                "ClassCount": CLASS_COUNT[target],
                "Seed": seed,
                "Alpha": alpha,
            })
            rows.append(row)
    statistics = {
        "Target": target,
        "ClassCount": CLASS_COUNT[target],
        "Seed": seed,
        "IDCount": len(id_msp),
        "OODCount": len(ood_msp),
        "ID_MSP_Mean": float(id_msp.mean()),
        "OOD_MSP_Mean": float(ood_msp.mean()),
        "MSP_MeanGap": float(id_msp.mean() - ood_msp.mean()),
        "MSP_StandardizedGap": _standardized_gap(id_msp, ood_msp),
        "ID_Geometry_Mean": float(id_geometry.mean()),
        "OOD_Geometry_Mean": float(ood_geometry.mean()),
        "Geometry_MeanGap": float(id_geometry.mean() - ood_geometry.mean()),
        "Geometry_StandardizedGap": _standardized_gap(
            id_geometry, ood_geometry
        ),
        "ID_MSP_Geometry_Correlation": _correlation(id_msp, id_geometry),
        "OOD_MSP_Geometry_Correlation": _correlation(ood_msp, ood_geometry),
        "CalibrationIDCount": int(id_tune.sum()),
        "TuneOODCount": int(ood_tune.sum()),
    }
    return rows, statistics


def main() -> None:
    args = build_parser().parse_args()
    for field in (
        "openood_root", "data_root", "openood_results_root",
        "hamiltonian_root", "journal_root", "output_root",
        "densenet_cache_root",
    ):
        setattr(args, field, getattr(args, field).resolve())
    args.targets = list(dict.fromkeys(args.targets))
    if sorted(set(args.seeds)) != [0, 1, 2]:
        raise ValueError("The mechanism protocol requires seeds 0, 1, and 2")
    args.seeds = [0, 1, 2]
    if not torch.cuda.is_available():
        raise RuntimeError("Validation mechanism analysis requires CUDA")

    decision_path = (
        args.decision_path.resolve()
        if args.decision_path is not None
        else _default_decision_path(args.hamiltonian_root)
    )
    decision, decision_hash = _load_locked_decision(decision_path)
    gate_path = args.journal_root / "locked_imagenet1k_fusion/validation_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("decision_sha256") != decision_hash:
        raise RuntimeError("Validation gate and locked decision hashes differ")

    from Imagenet_ood_experiment import (
        add_openood_to_path,
        _limit_loader,
    )

    add_openood_to_path(args.openood_root)
    evaluator_module = importlib.import_module("openood.evaluation_api.evaluator")
    evaluator_module.data_setup = lambda *_args, **_kwargs: None
    Evaluator = evaluator_module.Evaluator
    BasePostprocessor = importlib.import_module(
        "openood.postprocessors.base_postprocessor"
    ).BasePostprocessor
    postprocessor_module = importlib.import_module(
        "openood_hamiltonian_postprocessor"
    )
    forward_with_feature = postprocessor_module._forward_with_feature
    stratified_reservoir_indices = postprocessor_module.stratified_reservoir_indices

    mode = args.stage
    evaluation_limit = args.max_eval_samples if mode == "smoke" else 0
    completion_path = args.output_root / f"validation_mechanism_{mode}_completed.json"
    if completion_path.is_file():
        print(f"Validation mechanism {mode} already complete: {completion_path}")
        return

    dense_weights = args.densenet_cache_root / "densenet121_tv_in1k_pinned.bin"
    if "imagenet1k_densenet121" in args.targets:
        _download_pinned_weights(dense_weights)

    all_rows, all_statistics = [], []
    for target in args.targets:
        benchmark = BASE_BENCHMARK[target]
        for seed in args.seeds:
            raw_path = _raw_path(args.output_root, mode, target, seed)
            config_path = raw_path.with_suffix(".json")
            if raw_path.is_file() != config_path.is_file():
                raise RuntimeError(
                    "Incomplete validation-score cache; remove or restore both "
                    f"files before resuming: {raw_path}, {config_path}"
                )
            if raw_path.is_file() and config_path.is_file():
                print(f"[resume] validation {target} seed={seed}", flush=True)
                with np.load(raw_path) as saved:
                    arrays = {name: saved[name] for name in saved.files}
                rows, statistics = _analyse_scores(target, seed, arrays)
                all_rows.extend(rows)
                all_statistics.append(statistics)
                continue

            if target == "imagenet1k_densenet121":
                network, preprocessor, data_config = _build_densenet(dense_weights)
                network = network.cuda().eval()
                postprocessor = _make_densenet_postprocessor(
                    output_root=args.journal_root / "backbone_densenet121",
                    seed=seed,
                    setup_samples_per_class=args.setup_samples_per_class,
                    alpha=float(decision["geometry_weight"]),
                    calibration_seed=args.calibration_seed,
                    setup_batch_size=args.setup_batch_size,
                    num_workers=args.num_workers,
                    model_sha256=WEIGHT_SHA256,
                    forward_with_feature=forward_with_feature,
                    stratified_reservoir_indices=stratified_reservoir_indices,
                    BasePostprocessor=BasePostprocessor,
                )
                model_tag = "densenet121.tv_in1k"
                geometry_source = postprocessor
            else:
                network, preprocessor, model_tag, _ = _model_for_benchmark(
                    benchmark, args, seed
                )
                network = network.cuda().eval()
                if target == "imagenet1k_resnet50":
                    detector, _ = _load_imagenet1k_detector(
                        args.hamiltonian_root, seed, torch.device("cuda")
                    )
                else:
                    detector, _ = _load_transfer_detector(args, benchmark, seed)
                postprocessor = _make_locked_postprocessor(
                    detector, float(decision["geometry_weight"]),
                    args.calibration_seed, forward_with_feature, BasePostprocessor,
                )
                data_config = None
                geometry_source = detector

            evaluator = Evaluator(
                network,
                id_name=OPENOOD_ID_NAME[benchmark],
                data_root=str(args.data_root),
                config_root=str(args.openood_root / "configs"),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                batch_size=_batch_size_for(benchmark, args),
                shuffle=False,
                num_workers=args.num_workers,
            )
            if evaluation_limit:
                # Evaluator.setup() has already used the complete ID validation
                # set for the locked, ID-only normalisation.  Limit only the
                # subsequent engineering smoke extraction.  No near/far test
                # loader is accessed here or anywhere else in this program.
                evaluator.dataloader_dict["id"]["val"] = _limit_loader(
                    evaluator.dataloader_dict["id"]["val"], evaluation_limit
                )
                evaluator.dataloader_dict["ood"]["val"] = _limit_loader(
                    evaluator.dataloader_dict["ood"]["val"], evaluation_limit
                )
            print(f"[extract] validation {target} seed={seed} mode={mode}", flush=True)
            id_features, id_logits, id_labels_t = _extract_split(
                evaluator.net, evaluator.dataloader_dict["id"]["val"],
                forward_with_feature, f"{target} ID validation",
            )
            ood_features, ood_logits, _ = _extract_split(
                evaluator.net, evaluator.dataloader_dict["ood"]["val"],
                forward_with_feature, f"{target} OOD validation",
            )
            id_geometry = _centroid_scores(
                geometry_source, id_features, _batch_size_for(benchmark, args)
            )
            ood_geometry = _centroid_scores(
                geometry_source, ood_features, _batch_size_for(benchmark, args)
            )
            id_labels = id_labels_t.numpy()
            id_tune, _, ood_tune, _ = _validation_masks(
                id_labels, len(ood_features), args.calibration_seed
            )
            arrays = {
                "id_msp": id_logits.softmax(1).max(1).values.numpy(),
                "ood_msp": ood_logits.softmax(1).max(1).values.numpy(),
                "id_geometry": id_geometry,
                "ood_geometry": ood_geometry,
                "id_prediction": id_logits.argmax(1).numpy(),
                "id_labels": id_labels,
                "id_tune": id_tune,
                "ood_tune": ood_tune,
            }
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(raw_path, **arrays)
            config_path.write_text(json.dumps({
                "protocol": "validation-only; no near/far test iteration",
                "decision_sha256": decision_hash,
                "locked_alpha": float(decision["geometry_weight"]),
                "target": target,
                "class_count": CLASS_COUNT[target],
                "seed": seed,
                "model_tag": model_tag,
                "data_config": data_config,
                "max_eval_samples": evaluation_limit,
            }, indent=2), encoding="utf-8")
            rows, statistics = _analyse_scores(target, seed, arrays)
            all_rows.extend(rows)
            all_statistics.append(statistics)
            print(f"[saved] {raw_path}", flush=True)

            del evaluator, postprocessor, network
            if target != "imagenet1k_densenet121":
                del detector
            torch.cuda.empty_cache()

    sweep = pd.DataFrame(all_rows)
    statistics = pd.DataFrame(all_statistics)
    expected_runs = len(args.targets) * len(args.seeds)
    expected_sweep_rows = expected_runs * len(ALPHAS) * 3
    if len(statistics) != expected_runs or len(sweep) != expected_sweep_rows:
        raise RuntimeError(
            "Incomplete mechanism results: "
            f"statistics={len(statistics)}/{expected_runs}, "
            f"sweep={len(sweep)}/{expected_sweep_rows}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    sweep_path = args.output_root / f"validation_alpha_sweep_{mode}.csv"
    statistics_path = args.output_root / f"validation_score_statistics_{mode}.csv"
    sweep.to_csv(sweep_path, index=False, float_format="%.6f")
    statistics.to_csv(statistics_path, index=False, float_format="%.9f")
    completion_path.write_text(json.dumps({
        "protocol": "validation-only; no near/far test iteration",
        "decision_sha256": decision_hash,
        "targets": args.targets,
        "seeds": args.seeds,
        "alphas": list(ALPHAS),
        "max_eval_samples": evaluation_limit,
        "sweep_csv": str(sweep_path.resolve()),
        "statistics_csv": str(statistics_path.resolve()),
        "completed": True,
    }, indent=2), encoding="utf-8")
    print("\n===== Validation alpha sweep: holdout split =====")
    print(
        sweep[sweep["Split"].eq("holdout")]
        .groupby(["Target", "ClassCount", "Alpha"])[["AUROC", "FPR95"]]
        .mean().round(4).to_string()
    )
    print("\n===== Validation score statistics =====")
    print(statistics.round(5).to_string(index=False))
    print(f"\nSaved: {sweep_path}")
    print(f"Saved: {statistics_path}")
    print("No near/far OOD test loader was iterated.")


if __name__ == "__main__":
    main()
