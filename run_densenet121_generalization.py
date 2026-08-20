"""ImageNet-1K backbone generalisation with a pinned timm DenseNet-121.

The locked centroid/MSP rule and geometry weight are inherited unchanged from
the ResNet-50 validation decision.  Three seeds change only the class-balanced
centroid sample.  MSP and Scale are reproduced locally for the same backbone.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from hamiltonian_detector import load_torch_checkpoint
from run_locked_imagenet1k_fusion import (
    _default_decision_path,
    _load_locked_decision,
)
from diagnose_imagenet1k_score_rules import _validation_masks


MODEL_NAME = "densenet121.tv_in1k"
HF_REPO = "timm/densenet121.tv_in1k"
HF_REVISION = "d42603aa2960ebf044e46628536b6f0edf7c0c6b"
HF_FILENAME = "pytorch_model.bin"
WEIGHT_SHA256 = "42bd2c384fae0b346f977fe9feed1bcfdbd6ea4727478c6c76177d725b218e89"
NUM_CLASSES = 1000
FEATURE_DIM = 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("download", "smoke", "full"),
                        required=True)
    parser.add_argument("--openood-root", type=Path, default=Path("OpenOOD"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--hamiltonian-root", type=Path,
                        default=Path("results_openood"))
    parser.add_argument("--journal-root", type=Path,
                        default=Path("results/journal"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("results/journal/backbone_densenet121"))
    parser.add_argument("--cache-root", type=Path,
                        default=Path("cache/pretrained"))
    parser.add_argument("--decision-path", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--setup-samples-per-class", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--setup-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=128)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_pinned_weights(destination: Path, attempts: int = 5) -> Path:
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        digest = _sha256(destination)
        if digest == WEIGHT_SHA256:
            print(f"Verified cached DenseNet-121 weights: {destination}")
            return destination
        raise RuntimeError(
            f"Existing DenseNet weight hash mismatch: {destination}\n"
            f"expected {WEIGHT_SHA256}, found {digest}"
        )

    partial = destination.with_name(destination.name + ".part")
    url = (
        f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/"
        f"{HF_FILENAME}?download=true"
    )
    for attempt in range(1, attempts + 1):
        start = partial.stat().st_size if partial.is_file() else 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        try:
            with requests.get(
                url, headers=headers, stream=True, allow_redirects=True,
                timeout=(30, 120),
            ) as response:
                response.raise_for_status()
                append = start > 0 and response.status_code == 206
                mode = "ab" if append else "wb"
                downloaded = start if append else 0
                last_report = time.monotonic()
                with partial.open(mode) as stream:
                    for block in response.iter_content(8 * 1024 * 1024):
                        if not block:
                            continue
                        stream.write(block)
                        downloaded += len(block)
                        if time.monotonic() - last_report >= 5:
                            print(
                                f"DenseNet weights: {downloaded / 1024**2:.1f} MB",
                                flush=True,
                            )
                            last_report = time.monotonic()
        except requests.RequestException as error:
            if attempt == attempts:
                raise RuntimeError(
                    f"Pinned DenseNet weight download failed: {url}"
                ) from error
            delay = min(2**attempt, 30)
            print(
                f"[retry {attempt}/{attempts}] {error}; resuming in {delay}s",
                flush=True,
            )
            time.sleep(delay)
            continue

        digest = _sha256(partial)
        if digest == WEIGHT_SHA256:
            partial.replace(destination)
            print(f"Downloaded and verified DenseNet-121 weights: {destination}")
            return destination
        if attempt == attempts:
            raise RuntimeError(
                f"DenseNet weight hash mismatch after download: expected "
                f"{WEIGHT_SHA256}, found {digest}. Keep {partial} for inspection."
            )
        # A complete but invalid response cannot be resumed safely.
        partial.unlink()
        print(
            f"[retry {attempt}/{attempts}] invalid SHA256 {digest}; restarting",
            flush=True,
        )
    raise AssertionError("unreachable")


class TimmDenseNetAdapter(nn.Module):
    """Expose OpenOOD's return_feature/get_fc interface for timm DenseNet."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @property
    def fc(self):
        return self.model.get_classifier()

    def get_fc_layer(self):
        return self.model.get_classifier()

    def get_fc(self):
        layer = self.get_fc_layer()
        return (
            layer.weight.detach().cpu().numpy(),
            layer.bias.detach().cpu().numpy(),
        )

    def _forward_pre_logits(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Return pooled DenseNet features across old and new timm APIs."""
        forward_head = getattr(self.model, "forward_head", None)
        if callable(forward_head):
            return forward_head(feature_map, pre_logits=True)

        # timm 0.9.x DenseNet variants can expose ``forward_features`` without
        # the newer ``forward_head`` helper.  Their classifier path follows the
        # original DenseNet implementation: final ReLU, global average pooling,
        # then flattening.  Prefer the model's configured global-pool module
        # when it exists and otherwise use the canonical adaptive average pool.
        activated = F.relu(feature_map, inplace=False)
        global_pool = getattr(self.model, "global_pool", None)
        if callable(global_pool):
            feature = global_pool(activated)
        else:
            feature = F.adaptive_avg_pool2d(activated, output_size=(1, 1))
        if feature.ndim > 2:
            feature = torch.flatten(feature, 1)
        if feature.ndim != 2:
            raise RuntimeError(
                "DenseNet pre-logit feature must be two-dimensional, found "
                f"shape={tuple(feature.shape)}"
            )
        return feature

    def forward(self, x, return_feature=False, return_feature_list=False):
        feature_map = self.model.forward_features(x)
        feature = self._forward_pre_logits(feature_map)
        logits = self.get_fc_layer()(feature)
        if return_feature_list:
            return logits, [feature_map, feature]
        if return_feature:
            return logits, feature
        return logits


def _build_model(weights_path: Path):
    import timm

    try:
        model = timm.create_model(MODEL_NAME, pretrained=False)
    except RuntimeError:
        model = timm.create_model("densenet121", pretrained=False)
    raw = load_torch_checkpoint(weights_path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    result = model.load_state_dict(raw, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"DenseNet strict load failed: {result}")
    if int(getattr(model, "num_features", -1)) != FEATURE_DIM:
        raise RuntimeError(f"Unexpected DenseNet feature dimension: {model.num_features}")

    try:
        from timm.data import resolve_model_data_config
        data_config = resolve_model_data_config(model)
    except ImportError:
        from timm.data import resolve_data_config
        data_config = resolve_data_config({}, model=model)
    from timm.data import create_transform
    preprocessor = create_transform(**data_config, is_training=False)
    return TimmDenseNetAdapter(model), preprocessor, data_config


def _accumulate_centroids(
    features: torch.Tensor, labels: torch.Tensor, n_classes: int
) -> torch.Tensor:
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("features/labels have incompatible shapes")
    sums = torch.zeros(n_classes, features.shape[1], dtype=features.dtype)
    counts = torch.zeros(n_classes, dtype=torch.long)
    sums.index_add_(0, labels, features)
    counts.index_add_(0, labels, torch.ones_like(labels))
    if (counts == 0).any():
        missing = torch.where(counts == 0)[0].tolist()
        raise RuntimeError(f"Centroid classes without samples: {missing[:20]}")
    return F.normalize(sums / counts[:, None], dim=-1)


def _max_centroid_cosine(
    features: torch.Tensor, centroids: torch.Tensor
) -> torch.Tensor:
    """Apply the audited max-cosine geometry rule to a feature batch."""
    if features.ndim != 2 or centroids.ndim != 2:
        raise ValueError("features and centroids must both be two-dimensional")
    if features.shape[1] != centroids.shape[1]:
        raise ValueError("features and centroids have incompatible dimensions")
    return (F.normalize(features, dim=-1) @ centroids.T).max(1).values


def _make_locked_postprocessor(
    *,
    output_root: Path,
    seed: int,
    setup_samples_per_class: int,
    alpha: float,
    calibration_seed: int,
    setup_batch_size: int,
    num_workers: int,
    model_sha256: str,
    forward_with_feature,
    stratified_reservoir_indices,
    BasePostprocessor,
):
    cache_path = output_root / "centroids" / f"densenet121_seed{seed}.pt"

    class LockedDenseNetCentroidPostprocessor(BasePostprocessor):
        def __init__(self):
            super().__init__(config=None)
            self.APS_mode = False
            self.hyperparam_search_done = True
            self.setup_flag = False
            self.calibration = None
            self.centroids = None

        @torch.no_grad()
        def setup(self, net, id_loader_dict, ood_loader_dict):
            del ood_loader_dict
            device = next(net.parameters()).device
            expected = {
                "format_version": 1,
                "model_sha256": model_sha256,
                "seed": seed,
                "samples_per_class": setup_samples_per_class,
                "n_classes": NUM_CLASSES,
                "feature_dim": FEATURE_DIM,
            }
            if cache_path.is_file():
                cached = load_torch_checkpoint(cache_path, map_location="cpu")
                if cached.get("metadata") != expected:
                    raise RuntimeError(f"Stale DenseNet centroid cache: {cache_path}")
                centroids = cached["centroids"]
                print(f"Loaded DenseNet centroid cache: {cache_path}")
            else:
                train_dataset = id_loader_dict["train"].dataset
                indices = stratified_reservoir_indices(
                    train_dataset.imglist,
                    NUM_CLASSES,
                    setup_samples_per_class,
                    seed,
                )
                loader = DataLoader(
                    Subset(train_dataset, indices),
                    batch_size=setup_batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=num_workers > 0,
                )
                features, labels = [], []
                for batch in tqdm(loader, desc=f"DenseNet centroids seed {seed}"):
                    _, feature = forward_with_feature(
                        net, batch["data"].to(device, non_blocking=True)
                    )
                    features.append(feature.cpu())
                    labels.append(batch["label"].long().cpu())
                centroids = _accumulate_centroids(
                    torch.cat(features), torch.cat(labels), NUM_CLASSES
                )
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".tmp")
                torch.save(
                    {"metadata": expected, "centroids": centroids}, temporary
                )
                temporary.replace(cache_path)
                print(f"Saved DenseNet centroid cache: {cache_path}")
            self.centroids = centroids.to(device)

            msp_parts, geometry_parts, label_parts = [], [], []
            for batch in tqdm(
                id_loader_dict["val"], desc="DenseNet ID-only calibration"
            ):
                logits, feature = forward_with_feature(
                    net, batch["data"].to(device, non_blocking=True)
                )
                msp_parts.append(logits.softmax(1).max(1).values.cpu())
                geometry_parts.append(
                    _max_centroid_cosine(feature, self.centroids).cpu()
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
            msp = logits.softmax(1).max(1).values
            geometry = _max_centroid_cosine(feature, self.centroids)
            confidence = (
                (1.0 - alpha) * (msp - self.msp_mean) / self.msp_std
                + alpha
                * (geometry - self.geometry_mean)
                / self.geometry_std
            )
            return logits.argmax(1), confidence

    return LockedDenseNetCentroidPostprocessor()


def _target(output_root: Path, mode: str, method: str, seed: int) -> Path:
    return output_root / mode / method / f"seed{seed}" / f"{method}.csv"


def _normalise_saved(path: Path, method: str, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    unnamed = [name for name in frame if name.startswith("Unnamed:")]
    if unnamed:
        frame = frame.rename(columns={unnamed[0]: "Dataset"})
    elif "Dataset" not in frame:
        frame = frame.rename(columns={frame.columns[0]: "Dataset"})
    frame.insert(0, "Seed", seed)
    frame.insert(0, "Method", method)
    return frame


def main() -> None:
    args = build_parser().parse_args()
    for field in (
        "openood_root", "data_root", "hamiltonian_root", "journal_root",
        "output_root", "cache_root",
    ):
        setattr(args, field, getattr(args, field).resolve())
    if sorted(set(args.seeds)) != [0, 1, 2]:
        raise ValueError("DenseNet locked evaluation requires seeds 0, 1, and 2")
    args.seeds = [0, 1, 2]
    weights_path = args.cache_root / "densenet121_tv_in1k_pinned.bin"
    _download_pinned_weights(weights_path)
    if args.stage == "download":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("DenseNet generalisation requires CUDA")

    decision_path = (
        args.decision_path.resolve()
        if args.decision_path is not None
        else _default_decision_path(args.hamiltonian_root)
    )
    decision, decision_hash = _load_locked_decision(decision_path)
    gate_path = args.journal_root / "locked_imagenet1k_fusion/validation_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("decision_sha256") != decision_hash:
        raise RuntimeError("Locked decision and validation gate differ")
    alpha = float(decision["geometry_weight"])

    from Imagenet_ood_experiment import (
        add_openood_to_path,
        limit_evaluator_for_smoke_test,
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
    stratified_reservoir_indices = (
        postprocessor_module.stratified_reservoir_indices
    )

    mode = args.stage
    evaluation_limit = args.max_eval_samples if mode == "smoke" else 0
    completed = args.output_root / f"{mode}_completed.json"
    if completed.is_file():
        print(f"DenseNet {mode} already complete: {completed}")
        return

    frames = []
    # Locked fusion: centroid sampling varies over three seeds.
    for seed in args.seeds:
        target = _target(args.output_root, mode, "locked_centroid_msp", seed)
        config_path = target.with_suffix(".json")
        if target.is_file() and config_path.is_file():
            frames.append(_normalise_saved(target, "locked_centroid_msp", seed))
            print(f"[resume] DenseNet locked seed={seed}")
            continue
        network, preprocessor, data_config = _build_model(weights_path)
        network = network.cuda().eval()
        postprocessor = _make_locked_postprocessor(
            output_root=args.output_root,
            seed=seed,
            setup_samples_per_class=args.setup_samples_per_class,
            alpha=alpha,
            calibration_seed=args.calibration_seed,
            setup_batch_size=args.setup_batch_size,
            num_workers=args.num_workers,
            model_sha256=WEIGHT_SHA256,
            forward_with_feature=forward_with_feature,
            stratified_reservoir_indices=stratified_reservoir_indices,
            BasePostprocessor=BasePostprocessor,
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
        if evaluation_limit:
            limit_evaluator_for_smoke_test(evaluator, evaluation_limit)
        print(f"[run] DenseNet locked centroid/MSP seed={seed} mode={mode}")
        metrics = evaluator.eval_ood(fsood=False, progress=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(target, float_format="%.6f")
        config_path.write_text(json.dumps({
            "model": MODEL_NAME,
            "model_weights_sha256": WEIGHT_SHA256,
            "model_revision": HF_REVISION,
            "data_config": data_config,
            "method": "locked_centroid_msp",
            "decision_sha256": decision_hash,
            "geometry_weight": alpha,
            "geometry_score_rule": "centroid/max_cosine",
            "query_feature_normalization": "l2",
            "centroid_normalization": "l2_after_class_mean",
            "seed": seed,
            "setup_samples_per_class": args.setup_samples_per_class,
            "calibration": postprocessor.calibration,
            "max_eval_samples": evaluation_limit,
        }, indent=2), encoding="utf-8")
        frames.append(_normalise_saved(target, "locked_centroid_msp", seed))
        del evaluator, postprocessor, network
        torch.cuda.empty_cache()

    # Deterministic classifier baselines need one run each.
    for method in ("msp", "scale"):
        seed = 0
        target = _target(args.output_root, mode, method, seed)
        config_path = target.with_suffix(".json")
        if target.is_file() and config_path.is_file():
            frames.append(_normalise_saved(target, method, seed))
            print(f"[resume] DenseNet {method}")
            continue
        network, preprocessor, data_config = _build_model(weights_path)
        network = network.cuda().eval()
        evaluator = Evaluator(
            network,
            id_name="imagenet",
            data_root=str(args.data_root),
            config_root=str(args.openood_root / "configs"),
            preprocessor=preprocessor,
            postprocessor_name=method,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if evaluation_limit:
            limit_evaluator_for_smoke_test(evaluator, evaluation_limit)
        print(f"[run] DenseNet {method} mode={mode}")
        metrics = evaluator.eval_ood(fsood=False, progress=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(target, float_format="%.6f")
        config_path.write_text(json.dumps({
            "model": MODEL_NAME,
            "model_weights_sha256": WEIGHT_SHA256,
            "model_revision": HF_REVISION,
            "data_config": data_config,
            "method": method,
            "seed": 0,
            "max_eval_samples": evaluation_limit,
        }, indent=2), encoding="utf-8")
        frames.append(_normalise_saved(target, method, 0))
        del evaluator, network
        torch.cuda.empty_cache()

    combined = pd.concat(frames, ignore_index=True)
    combined_path = args.output_root / f"densenet121_{mode}_all_runs.csv"
    combined.to_csv(combined_path, index=False, float_format="%.6f")
    completed.parent.mkdir(parents=True, exist_ok=True)
    completed.write_text(json.dumps({
        "model": MODEL_NAME,
        "weights_sha256": WEIGHT_SHA256,
        "decision_sha256": decision_hash,
        "mode": mode,
        "locked_seeds": args.seeds,
        "baselines": ["msp", "scale"],
        "combined_csv": str(combined_path.resolve()),
        "completed": True,
    }, indent=2), encoding="utf-8")
    print(f"Saved DenseNet-121 generalisation: {combined_path}")


if __name__ == "__main__":
    main()
