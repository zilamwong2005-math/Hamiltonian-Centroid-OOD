"""OpenOOD v1.5 adapter for the Hamiltonian OOD detector."""

from __future__ import annotations

import ast
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from hamiltonian_detector import (
    BANDWIDTH_LOSSES,
    MASS_MODES,
    MASS_NORMALIZATIONS,
    HamiltonianDetector,
    image_effective_rank,
    load_torch_checkpoint,
    normalize_anchor_masses,
)
from openood.postprocessors import BasePostprocessor


def _label_from_imglist_line(line: str) -> int:
    tokens = line.strip().split(" ", 1)
    if len(tokens) != 2:
        raise ValueError(f"Malformed OpenOOD imglist line: {line!r}")
    extra = ast.literal_eval(tokens[1])
    if isinstance(extra, dict):
        if "label" not in extra:
            raise ValueError("Dictionary imglist entry does not contain a label")
        return int(extra["label"])
    return int(extra)


def stratified_reservoir_indices(
    imglist: Sequence[str],
    n_classes: int,
    samples_per_class: int,
    seed: int,
) -> List[int]:
    """Choose a bounded, reproducible class-balanced subset in one pass."""

    rng = random.Random(seed)
    seen = [0] * n_classes
    reservoirs: List[List[int]] = [[] for _ in range(n_classes)]
    for index, line in enumerate(imglist):
        label = _label_from_imglist_line(line)
        if not 0 <= label < n_classes:
            continue
        seen[label] += 1
        bucket = reservoirs[label]
        if len(bucket) < samples_per_class:
            bucket.append(index)
        else:
            replacement = rng.randrange(seen[label])
            if replacement < samples_per_class:
                bucket[replacement] = index

    missing = [c for c, bucket in enumerate(reservoirs) if not bucket]
    if missing:
        raise RuntimeError(f"Training imglist has no samples for classes: {missing[:20]}")
    return sorted(index for bucket in reservoirs for index in bucket)


def _forward_with_feature(net, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        output = net(data, return_feature=True)
    except TypeError as exc:
        raise TypeError(
            "The backbone must implement forward(x, return_feature=True), as the "
            "OpenOOD ResNet18_224x224 and ResNet50 networks do."
        ) from exc
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise TypeError("Backbone did not return (logits, feature)")
    logits, feature = output[0], output[1]
    if feature.ndim > 2:
        feature = torch.flatten(feature, 1)
    return logits, F.normalize(feature, dim=-1)


class HamiltonianPostprocessor(BasePostprocessor):
    """A standard OpenOOD postprocessor backed by Hamiltonian trajectories.

    OpenOOD calls :meth:`setup` once with its official train/val/test loaders,
    then calls :meth:`postprocess` for every ID and OOD batch.  Only a small,
    class-balanced subset of the ID training split is decoded for anchors and
    bandwidth fitting.  The extracted features are cached and reused across
    potential-function ablations.
    """

    def __init__(
        self,
        *,
        n_classes: int,
        potential: str,
        output_dir: Path,
        cache_tag: str,
        n_anchors: int = 5,
        setup_samples_per_class: int = 12,
        ham_epochs: int = 10,
        ham_lr: float = 1e-3,
        ham_batch_size: int = 32,
        bandwidth_loss: str = "static",
        trajectory_train_steps: int = 0,
        n_steps: int = 10,
        dt: float = 0.05,
        candidate_k: int = 20,
        sim_batch: int = 32,
        setup_batch_size: int = 128,
        num_workers: int = 8,
        sigma_init: float = 0.5,
        sigma_min: float = 0.05,
        sigma_max: float = 4.0,
        mass_mode: str = "uniform",
        mass_normalization: str = "none",
        mass_resolution: int = 0,
        prediction_source: str = "backbone",
        seed: int = 42,
        force_retrain: bool = False,
    ) -> None:
        super().__init__(config=None)
        if setup_samples_per_class < n_anchors:
            raise ValueError("setup_samples_per_class must be >= n_anchors")
        if n_classes > 1 and candidate_k == 1:
            raise ValueError("candidate_k must be 0 or at least 2")
        if ham_batch_size <= 0:
            raise ValueError("ham_batch_size must be positive")
        if trajectory_train_steps < 0:
            raise ValueError("trajectory_train_steps must be zero or positive")
        if bandwidth_loss not in BANDWIDTH_LOSSES:
            raise ValueError(f"bandwidth_loss must be one of {BANDWIDTH_LOSSES}")
        if mass_mode not in MASS_MODES:
            raise ValueError(f"mass_mode must be one of {MASS_MODES}")
        if mass_normalization not in MASS_NORMALIZATIONS:
            raise ValueError(
                f"mass_normalization must be one of {MASS_NORMALIZATIONS}"
            )
        if mass_resolution < 0:
            raise ValueError("mass_resolution must be zero or positive")
        if prediction_source not in ("backbone", "hamiltonian"):
            raise ValueError("prediction_source must be backbone or hamiltonian")

        self.n_classes = int(n_classes)
        self.potential = potential
        self.output_dir = Path(output_dir)
        self.cache_tag = cache_tag
        self.n_anchors = int(n_anchors)
        self.setup_samples_per_class = int(setup_samples_per_class)
        self.ham_epochs = int(ham_epochs)
        self.ham_lr = float(ham_lr)
        self.ham_batch_size = int(ham_batch_size)
        self.bandwidth_loss = bandwidth_loss
        self.trajectory_train_steps = (
            int(n_steps) if int(trajectory_train_steps) == 0 else int(trajectory_train_steps)
        )
        self.n_steps = int(n_steps)
        self.dt = float(dt)
        self.candidate_k = int(candidate_k)
        self.sim_batch = int(sim_batch)
        self.setup_batch_size = int(setup_batch_size)
        self.num_workers = int(num_workers)
        self.sigma_init = float(sigma_init)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.mass_mode = mass_mode
        self.mass_normalization = mass_normalization
        self.mass_resolution = int(mass_resolution)
        self.prediction_source = prediction_source
        self.seed = int(seed)
        self.force_retrain = bool(force_retrain)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.detector: HamiltonianDetector | None = None
        self.APS_mode = False
        self.hyperparam_search_done = True
        self.setup_flag = False

    @property
    def feature_cache_path(self) -> Path:
        return self.output_dir / (
            f"setup_features_{self.cache_tag}_seed{self.seed}"
            f"_m{self.setup_samples_per_class}.pt"
        )

    @property
    def detector_path(self) -> Path:
        dt_tag = str(self.dt).replace(".", "p")
        sigma_tag = str(self.sigma_init).replace(".", "p")
        sigma_min_tag = str(self.sigma_min).replace(".", "p")
        sigma_max_tag = str(self.sigma_max).replace(".", "p")
        lr_tag = str(self.ham_lr).replace(".", "p")
        return self.output_dir / (
            f"detector_{self.cache_tag}_seed{self.seed}_{self.potential}"
            f"_a{self.n_anchors}"
            f"_mass-{self.mass_mode}-{self.mass_normalization}"
            f"_loss-{self.bandwidth_loss}-ts{self.trajectory_train_steps}"
            f"_m{self.setup_samples_per_class}_t{self.n_steps}_dt{dt_tag}"
            f"_k{self.candidate_k}_s{sigma_tag}_{sigma_min_tag}-{sigma_max_tag}"
            f"_e{self.ham_epochs}_lr{lr_tag}.pt"
        )

    def _cache_metadata(self) -> Dict[str, object]:
        return {
            "format_version": 2,
            "cache_tag": self.cache_tag,
            "n_classes": self.n_classes,
            "samples_per_class": self.setup_samples_per_class,
            "seed": self.seed,
        }

    @torch.no_grad()
    def _extract_setup_features(
        self, net, train_loader
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dataset = train_loader.dataset
        if not hasattr(dataset, "imglist"):
            raise TypeError("Expected OpenOOD ImglistDataset with an imglist attribute")

        indices = stratified_reservoir_indices(
            dataset.imglist,
            self.n_classes,
            self.setup_samples_per_class,
            self.seed,
        )
        subset_loader = DataLoader(
            Subset(dataset, indices),
            batch_size=self.setup_batch_size,
            shuffle=False,
            generator=torch.Generator().manual_seed(self.seed + 10_000),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
        device = next(net.parameters()).device
        features, labels, source_indices = [], [], []
        net.eval()
        print(
            f"Hamiltonian setup: extracting {len(indices):,} balanced ID features...",
            flush=True,
        )
        for batch_index, batch in enumerate(subset_loader, 1):
            data = batch["data"].to(device, non_blocking=True)
            _, feature = _forward_with_feature(net, data)
            features.append(feature.cpu())
            labels.append(batch["label"].cpu().long())
            source_indices.append(batch["index"].cpu().long())
            if batch_index % 25 == 0:
                print(f"  setup batches: {batch_index}/{len(subset_loader)}", flush=True)

        return torch.cat(features), torch.cat(labels), torch.cat(source_indices)

    def _load_or_extract_features(
        self, net, train_loader
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = self._cache_metadata()
        # Feature extraction is independent of the potential and detector
        # bandwidths, so even --force-retrain should reuse this fair shared cache.
        if self.feature_cache_path.is_file():
            cached = load_torch_checkpoint(self.feature_cache_path, map_location="cpu")
            if cached.get("metadata") == expected:
                print(f"Loaded shared setup feature cache: {self.feature_cache_path}", flush=True)
                return cached["features"], cached["labels"], cached["source_indices"]

        features, labels, source_indices = self._extract_setup_features(net, train_loader)
        temporary = self.feature_cache_path.with_suffix(".tmp")
        torch.save(
            {
                "metadata": expected,
                "features": features,
                "labels": labels,
                "source_indices": source_indices,
            },
            temporary,
        )
        temporary.replace(self.feature_cache_path)
        print(f"Saved shared setup feature cache: {self.feature_cache_path}", flush=True)
        return features, labels, source_indices

    def _select_anchor_rows(self, labels: torch.Tensor) -> torch.Tensor:
        rows = []
        for class_index in range(self.n_classes):
            class_rows = torch.where(labels == class_index)[0]
            if len(class_rows) < self.n_anchors:
                raise RuntimeError(
                    f"Class {class_index} has {len(class_rows)} setup samples, "
                    f"but {self.n_anchors} anchors were requested"
                )
            generator = torch.Generator().manual_seed(self.seed + class_index)
            order = torch.randperm(len(class_rows), generator=generator)[: self.n_anchors]
            rows.append(class_rows[order])
        return torch.stack(rows)

    @property
    def mass_cache_path(self) -> Path:
        return self.output_dir / (
            f"anchor_image_effective_rank_{self.cache_tag}_seed{self.seed}"
            f"_a{self.n_anchors}_r{self.mass_resolution}.pt"
        )

    @torch.no_grad()
    def _load_or_compute_anchor_masses(
        self,
        dataset,
        anchor_source_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if self.mass_mode == "uniform":
            return torch.ones(
                self.n_classes, self.n_anchors, dtype=torch.float32
            )

        flat_indices = anchor_source_indices.reshape(-1).tolist()
        metadata = {
            "format_version": 1,
            "cache_tag": self.cache_tag,
            "seed": self.seed,
            "n_anchors": self.n_anchors,
            "mass_resolution": self.mass_resolution,
            "source_indices": flat_indices,
        }
        if self.mass_cache_path.is_file():
            cached = load_torch_checkpoint(self.mass_cache_path, map_location="cpu")
            if cached.get("metadata") == metadata:
                print(f"Loaded image effective-rank cache: {self.mass_cache_path}", flush=True)
                raw_masses = cached["masses"]
                return normalize_anchor_masses(raw_masses, self.mass_normalization)

        loader = DataLoader(
            Subset(dataset, flat_indices),
            batch_size=self.ham_batch_size,
            shuffle=False,
            generator=torch.Generator().manual_seed(self.seed + 20_000),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
        mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
        parts = []
        print(
            f"Hamiltonian setup: computing effective rank for {len(flat_indices):,} anchors...",
            flush=True,
        )
        for batch in loader:
            # data_aux is OpenOOD's deterministic test preprocessor.  Undo the
            # standard ImageNet normalisation before applying Definition 4.
            images = batch.get("data_aux", batch["data"]).to(device, non_blocking=True)
            images = (images * std + mean).clamp(0.0, 1.0)
            if self.mass_resolution > 0 and images.shape[-2:] != (
                self.mass_resolution,
                self.mass_resolution,
            ):
                images = F.interpolate(
                    images,
                    size=(self.mass_resolution, self.mass_resolution),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            parts.append(image_effective_rank(images).cpu())

        raw_masses = torch.cat(parts).reshape(self.n_classes, self.n_anchors)
        temporary = self.mass_cache_path.with_suffix(".tmp")
        torch.save({"metadata": metadata, "masses": raw_masses}, temporary)
        temporary.replace(self.mass_cache_path)
        print(f"Saved image effective-rank cache: {self.mass_cache_path}", flush=True)
        return normalize_anchor_masses(raw_masses, self.mass_normalization)

    @torch.no_grad()
    def _initialize_anchors(
        self,
        detector: HamiltonianDetector,
        features: torch.Tensor,
        labels: torch.Tensor,
        anchor_rows: torch.Tensor,
        anchor_masses: torch.Tensor,
    ) -> None:
        for class_index in range(self.n_classes):
            class_features = features[labels == class_index]
            anchors = F.normalize(features[anchor_rows[class_index]], dim=-1)
            detector.anchors[class_index].copy_(anchors.to(detector.anchors.device))
            detector.masses[class_index].copy_(
                anchor_masses[class_index].to(detector.masses.device)
            )
            detector.centroids[class_index].copy_(
                F.normalize(class_features.mean(0), dim=0).to(detector.centroids.device)
            )

    def _fit_bandwidths(
        self,
        detector: HamiltonianDetector,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        dataset = TensorDataset(features, labels)
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            dataset,
            batch_size=self.ham_batch_size,
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.Adam([detector.log_sigma], lr=self.ham_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.ham_epochs)
        )
        device = detector.anchors.device
        detector.train()
        for epoch in range(1, self.ham_epochs + 1):
            started = time.time()
            loss_sum, sample_count = 0.0, 0
            for feature, label in loader:
                feature = feature.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss, _ = detector.training_loss(
                    feature,
                    label,
                    objective=self.bandwidth_loss,
                    trajectory_steps=self.trajectory_train_steps,
                )
                loss.backward()
                optimizer.step()
                detector.clamp_bandwidths_()
                loss_sum += float(loss.detach()) * len(feature)
                sample_count += len(feature)
            scheduler.step()
            print(
                f"  {self.potential} sigma epoch {epoch:02d}/{self.ham_epochs}: "
                f"loss={loss_sum / sample_count:.6f}, {time.time() - started:.1f}s",
                flush=True,
            )

    def setup(self, net, id_loader_dict, ood_loader_dict) -> None:
        del ood_loader_dict
        device = next(net.parameters()).device
        if self.detector_path.is_file() and not self.force_retrain:
            checkpoint = load_torch_checkpoint(
                self.detector_path, map_location="cpu"
            )
            detector = HamiltonianDetector.from_checkpoint(checkpoint).to(device)
            if detector.potential != self.potential:
                raise RuntimeError("Detector checkpoint potential does not match request")
            self.detector = detector.eval()
            self.setup_flag = True
            print(f"Loaded Hamiltonian detector: {self.detector_path}", flush=True)
            return

        train_loader = id_loader_dict["train"]
        features, labels, source_indices = self._load_or_extract_features(
            net, train_loader
        )
        anchor_rows = self._select_anchor_rows(labels)
        anchor_source_indices = source_indices[anchor_rows]
        anchor_masses = self._load_or_compute_anchor_masses(
            train_loader.dataset, anchor_source_indices, device
        )
        feat_dim = int(features.shape[1])
        detector = HamiltonianDetector(
            feat_dim=feat_dim,
            n_classes=self.n_classes,
            n_anchors_per_class=self.n_anchors,
            n_steps=self.n_steps,
            dt=self.dt,
            sim_batch=self.sim_batch,
            potential=self.potential,
            candidate_k=self.candidate_k,
            sigma_init=self.sigma_init,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
        ).to(device)
        self._initialize_anchors(
            detector, features, labels, anchor_rows, anchor_masses
        )
        self._fit_bandwidths(detector, features, labels)
        detector.eval()

        checkpoint = detector.export_checkpoint()
        checkpoint["setup_metadata"] = self._cache_metadata()
        temporary = self.detector_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(self.detector_path)
        print(f"Saved Hamiltonian detector: {self.detector_path}", flush=True)
        self.detector = detector
        self.setup_flag = True

    @torch.no_grad()
    def postprocess(self, net, data: torch.Tensor):
        if self.detector is None:
            raise RuntimeError("HamiltonianPostprocessor.setup() has not run")
        logits, feature = _forward_with_feature(net, data)
        prediction = (
            logits.argmax(dim=1)
            if self.prediction_source == "backbone"
            else self.detector.predict(feature)
        )
        confidence = self.detector.score(feature)
        return prediction, confidence

    def save_run_config(self, path: Path, extra: Dict[str, object]) -> None:
        payload = {
            **self._cache_metadata(),
            "potential": self.potential,
            "n_anchors": self.n_anchors,
            "ham_epochs": self.ham_epochs,
            "ham_lr": self.ham_lr,
            "ham_batch_size": self.ham_batch_size,
            "bandwidth_loss": self.bandwidth_loss,
            "trajectory_train_steps": self.trajectory_train_steps,
            "n_steps": self.n_steps,
            "dt": self.dt,
            "candidate_k": self.candidate_k,
            "sim_batch": self.sim_batch,
            "sigma_init": self.sigma_init,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "mass_mode": self.mass_mode,
            "mass_normalization": self.mass_normalization,
            "mass_resolution": self.mass_resolution,
            "prediction_source": self.prediction_source,
            **extra,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
