"""Reusable Hamiltonian OOD detector with interchangeable radial potentials.

The paper defines a negative potential well ``U(q) = -A(q)``.  This module
stores and reports the positive affinity ``A`` (larger means more ID-like) and
integrates the physical force ``-grad(U) = grad(A)``.  Keeping that convention
explicit prevents the sign error that is easy to introduce when translating
the leapfrog equations.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


POTENTIAL_NAMES = ("gaussian", "laplacian", "cauchy", "imq", "matern32")
MASS_MODES = ("uniform", "effective_rank")
MASS_NORMALIZATIONS = ("none", "class_mean", "global_mean")
BANDWIDTH_LOSSES = ("static", "trajectory")


def image_effective_rank(images: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute the paper's effective-rank mass for a batch of images.

    ``images`` must have shape ``[B, C, H, W]`` and contain unnormalised image
    intensities.  RGB inputs are converted to luminance, each image matrix is
    mean-centred, and ``exp(entropy(s / sum(s)))`` is returned.  A constant
    image has a zero centred matrix; following the paper's stated convention,
    its mass is defined as one.
    """

    if images.ndim != 4 or images.shape[1] not in (1, 3):
        raise ValueError("images must have shape [B, 1|3, H, W]")
    images = images.float()
    if images.shape[1] == 3:
        weights = images.new_tensor((0.2989, 0.5870, 0.1140)).view(1, 3, 1, 1)
        matrix = (images * weights).sum(dim=1)
    else:
        matrix = images[:, 0]
    matrix = matrix - matrix.mean(dim=(-2, -1), keepdim=True)
    singular_values = torch.linalg.svdvals(matrix)
    totals = singular_values.sum(dim=-1, keepdim=True)
    probabilities = singular_values / totals.clamp_min(eps)
    entropy = -(torch.special.xlogy(probabilities, probabilities)).sum(dim=-1)
    ranks = entropy.exp()
    return torch.where(totals.squeeze(-1) > eps, ranks, torch.ones_like(ranks))


def normalize_anchor_masses(
    masses: torch.Tensor, normalization: str = "none", eps: float = 1e-8
) -> torch.Tensor:
    """Apply an explicit mass normalisation used by the ablation scripts."""

    if masses.ndim != 2:
        raise ValueError("anchor masses must have shape [classes, anchors]")
    if normalization not in MASS_NORMALIZATIONS:
        raise ValueError(
            f"Unknown mass normalization '{normalization}'; "
            f"choose one of {MASS_NORMALIZATIONS}"
        )
    if normalization == "none":
        return masses
    if normalization == "class_mean":
        return masses / masses.mean(dim=1, keepdim=True).clamp_min(eps)
    return masses / masses.mean().clamp_min(eps)


def load_torch_checkpoint(path: str | Path, map_location="cpu"):
    """Load trusted experiment checkpoints across PyTorch 1.13 through 2.x."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only keyword.
        return torch.load(path, map_location=map_location)


@dataclass(frozen=True)
class DetectorConfig:
    feat_dim: int
    n_classes: int
    n_anchors_per_class: int = 20
    n_steps: int = 20
    dt: float = 0.1
    anchor_chunk: int = 10
    class_chunk: int = 50
    sim_batch: int = 64
    potential: str = "gaussian"
    candidate_k: int = 0
    sigma_init: float = 1.0
    sigma_min: float = 0.05
    sigma_max: float = 4.0
    eps: float = 1e-8


def _validate_potential(name: str) -> str:
    name = name.lower().replace("-", "_")
    aliases = {
        "rbf": "gaussian",
        "inverse_multiquadric": "imq",
        "inverse_quadratic": "cauchy",
        "matern_32": "matern32",
    }
    name = aliases.get(name, name)
    if name not in POTENTIAL_NAMES:
        raise ValueError(
            f"Unknown potential '{name}'. Choose one of: {', '.join(POTENTIAL_NAMES)}"
        )
    return name


def radial_affinity_and_force_coefficient(
    dist2: torch.Tensor,
    sigma: torch.Tensor,
    potential: str,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return radial affinity and the coefficient of ``(q - anchor)``.

    If ``A(q) = phi(||q-a|| / sigma)``, the returned ``coefficient`` obeys
    ``grad_q A = coefficient * (q-a)``.  It is non-positive for all supported
    attractive wells.
    """

    potential = _validate_potential(potential)
    sigma2 = sigma.square().clamp_min(eps)
    scaled2 = dist2.clamp_min(0.0) / sigma2

    if potential == "gaussian":
        affinity = torch.exp(-0.5 * scaled2)
        coefficient = -affinity / sigma2
    elif potential == "laplacian":
        scaled_r = torch.sqrt(scaled2 + eps)
        affinity = torch.exp(-scaled_r)
        coefficient = -affinity / (sigma2 * scaled_r.clamp_min(math.sqrt(eps)))
    elif potential == "cauchy":
        base = 1.0 + scaled2
        affinity = base.reciprocal()
        coefficient = -2.0 * base.square().reciprocal() / sigma2
    elif potential == "imq":
        base = 1.0 + scaled2
        affinity = torch.rsqrt(base)
        coefficient = -base.pow(-1.5) / sigma2
    else:  # Matern-3/2
        scaled_r = torch.sqrt(scaled2 + eps)
        root3_r = math.sqrt(3.0) * scaled_r
        exp_term = torch.exp(-root3_r)
        affinity = (1.0 + root3_r) * exp_term
        coefficient = -3.0 * exp_term / sigma2

    return affinity, coefficient


class HamiltonianDetector(nn.Module):
    """Class-balanced radial potential field with leapfrog OOD scoring.

    ``candidate_k=0`` evaluates every class exactly.  A positive value keeps
    the nearest centroid classes fixed during each trajectory.  The latter is
    an explicit large-scale approximation intended for ImageNet-200/1K.
    """

    def __init__(
        self,
        feat_dim: int,
        n_classes: int,
        n_anchors_per_class: int = 20,
        n_steps: int = 20,
        dt: float = 0.1,
        anchor_chunk: int = 10,
        class_chunk: int = 50,
        sim_batch: int = 64,
        potential: str = "gaussian",
        candidate_k: int = 0,
        sigma_init: float = 1.0,
        sigma_min: float = 0.05,
        sigma_max: float = 4.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if feat_dim <= 0 or n_classes <= 0 or n_anchors_per_class <= 0:
            raise ValueError("feat_dim, n_classes, and n_anchors_per_class must be positive")
        if n_steps < 0 or dt <= 0 or sim_batch <= 0:
            raise ValueError("n_steps must be non-negative; dt and sim_batch must be positive")
        if not 0 < sigma_min <= sigma_init <= sigma_max:
            raise ValueError("Require 0 < sigma_min <= sigma_init <= sigma_max")

        self.feat_dim = int(feat_dim)
        self.n_classes = int(n_classes)
        self.K = int(n_anchors_per_class)
        self.T = int(n_steps)
        self.dt = float(dt)
        self.anchor_chunk = int(anchor_chunk)
        self.class_chunk = int(class_chunk)
        self.sim_batch = int(sim_batch)
        self.potential = _validate_potential(potential)
        self.candidate_k = min(max(int(candidate_k), 0), self.n_classes)
        self.sigma_init = float(sigma_init)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.eps = float(eps)

        initial_log_sigma = math.log(float(sigma_init))
        self.log_sigma = nn.Parameter(
            torch.full((self.n_classes, self.K), initial_log_sigma)
        )
        self.register_buffer(
            "anchors", torch.zeros(self.n_classes, self.K, self.feat_dim)
        )
        self.register_buffer("masses", torch.ones(self.n_classes, self.K))
        self.register_buffer("centroids", torch.zeros(self.n_classes, self.feat_dim))

    @property
    def config(self) -> DetectorConfig:
        return DetectorConfig(
            feat_dim=self.feat_dim,
            n_classes=self.n_classes,
            n_anchors_per_class=self.K,
            n_steps=self.T,
            dt=self.dt,
            anchor_chunk=self.anchor_chunk,
            class_chunk=self.class_chunk,
            sim_batch=self.sim_batch,
            potential=self.potential,
            candidate_k=self.candidate_k,
            sigma_init=self.sigma_init,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            eps=self.eps,
        )

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.clamp(
            math.log(self.sigma_min), math.log(self.sigma_max)
        ).exp()

    def clamp_bandwidths_(self) -> None:
        with torch.no_grad():
            self.log_sigma.clamp_(math.log(self.sigma_min), math.log(self.sigma_max))

    @torch.no_grad()
    def refresh_centroids(self) -> None:
        self.anchors.copy_(F.normalize(self.anchors, dim=-1))
        weighted = (self.anchors * self.masses[..., None]).sum(dim=1)
        self.centroids.copy_(F.normalize(weighted, dim=-1))

    def export_checkpoint(self) -> Dict[str, object]:
        return {
            "format_version": 2,
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
        }

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Dict[str, object], map_location: Optional[str] = None
    ) -> "HamiltonianDetector":
        del map_location  # torch.load handles device placement before this method.
        config = dict(checkpoint["config"])
        detector = cls(**config)
        detector.load_state_dict(checkpoint["state_dict"])
        return detector

    def _flat_field(
        self, q: torch.Tensor, need_force: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Exact all-class affinity using a GEMM rather than a 4-D difference."""

        q = F.normalize(q, dim=-1)
        flat_anchors = self.anchors.reshape(-1, self.feat_dim)
        flat_sigma = self.sigma.reshape(1, -1)
        flat_masses = self.masses.reshape(1, -1)
        dist2 = (2.0 - 2.0 * (q @ flat_anchors.T)).clamp_min(0.0)
        affinity, coefficient = radial_affinity_and_force_coefficient(
            dist2, flat_sigma, self.potential, self.eps
        )
        weighted_affinity = affinity * flat_masses
        per_class = weighted_affinity.reshape(-1, self.n_classes, self.K).sum(-1)

        if not need_force:
            return per_class, None

        weighted_coefficient = coefficient * flat_masses
        force = (
            q * weighted_coefficient.sum(dim=1, keepdim=True)
            - weighted_coefficient @ flat_anchors
        )
        return per_class, force

    def select_candidates(self, q: torch.Tensor, candidate_k: Optional[int] = None) -> torch.Tensor:
        k = self.candidate_k if candidate_k is None else int(candidate_k)
        if k <= 0 or k >= self.n_classes:
            return torch.arange(self.n_classes, device=q.device).expand(q.shape[0], -1)
        similarity = F.normalize(q, dim=-1) @ self.centroids.T
        return similarity.topk(k, dim=1, largest=True, sorted=True).indices

    def _candidate_field(
        self, q: torch.Tensor, candidates: torch.Tensor, need_force: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        q = F.normalize(q, dim=-1)
        anchors = self.anchors[candidates]
        sigma = self.sigma[candidates]
        masses = self.masses[candidates]
        # Avoid materialising [B, candidate_k, K, D] differences.  This is
        # especially important when differentiating trajectories on ImageNet.
        similarity = torch.einsum("bd,bckd->bck", q, anchors)
        dist2 = (2.0 - 2.0 * similarity).clamp_min(0.0)
        affinity, coefficient = radial_affinity_and_force_coefficient(
            dist2, sigma, self.potential, self.eps
        )
        weighted_affinity = affinity * masses
        per_class = weighted_affinity.sum(-1)
        if not need_force:
            return per_class, None
        weighted_coefficient = coefficient * masses
        force = (
            q * weighted_coefficient.sum(dim=(1, 2), keepdim=False)[:, None]
            - torch.einsum("bck,bckd->bd", weighted_coefficient, anchors)
        )
        return per_class, force

    def affinity_per_class(
        self, q: torch.Tensor, candidates: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if candidates is None:
            return self._flat_field(q, need_force=False)[0]
        return self._candidate_field(q, candidates, need_force=False)[0]

    # Compatibility with the original experiment helper.
    def _affinity_per_class(
        self, q: torch.Tensor, detach_sigma: bool = False
    ) -> torch.Tensor:
        if detach_sigma:
            with torch.no_grad():
                return self.affinity_per_class(q)
        return self.affinity_per_class(q)

    def training_loss(
        self,
        q: torch.Tensor,
        labels: torch.Tensor,
        candidate_k: Optional[int] = None,
        *,
        objective: str = "static",
        trajectory_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fit bandwidths with either the static surrogate or Eq. 13--14.

        ``objective='trajectory'`` keeps the leapfrog graph differentiable and
        applies cross entropy to the historical per-class maximum affinity.
        Candidate mode always includes the true class plus hard negatives.
        """

        if objective not in BANDWIDTH_LOSSES:
            raise ValueError(
                f"Unknown bandwidth objective '{objective}'; choose from {BANDWIDTH_LOSSES}"
            )
        q = F.normalize(q, dim=-1)
        k = self.candidate_k if candidate_k is None else int(candidate_k)
        candidates = None
        if k <= 0 or k >= self.n_classes:
            target = labels
        else:
            # Use the true class plus deterministic hard negatives.  The target
            # is column zero, so every class bandwidth still receives positives.
            negative_k = min(k - 1, self.n_classes - 1)
            similarity = q @ self.centroids.T
            similarity = similarity.clone()
            similarity.scatter_(1, labels[:, None], float("-inf"))
            negatives = similarity.topk(negative_k, dim=1).indices
            candidates = torch.cat((labels[:, None], negatives), dim=1)
            target = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)

        if objective == "static":
            affinity = self.affinity_per_class(q, candidates)
        else:
            affinity = self.trajectory_affinity(
                q, candidates=candidates, n_steps=trajectory_steps
            )
        return F.cross_entropy(affinity, target), affinity

    def trajectory_affinity(
        self,
        q0: torch.Tensor,
        *,
        candidates: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Differentiable historical affinity used by Eq. 13--14 training."""

        steps = self.T if n_steps is None else int(n_steps)
        if steps < 0:
            raise ValueError("trajectory training steps must be non-negative")
        q = F.normalize(q0, dim=-1)
        p = torch.zeros_like(q)
        best, force = self._field(q, candidates)
        for _ in range(steps):
            p = p + 0.5 * self.dt * force
            q = F.normalize(q + self.dt * p, dim=-1)
            current, force = self._field(q, candidates)
            p = p + 0.5 * self.dt * force
            best = torch.maximum(best, current)
        return best

    def forward(self, q: torch.Tensor, labels: torch.Tensor):
        return self.training_loss(q, labels)

    def _field(
        self, q: torch.Tensor, candidates: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if candidates is None:
            affinity, force = self._flat_field(q, need_force=True)
        else:
            affinity, force = self._candidate_field(q, candidates, need_force=True)
        assert force is not None
        return affinity, force

    @torch.no_grad()
    def _simulate_chunk(
        self, q0: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        q = F.normalize(q0, dim=-1).clone()
        p = torch.zeros_like(q)
        candidates = None
        if 0 < self.candidate_k < self.n_classes:
            candidates = self.select_candidates(q)

        best, force = self._field(q, candidates)
        for _ in range(self.T):
            # force = -grad(U) = grad(A), hence the plus sign.
            p = p + 0.5 * self.dt * force
            q = F.normalize(q + self.dt * p, dim=-1)
            current, force = self._field(q, candidates)
            p = p + 0.5 * self.dt * force
            best = torch.maximum(best, current)
        return best, candidates

    @torch.no_grad()
    def simulate(self, q0: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        affinity_parts = []
        candidate_parts = []
        has_candidates = 0 < self.candidate_k < self.n_classes
        for start in range(0, q0.shape[0], self.sim_batch):
            affinity, candidates = self._simulate_chunk(q0[start : start + self.sim_batch])
            affinity_parts.append(affinity)
            if has_candidates and candidates is not None:
                candidate_parts.append(candidates)
        all_candidates = torch.cat(candidate_parts) if candidate_parts else None
        return torch.cat(affinity_parts), all_candidates

    @torch.no_grad()
    def score(self, q: torch.Tensor) -> torch.Tensor:
        affinity, _ = self.simulate(q)
        return affinity.max(dim=-1).values

    @torch.no_grad()
    def predict(self, q: torch.Tensor) -> torch.Tensor:
        affinity, candidates = self.simulate(q)
        local_pred = affinity.argmax(dim=-1)
        if candidates is None:
            return local_pred
        return candidates.gather(1, local_pred[:, None]).squeeze(1)
