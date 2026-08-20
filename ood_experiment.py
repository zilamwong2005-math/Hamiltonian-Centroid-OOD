"""
Hamiltonian Image Classification System — OOD Detection Experiments
====================================================================
Reproduces the CIFAR benchmark table from the paper and supports the strict
OpenOOD v1.5 file lists:
  In-dist (model) x OOD dataset → AUROC / AUPR / FPR95

Supported in-dist sets : CIFAR-10, CIFAR-100 (SVHN in legacy mode only)
Supported OOD sets      : CIFAR counterpart, TinyImageNet, MNIST, SVHN,
                          Places365, Texture
Supported backbones     : OpenOOD ResNet-18 (32x32), DenseNet-BC-100

Usage
-----
# Single run:
python ood_experiment.py --in_dist cifar10 --model densenet --epochs 100

# Full table (all 6 in-dist × model combos):
python ood_experiment.py --run_all --epochs 100

# Resume an interrupted run (encoder checkpoint is reloaded automatically):
python ood_experiment.py --in_dist cifar10 --model resnet18 \
  --protocol openood --encoder_source official

# Fair potential-function ablation with a shared encoder and anchor seed:
python ood_experiment.py --in_dist cifar10 --model resnet \
  --potentials gaussian laplacian cauchy imq matern32

Output layout
-------------
results/
  ood_results_potentials.csv   ← numeric table, appended each run
  run_<timestamp>.log          ← full console mirror for this session
  checkpoints/
    cifar10_densenet_encoder.pt   ← encoder weights + metadata
    cifar10_densenet_<potential>_detector.pt  ← detector state
"""

import argparse
import hashlib
import json
import logging
import os
import random
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from hamiltonian_detector import (
    BANDWIDTH_LOSSES,
    MASS_MODES,
    MASS_NORMALIZATIONS,
    POTENTIAL_NAMES,
    HamiltonianDetector,
    image_effective_rank,
    normalize_anchor_masses,
)
from openood_cifar import (
    OOD_GROUPS,
    build_openood_dataset,
    canonical_ood_name,
    discover_cifar_checkpoint,
    iter_ood_names,
)

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT   = Path("./data")
RESULTS_DIR = Path("./results")
CKPT_DIR    = RESULTS_DIR / "checkpoints"
RESULTS_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)
_FILE_DIGEST_CACHE: dict[Path, str] = {}


def configure_paths(data_root: Path, results_dir: Path) -> None:
    """Set all runtime paths before logging or checkpoints are opened."""

    global DATA_ROOT, RESULTS_DIR, CKPT_DIR, CSV_PATH
    DATA_ROOT = Path(data_root)
    RESULTS_DIR = Path(results_dir)
    CKPT_DIR = RESULTS_DIR / "checkpoints"
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_PATH = RESULTS_DIR / "ood_results_cifar_v3.csv"


def file_sha256(path: Path) -> str:
    path = Path(path).resolve()
    if path not in _FILE_DIGEST_CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        _FILE_DIGEST_CACHE[path] = digest.hexdigest()
    return _FILE_DIGEST_CACHE[path]


# ══════════════════════════════════════════════════════════════════════════════
# 0.  Logging — tee every print() to both console and a timestamped .log file
# ══════════════════════════════════════════════════════════════════════════════

def setup_logger() -> logging.Logger:
    """
    Creates a logger that writes to:
      • stdout  (INFO level, human-readable)
      • results/run_<timestamp>.log  (DEBUG level, full detail)
    Returns the logger.  Use log.info() / log.debug() throughout.
    """
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = RESULTS_DIR / f"run_{ts}.log"

    fmt     = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger = logging.getLogger("hamiltonian_ood")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file : {logfile.resolve()}")
    logger.info(f"Device   : {DEVICE}")
    logger.info(f"PyTorch  : {torch.__version__}")
    return logger


# global logger — initialised in __main__, usable everywhere after that
log: logging.Logger = logging.getLogger("hamiltonian_ood")


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def vram_flush():
    """Release all cached (but unoccupied) VRAM back to the allocator."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def vram_stats() -> str:
    """One-line VRAM summary for the log."""
    if not torch.cuda.is_available():
        return "CPU mode"
    alloc   = torch.cuda.memory_allocated()  / 1024 ** 3
    reserved= torch.cuda.memory_reserved()   / 1024 ** 3
    total   = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    return f"VRAM  alloc={alloc:.2f} GB  reserved={reserved:.2f} GB  total={total:.2f} GB"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Backbone Definitions
# ══════════════════════════════════════════════════════════════════════════════

class _Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_ch, growth):
        super().__init__()
        inter = self.expansion * growth
        self.bn1  = nn.BatchNorm2d(in_ch);  self.conv1 = nn.Conv2d(in_ch,  inter, 1, bias=False)
        self.bn2  = nn.BatchNorm2d(inter);  self.conv2 = nn.Conv2d(inter, growth, 3, padding=1, bias=False)
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], 1)


class _Transition(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn   = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pool = nn.AvgPool2d(2)
    def forward(self, x):
        return self.pool(self.conv(F.relu(self.bn(x))))


class DenseNet(nn.Module):
    """DenseNet-BC with depth=100, growth_rate=12."""
    def __init__(self, num_classes=10, depth=100, growth_rate=12, reduction=0.5):
        super().__init__()
        assert (depth - 4) % 3 == 0
        n_blocks = (depth - 4) // 6
        num_ch   = 2 * growth_rate
        self.conv0 = nn.Conv2d(3, num_ch, 3, padding=1, bias=False)

        self.dense1, num_ch = self._make_dense(num_ch, growth_rate, n_blocks)
        num_ch_t = int(num_ch * reduction)
        self.trans1 = _Transition(num_ch, num_ch_t); num_ch = num_ch_t

        self.dense2, num_ch = self._make_dense(num_ch, growth_rate, n_blocks)
        num_ch_t = int(num_ch * reduction)
        self.trans2 = _Transition(num_ch, num_ch_t); num_ch = num_ch_t

        self.dense3, num_ch = self._make_dense(num_ch, growth_rate, n_blocks)
        self.bn_final = nn.BatchNorm2d(num_ch)
        self.fc       = nn.Linear(num_ch, num_classes)
        self.feat_dim = num_ch

        for m in self.modules():
            if isinstance(m, nn.Conv2d):        nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):      nn.init.constant_(m.bias, 0)

    def _make_dense(self, in_ch, growth, n):
        layers = []
        for _ in range(n):
            layers.append(_Bottleneck(in_ch, growth)); in_ch += growth
        return nn.Sequential(*layers), in_ch

    def forward(self, x, return_feat=False):
        out = self.conv0(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = F.relu(self.bn_final(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return out if return_feat else self.fc(out)


class _OpenOODBasicBlock(nn.Module):
    """BasicBlock matching OpenOOD's ``resnet18_32x32`` key layout."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class OpenOODResNet18(nn.Module):
    """Exact OpenOOD v1.5 ResNet-18 architecture for 32x32 inputs."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.feat_dim = 512

    def _make_layer(self, planes: int, blocks: int, stride: int):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(
                _OpenOODBasicBlock(self.in_planes, planes, block_stride)
            )
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _forward_feature_list(self, x):
        feature1 = F.relu(self.bn1(self.conv1(x)))
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.avgpool(self.layer4(feature4))
        return [feature1, feature2, feature3, feature4, feature5]

    def forward_features(self, x):
        return self._forward_feature_list(x)[-1].flatten(1)

    def forward_threshold(self, x, threshold):
        """OpenOOD-compatible clipped-feature forward pass."""
        feature = self._forward_feature_list(x)[-1].clip(max=threshold)
        return self.fc(feature.flatten(1))

    def intermediate_forward(self, x, layer_index: int):
        """Return a residual-stage feature map for RankFeat."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if layer_index == 1:
            return out
        out = self.layer2(out)
        if layer_index == 2:
            return out
        out = self.layer3(out)
        if layer_index == 3:
            return out
        out = self.layer4(out)
        if layer_index == 4:
            return out
        raise ValueError(f"layer_index must be in 1..4, got {layer_index}")

    def forward(
        self,
        x,
        return_feature: bool = False,
        return_feature_list: bool = False,
        return_feat: bool = False,
    ):
        """Support both the local and official OpenOOD feature APIs.

        ``return_feat=True`` is the historical local API and returns only the
        pooled feature tensor.  OpenOOD postprocessors use
        ``return_feature=True`` and expect ``(logits, feature)``; wrappers such
        as ReAct and Scale also pass these flags positionally.
        """
        feature_list = self._forward_feature_list(x)
        feature = feature_list[-1].flatten(1)
        logits = self.fc(feature)
        if return_feature:
            return logits, feature
        if return_feature_list:
            return logits, feature_list
        if return_feat:
            return feature
        return logits

    def get_fc(self):
        return (
            self.fc.weight.detach().cpu().numpy(),
            self.fc.bias.detach().cpu().numpy(),
        )

    def get_fc_layer(self):
        return self.fc


def canonical_model_name(model_name: str) -> str:
    aliases = {
        "resnet": "resnet18",
        "resnet18": "resnet18",
        "resnet-18": "resnet18",
        "densenet": "densenet100",
        "dn": "densenet100",
        "densenet100": "densenet100",
    }
    try:
        return aliases[model_name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown backbone {model_name!r}; choose resnet18 or densenet100"
        ) from exc


def get_backbone(model_name: str, num_classes: int) -> nn.Module:
    canonical = canonical_model_name(model_name)
    if canonical == "densenet100":
        return DenseNet(num_classes=num_classes)
    return OpenOODResNet18(num_classes=num_classes)


def get_resnet(num_classes):
    """Backward-compatible constructor; the architecture is ResNet-18."""

    return OpenOODResNet18(num_classes=num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Hamiltonian OOD Detector
# ══════════════════════════════════════════════════════════════════════════════

# Historical implementation retained for comparison with old logs.  New runs
# use the tested multi-potential HamiltonianDetector imported above.
class LegacyGaussianHamiltonianDetector(nn.Module):
    """
    Implements Sections 3.2–3.3.

    Training vs inference split
    ----------------------------
    The paper optimises σ via L_dynamics (Eq. 14), which requires evaluating
    the maximum affinity along a T-step Hamiltonian trajectory.  Differentiating
    through the trajectory w.r.t. σ is theoretically sound but practically
    intractable: with T=100 the unrolled graph exhausts VRAM on any consumer
    GPU, and detaching the trajectory (previous attempt) kills the gradient
    because the Gaussian wells are flat at initialisation.

    We therefore separate the two roles:

      Training  — optimise σ with a *static* loss evaluated directly at each
                  sample's position on the hypersphere.  The loss has the same
                  contrastive form as L_dynamics but uses the instantaneous
                  affinity U(q) rather than the trajectory maximum U*_c.
                  This gives a dense, well-conditioned gradient to log_sigma
                  without any simulation overhead.

                  L_static = −log[ exp(U_{y}(q)) / Σ_c exp(U_c(q)) ]

                  This is equivalent to L_dynamics in the limit where the
                  initial position q is already near the correct attractor —
                  which is enforced by the encoder's cross-entropy training.

      Inference — run the full leapfrog simulation (Eq. 12) with the learned
                  σ values to compute U*_c, then score/classify as in Eq. 13–15.
                  No gradients needed; runs under torch.no_grad().

    Memory knobs
    ------------
    anchor_chunk   Slice over K anchors to cap [B,C,ck,D] tensor size.
    sim_batch      Slice over the query batch during inference simulation.

    Recommended settings
    --------------------
    8 GB GPU : n_anchors=100, n_steps=100, anchor_chunk=5,  sim_batch=64
    4 GB GPU : n_anchors=50,  n_steps=50,  anchor_chunk=5,  sim_batch=32
    CPU only : n_anchors=20,  n_steps=20,  anchor_chunk=5,  sim_batch=32
    """

    def __init__(self, feat_dim: int, n_classes: int,
                 n_anchors_per_class: int = 20, n_steps: int = 20, dt: float = 0.1,
                 anchor_chunk: int = 5, sim_batch: int = 64):
        super().__init__()
        self.feat_dim     = feat_dim
        self.n_classes    = n_classes
        self.K            = n_anchors_per_class
        self.T            = n_steps
        self.dt           = dt
        self.anchor_chunk = anchor_chunk
        self.sim_batch    = sim_batch
        # log_sigma: one bandwidth per anchor, initialised to 0 (σ=1)
        self.log_sigma = nn.Parameter(torch.zeros(n_classes, n_anchors_per_class))
        self.register_buffer("anchors", torch.zeros(n_classes, n_anchors_per_class, feat_dim))
        self.register_buffer("masses",  torch.ones(n_classes,  n_anchors_per_class))

    # ── shared primitive: per-class affinity U_c(q) ──────────────────────────

    def _affinity_per_class(self, q: torch.Tensor,
                            detach_sigma: bool = False) -> torch.Tensor:
        """
        q            : [B, D]
        detach_sigma : if True, σ is treated as a constant (used inside
                       the leapfrog force calculation at inference).
        Returns      : [B, C]  — summed Gaussian affinity per class.
        """
        sigma  = (self.log_sigma.detach() if detach_sigma
                  else self.log_sigma).exp()          # [C, K]
        B      = q.shape[0]
        result = torch.zeros(B, self.n_classes, device=q.device, dtype=q.dtype)
        for ks in range(0, self.K, self.anchor_chunk):
            ke     = min(ks + self.anchor_chunk, self.K)
            anch_c = self.anchors[:, ks:ke, :]            # [C, ck, D]
            sig_c  = sigma[:, ks:ke]                      # [C, ck]
            mas_c  = self.masses[:, ks:ke]                # [C, ck]
            diff   = q[:, None, None, :] - anch_c[None]   # [B, C, ck, D]
            dist2  = (diff ** 2).sum(-1)                   # [B, C, ck]
            aff    = mas_c[None] * torch.exp(
                         -dist2 / (2 * sig_c[None] ** 2))
            result = result + aff.sum(-1)
        return result                                      # [B, C]

    # ── training forward: static contrastive loss on σ ───────────────────────

    def forward(self, q: torch.Tensor, labels: torch.Tensor):
        """
        Static loss — no simulation required.

        L_static = cross_entropy( U(q), y )   where U(q) is [B, C].

        Gradient w.r.t. log_sigma:
          ∂L/∂log_σ_{c,k} ∝  dist²_{c,k}(q) / σ²_{c,k}  · (p_c - 1_{c=y})
        This is dense (every σ gets signal from every sample) and O(BxCxK)
        memory — no trajectory storage at all.
        """
        U      = self._affinity_per_class(q)               # [B, C]; σ is live
        loss   = F.nll_loss(F.log_softmax(U, dim=-1), labels)
        return loss, U

    # ── inference: full leapfrog simulation ──────────────────────────────────

    @torch.no_grad()
    def _grad_U_inference(self, q: torch.Tensor) -> torch.Tensor:
        """Force −∇_q U(q) for leapfrog; σ is fixed (detached)."""
        sigma = self.log_sigma.detach().exp()
        grad  = torch.zeros_like(q)
        for ks in range(0, self.K, self.anchor_chunk):
            ke     = min(ks + self.anchor_chunk, self.K)
            anch_c = self.anchors[:, ks:ke, :]
            sig_c  = sigma[:, ks:ke]
            mas_c  = self.masses[:, ks:ke]
            diff   = q[:, None, None, :] - anch_c[None]
            dist2  = (diff ** 2).sum(-1, keepdim=True)
            w      = (mas_c[None, :, :, None]
                      / sig_c[None, :, :, None] ** 2
                      * torch.exp(-dist2 / (2 * sig_c[None, :, :, None] ** 2)))
            grad   = grad + (-(w * diff).sum(dim=[1, 2]))
        return grad

    @torch.no_grad()
    def _simulate_chunk(self, q0: torch.Tensor) -> torch.Tensor:
        """Leapfrog for a batch of ≤ sim_batch points (Eq. 12)."""
        q    = q0.clone()
        p    = torch.zeros_like(q)
        best = self._affinity_per_class(q, detach_sigma=True)
        dt   = self.dt
        for _ in range(self.T):
            p    = p - 0.5 * dt * self._grad_U_inference(q)
            q    = F.normalize(q + dt * p, dim=-1)
            p    = p - 0.5 * dt * self._grad_U_inference(q)
            best = torch.max(best, self._affinity_per_class(q, detach_sigma=True))
        return best    # [B, C]

    @torch.no_grad()
    def _simulate(self, q0: torch.Tensor) -> torch.Tensor:
        parts = []
        for s in range(0, q0.shape[0], self.sim_batch):
            parts.append(self._simulate_chunk(q0[s: s + self.sim_batch]))
        return torch.cat(parts, dim=0)

    # ── public scoring API ────────────────────────────────────────────────────

    @torch.no_grad()
    def score(self, q: torch.Tensor) -> torch.Tensor:
        """Max per-class trajectory affinity (Eq. 13). Higher = more in-dist."""
        return self._simulate(q).max(dim=-1).values

    @torch.no_grad()
    def predict(self, q: torch.Tensor) -> torch.Tensor:
        return self._simulate(q).argmax(dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════

def ckpt_path(in_dist: str, model_name: str, kind: str,
              potential: str | None = None, seed: int = SEED,
              detector_tag: str | None = None) -> Path:
    """kind: 'encoder' | 'detector'"""
    if kind == "detector" and potential:
        tag = f"_{detector_tag}" if detector_tag else ""
        return CKPT_DIR / f"{in_dist}_{model_name}_s{seed}_{potential}{tag}_{kind}.pt"
    return CKPT_DIR / f"{in_dist}_{model_name}_s{seed}_{kind}.pt"


def save_encoder(model, in_dist: str, model_name: str, epoch: int,
                 acc: float, args_dict: dict, seed: int = SEED):
    """Save encoder weights + training metadata."""
    path = ckpt_path(in_dist, model_name, "encoder", seed=seed)
    torch.save({
        "model_state":  model.state_dict(),
        "in_dist":      in_dist,
        "model_name":   model_name,
        "epoch":        epoch,
        "test_acc":     acc,
        "feat_dim":     model.feat_dim,
        "num_classes":  model.fc.out_features if hasattr(model, "fc") else None,
        "saved_at":     datetime.now().isoformat(),
        "args":         args_dict,
    }, path)
    log.info(f"Encoder checkpoint saved → {path}")


def load_encoder(model, in_dist: str, model_name: str, seed: int = SEED) -> dict | None:
    """
    Load encoder weights if a checkpoint exists.
    Returns the metadata dict on success, None if no checkpoint found.
    """
    path = ckpt_path(in_dist, model_name, "encoder", seed=seed)
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    log.info(f"Encoder loaded from {path}  "
             f"(epoch={ckpt['epoch']}, acc={ckpt['test_acc']:.2f}%)")
    return ckpt


def load_openood_encoder_direct(model, ckpt_path_str: str,
                                prefix: str = None,
                                key_remap: dict = None) -> dict:
    """
    Robustly load an OpenOOD pre-trained checkpoint into our model.

    Parameters
    ----------
    ckpt_path_str : path to the checkpoint file
    prefix        : if given, strip this prefix from all checkpoint keys
                    (e.g. "backbone." or "network.").  If None, auto-detect.
    key_remap     : dict mapping checkpoint key → model key for manual remaps
                    (e.g. {"linear.weight": "fc.weight"}).  Applied after
                    prefix stripping.
    """
    if not os.path.exists(ckpt_path_str):
        raise FileNotFoundError(f"OpenOOD checkpoint not found: {ckpt_path_str}")

    log.info(f"Loading OpenOOD checkpoint: {ckpt_path_str}")
    raw = torch.load(ckpt_path_str, map_location=DEVICE, weights_only=False)

    # ── unwrap outer container ────────────────────────────────────────────────
    if isinstance(raw, dict):
        for top_key in ("state_dict", "model", "net", "network",
                        "backbone", "params", "weights", "model_state_dict"):
            if top_key in raw and isinstance(raw[top_key], dict):
                state_dict = raw[top_key]
                log.info(f"  Unwrapped top-level key '{top_key}'")
                break
        else:
            state_dict = raw
    else:
        state_dict = raw

    # ── log ALL checkpoint keys (essential for diagnosis) ─────────────────────
    raw_keys = list(state_dict.keys())
    log.info(f"  Checkpoint has {len(raw_keys)} keys.  Full key list:")
    for k in raw_keys:
        v = state_dict[k]
        shape = tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__
        log.info(f"    CKPT  {k:<55}  {shape}")

    target_sd   = model.state_dict()
    target_keys = set(target_sd.keys())
    log.info(f"  Target model has {len(target_keys)} keys.")

    # ── prefix stripping ──────────────────────────────────────────────────────
    if prefix is not None:
        # User specified an explicit prefix — trust it
        stripped = {
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()
        }
        hits = len(set(stripped.keys()) & target_keys)
        log.info(f"  User-specified prefix '{prefix}': {hits}/{len(target_keys)} keys matched")
    else:
        # Auto-detect best prefix
        prefix_candidates = [
            "", "backbone.", "network.", "module.", "module.backbone.",
            "module.network.", "model.", "encoder.", "net.", "base_model.",
        ]
        best_stripped, best_hits, best_prefix = {}, -1, ""
        for pfx in prefix_candidates:
            candidate = {
                (k[len(pfx):] if k.startswith(pfx) else k): v
                for k, v in state_dict.items()
            }
            hits = len(set(candidate.keys()) & target_keys)
            if hits > best_hits:
                best_hits, best_stripped, best_prefix = hits, candidate, pfx

        # fallback: strip everything before first dot
        fallback = {(k.split(".", 1)[1] if "." in k else k): v
                    for k, v in state_dict.items()}
        if len(set(fallback.keys()) & target_keys) > best_hits:
            best_stripped = fallback
            best_prefix   = "<strip-first-segment>"
            best_hits     = len(set(fallback.keys()) & target_keys)

        stripped = best_stripped
        hits     = best_hits
        log.info(f"  Auto-detected prefix: '{best_prefix}'  →  "
                 f"{hits}/{len(target_keys)} keys matched")

    # ── apply manual key remaps ───────────────────────────────────────────────
    # Built-in common aliases (always applied)
    builtin_remap = {
        "linear.weight":     "fc.weight",
        "linear.bias":       "fc.bias",
        "classifier.weight": "fc.weight",
        "classifier.bias":   "fc.bias",
        "head.weight":       "fc.weight",
        "head.bias":         "fc.bias",
    }
    all_remaps = {**builtin_remap, **(key_remap or {})}
    for src, dst in all_remaps.items():
        if src in stripped and dst not in stripped:
            stripped[dst] = stripped.pop(src)
            log.info(f"  Remapped: '{src}' → '{dst}'")

    # ── load ──────────────────────────────────────────────────────────────────
    missing, unexpected = model.load_state_dict(stripped, strict=False)

    head_keys       = [k for k in missing if "fc" in k or "classifier" in k or "linear" in k]
    backbone_missing = [k for k in missing if k not in head_keys]

    loaded_count = len(target_keys) - len(missing)
    log.info(f"  Loaded {loaded_count}/{len(target_keys)} keys successfully.")

    if backbone_missing:
        log.warning(f"  {len(backbone_missing)} BACKBONE keys not loaded:")
        for k in backbone_missing:
            log.warning(f"    missing: {k}  expected shape: {tuple(target_sd[k].shape)}")
        log.warning("  → Run:  python inspect_checkpoint.py"
                    f" --ckpt \"{ckpt_path_str}\""
                    f" --in_dist <dataset> --model <model>")
        log.warning("    to get an exact --key_remap argument.")

    if head_keys:
        log.warning(f"  Classifier head NOT loaded: {head_keys}")
        log.warning(f"  ← THIS causes the ~52% accuracy.")
        log.warning(f"  Fix: the checkpoint probably stores the head under a different name.")
        log.warning(f"  Run inspect_checkpoint.py to find the correct name, then re-run with:")
        log.warning(f"    --key_remap <ckpt_head_name>:fc.weight <ckpt_head_name>:fc.bias")
        log.warning(f"  Or skip head loading and fine-tune it:")
        log.warning(f"    --finetune_head_epochs 10")

    if unexpected:
        log.debug(f"  {len(unexpected)} unexpected keys ignored.")

    log.info("OpenOOD loading complete.")
    return {"head_missing": len(head_keys) > 0,
            "backbone_missing": len(backbone_missing),
            "unexpected": len(unexpected),
            "loaded": loaded_count,
            "total": len(target_keys)}


def finetune_head_only(model, tr_loader, epochs: int = 10, lr: float = 0.01):
    """
    Fine-tune ONLY the classifier head (fc layer) with backbone frozen.

    Use when the OpenOOD checkpoint omits classifier weights.
    10 epochs typically recovers full accuracy because the backbone
    features are already well-structured.
    """
    log.info(f"  Fine-tuning classifier head only ({epochs} ep, backbone frozen) …")
    for name, param in model.named_parameters():
        param.requires_grad = ("fc" in name or "classifier" in name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Trainable params: {trainable:,}")

    opt   = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.train()
    for ep in range(1, epochs + 1):
        correct, total, ls = 0, 0, 0.0
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out  = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward(); opt.step()
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0); ls += loss.item() * y.size(0)
        sched.step()
        log.info(f"    Head ep {ep:2d}/{epochs}  "
                 f"loss={ls/total:.4f}  acc={correct/total*100:.2f}%")

    for param in model.parameters():   # unfreeze everything
        param.requires_grad = True


def save_detector(
    detector, in_dist: str, model_name: str, ham_epochs: int, *,
    seed: int, mass_mode: str, mass_normalization: str,
    bandwidth_loss: str, trajectory_train_steps: int,
    cache_namespace: str, ham_train_samples_per_class: int,
):
    """Save the full Hamiltonian detector state (anchors + σ)."""
    detector_tag = (
        f"mass-{mass_mode}-{mass_normalization}_loss-{bandwidth_loss}"
        f"-ts{trajectory_train_steps}_{cache_namespace}"
    )
    path = ckpt_path(
        in_dist, model_name, "detector", detector.potential, seed, detector_tag
    )
    checkpoint = detector.export_checkpoint()
    checkpoint["saved_at"] = datetime.now().isoformat()
    checkpoint["ham_epochs"] = ham_epochs
    checkpoint["experiment_metadata"] = {
        "seed": seed,
        "mass_mode": mass_mode,
        "mass_normalization": mass_normalization,
        "bandwidth_loss": bandwidth_loss,
        "trajectory_train_steps": trajectory_train_steps,
        "cache_namespace": cache_namespace,
        "ham_train_samples_per_class": ham_train_samples_per_class,
    }
    torch.save(checkpoint, path)
    log.info(f"Detector checkpoint saved → {path}")


def load_detector(in_dist: str, model_name: str, potential: str,
                  n_anchors: int, n_steps: int, dt: float,
                  candidate_k: int, sigma_init: float,
                  ham_epochs: int, *, seed: int, mass_mode: str,
                  mass_normalization: str, bandwidth_loss: str,
                  trajectory_train_steps: int, cache_namespace: str,
                  ham_train_samples_per_class: int) -> HamiltonianDetector | None:
    """Reconstruct and load a HamiltonianDetector from checkpoint if it exists."""
    detector_tag = (
        f"mass-{mass_mode}-{mass_normalization}_loss-{bandwidth_loss}"
        f"-ts{trajectory_train_steps}_{cache_namespace}"
    )
    path = ckpt_path(
        in_dist, model_name, "detector", potential, seed, detector_tag
    )
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=DEVICE)
    if ckpt.get("ham_epochs") != ham_epochs:
        log.warning(
            "Detector checkpoint bandwidth-training epoch mismatch; retraining."
        )
        return None
    metadata = ckpt.get("experiment_metadata", {})
    if metadata.get("ham_train_samples_per_class", 0) != ham_train_samples_per_class:
        log.warning("Detector checkpoint training-subset mismatch; retraining.")
        return None
    det = HamiltonianDetector.from_checkpoint(ckpt).to(DEVICE)
    requested = (n_anchors, n_steps, dt, candidate_k, sigma_init)
    stored = (det.K, det.T, det.dt, det.candidate_k, det.sigma_init)
    if stored != requested:
        log.warning(
            f"Detector checkpoint config mismatch; retraining. "
            f"stored={stored}, requested={requested}"
        )
        return None
    log.info(f"Detector loaded from {path}")
    return det


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Data helpers
# ══════════════════════════════════════════════════════════════════════════════

CIFAR10_MEAN  = (0.4914, 0.4822, 0.4465)
CIFAR10_STD   = (0.2470, 0.2435, 0.2616)
CIFAR10_STD_LEGACY = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)
SVHN_MEAN     = (0.4377, 0.4438, 0.4728)
SVHN_STD      = (0.1980, 0.2010, 0.1970)


def get_transforms(mean, std, train=True):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _load_dataset(cls, root, download_kwargs, transform):
    """Try local first, then download."""
    try:
        return cls(root, download=False, transform=transform, **download_kwargs)
    except Exception:
        log.info("Dataset not found locally — attempting download …")
        return cls(root, download=True, transform=transform, **download_kwargs)


def _nw():
    """Safe num_workers: 0 on Windows to avoid multiprocessing pickle errors."""
    return 0 if os.name == "nt" else 4


def get_in_dist_loaders(
    name: str,
    batch_size=128,
    seed: int = SEED,
    *,
    protocol: str = "openood",
    max_eval_samples: int = 0,
):
    name = name.lower()
    if name == "cifar10":
        mean = CIFAR10_MEAN
        std = CIFAR10_STD if protocol == "openood" else CIFAR10_STD_LEGACY
        n = 10
    elif name == "cifar100":
        mean, std, n = CIFAR100_MEAN, CIFAR100_STD, 100
    elif name == "svhn":
        if protocol == "openood":
            raise ValueError("SVHN is not an ID benchmark in this CIFAR OpenOOD runner")
        mean, std, n = SVHN_MEAN, SVHN_STD, 10
        svhn_root = DATA_ROOT / "svhn"; svhn_root.mkdir(parents=True, exist_ok=True)
        tr = _load_dataset(datasets.SVHN, str(svhn_root), {"split": "train"}, get_transforms(mean, std, True))
        te = _load_dataset(datasets.SVHN, str(svhn_root), {"split": "test"},  get_transforms(mean, std, False))
    else:
        raise ValueError(f"Unknown in-dist dataset: {name}")

    if protocol == "openood":
        tr = build_openood_dataset(
            DATA_ROOT, name, "train", get_transforms(mean, std, True)
        )
        te = build_openood_dataset(
            DATA_ROOT,
            name,
            "test",
            get_transforms(mean, std, False),
            max_samples=max_eval_samples,
        )
    elif protocol == "legacy" and name == "cifar10":
        tr = _load_dataset(
            datasets.CIFAR10,
            DATA_ROOT,
            {"train": True},
            get_transforms(mean, std, True),
        )
        te = _load_dataset(
            datasets.CIFAR10,
            DATA_ROOT,
            {"train": False},
            get_transforms(mean, std, False),
        )
    elif protocol == "legacy" and name == "cifar100":
        tr = _load_dataset(
            datasets.CIFAR100,
            DATA_ROOT,
            {"train": True},
            get_transforms(mean, std, True),
        )
        te = _load_dataset(
            datasets.CIFAR100,
            DATA_ROOT,
            {"train": False},
            get_transforms(mean, std, False),
        )
    elif protocol not in {"openood", "legacy"}:
        raise ValueError(f"Unknown data protocol: {protocol}")

    if protocol == "legacy" and max_eval_samples and len(te) > max_eval_samples:
        te = Subset(te, range(max_eval_samples))
    generator = torch.Generator().manual_seed(seed)
    tr_loader = DataLoader(
        tr, batch_size=batch_size, shuffle=True, generator=generator,
        num_workers=_nw(), pin_memory=True,
    )
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=_nw(), pin_memory=True)
    return tr_loader, te_loader, n, mean, std


def get_deterministic_feature_loader(dataset, batch_size: int, seed: int):
    """Use an RNG stream independent of encoder training for fair ablations."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed + 10_000),
        num_workers=_nw(),
        pin_memory=True,
    )


def get_ood_loader(
    name: str, mean, std, batch_size=128, n_samples=10000, *,
    seed: int = SEED, allow_proxy_data: bool = False,
    protocol: str = "openood", in_dist: str | None = None,
    max_eval_samples: int = 0,
):
    name = name.lower()
    tf   = get_transforms(mean, std, train=False)
    if protocol == "openood":
        if in_dist is None:
            raise ValueError("in_dist is required for the OpenOOD protocol")
        canonical = canonical_ood_name(name)
        dataset = build_openood_dataset(
            DATA_ROOT,
            in_dist,
            canonical,
            tf,
            max_samples=max_eval_samples,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=_nw(),
            pin_memory=True,
        )
    if protocol != "legacy":
        raise ValueError(f"Unknown data protocol: {protocol}")

    tf_mnist = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    if name == "svhn":
        svhn_root = DATA_ROOT / "svhn"; svhn_root.mkdir(parents=True, exist_ok=True)
        ds = _load_dataset(datasets.SVHN, str(svhn_root), {"split": "test"}, tf)
    elif name == "cifar10":
        ds = _load_dataset(datasets.CIFAR10, DATA_ROOT, {"train": False}, tf)
    elif name == "cifar100":
        ds = _load_dataset(datasets.CIFAR100, DATA_ROOT, {"train": False}, tf)
    elif name in ("tinyimagenet", "tiny_imagenet"):
        tiny_path = DATA_ROOT / "tiny-imagenet-200" / "val"
        if tiny_path.exists():
            ds = datasets.ImageFolder(str(tiny_path), transform=tf)
        elif allow_proxy_data:
            log.warning("TinyImageNet missing: using STL-10 because --allow_proxy_data was set.")
            ds = _load_dataset(datasets.STL10, DATA_ROOT, {"split": "unlabeled"},
                               transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32),
                                                   transforms.ToTensor(), transforms.Normalize(mean, std)]))
        else:
            raise FileNotFoundError(
                f"TinyImageNet is required at {tiny_path}. Proxy datasets are disabled."
            )
    elif name == "lsun":
        lsun_path = DATA_ROOT / "LSUN_resize"
        if lsun_path.exists():
            ds = datasets.ImageFolder(str(lsun_path), transform=tf)
        elif allow_proxy_data:
            log.warning("LSUN_resize missing: using FakeData because --allow_proxy_data was set.")
            ds = datasets.FakeData(size=n_samples, image_size=(3,32,32), num_classes=10, transform=tf)
        else:
            raise FileNotFoundError(
                f"LSUN_resize is required at {lsun_path}. Proxy datasets are disabled."
            )

    elif name == "mnist":
        mnist_root = DATA_ROOT / "MNIST"
        mnist_root.mkdir(parents=True, exist_ok=True)
        ds = datasets.MNIST(str(mnist_root), train=False, download=True, transform=tf_mnist)

    elif name in ("texture", "dtd"):
        dtd_root = DATA_ROOT / "dtd"
        dtd_root.mkdir(parents=True, exist_ok=True)
        ds = datasets.DTD(str(dtd_root), split="test", download=True, transform=tf)

    elif name in ("places365", "places"):
        places_path = DATA_ROOT / "Places365"
        if places_path.exists():
            ds = datasets.ImageFolder(str(places_path), transform=tf)
        elif allow_proxy_data:
            log.warning("Places365 missing: using FakeData because --allow_proxy_data was set.")
            ds = datasets.FakeData(size=n_samples, image_size=(3, 32, 32), num_classes=365, transform=tf)
        else:
            raise FileNotFoundError(
                f"Places365 is required at {places_path}. Proxy datasets are disabled."
            )
    else:
        raise ValueError(f"Unknown OOD dataset: {name}")
    limit = max_eval_samples or n_samples
    if len(ds) > limit:
        ds = Subset(ds, random.Random(seed).sample(range(len(ds)), limit))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=_nw(), pin_memory=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Training
# ══════════════════════════════════════════════════════════════════════════════

def train_encoder(model, loader, epochs, lr=0.1, wd=5e-4):
    opt   = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.train()
    for ep in range(1, epochs + 1):
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward(); opt.step()
            loss_sum += loss.item() * x.size(0)
            correct  += output.argmax(1).eq(y).sum().item()
            total    += x.size(0)
        sched.step()
        if ep % 10 == 0 or ep == 1:
            log.info(f"  Ep {ep:3d}/{epochs}  loss={loss_sum/total:.4f}  "
                     f"acc={correct/total*100:.2f}%")


@torch.no_grad()
def evaluate_accuracy(model, loader) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += model(x).argmax(1).eq(y).sum().item(); total += y.size(0)
    return correct / total * 100


@torch.no_grad()
def extract_features(
    model,
    loader,
    *,
    return_image_masses: bool = False,
    mean=None,
    std=None,
    mass_resolution: int = 0,
):
    model.eval()
    feats, labels, image_masses = [], [], []
    for x, y in loader:
        x = x.to(DEVICE)
        if return_image_masses:
            if mean is None or std is None:
                raise ValueError("mean and std are required for effective-rank mass")
            mean_tensor = x.new_tensor(mean).view(1, -1, 1, 1)
            std_tensor = x.new_tensor(std).view(1, -1, 1, 1)
            mass_images = (x * std_tensor + mean_tensor).clamp(0.0, 1.0)
            if mass_resolution > 0 and mass_images.shape[-2:] != (
                mass_resolution, mass_resolution
            ):
                mass_images = F.interpolate(
                    mass_images,
                    size=(mass_resolution, mass_resolution),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            image_masses.append(image_effective_rank(mass_images).cpu())
        f = model(x, return_feat=True)
        feats.append(F.normalize(f, dim=-1).cpu()); labels.append(y)
    outputs = (torch.cat(feats), torch.cat(labels))
    if return_image_masses:
        return (*outputs, torch.cat(image_masses))
    return outputs


def build_anchors_from_feats(
    detector,
    tr_feats: torch.Tensor,
    tr_labels: torch.Tensor,
    n_anchors_per_class: int,
    *,
    sample_masses: torch.Tensor | None = None,
    mass_normalization: str = "none",
    seed: int = SEED,
):
    """
    Fill detector.anchors from pre-extracted CPU features.
    This avoids running the encoder a second time and keeps GPU memory free.
    """
    log.info("  Building Hamiltonian anchors from cached features …")
    selected_mass_rows = []
    for c in range(detector.n_classes):
        rows = torch.where(tr_labels == c)[0]
        if not len(rows):
            raise RuntimeError(f"No training features for class {c}")
        if len(rows) < n_anchors_per_class:
            repeats = (n_anchors_per_class + len(rows) - 1) // len(rows)
            rows = rows.repeat(repeats)
        generator = torch.Generator().manual_seed(seed + c)
        chosen = rows[
            torch.randperm(len(rows), generator=generator)[:n_anchors_per_class]
        ]
        detector.anchors[c].copy_(
            F.normalize(tr_feats[chosen], dim=-1).to(detector.anchors.device)
        )
        selected_mass_rows.append(
            torch.ones(n_anchors_per_class)
            if sample_masses is None
            else sample_masses[chosen]
        )
    masses = normalize_anchor_masses(
        torch.stack(selected_mass_rows), mass_normalization
    )
    detector.masses.copy_(masses.to(detector.masses.device))
    detector.refresh_centroids()
    log.info(f"  Anchors built — {detector.n_classes} classes × {n_anchors_per_class} anchors.")


def class_balanced_feature_subset(
    features: torch.Tensor,
    labels: torch.Tensor,
    samples_per_class: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically cap bandwidth training without changing anchors."""

    if samples_per_class <= 0:
        return features, labels
    selected = []
    for class_id in sorted(labels.unique().tolist()):
        rows = torch.where(labels == class_id)[0]
        generator = torch.Generator().manual_seed(seed + int(class_id))
        order = torch.randperm(len(rows), generator=generator)
        selected.append(rows[order[: min(samples_per_class, len(rows))]])
    indices = torch.cat(selected)
    return features[indices], labels[indices]


def train_detector(
    detector,
    train_feats,
    train_labels,
    epochs=20,
    lr=0.001,
    *,
    bandwidth_loss: str = "static",
    trajectory_train_steps: int | None = None,
    seed: int = SEED,
):
    """
    Optimise log_sigma via L_dynamics (Eq. 14).

    We deliberately keep the DataLoader batch_size equal to sim_batch so each
    forward call processes exactly one simulation chunk -- this makes the step
    time predictable and avoids a hidden nested loop inside _simulate().
    """
    opt      = optim.Adam([detector.log_sigma], lr=lr)
    sched    = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ds       = torch.utils.data.TensorDataset(train_feats, train_labels)
    # batch_size == sim_batch so _simulate() never needs to chunk internally
    generator = torch.Generator().manual_seed(seed)
    ldr = DataLoader(
        ds, batch_size=detector.sim_batch, shuffle=True, generator=generator
    )
    n_batches = len(ldr)
    detector.train()
    for ep in range(1, epochs + 1):
        loss_sum, n = 0.0, 0
        t0 = time.time()
        for batch_idx, (q, y) in enumerate(ldr, 1):
            q, y = q.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss, _ = detector.training_loss(
                q,
                y,
                objective=bandwidth_loss,
                trajectory_steps=trajectory_train_steps,
            )
            loss.backward(); opt.step(); detector.clamp_bandwidths_()
            loss_sum += loss.item() * q.size(0); n += q.size(0)
            # print a dot every 10 batches so the user can see progress
            if batch_idx % 10 == 0 or batch_idx == n_batches:
                print(f"\r    σ-opt ep {ep:3d}/{epochs}  "
                      f"batch {batch_idx:4d}/{n_batches}  "
                      f"loss={loss_sum/n:.8f}  "
                      f"({time.time()-t0:.0f}s)", end="", flush=True)
        sched.step()
        print()  # newline after the progress line
        log.info(f"    σ-opt ep {ep:3d}/{epochs}  L_dynamics={loss_sum/n:.8f}  "
                 f"({time.time()-t0:.0f}s)")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Evaluation metrics
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_ood(in_scores, ood_scores) -> dict:
    """Match ``openood.evaluators.metrics.auc_and_fpr_recall`` exactly.

    OpenOOD v1.5 treats OOD as the positive class for FPR@95 and negates the
    ID confidence.  ``FPR95_IDTPR`` is also retained because many papers use
    the alternative convention (OOD accepted at 95% ID recall).
    """

    confidence = np.concatenate([in_scores, ood_scores])
    ood_indicator = np.concatenate(
        [np.zeros(len(in_scores), dtype=int), np.ones(len(ood_scores), dtype=int)]
    )
    fpr, tpr, _ = roc_curve(ood_indicator, -confidence)
    openood_fpr95 = fpr[np.argmax(tpr >= 0.95)] * 100
    auroc = auc(fpr, tpr) * 100

    precision_in, recall_in, _ = precision_recall_curve(
        1 - ood_indicator, confidence
    )
    precision_out, recall_out, _ = precision_recall_curve(
        ood_indicator, -confidence
    )
    aupr_in = auc(recall_in, precision_in) * 100
    aupr_out = auc(recall_out, precision_out) * 100

    id_threshold = np.percentile(in_scores, 5)
    conventional_fpr95 = np.mean(ood_scores >= id_threshold) * 100
    return {
        "AUROC": auroc,
        "AUPR": aupr_in,
        "AUPR_OUT": aupr_out,
        "FPR95": openood_fpr95,
        "FPR95_IDTPR": conventional_fpr95,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Full pipeline for one (in_dist, model) pair
# ══════════════════════════════════════════════════════════════════════════════

OOD_SETS = {
    "cifar10": ["cifar100", "tinyimagenet", "mnist", "svhn", "places365", "texture"],
    "cifar100": ["cifar10", "tinyimagenet", "mnist", "svhn", "places365", "texture"],
    "svhn": ["cifar10", "cifar100", "tinyimagenet", "mnist", "places365", "texture"],
}


def run_experiment(in_dist: str, model_name: str, epochs: int,
                   n_anchors: int = 20, ham_epochs: int = 20,
                   n_steps: int = 20, dt: float = 0.1,
                   tau: float = 0.5, anchor_chunk: int = 5,
                   sim_batch: int = 64, args_dict: dict = None,
                   force_retrain: bool = False,
                   openood_ckpt: str = None,
                   openood_prefix: str = None,
                   openood_key_remap: dict = None,
                   potential: str = "gaussian",
                   candidate_k: int = 0,
                   sigma_init: float = 1.0,
                   mass_mode: str = "uniform",
                   mass_normalization: str = "none",
                   mass_resolution: int = 0,
                   bandwidth_loss: str = "static",
                   trajectory_train_steps: int = 0,
                   seed: int = SEED,
                   allow_proxy_data: bool = False,
                   force_encoder_retrain: bool | None = None,
                   protocol: str = "openood",
                   encoder_source: str = "official",
                   openood_ckpt_root: str | Path = "openood_pretrained",
                   max_eval_samples: int = 0,
                   ham_train_samples_per_class: int = 0):

    in_dist = in_dist.lower()
    model_name = canonical_model_name(model_name)
    if protocol == "openood" and in_dist not in ("cifar10", "cifar100"):
        raise ValueError("The strict OpenOOD CIFAR protocol supports cifar10/cifar100")
    if encoder_source not in {"official", "self_trained"}:
        raise ValueError(f"Unknown encoder source: {encoder_source}")
    args_dict = args_dict or {}
    if encoder_source == "official" and protocol != "openood":
        raise ValueError("Official OpenOOD checkpoints require --protocol openood")
    if encoder_source == "official" and model_name != "resnet18":
        raise ValueError(
            "OpenOOD v1.5 publishes CIFAR checkpoints for ResNet-18 only. "
            "Use --encoder_source self_trained for DenseNet-BC-100."
        )

    resolved_openood_ckpt = openood_ckpt
    if encoder_source == "official" and resolved_openood_ckpt is None:
        resolved_openood_ckpt = str(
            discover_cifar_checkpoint(Path(openood_ckpt_root), in_dist, seed)
        )
    checkpoint_identity = "self_trained"
    if resolved_openood_ckpt is not None:
        checkpoint_path = Path(resolved_openood_ckpt).resolve()
        checkpoint_identity = f"sha256:{file_sha256(checkpoint_path)}"
    namespace_payload = {
        "protocol": protocol,
        "encoder_source": encoder_source,
        "checkpoint": checkpoint_identity,
        "ham_train_samples_per_class": ham_train_samples_per_class,
    }
    cache_namespace = hashlib.sha1(
        json.dumps(namespace_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    model_cache_name = f"{model_name}_{protocol}_{encoder_source}"

    log.info("")
    log.info("=" * 70)
    log.info(f"  In-dist: {in_dist.upper()}   Backbone: {model_name.upper()}"
             f"   Potential: {potential}")
    log.info("=" * 70)
    set_seed(seed)
    train_steps = n_steps if trajectory_train_steps == 0 else trajectory_train_steps
    if force_encoder_retrain is None:
        force_encoder_retrain = force_retrain

    # ── data ──────────────────────────────────────────────────────────────────
    tr_loader, te_loader, n_cls, mean, std = get_in_dist_loaders(
        in_dist,
        seed=seed,
        protocol=protocol,
        max_eval_samples=max_eval_samples,
    )
    log.info(
        f"  Protocol: {protocol}  |  encoder: {encoder_source}  |  "
        f"ID train/test: {len(tr_loader.dataset):,}/{len(te_loader.dataset):,}"
    )

    # ── backbone ──────────────────────────────────────────────────────────────
    model = get_backbone(model_name, n_cls).to(DEVICE)

    # ── encoder: load checkpoint or train from scratch ────────────────────────

    if resolved_openood_ckpt is not None:
        diag = load_openood_encoder_direct(
            model, resolved_openood_ckpt,
            prefix    = openood_prefix,
            key_remap = openood_key_remap,
        )
        if encoder_source == "official" and (
            diag["head_missing"] or diag["backbone_missing"]
        ):
            raise RuntimeError(
                "Official checkpoint did not load strictly; refusing to produce a paper result."
            )
        if diag["head_missing"] and args_dict.get("finetune_head_epochs", 0) > 0:
            ft_ep = args_dict["finetune_head_epochs"]
            log.info(f"[1/4] Head missing — fine-tuning head for {ft_ep} epochs …")
            finetune_head_only(model, tr_loader, epochs=ft_ep, lr=0.01)
        elif diag["head_missing"]:
            log.warning("[1/4] Head not loaded and --finetune_head_epochs not set.")
            log.warning("      Run: python inspect_checkpoint.py --ckpt <your.ckpt>"
                        f" --in_dist {in_dist} --model {model_name}")
            log.warning("      to get the exact --key_remap argument, or add"
                        " --finetune_head_epochs 10.")
        model.eval()
        acc = evaluate_accuracy(model, te_loader)
        log.info(f"[1/4] OpenOOD model  |  ID test accuracy = {acc:.2f}%"
                 + ("  ✓" if acc > 80 else
                    "  ← still low — run inspect_checkpoint.py for diagnosis"))
        minimum_accuracy = 90.0 if in_dist == "cifar10" else 65.0
        if encoder_source == "official" and acc < minimum_accuracy:
            raise RuntimeError(
                f"Official {in_dist} checkpoint accuracy {acc:.2f}% is below the "
                f"safety threshold {minimum_accuracy:.1f}%; stopping before OOD scoring."
            )
    else:
        # 这里保留你原有的自训/本地恢复逻辑
        enc_ckpt = (None if force_encoder_retrain
                    else load_encoder(model, in_dist, model_cache_name, seed))
        if enc_ckpt is not None:
            log.info(f"[1/4] Encoder resumed from checkpoint ...")
            acc = enc_ckpt["test_acc"]
        else:
            log.info(f"[1/4] Training {model_name} encoder for {epochs} epochs …")
            t0 = time.time()
            train_encoder(model, tr_loader, epochs)
            elapsed = (time.time() - t0) / 60
            acc = evaluate_accuracy(model, te_loader)
            log.info(f"  Training done in {elapsed:.1f} min  |  test acc = {acc:.2f}%")
            save_encoder(
                model, in_dist, model_cache_name, epochs, acc, args_dict or {}, seed
            )

    vram_flush()
    log.debug(f"  After encoder phase  — {vram_stats()}")

    # ── extract features ──────────────────────────────────────────────────────
    log.info("[2/4] Extracting normalized features …")
    set_seed(seed + 10_000)
    feature_loader = get_deterministic_feature_loader(
        tr_loader.dataset, tr_loader.batch_size, seed
    )
    if mass_mode == "effective_rank":
        tr_feats, tr_labels, tr_image_masses = extract_features(
            model,
            feature_loader,
            return_image_masses=True,
            mean=mean,
            std=std,
            mass_resolution=mass_resolution,
        )
    else:
        tr_feats, tr_labels = extract_features(model, feature_loader)
        tr_image_masses = None
    te_feats, _         = extract_features(model, te_loader)

    # move model to CPU to free VRAM before the leapfrog training loop
    model.cpu(); vram_flush()
    log.debug(f"  After feature extraction — {vram_stats()}")

    # ── Hamiltonian detector: load checkpoint or build+train ──────────────────
    log.info("[3/4] Building Hamiltonian potential field …")
    detector = (None if force_retrain else load_detector(
        in_dist, model_cache_name, potential, n_anchors, n_steps, dt,
        candidate_k, sigma_init, ham_epochs,
        seed=seed,
        mass_mode=mass_mode,
        mass_normalization=mass_normalization,
        bandwidth_loss=bandwidth_loss,
        trajectory_train_steps=train_steps,
        cache_namespace=cache_namespace,
        ham_train_samples_per_class=ham_train_samples_per_class,
    ))

    if detector is None:
        detector = HamiltonianDetector(
            model.feat_dim, n_cls,
            n_anchors_per_class = n_anchors,
            n_steps             = n_steps,
            dt                  = dt,
            anchor_chunk        = anchor_chunk,
            sim_batch           = sim_batch,
            potential           = potential,
            candidate_k         = candidate_k,
            sigma_init          = sigma_init,
        ).to(DEVICE)
        # anchors are built from CPU features — no GPU needed for this step
        build_anchors_from_feats(
            detector,
            tr_feats,
            tr_labels,
            n_anchors,
            sample_masses=tr_image_masses,
            mass_normalization=mass_normalization,
            seed=seed,
        )
        ham_feats, ham_labels = class_balanced_feature_subset(
            tr_feats,
            tr_labels,
            ham_train_samples_per_class,
            seed=seed + 20_000,
        )
        if len(ham_labels) != len(tr_labels):
            log.info(
                f"  Bandwidth training subset: {len(ham_labels):,}/"
                f"{len(tr_labels):,} class-balanced features"
            )
        train_detector(
            detector,
            ham_feats,
            ham_labels,
            epochs=ham_epochs,
            bandwidth_loss=bandwidth_loss,
            trajectory_train_steps=train_steps,
            seed=seed,
        )
        save_detector(
            detector,
            in_dist,
            model_cache_name,
            ham_epochs,
            seed=seed,
            mass_mode=mass_mode,
            mass_normalization=mass_normalization,
            bandwidth_loss=bandwidth_loss,
            trajectory_train_steps=train_steps,
            cache_namespace=cache_namespace,
            ham_train_samples_per_class=ham_train_samples_per_class,
        )
    else:
        log.info("  Detector loaded from checkpoint — skipping σ optimisation.")

    vram_flush()
    log.debug(f"  After detector training — {vram_stats()}")

    # move encoder back to GPU for feature extraction during OOD scoring
    model.to(DEVICE)

    # ── OOD evaluation ────────────────────────────────────────────────────────
    log.info("[4/4] Scoring OOD datasets …")
    detector.eval()
    in_scores   = detector.score(te_feats.to(DEVICE)).cpu().numpy()
    ood_names = (
        list(iter_ood_names(in_dist))
        if protocol == "openood"
        else OOD_SETS[in_dist]
    )
    row_results = {}
    ood_sample_counts = {}

    for ood_name in ood_names:
        ood_loader = get_ood_loader(
            ood_name,
            mean,
            std,
            seed=seed,
            allow_proxy_data=allow_proxy_data,
            protocol=protocol,
            in_dist=in_dist,
            max_eval_samples=max_eval_samples,
        )
        ood_sample_counts[ood_name] = len(ood_loader.dataset)
        ood_feats, _     = extract_features(model, ood_loader)
        ood_scores       = detector.score(ood_feats.to(DEVICE)).cpu().numpy()
        metrics          = evaluate_ood(in_scores, ood_scores)
        row_results[ood_name] = metrics
        log.info(f"  vs {ood_name:>12s}:  AUROC={metrics['AUROC']:5.1f}  "
                 f"AUPR={metrics['AUPR']:5.1f}  FPR95={metrics['FPR95']:5.1f}")

    metadata_checkpoint_path = (
        Path(resolved_openood_ckpt).resolve()
        if resolved_openood_ckpt is not None
        else ckpt_path(in_dist, model_cache_name, "encoder", seed=seed).resolve()
    )
    metadata = {
        "Protocol": protocol,
        "EncoderSource": encoder_source,
        "Checkpoint": str(metadata_checkpoint_path),
        "CheckpointSHA256": file_sha256(metadata_checkpoint_path),
        "IDAccuracy": acc,
        "IDTestSamples": len(te_loader.dataset),
        "OODSampleCounts": ood_sample_counts,
        "Anchors": n_anchors,
        "Steps": n_steps,
        "Dt": dt,
        "CandidateK": candidate_k,
        "SigmaInit": sigma_init,
        "HamEpochs": ham_epochs,
        "TrajectoryTrainSteps": train_steps,
        "MassResolution": mass_resolution,
        "HamTrainSamplesPerClass": ham_train_samples_per_class,
        "MaxEvalSamples": max_eval_samples,
        "CacheNamespace": cache_namespace,
    }
    return {"metrics": row_results, "metadata": metadata}


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Results persistence
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    ("cifar10", "resnet18"),
    ("cifar100", "resnet18"),
    ("cifar10", "densenet100"),
    ("cifar100", "densenet100"),
]

OOD_DISPLAY = {
    "svhn":         "SVHN",
    "tinyimagenet": "TinyImageNet",
    "tin":          "TinyImageNet",
    "lsun":         "LSUN",
    "cifar10":      "CIFAR-10",
    "cifar100":     "CIFAR-100",
    "mnist":        "MNIST",
    "texture":      "Texture",
    "places365":    "Places365",
}

CSV_PATH = RESULTS_DIR / "ood_results_cifar_v3.csv"
CSV_FIELDS = [
    "timestamp", "ExperimentID", "Seed", "In-dist", "Model", "Protocol",
    "EncoderSource", "Checkpoint", "CheckpointSHA256", "IDAccuracy", "IDTestSamples",
    "Potential", "MassMode", "MassNormalization", "MassResolution",
    "BandwidthLoss", "TrajectoryTrainSteps", "HamTrainSamplesPerClass",
    "Anchors", "Steps", "Dt", "CandidateK", "SigmaInit", "HamEpochs",
    "MaxEvalSamples", "OODGroup", "OOD", "OODSamples", "AUROC",
    "AUPR_IN", "AUPR_OUT", "FPR95", "FPR95_IDTPR",
]


def append_csv(all_results):
    """Upsert rows by experiment ID, making completed runs safe to repeat."""

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_rows = []
    for key, result in all_results.items():
        (
            ind, mname, potential, seed, mass_mode,
            mass_normalization, bandwidth_loss,
        ) = key
        metadata = result["metadata"]
        identity_payload = {
            "in_dist": ind,
            "model": mname,
            "potential": potential,
            "seed": seed,
            "mass_mode": mass_mode,
            "mass_normalization": mass_normalization,
            "bandwidth_loss": bandwidth_loss,
            **{
                field: metadata[field]
                for field in (
                    "Protocol", "EncoderSource", "MassResolution", "Anchors",
                    "Steps", "Dt", "CandidateK", "SigmaInit", "HamEpochs",
                    "TrajectoryTrainSteps", "HamTrainSamplesPerClass",
                    "MaxEvalSamples", "CacheNamespace",
                )
            },
        }
        experiment_id = hashlib.sha1(
            json.dumps(identity_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        for ood, metrics in result["metrics"].items():
            canonical_ood = canonical_ood_name(ood)
            group = next(
                (
                    group_name
                    for group_name, names in OOD_GROUPS.get(ind, {}).items()
                    if canonical_ood in names
                ),
                "legacy",
            )
            new_rows.append({
                "timestamp": ts,
                "ExperimentID": experiment_id,
                "Seed": seed,
                "In-dist": ind.upper(),
                "Model": mname,
                "Protocol": metadata["Protocol"],
                "EncoderSource": metadata["EncoderSource"],
                "Checkpoint": metadata["Checkpoint"],
                "CheckpointSHA256": metadata["CheckpointSHA256"],
                "IDAccuracy": f"{metadata['IDAccuracy']:.4f}",
                "IDTestSamples": metadata["IDTestSamples"],
                "Potential": potential,
                "MassMode": mass_mode,
                "MassNormalization": mass_normalization,
                "MassResolution": metadata["MassResolution"],
                "BandwidthLoss": bandwidth_loss,
                "TrajectoryTrainSteps": metadata["TrajectoryTrainSteps"],
                "HamTrainSamplesPerClass": metadata["HamTrainSamplesPerClass"],
                "Anchors": metadata["Anchors"],
                "Steps": metadata["Steps"],
                "Dt": metadata["Dt"],
                "CandidateK": metadata["CandidateK"],
                "SigmaInit": metadata["SigmaInit"],
                "HamEpochs": metadata["HamEpochs"],
                "MaxEvalSamples": metadata["MaxEvalSamples"],
                "OODGroup": group,
                "OOD": OOD_DISPLAY.get(ood, ood),
                "OODSamples": metadata["OODSampleCounts"][ood],
                "AUROC": f"{metrics['AUROC']:.4f}",
                "AUPR_IN": f"{metrics['AUPR']:.4f}",
                "AUPR_OUT": f"{metrics['AUPR_OUT']:.4f}",
                "FPR95": f"{metrics['FPR95']:.4f}",
                "FPR95_IDTPR": f"{metrics['FPR95_IDTPR']:.4f}",
            })

    rows_by_key = {}
    if CSV_PATH.is_file():
        with CSV_PATH.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != CSV_FIELDS:
                raise RuntimeError(
                    f"Unexpected CSV schema in {CSV_PATH}; move it aside before rerunning."
                )
            for row in reader:
                rows_by_key[(row["ExperimentID"], row["OOD"])] = row
    for row in new_rows:
        rows_by_key[(row["ExperimentID"], row["OOD"])] = row
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows_by_key.values())
    log.info(f"Results upserted into {CSV_PATH.resolve()}")


def print_latex(all_results):
    lines = [
        "",
        "% ── LaTeX Table ──────────────────────────────────────────────────",
        r"\begin{table}[t]\centering",
        r"\caption{OOD Detection Results (Hamiltonian Image Classification System)}",
        r"\begin{tabular}{lllcllrrr}\toprule",
        r"In-dist & OOD & Model & Potential & Seed & Setting & AUROC$\uparrow$ & AUPR$\uparrow$ & FPR95$\downarrow$ \\ \midrule",
    ]
    for key, result in all_results.items():
        (
            ind, mname, potential, seed, mass_mode,
            mass_normalization, bandwidth_loss,
        ) = key
        for i, (ood, m) in enumerate(result["metrics"].items()):
            ind_str   = ind.upper()        if i == 0 else ""
            model_str = mname.capitalize() if i == 0 else ""
            potential_str = potential if i == 0 else ""
            seed_str = str(seed) if i == 0 else ""
            setting_str = (
                f"{mass_mode}/{mass_normalization}/{bandwidth_loss}"
                if i == 0 else ""
            )
            lines.append(
                f"{ind_str} & {OOD_DISPLAY.get(ood,ood)} & {model_str} & {potential_str} & "
                f"{seed_str} & {setting_str} & "
                f"{m['AUROC']:.2f} & {m['AUPR']:.2f} & {m['FPR95']:.2f} \\\\"
            )
        lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    latex = "\n".join(lines)
    log.info(latex)

    latex_path = RESULTS_DIR / "ood_table.tex"
    latex_path.write_text(latex, encoding="utf-8")
    log.info(f"LaTeX table saved → {latex_path.resolve()}")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hamiltonian OOD Detection Experiments")
    parser.add_argument(
        "--run_all", action="store_true",
        help="Run CIFAR-10 and CIFAR-100; official mode uses ResNet-18 only",
    )
    parser.add_argument("--in_dist", default="cifar10", help="cifar10 | cifar100")
    parser.add_argument(
        "--model", default="resnet18", help="resnet18 | densenet100"
    )
    parser.add_argument(
        "--protocol", choices=("openood", "legacy"), default="openood",
        help="openood uses exact v1.5 image lists; legacy reproduces old sampling",
    )
    parser.add_argument(
        "--encoder_source", choices=("official", "self_trained"),
        default="official",
    )
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--results_dir", type=Path, default=Path("results/cifar"))
    parser.add_argument(
        "--openood_ckpt_root", type=Path, default=Path("openood_pretrained"),
        help="Root containing official CIFAR checkpoint archives after extraction",
    )
    parser.add_argument("--epochs",        type=int,   default=100, help="Encoder training epochs")
    parser.add_argument("--ham_epochs",    type=int,   default=25,  help="σ optimisation epochs")
    parser.add_argument("--n_anchors",     type=int,   default=80,  help="Anchors per class")
    parser.add_argument("--n_steps",       type=int,   default=120,  help="Leapfrog steps T")
    parser.add_argument("--dt",            type=float, default=0.05, help="Leapfrog step size Δt")
    parser.add_argument("--tau",           type=float, default=0.5, help="OOD rejection threshold τ")
    parser.add_argument("--anchor_chunk",  type=int,   default=10,
                        help="Anchors processed per chunk in affinity/grad (lower = less RAM, slower)")
    parser.add_argument("--sim_batch",     type=int,   default=64,
                        help="Query points simulated per leapfrog pass (lower = less RAM)")
    parser.add_argument("--potential", choices=POTENTIAL_NAMES, default="gaussian",
                        help="Radial potential used for this run")
    parser.add_argument("--potentials", nargs="+", choices=POTENTIAL_NAMES,
                        help="Run a fair ablation over several potentials")
    parser.add_argument("--candidate_k", type=int, default=0,
                        help="Classes retained per trajectory; 0 evaluates all classes exactly")
    parser.add_argument("--sigma_init", type=float, default=1.0,
                        help="Initial radial bandwidth on the unit hypersphere")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        help="Run repeated experiments; overrides --seed",
    )
    parser.add_argument("--mass_mode", choices=MASS_MODES, default="uniform")
    parser.add_argument(
        "--mass_normalization", choices=MASS_NORMALIZATIONS, default="none"
    )
    parser.add_argument(
        "--mass_resolution", type=int, default=0,
        help="Effective-rank image size; 0 keeps native 32x32 inputs",
    )
    parser.add_argument(
        "--bandwidth_loss", choices=BANDWIDTH_LOSSES, default="static"
    )
    parser.add_argument(
        "--trajectory_train_steps", type=int, default=0,
        help="Differentiable Eq.13-14 steps; 0 uses --n_steps",
    )
    parser.add_argument(
        "--allow_proxy_data", action="store_true",
        help="Opt in to legacy STL10/FakeData fallbacks (never use for paper tables)",
    )
    parser.add_argument(
        "--max_eval_samples", type=int, default=0,
        help="Smoke-test cap per ID/OOD list; 0 is mandatory for paper tables",
    )
    parser.add_argument(
        "--ham_train_samples_per_class", type=int, default=0,
        help="Class-balanced cap for bandwidth training; 0 uses all ID features",
    )
    parser.add_argument("--force_retrain", action="store_true",
                        help="Ignore existing checkpoints and retrain from scratch")
    parser.add_argument("--openood_ckpt",         default=None,
                        help="Path to official OpenOOD .ckpt/.pth file")
    parser.add_argument("--openood_prefix",        default=None,
                        help="Key prefix to strip from checkpoint keys "
                             "(e.g. 'backbone.' or 'network.'). "
                             "Auto-detected if not set. "
                             "Run inspect_checkpoint.py to find the right value.")
    parser.add_argument("--key_remap",             default=None, nargs="+",
                        help="Manual key remaps as 'ckpt_key:model_key' pairs. "
                             "Example: --key_remap linear.weight:fc.weight linear.bias:fc.bias "
                             "Run inspect_checkpoint.py to get the exact values.")
    parser.add_argument("--finetune_head_epochs",  type=int, default=0,
                        help="Fine-tune only the classifier head after loading an "
                             "OpenOOD checkpoint whose head weights are missing. "
                             "Set to 0 to skip. Default: 0.")
    args = parser.parse_args()

    if args.protocol == "openood" and args.allow_proxy_data:
        parser.error("--allow_proxy_data is incompatible with --protocol openood")
    if args.encoder_source == "self_trained" and args.openood_ckpt is not None:
        parser.error("--openood_ckpt requires --encoder_source official")
    if args.max_eval_samples < 0 or args.ham_train_samples_per_class < 0:
        parser.error("sample caps must be non-negative")

    configure_paths(args.data_root, args.results_dir)

    # parse --key_remap into a dict
    key_remap_dict = {}
    if args.key_remap:
        for pair in args.key_remap:
            parts = pair.split(":", 1)
            if len(parts) != 2:
                print(f"ERROR: --key_remap entry '{pair}' must be 'ckpt_key:model_key'")
                sys.exit(1)
            key_remap_dict[parts[0]] = parts[1]
        if key_remap_dict:
            print(f"[INFO] Manual key remaps: {key_remap_dict}")

    # ── initialise logging (must happen before anything else) ────────────────
    log = setup_logger()
    log.info(f"Args: {vars(args)}")

    if args.run_all:
        targets = [
            target
            for target in EXPERIMENTS
            if args.encoder_source != "official" or target[1] == "resnet18"
        ]
    else:
        targets = [(args.in_dist, canonical_model_name(args.model))]
    potentials  = args.potentials or [args.potential]
    seeds       = args.seeds or [args.seed]
    if args.openood_ckpt is not None and len(seeds) > 1:
        parser.error(
            "A single --openood_ckpt cannot represent multiple seeds; use "
            "--openood_ckpt_root for automatic seed discovery"
        )
    all_results = {}

    for seed in seeds:
      for (ind, mname) in targets:
        for potential_index, potential in enumerate(potentials):
            res = run_experiment(
                in_dist           = ind,
                model_name        = mname,
                epochs            = args.epochs,
                n_anchors         = args.n_anchors,
                ham_epochs        = args.ham_epochs,
                n_steps           = args.n_steps,
                dt                = args.dt,
                tau               = args.tau,
                anchor_chunk      = args.anchor_chunk,
                sim_batch         = args.sim_batch,
                args_dict         = vars(args),
                force_retrain     = args.force_retrain,
                openood_ckpt      = args.openood_ckpt,
                openood_prefix    = args.openood_prefix,
                openood_key_remap = key_remap_dict or None,
                potential         = potential,
                candidate_k       = args.candidate_k,
                sigma_init        = args.sigma_init,
                mass_mode         = args.mass_mode,
                mass_normalization = args.mass_normalization,
                mass_resolution   = args.mass_resolution,
                bandwidth_loss    = args.bandwidth_loss,
                trajectory_train_steps = args.trajectory_train_steps,
                seed              = seed,
                allow_proxy_data  = args.allow_proxy_data,
                force_encoder_retrain = args.force_retrain and potential_index == 0,
                protocol          = args.protocol,
                encoder_source    = args.encoder_source,
                openood_ckpt_root = args.openood_ckpt_root,
                max_eval_samples  = args.max_eval_samples,
                ham_train_samples_per_class = args.ham_train_samples_per_class,
            )
            all_results[
                (
                    ind, mname, potential, seed, args.mass_mode,
                    args.mass_normalization, args.bandwidth_loss,
                )
            ] = res

    append_csv(all_results)
    print_latex(all_results)

    log.info("All done.")
