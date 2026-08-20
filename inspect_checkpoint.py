"""
OpenOOD Checkpoint Inspector
=============================
Run this BEFORE ood_experiment.py to diagnose key mismatches.

Usage
-----
python inspect_checkpoint.py --ckpt path/to/checkpoint.ckpt --in_dist cifar10 --model resnet18

Output
------
• Full list of checkpoint keys with shapes
• Full list of model keys with shapes
• Side-by-side match report
• Exact --key_remap argument to paste into ood_experiment.py
"""

import argparse
import sys
import os
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from ood_experiment import get_backbone


def load_raw(ckpt_path: str) -> dict:
    """Load checkpoint and unwrap all known outer containers."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        return {"<root>": raw}

    # Try every common top-level key
    for key in ("state_dict", "model", "net", "network", "backbone",
                "params", "weights", "model_state_dict"):
        if key in raw and isinstance(raw[key], dict):
            print(f"[INFO] Unwrapped top-level key: '{key}'")
            return raw[key]

    # If it looks flat (values are tensors), return as-is
    if all(isinstance(v, torch.Tensor) for v in raw.values()):
        print("[INFO] Checkpoint is already a flat state dict.")
        return raw

    # Otherwise print what keys are at the top level and return raw
    print(f"[INFO] Top-level keys: {list(raw.keys())}")
    return raw


def strip_prefix(state_dict: dict, prefix: str) -> dict:
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }


def count_matches(state_dict: dict, target_keys: set) -> int:
    return len(set(state_dict.keys()) & target_keys)


def build_model(in_dist: str, model_name: str) -> nn.Module:
    n_cls = {"cifar10": 10, "cifar100": 100, "svhn": 10}[in_dist.lower()]
    return get_backbone(model_name, n_cls)


def main():
    parser = argparse.ArgumentParser(description="Inspect OpenOOD checkpoint key structure")
    parser.add_argument("--ckpt",     required=True,  help="Path to .ckpt / .pth file")
    parser.add_argument("--in_dist",  default="cifar10", help="cifar10 | cifar100 | svhn")
    parser.add_argument(
        "--model", default="resnet18", help="resnet18 | densenet100"
    )
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"ERROR: file not found: {args.ckpt}"); sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  Checkpoint Inspector")
    print(f"  File   : {args.ckpt}")
    print(f"  Model  : {args.model.upper()} / {args.in_dist.upper()}")
    print(f"{'='*70}\n")

    # ── 1. Load checkpoint ────────────────────────────────────────────────────
    raw_sd = load_raw(args.ckpt)
    ckpt_keys = list(raw_sd.keys())

    print(f"Checkpoint has {len(ckpt_keys)} keys.\n")
    print("── All checkpoint keys ──────────────────────────────────────────────")
    for k in ckpt_keys:
        v = raw_sd[k]
        shape = tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__
        print(f"  {k:<60}  {shape}")

    # ── 2. Build target model ─────────────────────────────────────────────────
    model = build_model(args.in_dist, args.model)
    target_sd   = model.state_dict()
    target_keys = set(target_sd.keys())

    print(f"\n── Target model keys ({len(target_keys)} total) ─────────────────────────")
    for k, v in target_sd.items():
        print(f"  {k:<60}  {tuple(v.shape)}")

    # ── 3. Try all prefix strategies ─────────────────────────────────────────
    prefix_candidates = [
        "",
        "backbone.",
        "network.",
        "module.",
        "module.backbone.",
        "module.network.",
        "model.",
        "encoder.",
        "net.",
        "base_model.",
        "feature_extractor.",
    ]

    print("\n── Prefix stripping results ─────────────────────────────────────────")
    results = []
    for prefix in prefix_candidates:
        stripped = strip_prefix(raw_sd, prefix)
        hits     = count_matches(stripped, target_keys)
        results.append((hits, prefix, stripped))
        print(f"  prefix='{prefix:<25}'  matched {hits:3d}/{len(target_keys)} keys")

    # Also try stripping first segment
    fallback = {(k.split(".", 1)[1] if "." in k else k): v for k, v in raw_sd.items()}
    hits     = count_matches(fallback, target_keys)
    results.append((hits, "<strip-first-segment>", fallback))
    print(f"  prefix='<strip-first-segment>'  matched {hits:3d}/{len(target_keys)} keys")

    results.sort(key=lambda x: x[0], reverse=True)
    best_hits, best_prefix, best_sd = results[0]

    print(f"\n  ✓ Best: prefix='{best_prefix}'  →  {best_hits}/{len(target_keys)} matched")

    # ── 4. Detailed match / mismatch for best strategy ────────────────────────
    matched   = set(best_sd.keys()) & target_keys
    missing   = target_keys - set(best_sd.keys())
    extra     = set(best_sd.keys()) - target_keys

    print(f"\n── Detailed report (best prefix='{best_prefix}') ────────────────────")
    print(f"  Matched   : {len(matched)}")
    print(f"  Missing   : {len(missing)}  ← these will be randomly initialised")
    print(f"  Extra/unk : {len(extra)}   ← in checkpoint but not in model")

    if missing:
        print("\n  Missing keys (need remapping or fine-tuning):")
        for k in sorted(missing):
            print(f"    MODEL  ← ???  :  {k}  {tuple(target_sd[k].shape)}")

    if extra:
        print("\n  Extra checkpoint keys (possible remap sources):")
        for k in sorted(extra):
            v = best_sd[k]
            shape = tuple(v.shape) if isinstance(v, torch.Tensor) else "?"
            print(f"    CKPT   → ???  :  {k}  {shape}")

    # ── 5. Auto-suggest remaps ────────────────────────────────────────────────
    print("\n── Auto-suggested key remaps ────────────────────────────────────────")
    suggestions = []

    # Match by shape: for each missing model key, find extra checkpoint keys with same shape
    missing_by_shape: dict[tuple, list] = {}
    for k in sorted(missing):
        shape = tuple(target_sd[k].shape)
        missing_by_shape.setdefault(shape, []).append(k)

    extra_by_shape: dict[tuple, list] = {}
    for k in sorted(extra):
        v = best_sd[k]
        if isinstance(v, torch.Tensor):
            extra_by_shape.setdefault(tuple(v.shape), []).append(k)

    remap_args = []
    for shape, model_keys in missing_by_shape.items():
        ckpt_keys_match = extra_by_shape.get(shape, [])
        for mk, ck in zip(model_keys, ckpt_keys_match):
            print(f"  '{ck}'  →  '{mk}'   (shape {shape})")
            suggestions.append((ck, mk))
            remap_args.append(f"{ck}:{mk}")

    if not suggestions:
        print("  (No automatic shape-based remaps found — shapes don't match)")

    # ── 6. Print exact command to use ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Recommended command:")
    print(f"{'='*70}")

    cmd = (f"python ood_experiment.py"
           f" --in_dist {args.in_dist}"
           f" --model {args.model}"
           f" --encoder_source official"
           f" --openood_ckpt \"{args.ckpt}\"")

    if best_prefix and best_prefix != "<strip-first-segment>":
        cmd += f" --openood_prefix \"{best_prefix}\""

    if remap_args:
        cmd += f" --key_remap {' '.join(remap_args)}"

    if missing and not remap_args:
        cmd += " --finetune_head_epochs 10"
        print("  (Head keys could not be remapped automatically →")
        print("   adding --finetune_head_epochs 10 to fine-tune head after loading)\n")

    print(f"\n  {cmd}\n")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
