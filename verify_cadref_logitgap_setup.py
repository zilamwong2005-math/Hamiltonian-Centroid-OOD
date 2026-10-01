"""Audit the fresh-server data and checkpoint matrix for CADRef/LogitGap."""

from __future__ import annotations

import argparse
from pathlib import Path


LIST_COUNTS = {
    "cifar10/train_cifar10.txt": 50_000,
    "cifar10/val_cifar10.txt": 1_000,
    "cifar10/test_cifar10.txt": 9_000,
    "cifar10/test_cifar100.txt": 9_000,
    "cifar10/test_tin.txt": 7_793,
    "cifar10/test_mnist.txt": 70_000,
    "cifar10/test_svhn.txt": 26_032,
    "cifar10/test_places365.txt": 35_195,
    "cifar10/test_texture.txt": 5_640,
    "cifar100/train_cifar100.txt": 50_000,
    "cifar100/val_cifar100.txt": 1_000,
    "cifar100/test_cifar100.txt": 9_000,
    "cifar100/test_cifar10.txt": 10_000,
    "cifar100/test_tin.txt": 6_526,
    "cifar100/test_mnist.txt": 70_000,
    "cifar100/test_svhn.txt": 26_032,
    "cifar100/test_places365.txt": 33_773,
    "cifar100/test_texture.txt": 5_640,
    "imagenet/train_imagenet.txt": 1_281_167,
    "imagenet/val_imagenet.txt": 5_000,
    "imagenet/test_imagenet.txt": 45_000,
    "imagenet/val_openimage_o.txt": 1_763,
    "imagenet/test_ssb_hard.txt": 49_000,
    "imagenet/test_ninco.txt": 5_879,
    "imagenet/test_inaturalist.txt": 10_000,
    "imagenet/test_textures.txt": 5_160,
    "imagenet/test_openimage_o.txt": 15_869,
    "imagenet200/train_imagenet200.txt": 258_951,
    "imagenet200/val_imagenet200.txt": 1_000,
    "imagenet200/test_imagenet200.txt": 9_000,
}

CLASSIC_DATASETS = {
    "cifar10", "cifar100", "tin", "mnist", "svhn", "places365", "texture"
}

CHECKPOINTS = {
    "cifar10": "cifar10_resnet18_32x32_base_e100_lr0.1_default",
    "cifar100": "cifar100_resnet18_32x32_base_e100_lr0.1_default",
    "imagenet200": "imagenet200_resnet18_224x224_base_e90_lr0.1_default",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("openood_pretrained")
    )
    parser.add_argument(
        "--full-path-check", action="store_true",
        help="Check every listed image instead of deterministic boundary samples.",
    )
    return parser


def image_path(data_root: Path, relative: str) -> Path:
    dataset = relative.split("/", 1)[0]
    category = "images_classic" if dataset in CLASSIC_DATASETS else "images_largescale"
    return data_root / category / relative


def main() -> None:
    args = build_parser().parse_args()
    data_root = args.data_root.resolve()
    list_root = data_root / "benchmark_imglist"
    failures: list[str] = []

    print("===== OpenOOD data-list audit =====")
    for relative, expected in LIST_COUNTS.items():
        path = list_root / relative
        if not path.is_file():
            failures.append(f"missing list: {path}")
            print(f"MISSING  {relative}")
            continue
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        if len(lines) != expected:
            failures.append(f"wrong count: {relative}={len(lines)}, expected {expected}")
        selected = lines if args.full_path_check else (
            lines[:3] + lines[len(lines) // 2:len(lines) // 2 + 3] + lines[-3:]
        )
        missing = []
        for line in selected:
            relative_image = line.split(maxsplit=1)[0]
            if not image_path(data_root, relative_image).is_file():
                missing.append(relative_image)
                if len(missing) == 3:
                    break
        if missing:
            failures.append(f"missing images for {relative}: {missing}")
        status = "OK" if len(lines) == expected and not missing else "FAIL"
        scope = "all" if args.full_path_check else "sample"
        print(f"{status:4s}  {relative:36s} {len(lines):8d}/{expected:8d}  paths={scope}")

    print("\n===== OpenOOD checkpoint audit =====")
    checkpoint_root = args.checkpoint_root.resolve()
    for benchmark, directory in CHECKPOINTS.items():
        for seed in (0, 1, 2):
            path = checkpoint_root / directory / f"s{seed}" / "best.ckpt"
            ok = path.is_file() and path.stat().st_size > 1_000_000
            print(f"{'OK' if ok else 'FAIL':4s}  {benchmark:11s} seed={seed}  {path}")
            if not ok:
                failures.append(f"missing checkpoint: {path}")

    if failures:
        print("\n===== Problems =====")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(f"Setup audit failed with {len(failures)} problem(s).")
    print("\nFresh-server CADRef/LogitGap prerequisites passed.")


if __name__ == "__main__":
    main()
