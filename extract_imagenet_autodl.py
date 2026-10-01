"""Resumably extract AutoDL's public ILSVRC-2012 archives for OpenOOD.

The OpenOOD image lists expect class directories for ImageNet training images
and a flat validation directory.  AutoDL exposes the standard outer training
tar (containing one tar per synset) and the flat validation tar.
"""

from __future__ import annotations

import argparse
import re
import tarfile
from pathlib import Path


TRAIN_IMAGES = 1_281_167
VAL_IMAGES = 50_000
TRAIN_CLASSES = 1_000
SYNSET = re.compile(r"n\d{8}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path,
        default=Path("/root/autodl-pub/ImageNet/ILSVRC2012"),
    )
    parser.add_argument(
        "--destination-root", type=Path,
        default=Path("data/images_largescale/imagenet_1k"),
    )
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--val-only", action="store_true")
    return parser


def _safe_members(members, destination: Path):
    root = destination.resolve()
    for member in members:
        if member.issym() or member.islnk():
            raise RuntimeError(f"Refusing linked tar member: {member.name}")
        target = (destination / member.name).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"Unsafe tar member: {member.name}") from error
        yield member


def extract_train(source: Path, destination: Path, markers: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)
    markers.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source, mode="r:") as outer:
        members = [member for member in outer.getmembers() if member.isfile()]
        if len(members) != TRAIN_CLASSES:
            raise RuntimeError(
                f"Expected {TRAIN_CLASSES} class tars; found {len(members)}"
            )
        for index, member in enumerate(members, 1):
            class_name = Path(member.name).stem
            if not SYNSET.fullmatch(class_name):
                raise RuntimeError(f"Unexpected ImageNet class tar: {member.name}")
            marker = markers / f"{class_name}.complete"
            class_dir = destination / class_name
            if marker.is_file():
                print(f"[skip {index:04d}/{TRAIN_CLASSES}] {class_name}", flush=True)
                continue
            stream = outer.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Cannot read nested tar: {member.name}")
            class_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=stream, mode="r:") as nested:
                nested.extractall(
                    class_dir,
                    members=_safe_members(nested.getmembers(), class_dir),
                )
            count = sum(1 for path in class_dir.iterdir() if path.is_file())
            if count <= 0:
                raise RuntimeError(f"No images extracted for {class_name}")
            marker.write_text(f"images={count}\n", encoding="utf-8")
            print(
                f"[train {index:04d}/{TRAIN_CLASSES}] {class_name}: {count} images",
                flush=True,
            )


def extract_val(source: Path, destination: Path, marker: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)
    if marker.is_file():
        print("[skip] ImageNet validation archive already marked complete")
        return
    with tarfile.open(source, mode="r:") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if len(members) != VAL_IMAGES:
            raise RuntimeError(
                f"Expected {VAL_IMAGES} validation images; found {len(members)}"
            )
        for index, member in enumerate(_safe_members(members, destination), 1):
            target = destination / member.name
            if not target.is_file():
                archive.extract(member, destination)
            if index % 5_000 == 0:
                print(f"[val {index:,}/{VAL_IMAGES:,}]", flush=True)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"images={VAL_IMAGES}\n", encoding="utf-8")


def verify(destination: Path, *, check_train: bool, check_val: bool) -> None:
    train = destination / "train"
    val = destination / "val"
    print("===== ImageNet extraction audit =====")
    if check_train:
        if not train.is_dir():
            raise RuntimeError(f"Missing ImageNet training directory: {train}")
        classes = sum(1 for path in train.iterdir() if path.is_dir())
        train_images = sum(1 for path in train.rglob("*") if path.is_file())
        print(f"classes:      {classes:,}/{TRAIN_CLASSES:,}")
        print(f"train images: {train_images:,}/{TRAIN_IMAGES:,}")
        if (classes, train_images) != (TRAIN_CLASSES, TRAIN_IMAGES):
            raise RuntimeError("ImageNet training extraction is incomplete")
    if check_val:
        if not val.is_dir():
            raise RuntimeError(f"Missing ImageNet validation directory: {val}")
        val_images = sum(1 for path in val.iterdir() if path.is_file())
        print(f"val images:   {val_images:,}/{VAL_IMAGES:,}")
        if val_images != VAL_IMAGES:
            raise RuntimeError("ImageNet validation extraction is incomplete")
    print("ImageNet extraction complete.")


def main() -> None:
    args = build_parser().parse_args()
    if args.train_only and args.val_only:
        raise ValueError("--train-only and --val-only are mutually exclusive")
    source = args.source_root.resolve()
    destination = args.destination_root.resolve()
    markers = destination / ".extraction"
    if not args.val_only:
        extract_train(
            source / "ILSVRC2012_img_train.tar",
            destination / "train",
            markers / "train",
        )
    if not args.train_only:
        extract_val(
            source / "ILSVRC2012_img_val.tar",
            destination / "val",
            markers / "val.complete",
        )
    verify(
        destination,
        check_train=not args.val_only,
        check_val=not args.train_only,
    )


if __name__ == "__main__":
    main()
