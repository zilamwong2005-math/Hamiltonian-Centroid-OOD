from pathlib import Path

import pytest
from PIL import Image

from openood_cifar import (
    OpenOODImglistDataset,
    canonical_ood_name,
    discover_cifar_checkpoint,
    imglist_path,
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(path)


def test_imglist_dataset_reads_exact_order_and_labels(tmp_path):
    image_root = tmp_path / "images_classic"
    _write_image(image_root / "cifar10" / "train" / "airplane" / "0001.png")
    _write_image(image_root / "cifar10" / "train" / "cat" / "0002.png")
    imglist = tmp_path / "train.txt"
    imglist.write_text(
        "cifar10/train/airplane/0001.png 0\n"
        "cifar10/train/cat/0002.png 3\n",
        encoding="utf-8",
    )

    dataset = OpenOODImglistDataset(image_root, imglist, transform=None)
    assert len(dataset) == 2
    assert dataset[0][1] == 0
    assert dataset[1][1] == 3


def test_imglist_dataset_rejects_parent_traversal(tmp_path):
    image_root = tmp_path / "images_classic"
    image_root.mkdir()
    imglist = tmp_path / "unsafe.txt"
    imglist.write_text("../secret.png -1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe image path"):
        OpenOODImglistDataset(image_root, imglist, transform=None)


def test_standard_aliases_and_list_path(tmp_path):
    assert canonical_ood_name("TinyImageNet") == "tin"
    assert canonical_ood_name("DTD") == "texture"
    assert imglist_path(tmp_path, "cifar10", "tin") == (
        tmp_path / "benchmark_imglist" / "cifar10" / "test_tin.txt"
    )


def test_discover_cifar_checkpoint(tmp_path):
    checkpoint = (
        tmp_path
        / "cifar10_resnet18_32x32_base_e100_lr0.1_default"
        / "s1"
        / "best.ckpt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    assert discover_cifar_checkpoint(tmp_path, "cifar10", 1) == checkpoint
    with pytest.raises(FileNotFoundError):
        discover_cifar_checkpoint(tmp_path, "cifar10", 0)

