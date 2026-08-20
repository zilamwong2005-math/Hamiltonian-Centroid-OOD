from pathlib import Path

import pytest

from run_locked_centroid_transfer import _find_cifar_detector


def test_find_cifar_detector_requires_one_exact_t0_checkpoint(tmp_path: Path):
    directory = tmp_path / "cifar_trajectory" / "checkpoints"
    directory.mkdir(parents=True)
    target = directory / (
        "cifar10_resnet18_openood_official_s0_gaussian_"
        "mass-uniform-none_loss-static-ts0_abc123_detector.pt"
    )
    target.touch()
    assert _find_cifar_detector(tmp_path, "cifar10", 0) == target


def test_find_cifar_detector_rejects_missing_checkpoint(tmp_path: Path):
    with pytest.raises(RuntimeError, match="Expected one"):
        _find_cifar_detector(tmp_path, "cifar100", 2)
