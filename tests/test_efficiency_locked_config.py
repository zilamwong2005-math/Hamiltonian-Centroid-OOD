import json
from pathlib import Path

from benchmark_inference_efficiency import (
    _state_megabytes,
    discover_locked_config,
)
from hamiltonian_detector import HamiltonianDetector


def test_discover_locked_transfer_config(tmp_path: Path):
    path = (
        tmp_path / "locked_centroid_transfer" / "full" / "cifar10"
        / "seed0" / "locked_centroid_msp.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({}), encoding="utf-8")
    assert discover_locked_config(tmp_path, "cifar10", 0) == path


def test_discover_locked_imagenet1k_config(tmp_path: Path):
    path = (
        tmp_path / "locked_imagenet1k_fusion" / "test" / "seed2"
        / "locked_centroid_msp.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({}), encoding="utf-8")
    assert discover_locked_config(tmp_path, "imagenet1k", 2) == path


def test_state_size_is_positive():
    detector = HamiltonianDetector(8, 3, n_anchors_per_class=2)
    assert _state_megabytes(detector) > 0
