from ood_experiment import OpenOODResNet18
from run_openood_baselines import (
    SUPPORTED_METHODS,
    _method_batch_size,
    _tuning_protocol,
    _valid_vim_dimensions,
    build_parser,
)

import torch


def test_vim_dimensions_exclude_empty_residual_subspace():
    assert _valid_vim_dimensions(512, [256, 1000]) == [256]
    assert _valid_vim_dimensions(2048, [256, 1000]) == [256, 1000]


def test_vim_dimensions_have_safe_fallback():
    assert _valid_vim_dimensions(128, [128, 256, 1000]) == [127]


def test_extended_methods_are_exposed():
    assert {"ash", "dice", "she", "rmds", "rankfeat"}.issubset(
        SUPPORTED_METHODS
    )


def test_rankfeat_uses_conservative_method_specific_batches():
    args = build_parser().parse_args([])
    assert _method_batch_size("cifar10", "rankfeat", args) == 64
    assert _method_batch_size("imagenet200", "rankfeat", args) == 32
    assert _method_batch_size("imagenet1k", "rankfeat", args) == 8
    assert _method_batch_size("imagenet1k", "msp", args) == 64


def test_ash_tuning_protocol_is_disclosed():
    assert "OOD validation" in _tuning_protocol("ash")
    assert "no test-set tuning" in _tuning_protocol("dice")


def test_cifar_adapter_exposes_rankfeat_stages():
    network = OpenOODResNet18(num_classes=10).eval()
    inputs = torch.randn(2, 3, 32, 32)
    assert network.intermediate_forward(inputs, 3).shape == (2, 256, 8, 8)
    assert network.intermediate_forward(inputs, 4).shape == (2, 512, 4, 4)
    assert network.forward_threshold(inputs, 1.0).shape == (2, 10)
