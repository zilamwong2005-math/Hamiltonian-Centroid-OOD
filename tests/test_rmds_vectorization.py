from pathlib import Path

import torch

from Imagenet_ood_experiment import add_openood_to_path


add_openood_to_path(Path(__file__).resolve().parents[1] / "OpenOOD")

from openood.postprocessors.rmds_postprocessor import (  # noqa: E402
    precision_from_scatter,
    vectorized_relative_mahalanobis,
)


def _loop_reference(features, class_mean, precision,
                    whole_mean, whole_precision):
    centered = features - whole_mean.view(1, -1)
    background = -torch.diag(centered @ whole_precision @ centered.t())
    columns = []
    for mean in class_mean:
        residual = features - mean.view(1, -1)
        score = -torch.diag(residual @ precision @ residual.t())
        columns.append(score - background)
    return torch.stack(columns, dim=1)


def test_vectorized_rmds_matches_original_class_loop():
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(5, 6, generator=generator)
    class_mean = torch.randn(4, 6, generator=generator)
    whole_mean = torch.randn(6, generator=generator)
    matrix = torch.randn(6, 6, generator=generator)
    precision = matrix.t() @ matrix + torch.eye(6)
    matrix = torch.randn(6, 6, generator=generator)
    whole_precision = matrix.t() @ matrix + torch.eye(6)

    expected = _loop_reference(
        features, class_mean, precision, whole_mean, whole_precision
    )
    actual = vectorized_relative_mahalanobis(
        features, class_mean, precision, whole_mean, whole_precision
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_scatter_precision_is_symmetric():
    values = torch.tensor([
        [1.0, 2.0, -1.0],
        [2.0, -1.0, 0.5],
        [-1.0, 0.0, 3.0],
        [0.5, 1.0, 2.0],
    ])
    centered = values - values.mean(0)
    precision = precision_from_scatter(centered.t() @ centered, len(values))
    torch.testing.assert_close(precision, precision.t())
