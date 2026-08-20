import torch
from torch.utils.data import DataLoader, TensorDataset

from Imagenet_ood_experiment import _limit_loader


def test_smoke_loader_does_not_keep_persistent_workers():
    dataset = TensorDataset(torch.arange(10))
    source = DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,
        persistent_workers=True,
    )

    limited = _limit_loader(source, max_samples=3)

    assert len(limited.dataset) == 3
    assert limited.num_workers == 2
    assert limited.persistent_workers is False

