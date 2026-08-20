import torch

import run_centroid_sample_efficiency as module


def test_nested_setup_rows_are_class_balanced_and_deterministic(monkeypatch):
    monkeypatch.setattr(module, "EXPECTED_CLASSES", 3)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
    first = module._nested_setup_rows(labels, seed=0)
    repeated = module._nested_setup_rows(labels, seed=0)
    assert first.shape == (3, 3)
    assert torch.equal(first, repeated)
    for class_index in range(3):
        assert torch.equal(labels[first[class_index]], torch.full((3,), class_index))


def test_centroid_subsets_are_nested_and_normalized(monkeypatch):
    monkeypatch.setattr(module, "SAMPLE_COUNTS", (1, 2, 3))
    features = torch.tensor([
        [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
        [-1.0, 0.0], [0.0, -1.0], [-1.0, -1.0],
    ])
    rows = torch.tensor([[0, 1, 2], [3, 4, 5]])
    one = module._centroids_for_m(features, rows, 1)
    three = module._centroids_for_m(features, rows, 3)
    assert torch.allclose(one.norm(dim=1), torch.ones(2))
    assert torch.allclose(three.norm(dim=1), torch.ones(2))
    assert not torch.allclose(one, three)
