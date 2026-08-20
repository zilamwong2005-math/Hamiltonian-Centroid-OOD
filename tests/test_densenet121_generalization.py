import torch
import torch.nn as nn

from run_densenet121_generalization import (
    TimmDenseNetAdapter,
    _accumulate_centroids,
    _max_centroid_cosine,
)


class _FakeTimmDenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(4, 3)

    def get_classifier(self):
        return self.classifier

    def forward_features(self, x):
        return x.mean(dim=(2, 3))

    def forward_head(self, feature, pre_logits=False):
        assert pre_logits
        return feature


class _FakeLegacyTimmDenseNet(nn.Module):
    """Mimic timm DenseNet releases without a forward_head method."""

    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(4, 3)

    def get_classifier(self):
        return self.classifier

    def forward_features(self, x):
        return x


def test_timm_adapter_supports_openood_feature_api():
    adapter = TimmDenseNetAdapter(_FakeTimmDenseNet())
    data = torch.randn(2, 4, 3, 3)
    logits, feature = adapter(data, return_feature=True)
    assert logits.shape == (2, 3)
    assert feature.shape == (2, 4)
    positional = adapter(data, True, False)
    assert len(positional) == 2


def test_timm_adapter_supports_legacy_densenet_without_forward_head():
    model = _FakeLegacyTimmDenseNet()
    adapter = TimmDenseNetAdapter(model)
    data = torch.randn(2, 4, 3, 3)
    logits, feature = adapter(data, return_feature=True)
    expected_feature = torch.relu(data).mean(dim=(2, 3))
    expected_logits = model.classifier(expected_feature)
    assert feature.shape == (2, 4)
    assert torch.allclose(feature, expected_feature)
    assert torch.allclose(logits, expected_logits)


def test_accumulate_centroids_returns_normalized_class_means():
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 0, 1])
    centroids = _accumulate_centroids(features, labels, 2)
    assert torch.allclose(centroids, torch.eye(2))


def test_max_centroid_cosine_is_invariant_to_feature_magnitude():
    centroids = torch.eye(2)
    features = torch.tensor([[3.0, 4.0], [1.0, -2.0]])
    original = _max_centroid_cosine(features, centroids)
    rescaled = _max_centroid_cosine(features * torch.tensor([[7.0], [0.25]]), centroids)
    assert torch.allclose(original, rescaled)
