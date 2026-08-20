import importlib.util
import importlib.machinery
import sys
import types
from pathlib import Path

import torch


def _import_experiment_without_sklearn():
    metrics = types.ModuleType("sklearn.metrics")
    metrics.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", loader=None)
    metrics.auc = lambda *args, **kwargs: 0.5
    metrics.precision_recall_curve = lambda *args, **kwargs: ([], [], [])
    metrics.roc_curve = lambda *args, **kwargs: ([], [], [])
    sklearn = types.ModuleType("sklearn")
    sklearn.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
    sklearn.metrics = metrics
    sys.modules.setdefault("sklearn", sklearn)
    sys.modules.setdefault("sklearn.metrics", metrics)
    import ood_experiment

    return ood_experiment


def test_resnet18_matches_openood_checkpoint_key_layout():
    experiment = _import_experiment_without_sklearn()
    official_path = (
        Path(__file__).parents[1]
        / "OpenOOD"
        / "openood"
        / "networks"
        / "resnet18_32x32.py"
    )
    spec = importlib.util.spec_from_file_location("official_resnet18", official_path)
    official_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(official_module)

    ours = experiment.OpenOODResNet18(num_classes=100)
    official = official_module.ResNet18_32x32(num_classes=100)
    ours_shapes = {key: tuple(value.shape) for key, value in ours.state_dict().items()}
    official_shapes = {
        key: tuple(value.shape) for key, value in official.state_dict().items()
    }
    assert ours_shapes == official_shapes


def test_resnet18_feature_api():
    experiment = _import_experiment_without_sklearn()
    model = experiment.OpenOODResNet18(num_classes=10).eval()
    inputs = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert model(inputs).shape == (2, 10)
        assert model(inputs, return_feat=True).shape == (2, 512)
        logits, feature = model(inputs, return_feature=True)
        assert logits.shape == (2, 10)
        assert feature.shape == (2, 512)
        positional_logits, positional_feature = model(inputs, True, False)
        assert torch.equal(positional_logits, logits)
        assert torch.equal(positional_feature, feature)
        list_logits, feature_list = model(inputs, return_feature_list=True)
        assert list_logits.shape == (2, 10)
        assert len(feature_list) == 5
        assert feature_list[-1].shape == (2, 512, 1, 1)
        assert model.get_fc_layer() is model.fc


def test_class_balanced_feature_subset_is_deterministic():
    experiment = _import_experiment_without_sklearn()
    features = torch.arange(60, dtype=torch.float32).reshape(20, 3)
    labels = torch.tensor([0] * 10 + [1] * 10)
    first = experiment.class_balanced_feature_subset(
        features, labels, 3, seed=7
    )
    second = experiment.class_balanced_feature_subset(
        features, labels, 3, seed=7
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], torch.tensor([0, 0, 0, 1, 1, 1]))
