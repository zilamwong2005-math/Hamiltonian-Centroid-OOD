import numpy as np

import run_calibration_budget_sensitivity as module


def test_fixed_calibration_splits_are_nested_balanced_and_disjoint(monkeypatch):
    monkeypatch.setattr(module, "CLASS_COUNT", 3)
    labels = np.repeat(np.arange(3), module.IMAGES_PER_CLASS)
    evaluation, calibration = module._fixed_calibration_splits(labels)
    assert evaluation.sum() == 3
    for budget in module.BUDGETS:
        assert calibration[budget].sum() == budget * 3
        assert not np.any(evaluation & calibration[budget])
    for smaller, larger in zip(module.BUDGETS[:-1], module.BUDGETS[1:]):
        assert np.all(calibration[smaller] <= calibration[larger])


def test_fixed_split_is_deterministic(monkeypatch):
    monkeypatch.setattr(module, "CLASS_COUNT", 2)
    labels = np.repeat(np.arange(2), module.IMAGES_PER_CLASS)
    left = module._fixed_calibration_splits(labels, split_seed=0)
    right = module._fixed_calibration_splits(labels, split_seed=0)
    assert np.array_equal(left[0], right[0])
    for budget in module.BUDGETS:
        assert np.array_equal(left[1][budget], right[1][budget])
