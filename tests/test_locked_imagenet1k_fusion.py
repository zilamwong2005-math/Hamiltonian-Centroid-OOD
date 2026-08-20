import json

import numpy as np

from run_locked_imagenet1k_fusion import (
    _fuse_scores,
    _load_locked_decision,
)


def test_fusion_uses_only_id_reference_statistics():
    id_msp = np.array([0.9, 0.8, 0.7, 0.6])
    ood_msp = np.array([0.2, 0.3])
    id_geometry = np.array([0.8, 0.7, 0.6, 0.5])
    ood_geometry = np.array([0.1, 0.2])
    reference = np.array([True, True, False, False])
    first_id, first_ood, statistics = _fuse_scores(
        id_msp, ood_msp, id_geometry, ood_geometry, reference, 0.8
    )
    second_id, second_ood, _ = _fuse_scores(
        id_msp, np.array([100.0, 200.0]),
        id_geometry, np.array([300.0, 400.0]), reference, 0.8
    )
    assert np.allclose(first_id, second_id)
    assert not np.allclose(first_ood, second_ood)
    assert statistics["calibration_count"] == 2


def test_locked_decision_is_hashed_and_audited(tmp_path):
    path = tmp_path / "decision.json"
    path.write_text(json.dumps({
        "protocol": "validation-only tune/holdout; no near/far test access",
        "selected_geometry_method": "centroid/max_cosine",
        "geometry_weight": 0.8,
        "recommended_for_one_locked_test_run": True,
    }), encoding="utf-8")
    decision, digest = _load_locked_decision(path)
    assert decision["geometry_weight"] == 0.8
    assert len(digest) == 64
