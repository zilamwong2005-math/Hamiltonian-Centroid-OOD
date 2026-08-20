from pathlib import Path

import pandas as pd

from run_centroid_only_ablation import _metric_path, _normalise_saved


def test_centroid_only_metric_path_is_seed_scoped():
    path = _metric_path(Path("results"), "smoke", "cifar10", 2)
    assert path == Path(
        "results/smoke/cifar10/seed2/centroid_only.csv"
    )


def test_normalise_saved_adds_auditable_identifiers(tmp_path):
    path = tmp_path / "centroid_only.csv"
    pd.DataFrame(
        {"AUROC": [90.0], "FPR@95": [25.0]}, index=["nearood"]
    ).to_csv(path)
    frame = _normalise_saved(path, "imagenet200", 1)
    assert frame.loc[0, "Method"] == "centroid_only"
    assert frame.loc[0, "Target"] == "imagenet200"
    assert frame.loc[0, "Seed"] == 1
    assert frame.loc[0, "Dataset"] == "nearood"
