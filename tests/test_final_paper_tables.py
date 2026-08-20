from pathlib import Path

import pandas as pd

from build_final_paper_tables import (
    aggregate_method,
    load_cifar,
    load_imagenet,
    make_method_table,
)


def _cifar_rows():
    rows = []
    oods = {"near": ("a", "b"), "far": ("c", "d")}
    for benchmark in ("CIFAR10", "CIFAR100"):
        for potential in ("gaussian", "imq"):
            for seed in (0, 1, 2):
                for group, names in oods.items():
                    for index, name in enumerate(names):
                        rows.append({
                            "ExperimentID": f"{benchmark}-{potential}-{seed}",
                            "Seed": seed,
                            "In-dist": benchmark,
                            "Model": "resnet18",
                            "Protocol": "openood",
                            "EncoderSource": "official",
                            "Potential": potential,
                            "MassMode": "uniform",
                            "MassNormalization": "none",
                            "BandwidthLoss": "static",
                            "Anchors": 80,
                            "Steps": 120,
                            "Dt": 0.05,
                            "HamEpochs": 25,
                            "MaxEvalSamples": 0,
                            "OODGroup": group,
                            "OOD": name,
                            "IDAccuracy": 90 + seed,
                            "AUROC": 80 + seed + index,
                            "FPR95": 40 - seed + index,
                        })
    return rows


def _imagenet_rows(root: Path):
    for benchmark, steps in (("imagenet200", 10), ("imagenet1k", 1)):
        for seed in (0, 1, 2):
            output = root / benchmark / f"seed{seed}" / f"eval-full-{steps}"
            output.mkdir(parents=True)
            rows = []
            for potential in ("gaussian", "imq"):
                for dataset in ("nearood", "farood"):
                    rows.append({
                        "ID": benchmark,
                        "Potential": potential,
                        "Dataset": dataset,
                        "Seed": seed,
                        "MassMode": "uniform",
                        "MassNormalization": "none",
                        "BandwidthLoss": "static",
                        "TrajectoryTrainSteps": steps,
                        "PredictionSource": "backbone",
                        "ACC": 80.0,
                        "AUROC": 75.0 + seed,
                        "FPR@95": 55.0 - seed,
                    })
            pd.DataFrame(rows).to_csv(
                output / "openood_metrics_all_potentials.csv", index=False
            )


def test_builds_complete_cross_benchmark_table(tmp_path):
    cifar_path = tmp_path / "cifar.csv"
    pd.DataFrame(_cifar_rows()).to_csv(cifar_path, index=False)
    openood_root = tmp_path / "openood"
    _imagenet_rows(openood_root)

    expected = {0, 1, 2}
    cifar = load_cifar(cifar_path, expected, 80, 120, 0.05, 25)
    imagenet = load_imagenet(openood_root, expected)
    summary = aggregate_method(pd.concat([cifar, imagenet], ignore_index=True))
    table = make_method_table(summary)

    assert len(table) == 8
    assert set(table["Benchmark"].astype(str)) == {
        "CIFAR-10", "CIFAR-100", "ImageNet-200", "ImageNet-1K"
    }
    assert set(table["Potential"].astype(str)) == {"gaussian", "imq"}
    assert table["Near AUROC"].str.contains("±").all()
    assert table["Far FPR95"].str.contains("±").all()
