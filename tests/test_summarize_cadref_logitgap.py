import pandas as pd

from summarize_cadref_logitgap import (
    EXPECTED_RUNS,
    METHODS,
    TARGETS,
    load_and_validate,
    paper_table,
    summarise,
)


def _formal_frame():
    rows = []
    for target in TARGETS:
        row_count = 8 if target in {"cifar10", "cifar100"} else 7
        datasets = ["nearood", "farood"] + [
            f"detail_{index}" for index in range(row_count - 2)
        ]
        for method in METHODS:
            for seed in range(EXPECTED_RUNS[target]):
                for dataset in datasets:
                    rows.append(
                        {
                            "Target": target,
                            "Method": method,
                            "Seed": seed,
                            "Dataset": dataset,
                            "FPR@95": 20.0 + seed,
                            "AUROC": 90.0 - seed,
                            "AUPR_IN": 91.0,
                            "AUPR_OUT": 89.0,
                            "ACC": 80.0,
                        }
                    )
    return pd.DataFrame(rows)


def test_formal_matrix_validation_and_summary(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    _formal_frame().to_csv(
        root / "cadref_logitgap_full_all_runs.csv", index=False
    )
    frame = load_and_validate(root, "full")
    raw, summary = summarise(frame)
    assert len(frame) == 166
    assert len(raw) == 44
    assert len(summary) == 20
    assert set(summary["RunCount"]) == {1, 3}
    paper = paper_table(summary)
    assert len(paper) == 20
    assert set(paper["Method"]) == {"CADRef", "LogitGap"}
