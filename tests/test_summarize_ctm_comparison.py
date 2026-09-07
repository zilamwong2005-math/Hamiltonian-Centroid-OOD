import pandas as pd

from summarize_ctm_comparison import METRICS, _latex, _summarise


def _row(method, seed, auroc, fpr):
    return {
        "Target": "cifar10",
        "Method": method,
        "Seed": seed,
        "Dataset": "nearood",
        "FPR@95": fpr,
        "AUROC": auroc,
        "AUPR_IN": auroc,
        "AUPR_OUT": auroc,
        "ACC": 95.0,
    }


def test_summary_keeps_run_count_and_sample_std():
    raw = pd.DataFrame(
        [_row("ctm", 0, 80.0, 20.0), _row("ctm", 1, 82.0, 18.0)]
    )
    summary = _summarise(raw)
    assert summary.loc[0, "RunCount"] == 2
    assert summary.loc[0, "AUROC_mean"] == 81.0
    assert round(summary.loc[0, "AUROC_std"], 6) == round(2**0.5, 6)


def test_latex_bolds_best_auc_and_fpr():
    raw = pd.DataFrame(
        [
            _row("ctm", 0, 90.0, 20.0),
            _row("centroid_only", 0, 89.0, 21.0),
            _row("locked_centroid_msp", 0, 88.0, 22.0),
        ]
    )
    text = _latex(_summarise(raw))
    assert "\\textbf{20.00}" in text
    assert "\\textbf{90.00}" in text
    assert all(metric in METRICS for metric in ("AUROC", "FPR@95"))
