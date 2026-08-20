"""Build strict cross-benchmark paper tables from completed experiment CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


POTENTIALS = ("gaussian", "imq")
GROUPS = ("near", "far")
BENCHMARK_ORDER = ("CIFAR-10", "CIFAR-100", "ImageNet-200", "ImageNet-1K")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cifar-csv",
        type=Path,
        default=Path("results/cifar/ood_results_cifar_v3.csv"),
    )
    parser.add_argument(
        "--openood-output-root", type=Path, default=Path("results_openood")
    )
    parser.add_argument(
        "--msp-root", type=Path, default=Path("results_openood_baselines")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/paper_tables")
    )
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--strict-baselines", action="store_true")
    parser.add_argument("--cifar-anchors", type=int, default=80)
    parser.add_argument("--cifar-steps", type=int, default=120)
    parser.add_argument("--cifar-dt", type=float, default=0.05)
    parser.add_argument("--cifar-ham-epochs", type=int, default=25)
    return parser


def _require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise RuntimeError(f"{source} lacks required columns: {missing}")


def _assert_seed_coverage(
    frame: pd.DataFrame,
    identity: list[str],
    expected_seeds: set[int],
    label: str,
) -> None:
    errors = []
    for key, group in frame.groupby(identity, dropna=False):
        seeds = set(pd.to_numeric(group["Seed"]).astype(int))
        if seeds != expected_seeds:
            errors.append(f"{key}: got {sorted(seeds)}")
    if errors:
        raise RuntimeError(
            f"{label} seed coverage is incomplete (expected "
            f"{sorted(expected_seeds)}):\n" + "\n".join(errors[:20])
        )


def load_cifar(
    path: Path,
    expected_seeds: set[int],
    anchors: int,
    steps: int,
    dt: float,
    ham_epochs: int,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "ExperimentID", "Seed", "In-dist", "Model", "Protocol",
        "EncoderSource", "Potential", "MassMode", "MassNormalization",
        "BandwidthLoss", "Anchors", "Steps", "Dt", "HamEpochs",
        "MaxEvalSamples", "OODGroup", "OOD", "IDAccuracy", "AUROC", "FPR95",
    }
    _require_columns(frame, required, path)
    frame = frame.drop_duplicates(["ExperimentID", "OOD"], keep="last")
    mask = (
        frame["In-dist"].str.upper().isin(["CIFAR10", "CIFAR100"])
        & frame["Model"].str.lower().eq("resnet18")
        & frame["Protocol"].str.lower().eq("openood")
        & frame["EncoderSource"].str.lower().eq("official")
        & frame["Potential"].str.lower().isin(POTENTIALS)
        & frame["MassMode"].str.lower().eq("uniform")
        & frame["MassNormalization"].str.lower().eq("none")
        & frame["BandwidthLoss"].str.lower().eq("static")
        & pd.to_numeric(frame["Anchors"]).eq(anchors)
        & pd.to_numeric(frame["Steps"]).eq(steps)
        & np.isclose(pd.to_numeric(frame["Dt"]), dt)
        & pd.to_numeric(frame["HamEpochs"]).eq(ham_epochs)
        & pd.to_numeric(frame["MaxEvalSamples"]).eq(0)
        & frame["OODGroup"].str.lower().isin(GROUPS)
    )
    frame = frame.loc[mask].copy()
    if frame.empty:
        raise RuntimeError(
            "No formal CIFAR uniform/static Gaussian/IMQ rows matched. "
            "Check --cifar-anchors/--cifar-steps/--cifar-dt/--cifar-ham-epochs."
        )
    frame["Seed"] = pd.to_numeric(frame["Seed"]).astype(int)
    frame["Potential"] = frame["Potential"].str.lower()
    frame["OODGroup"] = frame["OODGroup"].str.lower()
    for column in ("IDAccuracy", "AUROC", "FPR95"):
        frame[column] = pd.to_numeric(frame[column])

    duplicates = frame.duplicated(
        ["In-dist", "Potential", "Seed", "OODGroup", "OOD"], keep=False
    )
    if duplicates.any():
        preview = frame.loc[
            duplicates,
            ["In-dist", "Potential", "Seed", "OODGroup", "OOD", "ExperimentID"],
        ]
        raise RuntimeError(
            "Multiple CIFAR configurations survived the formal filter:\n"
            + preview.head(20).to_string(index=False)
        )
    _assert_seed_coverage(
        frame,
        ["In-dist", "Potential", "OODGroup", "OOD"],
        expected_seeds,
        "CIFAR",
    )
    per_seed = (
        frame.groupby(["In-dist", "Potential", "Seed", "OODGroup"])[
            ["IDAccuracy", "AUROC", "FPR95"]
        ]
        .mean()
        .reset_index()
    )
    per_seed["Benchmark"] = per_seed["In-dist"].map(
        {"CIFAR10": "CIFAR-10", "CIFAR100": "CIFAR-100"}
    )
    per_seed["Backbone"] = "ResNet-18"
    per_seed["TrajectorySteps"] = steps
    return per_seed.rename(columns={"OODGroup": "Group"})


def _read_openood_runs(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.rglob("openood_metrics_all_potentials.csv")):
        # Smoke tests and the ImageNet-1K T=10 diagnosis are intentionally
        # excluded later, but keep SourceFile for a fully auditable selection.
        frame = pd.read_csv(path)
        frame["SourceFile"] = str(path.resolve())
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No openood_metrics_all_potentials.csv files found under {root}"
        )
    return pd.concat(frames, ignore_index=True)


def load_imagenet(root: Path, expected_seeds: set[int]) -> pd.DataFrame:
    frame = _read_openood_runs(root)
    required = {
        "ID", "Potential", "Dataset", "Seed", "MassMode",
        "MassNormalization", "BandwidthLoss", "TrajectoryTrainSteps",
        "PredictionSource", "ACC", "AUROC", "FPR@95", "SourceFile",
    }
    _require_columns(frame, required, root)
    source = frame["SourceFile"].str.replace("\\", "/", regex=False)
    desired_steps = frame["ID"].str.lower().map(
        {"imagenet200": 10, "imagenet1k": 1}
    )
    mask = (
        frame["ID"].str.lower().isin(["imagenet200", "imagenet1k"])
        & frame["Potential"].str.lower().isin(POTENTIALS)
        & frame["Dataset"].str.lower().isin(["nearood", "farood"])
        & frame["MassMode"].str.lower().eq("uniform")
        & frame["MassNormalization"].str.lower().eq("none")
        & frame["BandwidthLoss"].str.lower().eq("static")
        & frame["PredictionSource"].str.lower().eq("backbone")
        & pd.to_numeric(frame["TrajectoryTrainSteps"]).eq(desired_steps)
        & source.str.contains("eval-full", regex=False)
    )
    frame = frame.loc[mask].copy()
    if frame.empty:
        raise RuntimeError("No formal ImageNet-200 T=10 / ImageNet-1K T=1 rows matched")
    frame["Seed"] = pd.to_numeric(frame["Seed"]).astype(int)
    frame["Potential"] = frame["Potential"].str.lower()
    frame["Dataset"] = frame["Dataset"].str.lower()
    for column in ("ACC", "AUROC", "FPR@95"):
        frame[column] = pd.to_numeric(frame[column])
    frame = frame.drop_duplicates(
        ["ID", "Potential", "Dataset", "Seed"], keep="last"
    )
    _assert_seed_coverage(
        frame,
        ["ID", "Potential", "Dataset"],
        expected_seeds,
        "ImageNet",
    )
    frame["Benchmark"] = frame["ID"].str.lower().map(
        {"imagenet200": "ImageNet-200", "imagenet1k": "ImageNet-1K"}
    )
    frame["Backbone"] = frame["ID"].str.lower().map(
        {"imagenet200": "ResNet-18", "imagenet1k": "ResNet-50"}
    )
    frame["Group"] = frame["Dataset"].map(
        {"nearood": "near", "farood": "far"}
    )
    frame["TrajectorySteps"] = pd.to_numeric(
        frame["TrajectoryTrainSteps"]
    ).astype(int)
    return frame.rename(
        columns={"ACC": "IDAccuracy", "FPR@95": "FPR95"}
    )[[
        "Benchmark", "Backbone", "Potential", "Seed", "Group",
        "TrajectorySteps", "IDAccuracy", "AUROC", "FPR95",
    ]]


def aggregate_method(per_seed: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        per_seed.groupby(
            ["Benchmark", "Backbone", "Potential", "TrajectorySteps", "Group"]
        )[["IDAccuracy", "AUROC", "FPR95"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped.columns = [
        column if isinstance(column, str) else "_".join(x for x in column if x)
        for column in grouped.columns
    ]
    return grouped


def _fmt(mean: float, std: float, count: int) -> str:
    if int(count) <= 1 or pd.isna(std):
        return f"{mean:.2f}"
    return f"{mean:.2f}±{std:.2f}"


def make_method_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    key_columns = ["Benchmark", "Backbone", "Potential", "TrajectorySteps"]
    for key, group in summary.groupby(key_columns, sort=False):
        values = dict(zip(key_columns, key))
        for name, metric in (("ID Acc.", "IDAccuracy"),):
            first = group.iloc[0]
            values[name] = _fmt(
                first[f"{metric}_mean"],
                first[f"{metric}_std"],
                first[f"{metric}_count"],
            )
        for ood_group in GROUPS:
            match = group[group["Group"] == ood_group]
            if len(match) != 1:
                raise RuntimeError(f"Missing/duplicate {ood_group} row for {key}")
            row = match.iloc[0]
            title = "Near" if ood_group == "near" else "Far"
            values[f"{title} FPR95"] = _fmt(
                row["FPR95_mean"], row["FPR95_std"], row["FPR95_count"]
            )
            values[f"{title} AUROC"] = _fmt(
                row["AUROC_mean"], row["AUROC_std"], row["AUROC_count"]
            )
        rows.append(values)
    table = pd.DataFrame(rows)
    table["Benchmark"] = pd.Categorical(
        table["Benchmark"], BENCHMARK_ORDER, ordered=True
    )
    table["Potential"] = pd.Categorical(
        table["Potential"], POTENTIALS, ordered=True
    )
    return table.sort_values(["Benchmark", "Potential"]).reset_index(drop=True)


def load_msp(root: Path) -> pd.DataFrame:
    combined = root / "msp_all_runs.csv"
    if combined.is_file():
        frame = pd.read_csv(combined)
    else:
        frames = []
        for path in sorted(root.rglob("msp.csv")):
            item = pd.read_csv(path)
            unnamed = [c for c in item.columns if c.startswith("Unnamed:")]
            item = item.rename(columns={(unnamed or [item.columns[0]])[0]: "Dataset"})
            parts = [part.lower() for part in path.relative_to(root).parts]
            benchmark = next((x for x in parts if x in {
                "cifar10", "cifar100", "imagenet200", "imagenet1k"
            }), None)
            if benchmark is None:
                continue
            seed_part = next((x for x in parts if x.startswith("seed")), "seed0")
            item["Benchmark"] = benchmark
            item["Seed"] = int(seed_part.removeprefix("seed"))
            frames.append(item)
        if not frames:
            return pd.DataFrame()
        frame = pd.concat(frames, ignore_index=True)
    frame["Benchmark"] = frame["Benchmark"].str.lower().map(
        {
            "cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
            "imagenet200": "ImageNet-200", "imagenet1k": "ImageNet-1K",
        }
    )
    frame["Dataset"] = frame["Dataset"].str.lower()
    frame = frame[frame["Dataset"].isin(["nearood", "farood"])].copy()
    frame["Group"] = frame["Dataset"].map(
        {"nearood": "near", "farood": "far"}
    )
    frame = frame.drop_duplicates(["Benchmark", "Seed", "Group"], keep="last")
    return frame


def make_comparison(method: pd.DataFrame, msp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in method.iterrows():
        for group in GROUPS:
            title = "Near" if group == "near" else "Far"
            rows.append({
                "Benchmark": str(row["Benchmark"]),
                "Method": f"H-{str(row['Potential']).capitalize()}",
                "Group": f"{title}-OOD",
                "FPR95": row[f"{title} FPR95"],
                "AUROC": row[f"{title} AUROC"],
                "Source": "This work (3 seeds)",
            })
    if not msp.empty:
        for (benchmark, group), block in msp.groupby(["Benchmark", "Group"]):
            rows.append({
                "Benchmark": benchmark,
                "Method": "MSP",
                "Group": f"{group.capitalize()}-OOD",
                "FPR95": _fmt(block["FPR@95"].mean(), block["FPR@95"].std(), len(block)),
                "AUROC": _fmt(block["AUROC"].mean(), block["AUROC"].std(), len(block)),
                "Source": "Local OpenOOD v1.5 reproduction",
            })
    return pd.DataFrame(rows).sort_values(["Benchmark", "Group", "Method"])


def _latex(frame: pd.DataFrame, caption: str, label: str) -> str:
    return frame.to_latex(
        index=False,
        escape=True,
        caption=caption,
        label=label,
    ).replace("±", r"$\pm$")


def main() -> None:
    args = build_parser().parse_args()
    expected_seeds = set(args.expected_seeds)
    cifar = load_cifar(
        args.cifar_csv.resolve(), expected_seeds,
        args.cifar_anchors, args.cifar_steps, args.cifar_dt,
        args.cifar_ham_epochs,
    )
    imagenet = load_imagenet(args.openood_output_root.resolve(), expected_seeds)
    per_seed = pd.concat([cifar, imagenet], ignore_index=True)
    summary = aggregate_method(per_seed)
    method_table = make_method_table(summary)

    msp = load_msp(args.msp_root.resolve())
    observed_baselines = set(msp["Benchmark"]) if not msp.empty else set()
    missing_baselines = set(BENCHMARK_ORDER) - observed_baselines
    if missing_baselines and args.strict_baselines:
        raise RuntimeError(
            "Missing locally reproduced MSP baselines: "
            + ", ".join(sorted(missing_baselines))
        )
    comparison = make_comparison(method_table, msp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.output_dir / "selected_method_per_seed.csv", index=False)
    summary.to_csv(args.output_dir / "selected_method_mean_std.csv", index=False)
    method_table.to_csv(
        args.output_dir / "cross_benchmark_hamiltonian.csv", index=False
    )
    (args.output_dir / "cross_benchmark_hamiltonian.tex").write_text(
        _latex(
            method_table,
            "Hamiltonian OOD detection across OpenOOD v1.5 benchmarks.",
            "tab:hamiltonian-cross-benchmark",
        ),
        encoding="utf-8",
    )
    comparison.to_csv(args.output_dir / "msp_comparison.csv", index=False)
    (args.output_dir / "msp_comparison.tex").write_text(
        _latex(
            comparison,
            "Comparison with locally reproduced OpenOOD v1.5 MSP.",
            "tab:hamiltonian-vs-msp",
        ),
        encoding="utf-8",
    )
    missing_path = args.output_dir / "missing_msp_baselines.txt"
    missing_path.write_text(
        ("none" if not missing_baselines else "\n".join(sorted(missing_baselines)))
        + "\n",
        encoding="utf-8",
    )

    print("===== Cross-benchmark Hamiltonian table =====")
    print(method_table.to_string(index=False))
    print("\n===== MSP comparison (available local baselines) =====")
    print(comparison.to_string(index=False))
    if missing_baselines:
        print("\nMissing local MSP baselines: " + ", ".join(sorted(missing_baselines)))
    print(f"\nSaved under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
