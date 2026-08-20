"""Create journal-ready trajectory, mass, baseline and efficiency summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ("IDAccuracy", "AUROC", "FPR95")
POTENTIALS = ("gaussian", "imq")
BASELINE_METHODS = {
    "cifar10": {
        "msp", "ebo", "mls", "gen", "react", "scale", "knn", "vim",
        "ash", "dice", "she", "rmds", "rankfeat",
    },
    "cifar100": {
        "msp", "ebo", "mls", "gen", "react", "scale", "knn", "vim",
        "ash", "dice", "she", "rmds", "rankfeat",
    },
    "imagenet200": {
        "msp", "ebo", "mls", "gen", "react", "scale", "knn", "vim",
        "ash", "dice", "she", "rmds", "rankfeat",
    },
    # The ImageNet-1K classifier is deterministic and shared by detector seeds.
    # Exact KNN/ViM are intentionally omitted because the 1.28M x 2048 feature
    # bank makes them a different, prohibitively expensive resource regime.
    "imagenet1k": {
        "msp", "ebo", "mls", "gen", "react", "scale",
        "ash", "dice", "she", "rmds", "rankfeat",
    },
}
BASELINE_SEEDS = {
    "cifar10": {0, 1, 2},
    "cifar100": {0, 1, 2},
    "imagenet200": {0, 1, 2},
    "imagenet1k": {0},
}
IMAGE_TRAJECTORY_STEPS = {
    # ImageNet-200 already has a complete three-seed T=10 experiment.
    "ImageNet-200": {0, 1, 3, 10},
    # ImageNet-1K T=10 was run only for seed 0 as a scaling diagnosis and is
    # intentionally excluded from the three-seed journal table.
    "ImageNet-1K": {0, 1, 3},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal-root", type=Path, default=Path("results/journal"))
    parser.add_argument("--cifar-main-csv", type=Path,
                        default=Path("results/cifar/ood_results_cifar_v3.csv"))
    parser.add_argument("--openood-root", type=Path, default=Path("results_openood"))
    parser.add_argument("--baseline-root", type=Path,
                        default=Path("results_openood_baselines"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/journal/summary"))
    parser.add_argument("--expected-seeds", nargs="+", type=int,
                        default=[0, 1, 2])
    parser.add_argument(
        "--allow-incomplete-baselines",
        action="store_true",
        help="Write partial baseline tables instead of enforcing the journal matrix",
    )
    return parser


def _flatten(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [
        column if isinstance(column, str) else "_".join(x for x in column if x)
        for column in frame.columns
    ]
    return frame


def _check_seeds(frame: pd.DataFrame, identity: list[str], expected: set[int],
                 label: str) -> None:
    errors = []
    for key, block in frame.groupby(identity, dropna=False):
        actual = set(pd.to_numeric(block["Seed"]).astype(int))
        if actual != expected:
            errors.append(f"{key}: {sorted(actual)}")
    if errors:
        raise RuntimeError(
            f"{label}: expected seeds {sorted(expected)} but found:\n"
            + "\n".join(errors[:30])
        )


def _read_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if path.is_file():
            frame = pd.read_csv(path)
            frame["SourceFile"] = str(path.resolve())
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def cifar_per_seed(args, expected: set[int]) -> pd.DataFrame:
    paths = [args.cifar_main_csv]
    paths.extend(args.journal_root.rglob("ood_results_cifar_v3.csv"))
    frame = _read_csvs(list(dict.fromkeys(Path(p).resolve() for p in paths)))
    if frame.empty:
        raise FileNotFoundError("No CIFAR result CSVs found")
    required = {
        "ExperimentID", "Seed", "In-dist", "Model", "Protocol",
        "EncoderSource", "Potential", "MassMode", "MassNormalization",
        "BandwidthLoss", "Anchors", "Steps", "Dt", "HamEpochs",
        "MaxEvalSamples", "OODGroup", "OOD", *METRICS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"CIFAR CSVs lack columns: {missing}")
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
        & pd.to_numeric(frame["Anchors"]).eq(80)
        & pd.to_numeric(frame["HamEpochs"]).eq(25)
        & pd.to_numeric(frame["MaxEvalSamples"]).eq(0)
        & frame["OODGroup"].str.lower().isin(["near", "far"])
    )
    frame = frame.loc[mask].copy()
    for column in (*METRICS, "Steps", "Seed"):
        frame[column] = pd.to_numeric(frame[column])
    frame["Seed"] = frame["Seed"].astype(int)
    frame["TrajectorySteps"] = frame["Steps"].astype(int)
    frame["Potential"] = frame["Potential"].str.lower()
    frame["Group"] = frame["OODGroup"].str.lower()
    frame["Benchmark"] = frame["In-dist"].str.upper().map(
        {"CIFAR10": "CIFAR-10", "CIFAR100": "CIFAR-100"}
    )
    frame = frame.drop_duplicates(
        ["Benchmark", "Potential", "TrajectorySteps", "Seed", "Group", "OOD"],
        keep="last",
    )
    _check_seeds(
        frame,
        ["Benchmark", "Potential", "TrajectorySteps", "Group", "OOD"],
        expected,
        "CIFAR trajectory",
    )
    return (
        frame.groupby(
            ["Benchmark", "Potential", "TrajectorySteps", "Seed", "Group"]
        )[list(METRICS)].mean().reset_index()
    )


def openood_per_seed(args, expected: set[int]) -> pd.DataFrame:
    paths = sorted(args.openood_root.rglob("openood_metrics_all_potentials.csv"))
    frame = _read_csvs(paths)
    if frame.empty:
        raise FileNotFoundError("No ImageNet OpenOOD result CSVs found")
    required = {
        "ID", "Potential", "Dataset", "Seed", "MassMode",
        "MassNormalization", "BandwidthLoss", "TrajectoryTrainSteps",
        "PredictionSource", "ACC", "AUROC", "FPR@95", "SourceFile",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"ImageNet CSVs lack columns: {missing}")
    source = frame["SourceFile"].str.replace("\\", "/", regex=False)
    mask = (
        frame["ID"].str.lower().isin(["imagenet200", "imagenet1k"])
        & frame["Potential"].str.lower().isin(POTENTIALS)
        & frame["Dataset"].str.lower().isin(["nearood", "farood"])
        & frame["MassMode"].str.lower().eq("uniform")
        & frame["MassNormalization"].str.lower().eq("none")
        & frame["BandwidthLoss"].str.lower().eq("static")
        & frame["PredictionSource"].str.lower().eq("backbone")
        & source.str.contains("eval-full", regex=False)
    )
    frame = frame.loc[mask].copy()
    frame["Seed"] = pd.to_numeric(frame["Seed"]).astype(int)
    frame["TrajectorySteps"] = pd.to_numeric(
        frame["TrajectoryTrainSteps"]
    ).astype(int)
    frame["Potential"] = frame["Potential"].str.lower()
    frame["Group"] = frame["Dataset"].str.lower().map(
        {"nearood": "near", "farood": "far"}
    )
    frame["Benchmark"] = frame["ID"].str.lower().map(
        {"imagenet200": "ImageNet-200", "imagenet1k": "ImageNet-1K"}
    )
    frame = frame[
        frame.apply(
            lambda row: row["TrajectorySteps"]
            in IMAGE_TRAJECTORY_STEPS[row["Benchmark"]],
            axis=1,
        )
    ].copy()
    frame = frame.rename(columns={
        "ACC": "IDAccuracy", "FPR@95": "FPR95"
    })
    frame = frame.drop_duplicates(
        ["Benchmark", "Potential", "TrajectorySteps", "Seed", "Group"],
        keep="last",
    )
    _check_seeds(
        frame,
        ["Benchmark", "Potential", "TrajectorySteps", "Group"],
        expected,
        "ImageNet trajectory",
    )
    return frame[[
        "Benchmark", "Potential", "TrajectorySteps", "Seed", "Group", *METRICS
    ]]


def summarize_trajectory(per_seed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        per_seed.groupby(
            ["Benchmark", "Potential", "TrajectorySteps", "Group"]
        )[list(METRICS)].agg(["mean", "std", "count"]).reset_index()
    )
    summary = _flatten(summary)
    static = summary[summary["TrajectorySteps"] == 0][
        ["Benchmark", "Potential", "Group", "AUROC_mean", "FPR95_mean"]
    ].rename(columns={
        "AUROC_mean": "AUROC_T0", "FPR95_mean": "FPR95_T0"
    })
    delta = summary.merge(static, on=["Benchmark", "Potential", "Group"],
                          how="left")
    delta["Delta_AUROC_vs_T0"] = delta["AUROC_mean"] - delta["AUROC_T0"]
    delta["Delta_FPR95_vs_T0"] = delta["FPR95_mean"] - delta["FPR95_T0"]
    return summary, delta


def summarize_mass(args, expected: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cifar_path = args.journal_root / "cifar_mass" / "ood_results_cifar_v3.csv"
    trajectory_path = (
        args.journal_root / "cifar_trajectory" / "ood_results_cifar_v3.csv"
    )
    cifar = _read_csvs([cifar_path, trajectory_path])
    cifar_summary = pd.DataFrame()
    if not cifar.empty:
        mask = (
            cifar["Potential"].str.lower().eq("gaussian")
            & cifar["BandwidthLoss"].str.lower().eq("static")
            & pd.to_numeric(cifar["Steps"]).eq(0)
            & pd.to_numeric(cifar["MaxEvalSamples"]).eq(0)
            & cifar["OODGroup"].str.lower().isin(["near", "far"])
        )
        cifar = cifar.loc[mask].copy()
        cifar["MassSetting"] = (
            cifar["MassMode"].str.lower() + "/"
            + cifar["MassNormalization"].str.lower()
        )
        cifar["Seed"] = pd.to_numeric(cifar["Seed"]).astype(int)
        cifar["Group"] = cifar["OODGroup"].str.lower()
        cifar["Benchmark"] = cifar["In-dist"].str.upper().map(
            {"CIFAR10": "CIFAR-10", "CIFAR100": "CIFAR-100"}
        )
        for metric in METRICS:
            cifar[metric] = pd.to_numeric(cifar[metric])
        cifar = cifar.drop_duplicates(
            ["Benchmark", "MassSetting", "Seed", "Group", "OOD"], keep="last"
        )
        _check_seeds(cifar, ["Benchmark", "MassSetting", "Group", "OOD"],
                     expected, "CIFAR mass")
        cifar_seed = cifar.groupby(
            ["Benchmark", "MassSetting", "Seed", "Group"]
        )[list(METRICS)].mean().reset_index()
        cifar_summary = _flatten(cifar_seed.groupby(
            ["Benchmark", "MassSetting", "Group"]
        )[list(METRICS)].agg(["mean", "std", "count"]).reset_index())

    image_paths = sorted(args.openood_root.rglob("openood_metrics_all_potentials.csv"))
    image = _read_csvs(image_paths)
    image_summary = pd.DataFrame()
    if not image.empty:
        source = image["SourceFile"].str.replace("\\", "/", regex=False)
        mask = (
            image["ID"].str.lower().eq("imagenet200")
            & image["Potential"].str.lower().eq("gaussian")
            & image["Dataset"].str.lower().isin(["nearood", "farood"])
            & image["BandwidthLoss"].str.lower().eq("static")
            & pd.to_numeric(image["TrajectoryTrainSteps"]).eq(0)
            & source.str.contains("eval-full", regex=False)
        )
        image = image.loc[mask].copy()
        image["MassSetting"] = (
            image["MassMode"].str.lower() + "/"
            + image["MassNormalization"].str.lower()
        )
        image["Seed"] = pd.to_numeric(image["Seed"]).astype(int)
        image["Group"] = image["Dataset"].str.lower().map(
            {"nearood": "near", "farood": "far"}
        )
        image["Benchmark"] = "ImageNet-200"
        image = image.rename(columns={"ACC": "IDAccuracy", "FPR@95": "FPR95"})
        image = image.drop_duplicates(
            ["Benchmark", "MassSetting", "Seed", "Group"], keep="last"
        )
        _check_seeds(image, ["Benchmark", "MassSetting", "Group"], expected,
                     "ImageNet-200 mass")
        image_summary = _flatten(image.groupby(
            ["Benchmark", "MassSetting", "Group"]
        )[list(METRICS)].agg(["mean", "std", "count"]).reset_index())
    return cifar_summary, image_summary


def summarize_baselines(root: Path, *, strict: bool = True) -> pd.DataFrame:
    path = root / "baseline_all_runs.csv"
    if not path.is_file():
        path = root / "msp_all_runs.csv"
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "Method" not in frame:
        frame["Method"] = "msp"
    frame["Benchmark"] = frame["Benchmark"].astype(str).str.lower()
    frame["Method"] = frame["Method"].astype(str).str.lower()
    frame["Dataset"] = frame["Dataset"].astype(str).str.lower()
    frame["Seed"] = pd.to_numeric(frame["Seed"]).astype(int)
    frame = frame[frame["Dataset"].str.lower().isin(["nearood", "farood"])].copy()
    frame = frame.drop_duplicates(
        ["Benchmark", "Method", "Seed", "Dataset"], keep="last"
    )
    if strict:
        errors = []
        for benchmark, methods in BASELINE_METHODS.items():
            for method in sorted(methods):
                for dataset in ("nearood", "farood"):
                    actual = set(
                        frame.loc[
                            frame["Benchmark"].eq(benchmark)
                            & frame["Method"].eq(method)
                            & frame["Dataset"].eq(dataset),
                            "Seed",
                        ]
                    )
                    expected = BASELINE_SEEDS[benchmark]
                    if actual != expected:
                        errors.append(
                            f"{benchmark}/{method}/{dataset}: "
                            f"expected {sorted(expected)}, found {sorted(actual)}"
                        )
        if errors:
            raise RuntimeError(
                "Baseline matrix is incomplete:\n" + "\n".join(errors[:50])
            )
    frame["Group"] = frame["Dataset"].str.lower().map(
        {"nearood": "near", "farood": "far"}
    )
    return _flatten(frame.groupby(["Benchmark", "Method", "Group"])[
        ["FPR@95", "AUROC", "AUPR_IN", "AUPR_OUT", "ACC"]
    ].agg(["mean", "std", "count"]).reset_index())


def _write(frame: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    frame.to_csv(path.with_suffix(".csv"), index=False, float_format="%.6f")
    path.with_suffix(".tex").write_text(
        frame.to_latex(index=False, escape=True, caption=caption, label=label),
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    for field in ("journal_root", "cifar_main_csv", "openood_root",
                  "baseline_root", "output_dir"):
        setattr(args, field, getattr(args, field).resolve())
    expected = set(args.expected_seeds)
    cifar = cifar_per_seed(args, expected)
    imagenet = openood_per_seed(args, expected)
    per_seed = pd.concat([cifar, imagenet], ignore_index=True)
    trajectory, delta = summarize_trajectory(per_seed)
    cifar_mass, image_mass = summarize_mass(args, expected)
    baselines = summarize_baselines(
        args.baseline_root, strict=not args.allow_incomplete_baselines
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.output_dir / "trajectory_per_seed.csv", index=False)
    _write(trajectory, args.output_dir / "trajectory_mean_std",
           "Trajectory-length ablation on OpenOOD v1.5.",
           "tab:trajectory-ablation")
    _write(delta, args.output_dir / "trajectory_delta_vs_t0",
           "Change relative to the static potential score (T=0).",
           "tab:trajectory-delta")
    if not cifar_mass.empty:
        _write(cifar_mass, args.output_dir / "mass_cifar_mean_std",
               "Effective-rank mass ablation on CIFAR.", "tab:mass-cifar")
    if not image_mass.empty:
        _write(image_mass, args.output_dir / "mass_imagenet200_mean_std",
               "Effective-rank mass ablation on ImageNet-200.",
               "tab:mass-imagenet200")
    if not baselines.empty:
        _write(baselines, args.output_dir / "baseline_mean_std",
               "Locally reproduced post-hoc baselines.", "tab:baselines")
    efficiency = args.journal_root / "efficiency" / "efficiency.csv"
    if efficiency.is_file():
        pd.read_csv(efficiency).to_csv(
            args.output_dir / "efficiency.csv", index=False
        )
    print(f"Saved journal summaries under: {args.output_dir}")


if __name__ == "__main__":
    main()
