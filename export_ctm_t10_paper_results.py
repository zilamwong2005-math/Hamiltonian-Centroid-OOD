"""Export completed CTM, ImageNet-1K T=10, and bridge results for the paper.

Only CSV/JSON artifacts are read. Source files are never changed. The output
is a new, exclusively created .tar.gz under PROJECT/archives; no dependencies
outside the Python standard library are required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import secrets
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


T10_TAG = "mass-uniform-none_loss-static-ts10_pred-backbone_eval-full"
MODEL_TAG = "imagenet1k_resnet50_tvsv1"
BRIDGE_FILES = (
    "locked_calibration_statistics.csv",
    "radial_centroid_bound_audit.csv",
    "radial_centroid_class_bounds.csv",
    "stage_metrics.csv",
    "stage_metrics_mean_std.csv",
    "transition_diagnostics.csv",
    "transition_diagnostics_mean_std.csv",
    "transition_metric_deltas.csv",
)


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _csv_has_data(path: Path) -> None:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream)
        if not next(rows, None) or not any(any(cell.strip() for cell in row) for row in rows):
            raise ValueError(f"CSV has no data rows: {path}")


def collect_inputs(project: Path) -> tuple[list[Path], dict]:
    """Validate every required artifact before creating any output."""
    project = project.resolve(strict=True)
    journal = project / "results" / "journal"
    ctm = journal / "ctm"
    bridge = journal / "static_reduction_bridge"
    ctm_manifest = ctm / "ctm_full_completed.json"
    bridge_manifest = bridge / "static_reduction_bridge_full_completed.json"
    required = [ctm_manifest, ctm / "ctm_full_all_runs.csv", bridge_manifest]
    required.extend(bridge / "full" / name for name in BRIDGE_FILES)
    t10_dirs = []
    for seed in (0, 1, 2):
        run = project / "results_openood" / "imagenet1k" / MODEL_TAG / f"seed{seed}" / T10_TAG
        t10_dirs.append(run)
        required.extend((run / "openood_metrics_all_potentials.csv", run / "run_config.json"))
    ctm_runs = sorted((ctm / "full").rglob("ctm.csv"))
    required.extend(ctm_runs)
    required.extend(path.with_suffix(".json") for path in ctm_runs)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("Missing required input files:\n  " + "\n  ".join(missing))
    if len(ctm_runs) != 11:
        raise ValueError(f"Expected 11 formal CTM run CSVs, found {len(ctm_runs)} under {ctm / 'full'}")

    ctm_info = _json(ctm_manifest)
    if not (
        ctm_info.get("completed") is True
        and ctm_info.get("stage") == "full"
        and ctm_info.get("max_eval_samples") == 0
        and ctm_info.get("all_id_train_samples") is True
        and ctm_info.get("independent_runs") == 11
    ):
        raise ValueError(f"CTM manifest does not establish 11 completed, uncapped, full-train runs: {ctm_manifest}")
    for path in ctm_runs:
        config = _json(path.with_suffix(".json"))
        if not (
            config.get("stage") == "full"
            and config.get("method") == "ctm"
            and config.get("max_eval_samples") == 0
            and config.get("all_id_train_samples") is True
        ):
            raise ValueError(f"CTM run is not a full evaluation: {path.with_suffix('.json')}")
    for seed, run in enumerate(t10_dirs):
        config = _json(run / "run_config.json")
        if not (
            config.get("seed") == seed
            and config.get("n_steps") == 10
            and config.get("max_eval_samples") == 0
            and config.get("mass_mode") == "uniform"
            and config.get("mass_normalization") == "none"
            and config.get("bandwidth_loss") == "static"
            and {"gaussian", "imq"}.issubset(config.get("potentials") or [])
        ):
            raise ValueError(f"T=10 configuration does not match the requested formal run: {run / 'run_config.json'}")
    bridge_info = _json(bridge_manifest)
    if not (
        bridge_info.get("completed") is True
        and bridge_info.get("stage") == "full"
        and bridge_info.get("near_far_test_access") is False
        and set(bridge_info.get("seeds", [])) == {0, 1, 2}
    ):
        raise ValueError(f"Bridge manifest does not establish the completed validation-only three-seed run: {bridge_manifest}")

    files = set(required)
    # Include individual potential tables and any additional result JSON/CSV,
    # but do not recursively include detector weights, feature caches, or images.
    for folder in [ctm / "full", bridge / "full", *t10_dirs]:
        candidates = folder.rglob("*") if folder == ctm / "full" else folder.iterdir()
        files.update(path for path in candidates if path.is_file() and path.suffix.lower() in {".csv", ".json"})
    ordered = sorted(files, key=lambda path: path.relative_to(project).as_posix())
    for path in ordered:
        if path.is_symlink() or not path.resolve().is_relative_to(project):
            raise ValueError(f"Refusing linked result outside the input project: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Empty input file: {path}")
        if path.suffix.lower() == ".csv":
            _csv_has_data(path)
        else:
            _json(path)
    counts = {"ctm_formal_runs": len(ctm_runs), "imagenet1k_t10_runs": len(t10_dirs), "bridge_seeds": 3}
    return ordered, counts


def export_results(project: Path) -> Path:
    project = project.resolve(strict=True)
    files, counts = collect_inputs(project)
    now = datetime.now(timezone.utc)
    records = []
    for path in files:
        records.append({
            "path": path.relative_to(project).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    manifest = {
        "format_version": 1,
        "created_utc": now.isoformat(),
        "source_project": str(project),
        "archive_paths_relative_to_project": True,
        "counts": counts,
        "files": records,
    }
    payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    archive_dir = project / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / (
        f"hamood_ctm_t10_bridge_paper_results_{now:%Y%m%dT%H%M%S%fZ}_{secrets.token_hex(3)}.tar.gz"
    )
    created = False
    try:
        with archive.open("xb") as stream:
            created = True
            with tarfile.open(fileobj=stream, mode="w:gz") as bundle:
                for path, record in zip(files, records):
                    # Read and hash again before packaging to detect a result
                    # changing while an experiment is still writing it.
                    content = path.read_bytes()
                    if hashlib.sha256(content).hexdigest() != record["sha256"]:
                        raise ValueError(f"Input changed during export; wait for the experiment to finish: {path}")
                    info = tarfile.TarInfo(record["path"])
                    info.size = len(content)
                    info.mtime = int(path.stat().st_mtime)
                    info.mode = 0o644
                    bundle.addfile(info, io.BytesIO(content))
                info = tarfile.TarInfo("paper_results_export_manifest.json")
                info.size = len(payload)
                info.mtime = int(now.timestamp())
                info.mode = 0o644
                bundle.addfile(info, io.BytesIO(payload))
    except Exception:
        if created:
            archive.unlink(missing_ok=True)
        raise
    print(f"CTM formal runs: {counts['ctm_formal_runs']}")
    print(f"ImageNet-1K T=10 formal runs: {counts['imagenet1k_t10_runs']}")
    print(f"Bridge seeds: {counts['bridge_seeds']}")
    print(f"Source CSV/JSON files: {len(files)}; archive entries: {len(files) + 1}")
    print(f"Archive: {archive}")
    print(f"Size: {archive.stat().st_size / 1024**2:.2f} MiB")
    print(f"SHA256: {hashlib.sha256(archive.read_bytes()).hexdigest()}")
    print("Source results unchanged. Download this archive for manuscript integration.")
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        export_results(args.project)
    except (OSError, ValueError, TypeError) as error:
        print(f"Export failed; no new completed archive was created.\n{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
