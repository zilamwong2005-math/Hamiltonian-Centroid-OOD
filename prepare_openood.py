"""Resumable downloader for the OpenOOD v1.5 CIFAR and ImageNet benchmarks.

This intentionally does not disable TLS verification.  Rental servers often
run unattended, so a failed or non-ZIP download is left in place with a clear
error instead of being silently treated as a valid dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
import zipfile
from pathlib import Path
from typing import Sequence


DOWNLOAD_IDS = {
    "benchmark_imglist": "1lI1j0_fDDvjIt9JlWAw09X8ks-yrR_H1",
    "imagenet_1k": "1i1ipLDFARR-JZ9argXd2-0a6DXwVhXEj",
    "ssb_hard": "1PzkA-WGG8Z18h0ooL_pDdz9cO-DCIouE",
    "ninco": "1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6",
    "inaturalist": "1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj",
    "texture": "1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam",
    "openimage_o": "1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE",
    "cifar10": "1Co32RiiWe16lTaiOU6JMMnyUYS41IlO1",
    "cifar100": "1PGKheHUsf29leJPPGuXqzLBMwl8qMF8_",
    "tin": "1PZ-ixyx52U989IKsMA2OT-24fToTrelC",
    "mnist": "1CCHAGWqA1KJTFFswuF9cbhmB-j98Y1Sb",
    "svhn": "1DQfc11HOtB1nEwqS4pWUFp8vtQ3DczvI",
    "places365": "1Ec-LRSTf6u5vEctKX9vRp9OA6tqnJ0Ay",
    "imagenet_v2": "1akg2IiE22HcbvTBpwXQoD7tgfPCdkoho",
    "imagenet_c": "1JeXL9YH4BO8gCJ631c5BHbaSsl-lekHt",
    "imagenet_r": "1EzjMN2gq-bVV7lg-MEAdeuBuz-7jbGYU",
    "imagenet_es": "1ATz11vKmPqyzfEaEDRaPTF9TXiC244sw",
    "imagenet200_checkpoint": "1ddVmwc8zmzSjdLUO84EuV4Gz1c7vhIAs",
    "imagenet1k_checkpoint": "15PdDMNRfnJ7f2oxW6lI-Ge4QJJH3Z0Fy",
    "cifar10_checkpoint": "1byGeYxM_PlLjT72wZsMQvP6popJeWBgt",
    "cifar100_checkpoint": "1s-1oNrRtmA0pGefxXJOUVRYpaoAML0C-",
}

# These archives mirror the OpenOOD-layout ZIP files on Hugging Face.  The LFS
# object id is the SHA-256 of the downloaded payload.  AutoDL's network_turbo
# explicitly accelerates Hugging Face, making these mirrors much more reliable
# than Google Drive from mainland-China rental servers.
HF_MIRRORS = {
    # The CIFAR benchmark mirrors use the exact OpenOOD directory-layout ZIPs.
    # Every file is pinned by its public LFS size and SHA-256.
    "cifar10": {
        "repo": "torch-uncertainty/Cifar10",
        "filename": "cifar10.zip",
        "size": 142_903_414,
        "sha256": "5d3c480cd13e8791af7429fde3884f6ac45a7191c17fc1414eec550ed2e6582c",
    },
    "cifar100": {
        "repo": "torch-uncertainty/Cifar100",
        "filename": "cifar100.zip",
        "size": 141_321_419,
        "sha256": "db6301142ca4119cb104a194e31a8f1190a1873759c3aa409b04c7899e9868f8",
    },
    "tin": {
        "repo": "torch-uncertainty/tiny-imagenet-200",
        "filename": "tin.zip",
        "size": 237_497_520,
        "sha256": "e95af0741e02afeb62c58cac3b5ac53ada99bee67bde65b061a423e2e40deb9d",
    },
    "mnist": {
        "repo": "torch-uncertainty/MNIST",
        "filename": "mnist.zip",
        "size": 47_228_895,
        "sha256": "47e04388bccdcb0c5bb416e98720382129156c13f56dadeb97935f4a012cd0f9",
    },
    "svhn": {
        "repo": "torch-uncertainty/SVHN",
        "filename": "svhn.zip",
        "size": 18_989_682,
        "sha256": "56fdae7a5409712bcf10ce460c3e1f30550e73c89b70b9544bf136a84c61c0ae",
    },
    "places365": {
        "repo": "torch-uncertainty/Places365",
        "filename": "places365.zip",
        "size": 496_937_997,
        "sha256": "76639b253baca242da1b2746b56983c364e999d0eb63c4b901d8473f23b2e7ab",
    },
    "ssb_hard": {
        "repo": "torch-uncertainty/SSB_hard",
        "filename": "ssb_hard.zip",
        "size": 1_055_841_329,
        "sha256": "b7653e07a318852276208acab9a6d7950c05e17307a0f8b63aaf5f4c8ae96919",
    },
    "ninco": {
        "repo": "torch-uncertainty/Ninco",
        "filename": "ninco.zip",
        "size": 674_439_979,
        "sha256": "e6870bb704e18a19cf6308bdb297033a89bf02a6d8ad0eb1e770c7d570644c6e",
    },
    "inaturalist": {
        "repo": "torch-uncertainty/inaturalist",
        "filename": "inaturalist.zip",
        "size": 3_952_024_451,
        "sha256": "ad6ba788de0eb9f7125d5dc2c08a30a806ef0c6defc48a9cf571dd56b3e30a0b",
    },
    "texture": {
        "repo": "torch-uncertainty/Texture",
        "filename": "texture.zip",
        "size": 625_703_847,
        "sha256": "801fd2026c281090e6f42a5bddbc7ab55d8d03e79ee4bcc22d61bdef59862926",
    },
    "openimage_o": {
        "repo": "torch-uncertainty/Openimage-O",
        "filename": "openimage_o.zip",
        "size": 619_605_926,
        "sha256": "1b16b9b7cce2972d1bf988c5eea38cb30be736f2c77b84b707036bd8c0eaa9d6",
    },
}

BENCHMARK_DATASETS = {
    "cifar10": (
        "cifar10",
        "cifar100",
        "tin",
        "mnist",
        "svhn",
        "texture",
        "places365",
    ),
    "cifar100": (
        "cifar100",
        "cifar10",
        "tin",
        "mnist",
        "svhn",
        "texture",
        "places365",
    ),
    "imagenet200": (
        "imagenet_1k",
        "ssb_hard",
        "ninco",
        "inaturalist",
        "texture",
        "openimage_o",
        "imagenet_v2",
        "imagenet_c",
        "imagenet_r",
    ),
    "imagenet1k": (
        "imagenet_1k",
        "ssb_hard",
        "ninco",
        "inaturalist",
        "texture",
        "openimage_o",
        "imagenet_v2",
        "imagenet_c",
        "imagenet_r",
        "imagenet_es",
    ),
}

CLASSIC_DATASETS = {
    "cifar10",
    "cifar100",
    "tin",
    "mnist",
    "svhn",
    "texture",
    "places365",
}
CHECKPOINT_BENCHMARKS = {"cifar10", "cifar100", "imagenet200", "imagenet1k"}
DATASET_NAMES = tuple(dict.fromkeys(
    dataset
    for datasets in BENCHMARK_DATASETS.values()
    for dataset in datasets
))


def _import_gdown():
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("gdown is required: python -m pip install 'gdown>=4.7.1'") from exc
    return gdown


def _has_payload(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    visited = 0
    for root, dirs, files in os.walk(directory):
        visited += 1
        if any(not name.endswith((".zip", ".part")) for name in files):
            return True
        # A few directory levels distinguish an empty target from a mounted
        # ImageNet tree without crawling millions of files.
        if visited >= 32:
            break
    return False


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            member_path = (destination / member.filename).resolve()
            if destination_resolved != member_path and destination_resolved not in member_path.parents:
                raise RuntimeError(f"Unsafe path in archive {archive}: {member.filename}")
        source.extractall(destination)


def _verify_sha256(archive: Path, expected: str | None) -> str:
    if expected is None:
        return "not_checked"
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if expected is not None and actual.lower() != expected.lower():
        raise RuntimeError(
            f"SHA256 mismatch for {archive}: expected {expected}, got {actual}"
        )
    return actual


def _http_session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _huggingface_file_metadata(mirror: dict[str, object]) -> tuple[int, str]:
    """Resolve immutable LFS/Xet size and SHA-256 from the Hub API."""

    repo = str(mirror["repo"])
    filename = str(mirror["filename"])
    api_url = (
        f"https://huggingface.co/api/datasets/{repo}/tree/main"
        "?recursive=false&expand=false"
    )
    with _http_session() as session:
        response = session.get(api_url, timeout=(30, 120))
        response.raise_for_status()
        payload = response.json()
    if isinstance(payload, dict):
        payload = [payload]
    match = next(
        (item for item in payload if item.get("path") == filename), None
    )
    if match is None:
        raise RuntimeError(
            f"Hugging Face mirror file is missing: {repo}/{filename}"
        )
    size = int(match.get("size") or 0)
    lfs = match.get("lfs") or {}
    digest = str(lfs.get("sha256") or lfs.get("oid") or "")
    if digest.startswith("sha256:"):
        digest = digest.split(":", 1)[1]
    if size <= 0 or len(digest) != 64:
        raise RuntimeError(
            f"Incomplete Hugging Face metadata for {repo}/{filename}: {match}"
        )
    return size, digest.lower()


def _download_huggingface(
    mirror: dict[str, object], archive: Path, *, attempts: int = 5
) -> None:
    """Download one public Hugging Face file with safe on-disk resume."""
    import requests

    repo = str(mirror["repo"])
    filename = str(mirror["filename"])
    if "size" not in mirror or "sha256" not in mirror:
        expected_size, expected_sha256 = _huggingface_file_metadata(mirror)
        mirror["size"] = expected_size
        mirror["sha256"] = expected_sha256
    else:
        expected_size = int(mirror["size"])
    url = (
        f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
        "?download=true"
    )
    partial = archive.with_name(archive.name + ".part")
    archive.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, attempts + 1):
        start = partial.stat().st_size if partial.is_file() else 0
        if start > expected_size:
            raise RuntimeError(
                f"Partial download is larger than expected: {partial} "
                f"({start} > {expected_size}). Remove only this .part file and retry."
            )
        if start == expected_size:
            partial.replace(archive)
            return

        headers = {"Range": f"bytes={start}-"} if start else {}
        try:
            with _http_session() as session:
                response = session.get(
                    url,
                    headers=headers,
                    stream=True,
                    allow_redirects=True,
                    timeout=(30, 120),
                )
                if response.status_code == 416 and start == expected_size:
                    partial.replace(archive)
                    return
                response.raise_for_status()

                # A 200 response to a Range request means the server ignored
                # resume, so restart the partial file instead of corrupting it.
                append = start > 0 and response.status_code == 206
                mode = "ab" if append else "wb"
                downloaded = start if append else 0
                last_report = time.monotonic()
                with partial.open(mode) as stream:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if not chunk:
                            continue
                        stream.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 10:
                            percent = downloaded / expected_size * 100
                            print(
                                f"  {archive.name}: {downloaded / 1024**3:.2f}/"
                                f"{expected_size / 1024**3:.2f} GiB ({percent:.1f}%)",
                                flush=True,
                            )
                            last_report = now
        except requests.RequestException as exc:
            if attempt == attempts:
                raise RuntimeError(
                    f"Hugging Face download failed after {attempts} attempts: {url}"
                ) from exc
            delay = min(2**attempt, 30)
            print(
                f"[retry {attempt}/{attempts}] {archive.name}: {exc}; "
                f"resuming in {delay}s",
                flush=True,
            )
            time.sleep(delay)
            continue

        actual_size = partial.stat().st_size
        if actual_size != expected_size:
            if attempt == attempts:
                raise RuntimeError(
                    f"Incomplete Hugging Face download: {partial} has {actual_size} "
                    f"bytes, expected {expected_size}. Keep the .part file and retry."
                )
            print(
                f"[retry {attempt}/{attempts}] {archive.name}: received "
                f"{actual_size}/{expected_size} bytes; resuming",
                flush=True,
            )
            continue
        partial.replace(archive)
        return


def _checkpoint_present(results_root: Path, benchmark: str) -> bool:
    if benchmark in {"cifar10", "cifar100"}:
        patterns = (
            f"**/{benchmark}_resnet18_32x32_base*/s*/best.ckpt",
            f"**/{benchmark}_res18*/s*/best.ckpt",
        )
    elif benchmark == "imagenet200":
        patterns = (
            "**/imagenet200_resnet18_224x224_base*/s*/best.ckpt",
            "**/imagenet200_res18_v1.5/**/best.ckpt",
        )
    else:
        patterns = (
            "**/imagenet_resnet50_tvsv1_base_default/ckpt.pth",
            "**/resnet50_imagenet1k_v1.pth",
        )
    matches = {
        path.resolve()
        for pattern in patterns
        for path in results_root.glob(pattern)
        if path.is_file()
    }
    if benchmark == "imagenet1k":
        return bool(matches)
    available_seeds = {
        int(path.parent.name[1:])
        for path in matches
        if path.parent.name.startswith("s") and path.parent.name[1:].isdigit()
    }
    return {0, 1, 2}.issubset(available_seeds)


def _download_zip(
    name: str,
    file_id: str,
    destination: Path,
    *,
    force: bool,
    keep_archive: bool,
    marker: Path,
    accept_existing_payload: bool = True,
    archive_dir: Path | None = None,
    expected_sha256: str | None = None,
    download_backend: str = "auto",
    hf_mirror: dict[str, object] | None = None,
) -> None:
    if marker.is_file() and not force:
        print(f"[ready] {name}: {destination}")
        return
    if accept_existing_payload and _has_payload(destination) and not force:
        print(f"[existing] {name}: {destination} (accepted without re-download)")
        return

    destination.mkdir(parents=True, exist_ok=True)
    download_target = (
        Path(archive_dir) / f"{name}.zip"
        if archive_dir is not None
        else destination / f"{name}.zip"
    )
    local_candidates = []
    if archive_dir is not None:
        local_candidates.append(Path(archive_dir) / f"{name}.zip")
    local_candidates.append(download_target)
    archive = next((path for path in local_candidates if path.is_file()), None)
    downloaded_this_run = False
    source = "local"
    if archive is None:
        archive = download_target
        hf_error = None
        if download_backend in {"auto", "huggingface"} and hf_mirror is not None:
            print(f"[download:huggingface] {name} -> {archive}", flush=True)
            try:
                _download_huggingface(hf_mirror, archive)
                source = f"huggingface:{hf_mirror['repo']}/{hf_mirror['filename']}"
            except RuntimeError as exc:
                hf_error = exc
                if download_backend == "huggingface":
                    raise
                print(f"[warning] {exc}", flush=True)
        elif download_backend == "huggingface":
            raise RuntimeError(
                f"No Hugging Face mirror is configured for {name}. "
                "Use --download-backend auto/google or upload the ZIP to --archive-dir."
            )

        if not archive.is_file():
            print(f"[download:google] {name} -> {archive}", flush=True)
            gdown = _import_gdown()
            try:
                result = gdown.download(
                    id=file_id, output=str(archive), quiet=False, resume=True
                )
            except Exception as exc:
                context = f" Hugging Face also failed: {hf_error}" if hf_error else ""
                raise RuntimeError(
                    f"Download failed for {name} (Google Drive id {file_id}).{context} "
                    f"You can upload {name}.zip and pass --archive-dir."
                ) from exc
            if result is None or not archive.is_file():
                raise RuntimeError(
                    f"Download failed for {name} (Google Drive id {file_id}). "
                    f"You can upload {name}.zip and pass --archive-dir."
                )
            source = f"google_drive:{file_id}"
        downloaded_this_run = True
    else:
        print(f"[local archive] {name}: {archive}", flush=True)
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(
            f"File for {name} is not a ZIP archive: {archive}. "
            "Google Drive may have returned a quota page, or the upload is incomplete."
        )

    verified_sha256 = expected_sha256
    if source.startswith("huggingface:") and verified_sha256 is None:
        verified_sha256 = str(hf_mirror["sha256"])
    actual_sha256 = _verify_sha256(archive, verified_sha256)

    print(f"[extract] {archive} -> {destination}", flush=True)
    _safe_extract(archive, destination)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        f"source={source}\ngoogle_drive_id={file_id}\n"
        f"archive={archive.resolve()}\nsha256={actual_sha256}\n",
        encoding="utf-8",
    )
    if downloaded_this_run and not keep_archive:
        archive.unlink()


def _download_imglists(
    data_root: Path,
    *,
    force: bool,
    keep_archive: bool,
    archive_dir: Path | None,
    checksums: dict[str, str],
    download_backend: str,
) -> None:
    target = data_root
    expected = data_root / "benchmark_imglist"
    marker = data_root / ".downloads" / "benchmark_imglist.complete"
    if expected.is_dir() and not force:
        print(f"[ready] benchmark_imglist: {expected}")
        return
    _download_zip(
        "benchmark_imglist",
        DOWNLOAD_IDS["benchmark_imglist"],
        target,
        force=force,
        keep_archive=keep_archive,
        marker=marker,
        accept_existing_payload=False,
        archive_dir=archive_dir,
        expected_sha256=checksums.get("benchmark_imglist"),
        download_backend=download_backend,
    )


def prepare_benchmarks(
    benchmarks: Sequence[str],
    data_root: Path,
    results_root: Path,
    *,
    download_datasets: bool = True,
    download_checkpoints: bool = True,
    force: bool = False,
    keep_archives: bool = False,
    archive_dir: Path | None = None,
    checksums: dict[str, str] | None = None,
    download_backend: str = "auto",
    dataset_names: Sequence[str] | None = None,
) -> None:
    invalid = sorted(set(benchmarks) - set(BENCHMARK_DATASETS))
    if invalid:
        raise ValueError(f"Unknown benchmarks: {invalid}")

    data_root = Path(data_root)
    results_root = Path(results_root)
    archive_dir = Path(archive_dir) if archive_dir is not None else None
    checksums = dict(checksums or {})
    if download_backend not in {"auto", "huggingface", "google"}:
        raise ValueError(f"Unknown download backend: {download_backend}")
    data_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    if download_datasets:
        _download_imglists(
            data_root,
            force=force,
            keep_archive=keep_archives,
            archive_dir=archive_dir,
            checksums=checksums,
            download_backend=download_backend,
        )
        if dataset_names is None:
            datasets = []
            for benchmark in benchmarks:
                datasets.extend(BENCHMARK_DATASETS[benchmark])
        else:
            invalid_datasets = sorted(set(dataset_names) - set(DATASET_NAMES))
            if invalid_datasets:
                raise ValueError(f"Unknown datasets: {invalid_datasets}")
            datasets = list(dataset_names)
        for dataset in dict.fromkeys(datasets):
            category = "images_classic" if dataset in CLASSIC_DATASETS else "images_largescale"
            destination = data_root / category / dataset
            marker = data_root / ".downloads" / f"{dataset}.complete"
            _download_zip(
                dataset,
                DOWNLOAD_IDS[dataset],
                destination,
                force=force,
                keep_archive=keep_archives,
                marker=marker,
                archive_dir=archive_dir,
                expected_sha256=checksums.get(dataset),
                download_backend=download_backend,
                hf_mirror=HF_MIRRORS.get(dataset),
            )

    if download_checkpoints:
        for benchmark in dict.fromkeys(benchmarks):
            if benchmark not in CHECKPOINT_BENCHMARKS:
                continue
            archive_name = f"{benchmark}_checkpoint"
            marker = results_root / ".downloads" / f"{archive_name}.complete"
            if _checkpoint_present(results_root, benchmark) and not force:
                print(f"[ready] {archive_name}: existing checkpoint found")
                continue
            # Checkpoint archives contain their own experiment directories, so
            # they are extracted at the OpenOOD results root.
            _download_zip(
                archive_name,
                DOWNLOAD_IDS[archive_name],
                results_root,
                force=force,
                keep_archive=keep_archives,
                marker=marker,
                accept_existing_payload=False,
                archive_dir=archive_dir,
                expected_sha256=checksums.get(archive_name),
                download_backend=download_backend,
            )


def _parse_checksums(values: Sequence[str]) -> dict[str, str]:
    checksums = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --sha256 value {value!r}; expected NAME=HEX")
        name, digest = value.split("=", 1)
        if name not in DOWNLOAD_IDS:
            raise ValueError(f"Unknown download name in --sha256: {name}")
        if len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            raise ValueError(f"Invalid SHA256 digest for {name}")
        checksums[name] = digest.lower()
    return checksums


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=sorted(BENCHMARK_DATASETS),
        default=["imagenet200", "imagenet1k"],
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--no-datasets", action="store_true")
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument(
        "--download-backend",
        choices=("auto", "huggingface", "google"),
        default="auto",
        help="auto prefers a configured Hugging Face mirror, then falls back to Google Drive",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_NAMES,
        help="Prepare only these datasets instead of every dataset in --benchmarks",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        help="Directory containing pre-uploaded <download-name>.zip files",
    )
    parser.add_argument(
        "--sha256",
        action="append",
        default=[],
        metavar="NAME=HEX",
        help="Optional archive checksum; repeat for multiple downloads",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.no_datasets and args.datasets:
        raise ValueError("--no-datasets and --datasets cannot be used together")
    prepare_benchmarks(
        args.benchmarks,
        args.data_root,
        args.results_root,
        download_datasets=not args.no_datasets,
        download_checkpoints=not args.no_checkpoints,
        force=args.force,
        keep_archives=args.keep_archives,
        archive_dir=args.archive_dir,
        checksums=_parse_checksums(args.sha256),
        download_backend=args.download_backend,
        dataset_names=args.datasets,
    )
    print("OpenOOD preparation complete.")


if __name__ == "__main__":
    main()
