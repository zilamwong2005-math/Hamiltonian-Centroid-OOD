import hashlib
import zipfile

import pytest

from prepare_openood import (
    HF_MIRRORS,
    _checkpoint_present,
    _download_zip,
    _parse_checksums,
    _safe_extract,
)


def test_preuploaded_archive_is_used_without_deleting_it(tmp_path):
    archive_dir = tmp_path / "archives"
    destination = tmp_path / "destination"
    archive_dir.mkdir()
    archive = archive_dir / "example.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("payload/file.txt", "ready")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    marker = tmp_path / "markers" / "example.complete"

    _download_zip(
        "example",
        "unused-google-id",
        destination,
        force=False,
        keep_archive=False,
        marker=marker,
        accept_existing_payload=False,
        archive_dir=archive_dir,
        expected_sha256=digest,
    )

    assert (destination / "payload" / "file.txt").read_text() == "ready"
    assert archive.is_file()
    assert digest in marker.read_text()


def test_safe_extract_rejects_parent_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../escape.txt", "bad")
    with pytest.raises(RuntimeError, match="Unsafe path"):
        _safe_extract(archive, tmp_path / "target")


def test_checksum_cli_parser():
    digest = "a" * 64
    assert _parse_checksums([f"imagenet200_checkpoint={digest}"]) == {
        "imagenet200_checkpoint": digest
    }
    with pytest.raises(ValueError):
        _parse_checksums(["imagenet200_checkpoint=short"])


def test_huggingface_mirror_metadata_is_complete():
    assert set(HF_MIRRORS) == {
        "cifar10",
        "cifar100",
        "tin",
        "mnist",
        "svhn",
        "places365",
        "ssb_hard",
        "ninco",
        "inaturalist",
        "texture",
        "openimage_o",
    }
    for mirror in HF_MIRRORS.values():
        assert mirror["repo"].startswith("torch-uncertainty/")
        assert mirror["filename"].endswith(".zip")
        assert mirror["size"] > 0
        digest = mirror["sha256"]
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)


def test_cifar_checkpoint_requires_all_three_seeds(tmp_path):
    base = tmp_path / "cifar10_resnet18_32x32_base_e100_lr0.1_default"
    for seed in (0, 1):
        checkpoint = base / f"s{seed}" / "best.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"ready")
    assert not _checkpoint_present(tmp_path, "cifar10")
    checkpoint = base / "s2" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"ready")
    assert _checkpoint_present(tmp_path, "cifar10")
