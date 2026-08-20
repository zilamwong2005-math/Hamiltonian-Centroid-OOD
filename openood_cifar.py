"""Strict OpenOOD v1.5 data helpers for the CIFAR benchmarks.

The official benchmark distributes every image as a regular file and fixes the
evaluation split through ``benchmark_imglist`` text files.  Reading those lists
directly is important: torchvision's native test split, DTD's ``test`` split,
and random 10k subsampling are not equivalent to the OpenOOD protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

from PIL import Image, ImageFile
from torch.utils.data import Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True


OOD_GROUPS = {
    "cifar10": {
        "near": ("cifar100", "tin"),
        "far": ("mnist", "svhn", "places365", "texture"),
    },
    "cifar100": {
        "near": ("cifar10", "tin"),
        "far": ("mnist", "svhn", "places365", "texture"),
    },
}


IMGLIST_FILES = {
    "cifar10": {
        "train": "train_cifar10.txt",
        "val": "val_cifar10.txt",
        "test": "test_cifar10.txt",
        "cifar100": "test_cifar100.txt",
        "tin": "test_tin.txt",
        "mnist": "test_mnist.txt",
        "svhn": "test_svhn.txt",
        "places365": "test_places365.txt",
        "texture": "test_texture.txt",
    },
    "cifar100": {
        "train": "train_cifar100.txt",
        "val": "val_cifar100.txt",
        "test": "test_cifar100.txt",
        "cifar10": "test_cifar10.txt",
        "tin": "test_tin.txt",
        "mnist": "test_mnist.txt",
        "svhn": "test_svhn.txt",
        "places365": "test_places365.txt",
        "texture": "test_texture.txt",
    },
}


# Counts from the OpenOOD v1.5 benchmark_imglist archive.  These constants make
# accidental use of a torchvision/full/random split fail before a long run.
EXPECTED_COUNTS = {
    "cifar10": {
        "train": 50_000,
        "val": 1_000,
        "test": 9_000,
        "cifar100": 9_000,
        "tin": 7_793,
        "mnist": 70_000,
        "svhn": 26_032,
        "places365": 35_195,
        "texture": 5_640,
    },
    "cifar100": {
        "train": 50_000,
        "val": 1_000,
        "test": 9_000,
        "cifar10": 10_000,
        "tin": 6_526,
        "mnist": 70_000,
        "svhn": 26_032,
        "places365": 33_773,
        "texture": 5_640,
    },
}


def canonical_ood_name(name: str) -> str:
    aliases = {
        "tinyimagenet": "tin",
        "tiny_imagenet": "tin",
        "dtd": "texture",
        "places": "places365",
    }
    return aliases.get(name.lower(), name.lower())


def imglist_path(data_root: Path, benchmark: str, split_or_dataset: str) -> Path:
    benchmark = benchmark.lower()
    key = canonical_ood_name(split_or_dataset)
    try:
        filename = IMGLIST_FILES[benchmark][key]
    except KeyError as exc:
        raise ValueError(
            f"No OpenOOD CIFAR imglist for benchmark={benchmark!r}, key={key!r}"
        ) from exc
    return Path(data_root) / "benchmark_imglist" / benchmark / filename


def _parse_imglist_line(line: str, *, source: Path, line_number: int) -> tuple[str, int]:
    stripped = line.strip()
    if not stripped:
        raise ValueError(f"Blank line in {source} at line {line_number}")
    try:
        image_name, label_text = stripped.rsplit(maxsplit=1)
        label = int(label_text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid OpenOOD imglist entry in {source} at line {line_number}: {stripped!r}"
        ) from exc

    posix = PurePosixPath(image_name)
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(
            f"Unsafe image path in {source} at line {line_number}: {image_name!r}"
        )
    return image_name, label


class OpenOODImglistDataset(Dataset):
    """Return ``(image_tensor, label)`` from one official OpenOOD imglist."""

    def __init__(
        self,
        image_root: Path,
        list_path: Path,
        transform,
        *,
        max_samples: int = 0,
    ) -> None:
        self.image_root = Path(image_root)
        self.list_path = Path(list_path)
        self.transform = transform
        if not self.list_path.is_file():
            raise FileNotFoundError(
                f"OpenOOD imglist not found: {self.list_path}. "
                "Run prepare_openood.py for cifar10/cifar100 first."
            )
        if not self.image_root.is_dir():
            raise FileNotFoundError(
                f"OpenOOD image root not found: {self.image_root}. "
                "Run prepare_openood.py for cifar10/cifar100 first."
            )

        entries = []
        with self.list_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                entries.append(
                    _parse_imglist_line(
                        line, source=self.list_path, line_number=line_number
                    )
                )
        if max_samples < 0:
            raise ValueError("max_samples must be non-negative")
        self.entries = entries[:max_samples] if max_samples else entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        image_name, label = self.entries[index]
        path = self.image_root.joinpath(*PurePosixPath(image_name).parts)
        try:
            with path.open("rb") as stream:
                image = Image.open(stream).convert("RGB")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Image listed by {self.list_path} is missing: {path}"
            ) from exc
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_openood_dataset(
    data_root: Path,
    benchmark: str,
    split_or_dataset: str,
    transform,
    *,
    max_samples: int = 0,
) -> OpenOODImglistDataset:
    return OpenOODImglistDataset(
        Path(data_root) / "images_classic",
        imglist_path(Path(data_root), benchmark, split_or_dataset),
        transform,
        max_samples=max_samples,
    )


@dataclass(frozen=True)
class VerificationRow:
    benchmark: str
    name: str
    list_path: Path
    expected: int
    actual: int
    missing: int
    missing_examples: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.actual == self.expected and self.missing == 0


def verify_openood_cifar_data(
    data_root: Path,
    benchmarks: Sequence[str] = ("cifar10", "cifar100"),
    *,
    check_images: bool = True,
    missing_example_limit: int = 5,
) -> list[VerificationRow]:
    """Verify official list counts and, optionally, every referenced image."""

    data_root = Path(data_root)
    image_root = data_root / "images_classic"
    rows = []
    for benchmark in benchmarks:
        benchmark = benchmark.lower()
        if benchmark not in IMGLIST_FILES:
            raise ValueError(f"Unknown CIFAR benchmark: {benchmark}")
        for name, expected in EXPECTED_COUNTS[benchmark].items():
            path = imglist_path(data_root, benchmark, name)
            if not path.is_file():
                rows.append(
                    VerificationRow(
                        benchmark, name, path, expected, 0, expected, (str(path),)
                    )
                )
                continue

            actual = 0
            missing = 0
            examples = []
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    image_name, _ = _parse_imglist_line(
                        line, source=path, line_number=line_number
                    )
                    actual += 1
                    if check_images:
                        image_path = image_root.joinpath(
                            *PurePosixPath(image_name).parts
                        )
                        if not image_path.is_file():
                            missing += 1
                            if len(examples) < missing_example_limit:
                                examples.append(str(image_path))
            rows.append(
                VerificationRow(
                    benchmark,
                    name,
                    path,
                    expected,
                    actual,
                    missing,
                    tuple(examples),
                )
            )
    return rows


def discover_cifar_checkpoint(root: Path, benchmark: str, seed: int) -> Path:
    """Locate one official OpenOOD v1.5 ResNet-18 checkpoint."""

    root = Path(root)
    benchmark = benchmark.lower()
    if benchmark not in ("cifar10", "cifar100"):
        raise ValueError(f"Official CIFAR checkpoint is unavailable for {benchmark}")
    if seed not in (0, 1, 2):
        raise ValueError("Official OpenOOD CIFAR checkpoints only provide seeds 0, 1, 2")

    patterns = (
        f"**/{benchmark}_resnet18_32x32_base_e100_lr0.1_default/s{seed}/best.ckpt",
        f"**/{benchmark}_resnet18_32x32_base*/s{seed}/best.ckpt",
        f"**/{benchmark}*/s{seed}/best.ckpt",
    )
    matches = []
    for pattern in patterns:
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if matches:
            break
    if not matches:
        raise FileNotFoundError(
            f"Official OpenOOD {benchmark} seed {seed} checkpoint not found under {root}. "
            f"Run prepare_openood.py --benchmarks {benchmark} --no-datasets or upload "
            f"the official checkpoint ZIP into the archive directory."
        )
    if len(matches) > 1:
        exact = [
            path
            for path in matches
            if "base_e100_lr0.1_default" in path.as_posix()
        ]
        if len(exact) == 1:
            return exact[0]
        raise RuntimeError(
            f"Multiple candidate checkpoints for {benchmark} seed {seed}: {matches}"
        )
    return matches[0]


def iter_ood_names(benchmark: str) -> Iterable[str]:
    groups = OOD_GROUPS[benchmark.lower()]
    yield from groups["near"]
    yield from groups["far"]

