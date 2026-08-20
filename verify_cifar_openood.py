"""Verify the complete OpenOOD v1.5 CIFAR image lists and image files."""

from __future__ import annotations

import argparse
from pathlib import Path

from openood_cifar import verify_openood_cifar_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=("cifar10", "cifar100"),
        default=("cifar10", "cifar100"),
    )
    parser.add_argument(
        "--lists-only",
        action="store_true",
        help="Check list files and counts without checking every image path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = verify_openood_cifar_data(
        args.data_root,
        args.benchmarks,
        check_images=not args.lists_only,
    )
    failures = []
    for row in rows:
        state = "正常" if row.ok else "失败"
        print(
            f"{row.benchmark:8s} {row.name:10s}  "
            f"列表={row.actual:6d}/{row.expected:6d}  "
            f"缺失={row.missing:6d}  {state}"
        )
        for example in row.missing_examples:
            print(f"  缺失样例: {example}")
        if not row.ok:
            failures.append(row)
    print("=" * 72)
    if failures:
        raise SystemExit(f"CIFAR OpenOOD 数据检查失败：{len(failures)} 个列表异常")
    print("CIFAR-10/CIFAR-100 OpenOOD v1.5 数据全部正常")


if __name__ == "__main__":
    main()

