"""CLI entrypoint for `docextract-datagen`."""

from __future__ import annotations

import argparse
from pathlib import Path

from datagen.generator import generate_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="docextract-datagen",
        description="Generate synthetic invoices/receipts with gold labels and fault injection.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=12,
        help="Number of synthetic documents to generate (default: 12)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(".data/synthetic"),
        help="Output directory for generated PDFs and gold JSON files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Starting random seed for deterministic generation",
    )
    args = parser.parse_args(argv)
    samples = generate_dataset(count=args.count, out_dir=args.out_dir, base_seed=args.seed)
    print(f"Generated {len(samples)} synthetic samples in {args.out_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
