#!/usr/bin/env python3
"""Generate pure color patch images from an aggregated RGB color dataset.

Default input is the strict aggregated XKCD color survey dataset:
  data/colorsurvey_rgb_aggregated_agreement_gt075_min2.csv

The script samples up to N unique RGB values per basic color, writes solid PNG
patches, and exports ground-truth metadata for downstream VLM experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import struct
import zlib
from collections import Counter, defaultdict
from pathlib import Path


BASIC_COLORS = (
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
)


DEFAULT_KEEP_COLUMNS = (
    "total_votes",
    "majority_count",
    "agreement_rate",
    "label_entropy_bits",
    "num_basic_labels",
    "basic_label_set",
    "example_colornames",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate N pure color patch PNGs per basic color from aggregated RGB CSV."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        nargs="?",
        default=Path("data/colorsurvey_rgb_aggregated_agreement_gt075_min2.csv"),
        help="Aggregated RGB CSV. Default: data/colorsurvey_rgb_aggregated_agreement_gt075_min2.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/color_patches_aggregated_100"),
        help="Output dataset directory. Default: data/color_patches_aggregated_100",
    )
    parser.add_argument(
        "--samples-per-color",
        type=int,
        default=100,
        help="Number of unique RGB patches to sample per basic color. Default: 100",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="Square patch size in pixels. Default: 224",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    parser.add_argument(
        "--label-col",
        default="majority_basic_color",
        help="Column containing the basic color label. Default: majority_basic_color",
    )
    parser.add_argument("--r-col", default="r", help="Red channel column. Default: r")
    parser.add_argument("--g-col", default="g", help="Green channel column. Default: g")
    parser.add_argument("--b-col", default="b", help="Blue channel column. Default: b")
    parser.add_argument(
        "--sample-strategy",
        choices=("random", "top"),
        default="random",
        help=(
            "Sampling strategy. random = reproducible random sample; "
            "top = highest agreement_rate, then most total_votes. Default: random"
        ),
    )
    parser.add_argument(
        "--on-shortfall",
        choices=("warn", "error"),
        default="warn",
        help="What to do if a color has fewer than requested samples. Default: warn",
    )
    return parser.parse_args()


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def write_solid_png(path: Path, rgb: tuple[int, int, int], size: int) -> None:
    width = height = size
    row = b"\x00" + bytes(rgb) * width
    raw = row * height

    png = b"\x89PNG\r\n\x1a\n"
    png += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += png_chunk(b"IDAT", zlib.compress(raw, level=9))
    png += png_chunk(b"IEND", b"")
    path.write_bytes(png)


def parse_channel(row: dict[str, str], column: str) -> int:
    value = int(float(row[column]))
    if value < 0 or value > 255:
        raise ValueError(f"RGB channel {column} out of range: {value}")
    return value


def safe_text(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower() or "value"


def validate_columns(fieldnames: list[str] | None, required: tuple[str, ...]) -> None:
    if not fieldnames:
        raise ValueError("Input CSV has no header row.")
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: "
            + ", ".join(missing)
            + ". Available columns: "
            + ", ".join(fieldnames)
        )


def load_candidates(args: argparse.Namespace) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    required = (args.r_col, args.g_col, args.b_col, args.label_col)

    with args.input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_columns(reader.fieldnames, required)

        seen_rgb: set[tuple[int, int, int]] = set()
        for row in reader:
            label = row[args.label_col].strip().lower()
            if label not in BASIC_COLORS:
                continue

            try:
                rgb = (
                    parse_channel(row, args.r_col),
                    parse_channel(row, args.g_col),
                    parse_channel(row, args.b_col),
                )
            except (KeyError, ValueError):
                continue

            if rgb in seen_rgb:
                continue
            seen_rgb.add(rgb)

            row = dict(row)
            row["r"] = str(rgb[0])
            row["g"] = str(rgb[1])
            row["b"] = str(rgb[2])
            row["hex"] = row.get("hex") or f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
            row["basic_color"] = label
            grouped[label].append(row)

    return grouped


def sort_for_top_sampling(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    def key(row: dict[str, str]) -> tuple[float, int, str]:
        agreement = float(row.get("agreement_rate") or 0.0)
        votes = int(float(row.get("total_votes") or 0))
        return (-agreement, -votes, row.get("hex", ""))

    return sorted(rows, key=key)


def sample_rows(grouped: dict[str, list[dict[str, str]]], args: argparse.Namespace) -> list[dict[str, str]]:
    rng = random.Random(args.seed)
    selected: list[dict[str, str]] = []
    shortfalls = []

    for color in BASIC_COLORS:
        rows = list(grouped.get(color, []))
        available = len(rows)
        if available < args.samples_per_color:
            shortfalls.append((color, available))
            chosen = rows
        elif args.sample_strategy == "top":
            chosen = sort_for_top_sampling(rows)[: args.samples_per_color]
        else:
            chosen = rng.sample(rows, args.samples_per_color)

        chosen.sort(key=lambda row: (int(row["r"]), int(row["g"]), int(row["b"])))
        selected.extend(chosen)

    if shortfalls and args.on_shortfall == "error":
        detail = ", ".join(f"{color}={available}" for color, available in shortfalls)
        raise ValueError(f"Not enough samples for requested count {args.samples_per_color}: {detail}")

    return selected


def build_dataset(args: argparse.Namespace) -> dict:
    if args.samples_per_color <= 0:
        raise ValueError("--samples-per-color must be positive.")
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be positive.")
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    grouped = load_candidates(args)
    selected = sample_rows(grouped, args)

    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_path = args.output_dir / "ground_truth.csv"
    metadata_path = args.output_dir / "metadata.json"

    counts = Counter()
    keep_columns = [
        column for column in DEFAULT_KEEP_COLUMNS if selected and column in selected[0]
    ]
    fieldnames = [
        "image_path",
        "filename",
        "basic_color",
        "r",
        "g",
        "b",
        "hex",
        *keep_columns,
    ]

    # Preserve order while removing duplicates if some keep columns are absent/present twice.
    fieldnames = list(dict.fromkeys(fieldnames))

    with ground_truth_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for row in selected:
            color = row["basic_color"]
            counts[color] += 1
            rgb = (int(row["r"]), int(row["g"]), int(row["b"]))
            hex_stem = safe_text(row["hex"])
            filename = f"{color}_{counts[color]:03d}_{hex_stem}.png"
            image_path = image_dir / filename
            write_solid_png(image_path, rgb, args.patch_size)

            writer.writerow(
                {
                    "image_path": str(image_path),
                    "filename": filename,
                    **row,
                }
            )

    available_by_color = {color: len(grouped.get(color, [])) for color in BASIC_COLORS}
    metadata = {
        "input_csv": str(args.input_csv),
        "output_dir": str(args.output_dir),
        "image_dir": str(image_dir),
        "ground_truth_csv": str(ground_truth_path),
        "samples_per_color_requested": args.samples_per_color,
        "patch_size": args.patch_size,
        "seed": args.seed,
        "sample_strategy": args.sample_strategy,
        "label_col": args.label_col,
        "available_by_color": available_by_color,
        "counts_by_color": {color: counts.get(color, 0) for color in BASIC_COLORS},
        "shortfall_by_color": {
            color: max(0, args.samples_per_color - counts.get(color, 0))
            for color in BASIC_COLORS
        },
        "total_images": sum(counts.values()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    metadata = build_dataset(args)
    print(f"Wrote {metadata['total_images']} patch images to: {metadata['image_dir']}")
    print(f"Wrote ground truth to: {metadata['ground_truth_csv']}")
    print("Counts by color:")
    for color, count in metadata["counts_by_color"].items():
        available = metadata["available_by_color"][color]
        shortfall = metadata["shortfall_by_color"][color]
        suffix = f" (available={available})"
        if shortfall:
            suffix += f" shortfall={shortfall}"
        print(f"  {color}: {count}{suffix}")


if __name__ == "__main__":
    main()
