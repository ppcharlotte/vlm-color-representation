#!/usr/bin/env python3
"""Build a pure color patch dataset from a baseline RGB/name CSV.

Expected CSV shape is flexible. By default the script auto-detects RGB columns
such as r/g/b or red/green/blue and a color-name column such as name/color.
It filters rows whose names map to one of the 11 basic color categories, samples
up to N rows per category, writes solid-color PNG patches, and exports ground
truth metadata.
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
from typing import Iterable


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

COLOR_ALIASES = {
    "grey": "gray",
}

RGB_COLUMN_CANDIDATES = {
    "r": ("r", "red", "rgb_r", "red_value", "redchannel"),
    "g": ("g", "green", "rgb_g", "green_value", "greenchannel"),
    "b": ("b", "blue", "rgb_b", "blue_value", "bluechannel"),
}

NAME_COLUMN_CANDIDATES = (
    "name",
    "color",
    "colour",
    "colorname",
    "colourname",
    "color_name",
    "colour_name",
    "label",
    "basic_color",
    "category",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pure color patch PNGs and ground truth from an RGB/name CSV."
    )
    parser.add_argument("csv_path", type=Path, help="Input baseline_color.csv path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/color_patches"),
        help="Output dataset directory. Default: data/color_patches",
    )
    parser.add_argument(
        "--samples-per-color",
        type=int,
        default=100,
        help="Maximum samples to keep for each basic color. Default: 100",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="Square patch size in pixels. Default: 224",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed. Default: 42")
    parser.add_argument("--r-col", help="RGB red column name. Auto-detected if omitted.")
    parser.add_argument("--g-col", help="RGB green column name. Auto-detected if omitted.")
    parser.add_argument("--b-col", help="RGB blue column name. Auto-detected if omitted.")
    parser.add_argument("--name-col", help="Color name column. Auto-detected if omitted.")
    parser.add_argument(
        "--match-mode",
        choices=("exact", "contains"),
        default="contains",
        help=(
            "How to map CSV names to 11 basic colors. "
            "'exact' accepts red/grey/etc.; 'contains' also accepts names like dark red. "
            "Default: contains"
        ),
    )
    parser.add_argument(
        "--rgb-scale",
        choices=("auto", "0-255", "0-1"),
        default="auto",
        help="Interpret RGB values. Default: auto",
    )
    return parser.parse_args()


def normalized_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def find_column(fieldnames: Iterable[str], candidates: Iterable[str], explicit: str | None) -> str:
    fields = list(fieldnames)
    if explicit:
        if explicit in fields:
            return explicit
        raise ValueError(f"Column {explicit!r} was requested but is not in the CSV.")

    by_normalized = {normalized_header(field): field for field in fields}
    for candidate in candidates:
        normalized = normalized_header(candidate)
        if normalized in by_normalized:
            return by_normalized[normalized]

    raise ValueError(
        "Could not auto-detect a required column. "
        f"Available columns: {', '.join(fields)}"
    )


def normalize_color_name(raw_name: str, match_mode: str) -> str | None:
    normalized = raw_name.strip().lower()
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = COLOR_ALIASES.get(normalized, normalized)

    if match_mode == "exact":
        return normalized if normalized in BASIC_COLORS else None

    tokens = re.findall(r"[a-z]+", normalized)
    mapped_tokens = [COLOR_ALIASES.get(token, token) for token in tokens]
    matches = [token for token in mapped_tokens if token in BASIC_COLORS]
    if len(set(matches)) == 1:
        return matches[0]
    return None


def parse_rgb_value(value: str) -> float:
    value = value.strip()
    if not value:
        raise ValueError("empty RGB value")
    return float(value)


def convert_rgb(values: tuple[float, float, float], scale: str) -> tuple[int, int, int]:
    if scale == "0-1":
        converted = [round(channel * 255) for channel in values]
    else:
        converted = [round(channel) for channel in values]

    if any(channel < 0 or channel > 255 for channel in converted):
        raise ValueError(f"RGB values out of range after conversion: {converted}")

    return tuple(int(channel) for channel in converted)


def infer_rgb_scale(rows: list[dict], columns: tuple[str, str, str], requested_scale: str) -> str:
    if requested_scale != "auto":
        return requested_scale

    max_value = 0.0
    for row in rows:
        for column in columns:
            max_value = max(max_value, parse_rgb_value(row[column]))

    return "0-1" if max_value <= 1.0 else "0-255"


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def write_solid_png(path: Path, rgb: tuple[int, int, int], size: int) -> None:
    """Write an RGB PNG using only the Python standard library."""
    width = height = size
    row = b"\x00" + bytes(rgb) * width
    raw = row * height

    png = b"\x89PNG\r\n\x1a\n"
    png += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += png_chunk(b"IDAT", zlib.compress(raw, level=9))
    png += png_chunk(b"IEND", b"")
    path.write_bytes(png)


def safe_stem(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return value or "color"


def detect_columns(args: argparse.Namespace) -> tuple[tuple[str, str, str], str]:
    with args.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row.")

        r_col = find_column(reader.fieldnames, RGB_COLUMN_CANDIDATES["r"], args.r_col)
        g_col = find_column(reader.fieldnames, RGB_COLUMN_CANDIDATES["g"], args.g_col)
        b_col = find_column(reader.fieldnames, RGB_COLUMN_CANDIDATES["b"], args.b_col)
        name_col = find_column(reader.fieldnames, NAME_COLUMN_CANDIDATES, args.name_col)

    return (r_col, g_col, b_col), name_col


def build_dataset(args: argparse.Namespace) -> dict:
    if args.samples_per_color <= 0:
        raise ValueError("--samples-per-color must be positive.")
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be positive.")
    if not args.csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.csv_path}")

    rgb_columns, name_col = detect_columns(args)
    rng = random.Random(args.seed)

    reservoirs: dict[str, list[dict]] = {color: [] for color in BASIC_COLORS}
    matched_counts = Counter()
    rows_total = 0
    skipped_name = 0
    skipped_rgb = 0
    max_rgb_value = 0.0

    with args.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for source_row, row in enumerate(reader, start=2):
            rows_total += 1
            basic_color = normalize_color_name(row.get(name_col, ""), args.match_mode)
            if basic_color is None:
                skipped_name += 1
                continue

            try:
                raw_rgb = tuple(parse_rgb_value(row[column]) for column in rgb_columns)
            except (KeyError, ValueError):
                skipped_rgb += 1
                continue

            if any(channel < 0 for channel in raw_rgb):
                skipped_rgb += 1
                continue

            max_rgb_value = max(max_rgb_value, *raw_rgb)
            matched_counts[basic_color] += 1
            candidate = {
                "source_row": source_row,
                "color_name_raw": row.get(name_col, ""),
                "basic_color": basic_color,
                "raw_rgb": raw_rgb,
            }

            bucket = reservoirs[basic_color]
            seen = matched_counts[basic_color]
            if len(bucket) < args.samples_per_color:
                bucket.append(candidate)
            else:
                replacement_index = rng.randrange(seen)
                if replacement_index < args.samples_per_color:
                    bucket[replacement_index] = candidate

    if args.rgb_scale == "auto":
        rgb_scale = "0-1" if max_rgb_value <= 1.0 else "0-255"
    else:
        rgb_scale = args.rgb_scale

    selected: list[dict] = []
    for color in BASIC_COLORS:
        for item in reservoirs[color]:
            try:
                rgb = convert_rgb(item["raw_rgb"], rgb_scale)
            except ValueError:
                skipped_rgb += 1
                continue

            selected.append(
                {
                    "source_row": item["source_row"],
                    "color_name_raw": item["color_name_raw"],
                    "basic_color": color,
                    "r": rgb[0],
                    "g": rgb[1],
                    "b": rgb[2],
                    "hex": f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}",
                }
            )

    selected.sort(key=lambda item: (item["basic_color"], item["source_row"]))

    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_path = args.output_dir / "ground_truth.csv"
    metadata_path = args.output_dir / "metadata.json"

    fieldnames = (
        "image_path",
        "filename",
        "basic_color",
        "color_name_raw",
        "r",
        "g",
        "b",
        "hex",
        "source_row",
    )

    counts = Counter()
    with ground_truth_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for item in selected:
            counts[item["basic_color"]] += 1
            filename = (
                f"{item['basic_color']}_"
                f"{counts[item['basic_color']]:03d}_"
                f"{safe_stem(item['hex'])}.png"
            )
            path = image_dir / filename
            write_solid_png(path, (item["r"], item["g"], item["b"]), args.patch_size)

            writer.writerow(
                {
                    "image_path": str(path),
                    "filename": filename,
                    **item,
                }
            )

    metadata = {
        "input_csv": str(args.csv_path),
        "output_dir": str(args.output_dir),
        "ground_truth_csv": str(ground_truth_path),
        "image_dir": str(image_dir),
        "basic_colors": list(BASIC_COLORS),
        "samples_per_color_requested": args.samples_per_color,
        "patch_size": args.patch_size,
        "seed": args.seed,
        "match_mode": args.match_mode,
        "rgb_scale": rgb_scale,
        "columns": {
            "r": rgb_columns[0],
            "g": rgb_columns[1],
            "b": rgb_columns[2],
            "name": name_col,
        },
        "rows_total": rows_total,
        "rows_matched_by_name": sum(matched_counts.values()),
        "rows_selected": len(selected),
        "rows_skipped_name": skipped_name,
        "rows_skipped_rgb": skipped_rgb,
        "available_by_color": {color: matched_counts.get(color, 0) for color in BASIC_COLORS},
        "counts_by_color": {color: counts.get(color, 0) for color in BASIC_COLORS},
        "shortfall_by_color": {
            color: max(0, args.samples_per_color - counts.get(color, 0))
            for color in BASIC_COLORS
        },
    }

    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata

def main() -> None:
    args = parse_args()
    metadata = build_dataset(args)

    print(f"Selected {metadata['rows_selected']} rows from {metadata['rows_total']} CSV rows.")
    print(f"Wrote images to: {metadata['image_dir']}")
    print(f"Wrote ground truth to: {metadata['ground_truth_csv']}")
    print("Counts by color:")
    for color, count in metadata["counts_by_color"].items():
        print(f"  {color}: {count}")


if __name__ == "__main__":
    main()
