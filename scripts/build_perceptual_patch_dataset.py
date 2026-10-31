#!/usr/bin/env python3
"""Build the strictly filtered basic-color patch dataset used in the thesis."""

from __future__ import annotations

import argparse
from pathlib import Path

from structured_color_utils import (
    BASIC_COLORS,
    generate_perceptual_patch_dataset,
    sample_strict_majority_rows,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply strict human-agreement filtering and generate solid patches.")
    parser.add_argument("--input-csv", type=Path, default=Path("data/colorsurvey_rgb_aggregated_new.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/perceptual_alignment"))
    parser.add_argument("--samples-per-color", type=int, default=100)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected, available = sample_strict_majority_rows(args.input_csv, args.samples_per_color, args.seed)
    dataset = generate_perceptual_patch_dataset(selected, args.output_dir / "data", args.patch_size)
    ground_truth = args.output_dir / "data" / "ground_truth.csv"
    fieldnames = list(dataset.rows[0]) if dataset.rows else []
    write_csv(ground_truth, fieldnames, dataset.rows)
    write_json(
        args.output_dir / "data" / "metadata.json",
        {
            "input_csv": str(args.input_csv),
            "ground_truth_csv": str(ground_truth),
            "sample_count": len(dataset.rows),
            "samples_per_color_requested": args.samples_per_color,
            "available_after_strict_filter": available,
            "selected_by_color": {color: sum(row["basic_color"] == color for row in dataset.rows) for color in BASIC_COLORS},
            "patch_size": args.patch_size,
            "seed": args.seed,
            "filter": "agreement_rate > 0.75 and majority_count >= 2",
            "sampling_method": "reproducible random sample of at most N unique RGB values per majority basic-color category",
            "fallback_policy": "No class-specific fallback or relaxed black/white rule is used.",
            "human_target": "Majority labels define the controlled categories; p_<color> columns are retained for soft-label sensitivity analyses.",
        },
    )
    print(f"Wrote {len(dataset.rows)} strictly filtered color patches to: {args.output_dir / 'data'}")


if __name__ == "__main__":
    main()
