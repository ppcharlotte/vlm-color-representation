#!/usr/bin/env python3
"""Aggregate basic-color survey answers to one row per RGB value.

Input is the extracted answers CSV from extract_basic_color_answers.py.
Output preserves soft label counts/proportions and adds a majority hard label.
"""

from __future__ import annotations

import argparse
import csv
import math
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


class RGBStats:
    def __init__(self, r: int, g: int, b: int) -> None:
        self.r = r
        self.g = g
        self.b = b
        self.counts = Counter()
        self.example_names = []

    def add(self, basic_color: str, colorname: str, max_examples: int) -> None:
        self.counts[basic_color] += 1
        if colorname and colorname not in self.example_names and len(self.example_names) < max_examples:
            self.example_names.append(colorname)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def majority(self) -> tuple[str, int]:
        return self.counts.most_common(1)[0]

    def agreement_rate(self) -> float:
        _, count = self.majority()
        return count / self.total if self.total else 0.0

    def entropy(self) -> float:
        if not self.total:
            return 0.0
        value = 0.0
        for count in self.counts.values():
            p = count / self.total
            value -= p * math.log2(p)
        return value

    def label_set(self) -> str:
        return "|".join(color for color in BASIC_COLORS if self.counts[color] > 0)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate basic-color answers by RGB.")
    parser.add_argument(
        "input_csv",
        type=Path,
        default=Path("data/colorsurvey_basic_answers.csv"),
        nargs="?",
        help="Input extracted answer CSV. Default: data/colorsurvey_basic_answers.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/colorsurvey_rgb_aggregated.csv"),
        help="Output aggregated CSV. Default: data/colorsurvey_rgb_aggregated.csv",
    )
    parser.add_argument(
        "--filtered-output",
        type=Path,
        default=Path("data/colorsurvey_rgb_aggregated_high_agreement.csv"),
        help="Output filtered high-agreement CSV.",
    )
    parser.add_argument(
        "--min-agreement",
        type=float,
        default=0.6,
        help="Agreement threshold for filtered output. Default: 0.6",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=1,
        help="Minimum total votes for filtered output. Default: 1",
    )
    parser.add_argument(
        "--max-example-names",
        type=int,
        default=8,
        help="Maximum original color names stored per RGB. Default: 8",
    )
    return parser.parse_args()



def row_for_stats(stats: RGBStats) -> dict:
    majority_color, majority_count = stats.majority()
    total = stats.total
    row = {
        "r": stats.r,
        "g": stats.g,
        "b": stats.b,
        "hex": f"#{stats.r:02X}{stats.g:02X}{stats.b:02X}",
        "total_votes": total,
        "majority_basic_color": majority_color,
        "majority_count": majority_count,
        "agreement_rate": round(stats.agreement_rate(), 6),
        "label_entropy_bits": round(stats.entropy(), 6),
        "num_basic_labels": sum(1 for color in BASIC_COLORS if stats.counts[color] > 0),
        "basic_label_set": stats.label_set(),
        "example_colornames": "|".join(stats.example_names),
    }

    for color in BASIC_COLORS:
        count = stats.counts[color]
        row[f"count_{color}"] = count
        row[f"p_{color}"] = round(count / total, 6) if total else 0.0

    return row



def aggregate(args: argparse.Namespace) -> tuple[int, int, int]:
    rgb_stats: dict[tuple[int, int, int], RGBStats] = {}
    input_rows = 0

    with args.input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            input_rows += 1
            r = int(row["r"])
            g = int(row["g"])
            b = int(row["b"])
            key = (r, g, b)
            if key not in rgb_stats:
                rgb_stats[key] = RGBStats(r, g, b)
            rgb_stats[key].add(row["basic_color"], row.get("colorname", ""), args.max_example_names)

    fieldnames = [
        "r",
        "g",
        "b",
        "hex",
        "total_votes",
        "majority_basic_color",
        "majority_count",
        "agreement_rate",
        "label_entropy_bits",
        "num_basic_labels",
        "basic_label_set",
        "example_colornames",
    ]
    for color in BASIC_COLORS:
        fieldnames.extend([f"count_{color}", f"p_{color}"])

    rows = [row_for_stats(stats) for stats in rgb_stats.values()]
    rows.sort(key=lambda row: (row["majority_basic_color"], -row["agreement_rate"], -row["total_votes"], row["r"], row["g"], row["b"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    filtered_rows = [
        row
        for row in rows
        if float(row["agreement_rate"]) >= args.min_agreement and int(row["total_votes"]) >= args.min_votes
    ]
    args.filtered_output.parent.mkdir(parents=True, exist_ok=True)
    with args.filtered_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    return input_rows, len(rows), len(filtered_rows)



def main() -> None:
    args = parse_args()
    input_rows, aggregated_rows, filtered_rows = aggregate(args)
    print(f"Input rows: {input_rows}")
    print(f"Aggregated unique RGB rows: {aggregated_rows}")
    print(f"Filtered rows: {filtered_rows}")
    print(f"Wrote aggregated CSV: {args.output}")
    print(f"Wrote filtered CSV: {args.filtered_output}")


if __name__ == "__main__":
    main()
