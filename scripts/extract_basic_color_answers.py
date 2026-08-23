#!/usr/bin/env python3
"""Extract basic-color answers from the XKCD color survey SQL dump.

The SQL dump contains lines like:
INSERT INTO "answers" VALUES(id,user_id,datestamp,r,g,b,'colorname');

This script streams the dump, filters answer rows whose free-text color name
maps unambiguously to one of the 11 basic color categories, and writes a CSV.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
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

ALIASES = {
    "grey": "gray",
}

ANSWER_PREFIX = 'INSERT INTO "answers" VALUES('


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract basic-color answers from mainsurvey_sqldump.txt into CSV."
    )
    parser.add_argument(
        "dump_path",
        type=Path,
        help="Path to mainsurvey_sqldump.txt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/colorsurvey_basic_answers.csv"),
        help="Output CSV path. Default: data/colorsurvey_basic_answers.csv",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )
    return parser.parse_args()


def split_sql_values(payload: str) -> list[str]:
    values: list[str] = []
    current: list[str] = []
    in_quote = False
    i = 0

    while i < len(payload):
        char = payload[i]
        if in_quote:
            if char == "'":
                if i + 1 < len(payload) and payload[i + 1] == "'":
                    current.append("'")
                    i += 2
                    continue
                in_quote = False
            else:
                current.append(char)
        else:
            if char == "'":
                in_quote = True
            elif char == ",":
                values.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        i += 1

    values.append("".join(current).strip())
    return values


def normalize_basic_color(colorname: str) -> str | None:
    normalized = colorname.strip().lower()
    normalized = normalized.replace("_", " ").replace("-", " ").replace("/", " ")
    tokens = re.findall(r"[a-z]+", normalized)
    mapped = [ALIASES.get(token, token) for token in tokens]
    matches = [token for token in mapped if token in BASIC_COLORS]

    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]
    return None


def as_int(value: str) -> int:
    return int(float(value))


def extract_answers(dump_path: Path, output_path: Path, max_rows: int | None = None) -> Counter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    total_answer_rows = 0
    selected_rows = 0
    skipped_ambiguous_or_nonbasic = 0
    skipped_parse_error = 0

    fieldnames = [
        "id",
        "user_id",
        "datestamp",
        "r",
        "g",
        "b",
        "hex",
        "colorname",
        "basic_color",
    ]

    with dump_path.open("r", encoding="utf-8", errors="replace") as source, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()

        for line in source:
            if not line.startswith(ANSWER_PREFIX):
                continue

            total_answer_rows += 1
            payload = line[len(ANSWER_PREFIX) :].rstrip(";\n")
            if payload.endswith(")"):
                payload = payload[:-1]

            try:
                values = split_sql_values(payload)
                if len(values) != 7:
                    raise ValueError(f"Expected 7 values, got {len(values)}")

                answer_id, user_id, datestamp, r_value, g_value, b_value, colorname = values
                basic_color = normalize_basic_color(colorname)
                if basic_color is None:
                    skipped_ambiguous_or_nonbasic += 1
                    continue

                r = as_int(r_value)
                g = as_int(g_value)
                b = as_int(b_value)
                if not all(0 <= channel <= 255 for channel in (r, g, b)):
                    raise ValueError("RGB channel out of range")

                writer.writerow(
                    {
                        "id": int(answer_id),
                        "user_id": int(user_id),
                        "datestamp": datestamp,
                        "r": r,
                        "g": g,
                        "b": b,
                        "hex": f"#{r:02X}{g:02X}{b:02X}",
                        "colorname": colorname,
                        "basic_color": basic_color,
                    }
                )
                counts[basic_color] += 1
                selected_rows += 1

                if max_rows is not None and selected_rows >= max_rows:
                    break
            except Exception:
                skipped_parse_error += 1
                continue

    counts["__total_answer_rows_seen"] = total_answer_rows
    counts["__selected_rows"] = selected_rows
    counts["__skipped_ambiguous_or_nonbasic"] = skipped_ambiguous_or_nonbasic
    counts["__skipped_parse_error"] = skipped_parse_error
    return counts


def main() -> None:
    args = parse_args()
    counts = extract_answers(args.dump_path, args.output, args.max_rows)
    print(f"Wrote: {args.output}")
    print(f"Answer rows scanned: {counts['__total_answer_rows_seen']}")
    print(f"Rows selected: {counts['__selected_rows']}")
    print(f"Skipped non-basic/ambiguous: {counts['__skipped_ambiguous_or_nonbasic']}")
    print(f"Skipped parse errors: {counts['__skipped_parse_error']}")
    print("Counts by basic color:")
    for color in BASIC_COLORS:
        print(f"  {color}: {counts[color]}")


if __name__ == "__main__":
    main()
