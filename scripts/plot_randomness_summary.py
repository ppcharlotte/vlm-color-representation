#!/usr/bin/env python3
"""Plot robustness across repeated random-sampling runs.

Each run directory should be a full pipeline output, for example:
experiments/llava_seed1, experiments/llava_seed2, experiments/llava_seed3.
The script overlays one thin line per seed/run and a thick mean +/- SD band.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

DEFAULT_SPECS = {
    "image_no_color_list_cielab_centroid_rsa": {
        "relative_csv": "rdm/image_cielab_alignment.csv",
        "filter_column": "representation",
        "filter_value": "image_no_color_list",
        "x_column": "layer_index",
        "y_column": "centroid_rsa_spearman",
        "ylabel": "Spearman r",
        "title": "Image vs CIEDE2000 Centroid RSA Across Random Seeds",
    },
    "image_no_color_list_cielab_sample_rsa": {
        "relative_csv": "rdm/image_cielab_alignment.csv",
        "filter_column": "representation",
        "filter_value": "image_no_color_list",
        "x_column": "layer_index",
        "y_column": "sample_rsa_spearman",
        "ylabel": "Spearman r",
        "title": "Image vs CIEDE2000 Sample RSA Across Random Seeds",
    },
    "query_no_color_list_text_rsa": {
        "relative_csv": "similarity/query_no_color_list_layerwise.csv",
        "x_column": "layer_index",
        "y_column": "rsa_spearman",
        "ylabel": "Spearman r",
        "title": "Query vs Text Color RSA Across Random Seeds",
    },
    "cee_text_rsa": {
        "relative_csv": "similarity/cee_ground_truth_layerwise.csv",
        "x_column": "layer_index",
        "y_column": "rsa_spearman",
        "ylabel": "Spearman r",
        "title": "CEe vs Text Color RSA Across Random Seeds",
    },
    "ceg_decision_cielab_centroid_rsa": {
        "relative_csv": "ceg_generation_side/rdm/ceg_decision_cielab_alignment.csv",
        "x_column": "layer_index",
        "y_column": "centroid_rsa_spearman",
        "ylabel": "Spearman r",
        "title": "CEg-decision vs CIEDE2000 Centroid RSA Across Random Seeds",
    },
    "ceg_token_from_generate_cielab_centroid_rsa": {
        "relative_csv": "ceg_generation_side/rdm/ceg_token_from_generate_cielab_alignment.csv",
        "x_column": "layer_index",
        "y_column": "centroid_rsa_spearman",
        "ylabel": "Spearman r",
        "title": "CEg-token-from-generate vs CIEDE2000 Centroid RSA Across Random Seeds",
    },
    "ceg_decision_image_centroid_rsa": {
        "relative_csv": "ceg_generation_side/rdm/ceg_decision_image_alignment.csv",
        "x_column": "layer_index",
        "y_column": "centroid_rsa_spearman",
        "ylabel": "Spearman r",
        "title": "CEg-decision vs Image Centroid RSA Across Random Seeds",
    },
    "ceg_token_from_generate_image_centroid_rsa": {
        "relative_csv": "ceg_generation_side/rdm/ceg_token_from_generate_image_alignment.csv",
        "x_column": "layer_index",
        "y_column": "centroid_rsa_spearman",
        "ylabel": "Spearman r",
        "title": "CEg-token-from-generate vs Image Centroid RSA Across Random Seeds",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot mean +/- SD across random-sampling experiment runs.")
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True, help="Pipeline output directories, one per seed/run.")
    parser.add_argument("--run-labels", default="", help="Optional comma-separated labels. Defaults to directory names.")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/randomness_plots"))
    parser.add_argument("--spec", action="append", choices=sorted(DEFAULT_SPECS), help="Preset plot spec. Can be repeated. Defaults to all available specs.")
    parser.add_argument("--relative-csv", default="", help="Custom CSV path relative to each run dir.")
    parser.add_argument("--x-column", default="layer_index")
    parser.add_argument("--y-column", default="")
    parser.add_argument("--filter-column", default="")
    parser.add_argument("--filter-value", default="")
    parser.add_argument("--title", default="")
    parser.add_argument("--ylabel", default="")
    parser.add_argument("--name", default="custom_randomness_plot")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def collect_series(run_dirs: list[Path], labels: list[str], spec: dict) -> tuple[list[tuple[str, np.ndarray, np.ndarray]], list[str]]:
    series = []
    warnings = []
    for label, run_dir in zip(labels, run_dirs):
        path = run_dir / spec["relative_csv"]
        if not path.exists():
            warnings.append(f"missing {label}: {path}")
            continue
        rows = read_rows(path)
        filter_column = spec.get("filter_column")
        filter_value = spec.get("filter_value")
        if filter_column and filter_value:
            rows = [row for row in rows if row.get(filter_column) == filter_value]
        x_col = spec.get("x_column", "layer_index")
        y_col = spec["y_column"]
        points = []
        for row in rows:
            x = finite_float(row.get(x_col, ""))
            y = finite_float(row.get(y_col, ""))
            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))
        if not points:
            warnings.append(f"no finite points {label}: {path}")
            continue
        points = sorted(points)
        xs = np.asarray([p[0] for p in points], dtype=float)
        ys = np.asarray([p[1] for p in points], dtype=float)
        series.append((label, xs, ys))
    return series, warnings


def aligned_matrix(series: list[tuple[str, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    common = sorted(set(series[0][1].tolist()).intersection(*[set(xs.tolist()) for _label, xs, _ys in series[1:]]))
    xs_common = np.asarray(common, dtype=float)
    rows = []
    for _label, xs, ys in series:
        mapping = {float(x): float(y) for x, y in zip(xs, ys)}
        rows.append([mapping[x] for x in xs_common])
    return xs_common, np.asarray(rows, dtype=float)


def plot_spec(run_dirs: list[Path], labels: list[str], output_dir: Path, name: str, spec: dict) -> Path | None:
    import matplotlib.pyplot as plt
    series, warnings = collect_series(run_dirs, labels, spec)
    for warning in warnings:
        print(f"Warning: {warning}")
    if not series:
        print(f"Skipping {name}: no data found.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for label, xs, ys in series:
        plt.plot(xs, ys, marker="o", linewidth=1.2, alpha=0.35, label=label)

    if len(series) >= 2:
        xs_common, matrix = aligned_matrix(series)
        mean = np.nanmean(matrix, axis=0)
        sd = np.nanstd(matrix, axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros_like(mean)
        plt.plot(xs_common, mean, color="black", marker="o", linewidth=2.8, label="mean")
        plt.fill_between(xs_common, mean - sd, mean + sd, color="black", alpha=0.16, label="mean +/- SD")

    plt.title(spec.get("title") or name)
    plt.xlabel(spec.get("x_column", "layer_index"))
    plt.ylabel(spec.get("ylabel") or spec["y_column"])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = output_dir / f"{name}.png"
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"Wrote {path}")
    return path


def main() -> None:
    args = parse_args()
    labels = [part.strip() for part in args.run_labels.split(",") if part.strip()]
    if not labels:
        labels = [path.name for path in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        raise ValueError("--run-labels must have the same number of entries as --run-dirs.")

    if args.relative_csv:
        if not args.y_column:
            raise ValueError("--y-column is required for a custom --relative-csv plot.")
        spec = {
            "relative_csv": args.relative_csv,
            "x_column": args.x_column,
            "y_column": args.y_column,
            "filter_column": args.filter_column,
            "filter_value": args.filter_value,
            "title": args.title or args.name,
            "ylabel": args.ylabel or args.y_column,
        }
        plot_spec(args.run_dirs, labels, args.output_dir, args.name, spec)
        return

    names = args.spec or list(DEFAULT_SPECS)
    for name in names:
        plot_spec(args.run_dirs, labels, args.output_dir, name, DEFAULT_SPECS[name])


if __name__ == "__main__":
    main()
