#!/usr/bin/env python3
"""Relate representation metrics to final color naming behavior."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from structured_color_utils import BASIC_COLORS, linear_regression_summary, logistic_regression_1d, permutation_pvalue, read_csv_rows, spearman_corr, write_csv, write_json


SOURCE_TO_CONDITION = {
    "image_force_choice": "force_choice",
    "query_force_choice": "force_choice",
    "answer_force_choice": "force_choice",
    "answer_lm_head_force_choice": "force_choice",
    "image_no_color_list": "no_color_list",
    "query_no_color_list": "no_color_list",
    "answer_no_color_list": "no_color_list",
    "answer_lm_head_no_color_list": "no_color_list",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze relationships between representation metrics and color naming behavior.")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/llava_structured_alignment"))
    parser.add_argument("--permutation-rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_behavior(output: Path, condition: str) -> tuple[dict[int, float], dict[str, float]]:
    rows = read_csv_rows(output / "color_naming" / f"{condition}_predictions.csv")
    sample_correct = {int(row["sample_index"]): float(row["correct"]) for row in rows}
    class_accuracy = {}
    for color in BASIC_COLORS:
        color_rows = [row for row in rows if row["true_color"] == color]
        class_accuracy[color] = float(np.mean([float(row["correct"]) for row in color_rows])) if color_rows else float("nan")
    return sample_correct, class_accuracy


def analyze_source(output: Path, source: str, condition: str, rounds: int, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    sample_correct, class_accuracy = load_behavior(output, condition)
    sample_path = output / "similarity" / f"{source}_sample_metrics.csv"
    class_path = output / "similarity" / f"{source}_class_metrics.csv"
    if not sample_path.exists() or not class_path.exists():
        print(f"Skipping missing source: {source}")
        return [], [], []
    sample_rows = read_csv_rows(sample_path)
    class_rows = read_csv_rows(class_path)
    layer_indices = sorted({int(row["layer_index"]) for row in sample_rows})
    sample_out = []
    class_out = []
    regression_out = []
    for layer_idx in layer_indices:
        layer_sample = [row for row in sample_rows if int(row["layer_index"]) == layer_idx]
        y = np.asarray([sample_correct.get(int(row["sample_index"]), np.nan) for row in layer_sample], dtype=float)
        for metric in ["correct_similarity", "pair_margin"]:
            x = np.asarray([float(row[metric]) for row in layer_sample], dtype=float)
            r, p, n = permutation_pvalue(x, y, spearman_corr, rounds, seed + layer_idx + (0 if metric == "correct_similarity" else 10000))
            logit = logistic_regression_1d(x, y)
            sample_out.append({"source": source, "condition": condition, "layer_index": layer_idx, "metric": metric, "target": "sample_correct", "spearman_r": r, "spearman_p": p, "n": n})
            regression_out.append({"source": source, "condition": condition, "layer_index": layer_idx, "metric": metric, "target": "sample_correct", "model": "logistic_regression_1d", **logit})

        layer_class = [row for row in class_rows if int(row["layer_index"]) == layer_idx]
        y_class = np.asarray([class_accuracy.get(row["color"], np.nan) for row in layer_class], dtype=float)
        for metric in ["mean_correct_similarity", "mean_pair_margin", "classwise_rsa_spearman"]:
            x_class = np.asarray([float(row[metric]) for row in layer_class], dtype=float)
            r, p, n = permutation_pvalue(x_class, y_class, spearman_corr, rounds, seed + layer_idx + len(metric) * 37)
            lin = linear_regression_summary(x_class, y_class)
            class_out.append({"source": source, "condition": condition, "layer_index": layer_idx, "metric": metric, "target": "class_accuracy", "spearman_r": r, "spearman_p": p, "n": n})
            regression_out.append({"source": source, "condition": condition, "layer_index": layer_idx, "metric": metric, "target": "class_accuracy", "model": "linear_regression_1d", **lin})
    return sample_out, class_out, regression_out


def main() -> None:
    args = parse_args()
    output = args.output_dir
    all_sample = []
    all_class = []
    all_reg = []
    for source, condition in SOURCE_TO_CONDITION.items():
        sample, class_level, reg = analyze_source(output, source, condition, args.permutation_rounds, args.seed)
        all_sample.extend(sample)
        all_class.extend(class_level)
        all_reg.extend(reg)
    relation_dir = output / "relation"
    sample_path = relation_dir / "sample_level_relations.csv"
    class_path = relation_dir / "class_level_relations.csv"
    reg_path = relation_dir / "regression_results.csv"
    write_csv(sample_path, ["source", "condition", "layer_index", "metric", "target", "spearman_r", "spearman_p", "n"], all_sample)
    write_csv(class_path, ["source", "condition", "layer_index", "metric", "target", "spearman_r", "spearman_p", "n"], all_class)
    write_csv(reg_path, ["source", "condition", "layer_index", "metric", "target", "model", "n", "coef", "intercept", "accuracy", "r2", "slope"], all_reg)
    best_sample = max(all_sample, key=lambda row: row["spearman_r"] if np.isfinite(row["spearman_r"]) else -999) if all_sample else {}
    best_class = max(all_class, key=lambda row: row["spearman_r"] if np.isfinite(row["spearman_r"]) else -999) if all_class else {}
    write_json(
        relation_dir / "summary.json",
        {
            "sample_level_relations_csv": str(sample_path),
            "class_level_relations_csv": str(class_path),
            "regression_results_csv": str(reg_path),
            "best_sample_relation": best_sample,
            "best_class_relation": best_class,
        },
    )
    print(f"Wrote relation analysis outputs to: {relation_dir}")


if __name__ == "__main__":
    main()
