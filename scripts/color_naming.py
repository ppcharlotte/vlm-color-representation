#!/usr/bin/env python3
"""Build color patches and run LLaVA color naming for two prompt conditions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from structured_color_utils import (
    BASIC_COLORS,
    FORCE_CHOICE_PROMPT,
    NO_COLOR_LIST_PROMPT,
    LlavaHelper,
    class_metrics,
    confusion_matrix,
    generate_patch_dataset,
    normalize_color_answer,
    sample_color_rows,
    save_confusion_plot,
    save_error_cases_plot,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pure color patches and evaluate LLaVA color naming.")
    parser.add_argument("--input-csv", type=Path, default=Path("data/colorsurvey_rgb_aggregated_new.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/llava_structured_alignment"))
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--samples-per-color", type=int, default=100)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt-condition", choices=("all", "force_choice", "no_color_list"), default="all")
    parser.add_argument("--skip-generation", action="store_true", help="Only create dataset/patches; do not run the VLM.")
    return parser.parse_args()


def run_condition(helper: LlavaHelper, dataset_rows: list[dict[str, str]], output_dir: Path, condition: str, prompt: str, args: argparse.Namespace) -> dict:
    rows = []
    for row in dataset_rows:
        generation = helper.generate_details(row["absolute_image_path"], prompt, args.max_new_tokens, args.temperature)
        raw = generation["text"]
        normalized, parse_status = normalize_color_answer(raw)
        true_color = row["basic_color"]
        rows.append(
            {
                "sample_index": row["sample_index"],
                "image_path": row["image_path"],
                "absolute_image_path": row["absolute_image_path"],
                "hex": row.get("hex"),
                "r": row.get("r"),
                "g": row.get("g"),
                "b": row.get("b"),
                "true_color": true_color,
                "raw_response": raw,
                "raw_response_with_special_tokens": generation["text_with_special_tokens"],
                "generated_token_ids": generation["generated_token_ids"],
                "input_token_count": generation["input_token_count"],
                "sequence_token_count": generation["sequence_token_count"],
                "output_token_count": generation["output_token_count"],
                "normalized_pred_color": normalized or "",
                "parse_status": parse_status,
                "correct": int(normalized == true_color),
            }
        )

    condition_dir = output_dir / "color_naming"
    pred_path = condition_dir / f"{condition}_predictions.csv"
    write_csv(
        pred_path,
        [
            "sample_index",
            "image_path",
            "absolute_image_path",
            "hex",
            "r",
            "g",
            "b",
            "true_color",
            "raw_response",
            "raw_response_with_special_tokens",
            "generated_token_ids",
            "input_token_count",
            "sequence_token_count",
            "output_token_count",
            "normalized_pred_color",
            "parse_status",
            "correct",
        ],
        rows,
    )
    metrics = class_metrics(rows)
    metrics_path = condition_dir / f"{condition}_class_metrics.csv"
    write_csv(metrics_path, ["color", "support", "predicted", "true_positive", "accuracy_recall", "precision", "f1"], metrics)

    labels = [row["true_color"] for row in rows]
    preds = [row["normalized_pred_color"] for row in rows]
    conf = confusion_matrix(labels, preds, BASIC_COLORS)
    conf_rows = []
    for color, values in zip(BASIC_COLORS, conf):
        out = {"true_color": color}
        out.update({pred_color: int(value) for pred_color, value in zip(BASIC_COLORS, values)})
        conf_rows.append(out)
    conf_path = condition_dir / f"{condition}_confusion_matrix.csv"
    write_csv(conf_path, ["true_color", *BASIC_COLORS], conf_rows)

    plot_path = output_dir / "plots" / f"{condition}_confusion_matrix.png"
    save_confusion_plot(plot_path, conf, BASIC_COLORS, f"{condition} Confusion Matrix")
    error_plot_path = output_dir / "plots" / f"{condition}_error_cases.png"
    save_error_cases_plot(error_plot_path, rows, output_dir / "data")

    correct = [int(row["correct"]) for row in rows]
    summary = {
        "condition": condition,
        "prompt": prompt,
        "n_samples": len(rows),
        "parse_rate": float(np.mean([bool(row["normalized_pred_color"]) for row in rows])) if rows else float("nan"),
        "accuracy": float(np.mean(correct)) if rows else float("nan"),
        "predictions_csv": str(pred_path),
        "class_metrics_csv": str(metrics_path),
        "confusion_matrix_csv": str(conf_path),
        "confusion_matrix_plot": str(plot_path) if plot_path.exists() else None,
        "error_cases_plot": str(error_plot_path) if error_plot_path.exists() else None,
    }
    write_json(condition_dir / f"{condition}_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    output = args.output_dir
    data_dir = output / "data"
    selected = sample_color_rows(args.input_csv, args.samples_per_color, args.seed)
    dataset = generate_patch_dataset(selected, data_dir, args.patch_size)
    fieldnames = list(dataset.rows[0].keys()) if dataset.rows else []
    write_csv(data_dir / "selected_color_samples.csv", fieldnames, dataset.rows)
    write_csv(data_dir / "ground_truth.csv", fieldnames, dataset.rows)

    selected_counts = Counter(row["basic_color"] for row in dataset.rows)
    rule_counts = defaultdict(Counter)
    for row in dataset.rows:
        rule_counts[row["basic_color"]][row.get("selection_majority_rule", "unknown")] += 1
    summaries = {
        "input_csv": str(args.input_csv),
        "output_dir": str(output),
        "model_id": args.model_id,
        "samples_per_color": args.samples_per_color,
        "patch_size": args.patch_size,
        "seed": args.seed,
        "n_samples": len(dataset.rows),
        "ground_truth_csv": str(data_dir / "ground_truth.csv"),
        "majority_count_filter": "Use majority_count > 2 by default; if a color has fewer than 10 such samples, fall back to majority_count >= 2 for that color. For black, fill from majority_count == 2 when strict samples are fewer than samples_per_color.",
        "selected_counts_by_color": {color: int(selected_counts[color]) for color in BASIC_COLORS},
        "selection_rule_counts_by_color": {color: dict(rule_counts[color]) for color in BASIC_COLORS},
        "conditions": {},
    }
    if not args.skip_generation:
        helper = LlavaHelper(args.model_id, args.device, args.dtype, args.seed)
        conditions = []
        if args.prompt_condition in {"all", "force_choice"}:
            conditions.append(("force_choice", FORCE_CHOICE_PROMPT))
        if args.prompt_condition in {"all", "no_color_list"}:
            conditions.append(("no_color_list", NO_COLOR_LIST_PROMPT))
        for condition, prompt in conditions:
            summaries["conditions"][condition] = run_condition(helper, dataset.rows, output, condition, prompt, args)
    write_json(output / "color_naming" / "summary.json", summaries)
    print(f"Wrote color naming outputs to: {output}")


if __name__ == "__main__":
    main()
