#!/usr/bin/env python3
"""One-command runner for the modular LLaVA structured color pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from structured_color_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full modular LLaVA color representation pipeline.")
    parser.add_argument("--input-csv", type=Path, default=Path("data/colorsurvey_rgb_aggregated_new.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/llava_structured_alignment"))
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--layer-indices", default="0,1,8,16,24,32")
    parser.add_argument("--samples-per-color", type=int, default=100)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved for future batching; current LLaVA extraction is sample-wise.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutation-rounds", type=int, default=1000)
    parser.add_argument("--control-rounds", type=int, default=100)
    parser.add_argument("--skip-color-naming", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-negative-control", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def run_step(name: str, cmd: list[str]) -> None:
    print(f"\n=== {name} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def maybe_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> None:
    args = parse_args()
    py = sys.executable
    script_dir = Path(__file__).resolve().parent
    output = args.output_dir
    common_model = ["--model-id", args.model_id, "--device", args.device, "--dtype", args.dtype, "--seed", str(args.seed)]

    if not args.skip_color_naming:
        run_step(
            "color naming",
            [
                py,
                str(script_dir / "color_naming.py"),
                "--input-csv",
                str(args.input_csv),
                "--output-dir",
                str(output),
                "--samples-per-color",
                str(args.samples_per_color),
                "--patch-size",
                str(args.patch_size),
                *common_model,
            ],
        )

    ground_truth = output / "data" / "ground_truth.csv"
    dataset_root = output / "data"
    if not args.skip_embeddings:
        plot_flag = ["--no-plots"] if args.no_plots else []
        run_step(
            "image embeddings",
            [
                py,
                str(script_dir / "image_embedding_extraction.py"),
                "--ground-truth",
                str(ground_truth),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(output),
                "--layer-indices",
                args.layer_indices,
                "--permutation-rounds",
                str(args.permutation_rounds),
                *common_model,
                *plot_flag,
            ],
        )
        run_step(
            "text embeddings",
            [py, str(script_dir / "text_embedding_extraction.py"), "--output-dir", str(output), "--layer-indices", args.layer_indices, *common_model],
        )
        run_step(
            "text perceptual RSA",
            [
                py,
                str(script_dir / "text_perceptual_rsa.py"),
                "--ground-truth",
                str(ground_truth),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(output),
                "--permutation-rounds",
                str(args.permutation_rounds),
                "--seed",
                str(args.seed),
                *plot_flag,
            ],
        )
        run_step(
            "query embeddings",
            [
                py,
                str(script_dir / "query_embedding_extraction.py"),
                "--ground-truth",
                str(ground_truth),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(output),
                "--layer-indices",
                args.layer_indices,
                *common_model,
            ],
        )

    plot_flag = ["--no-plots"] if args.no_plots else []
    if not args.skip_embeddings:
        run_step(
            "answer-position embeddings",
            [
                py,
                str(script_dir / "answer_position_embedding_analysis.py"),
                "--ground-truth",
                str(ground_truth),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(output),
                "--model-id",
                args.model_id,
                "--device",
                args.device,
                "--dtype",
                args.dtype,
                "--layer-indices",
                args.layer_indices,
                "--permutation-rounds",
                str(args.permutation_rounds),
                "--seed",
                str(args.seed),
                *plot_flag,
            ],
        )
        run_step(
            "CEe ground-truth encoding embeddings",
            [
                py,
                str(script_dir / "context_embedding_encoding_analysis.py"),
                "--ground-truth",
                str(ground_truth),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(output),
                "--model-id",
                args.model_id,
                "--device",
                args.device,
                "--dtype",
                args.dtype,
                "--layer-indices",
                args.layer_indices,
                "--permutation-rounds",
                str(args.permutation_rounds),
                "--seed",
                str(args.seed),
                *plot_flag,
            ],
        )
    run_step(
        "LM-head perceptual RSA",
        [
            py,
            str(script_dir / "lm_head_perceptual_rsa.py"),
            "--ground-truth",
            str(ground_truth),
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(output),
            "--permutation-rounds",
            str(args.permutation_rounds),
            "--seed",
            str(args.seed),
            *plot_flag,
        ],
    )
    run_step(
        "similarity analysis",
        [
            py,
            str(script_dir / "similarity_analysis.py"),
            "--ground-truth",
            str(ground_truth),
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(output),
            "--permutation-rounds",
            str(args.permutation_rounds),
            "--seed",
            str(args.seed),
            *plot_flag,
        ],
    )
    run_step(
        "relation analysis",
        [py, str(script_dir / "relation_analysis.py"), "--output-dir", str(output), "--permutation-rounds", str(args.permutation_rounds), "--seed", str(args.seed)],
    )
    if not args.skip_negative_control:
        run_step(
            "negative control",
            [
                py,
                str(script_dir / "negative_control.py"),
                "--ground-truth",
                str(ground_truth),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(output),
                "--permutation-rounds",
                str(args.permutation_rounds),
                "--control-rounds",
                str(args.control_rounds),
                "--seed",
                str(args.seed),
            ],
        )

    summary = {
        "model_id": args.model_id,
        "input_csv": str(args.input_csv),
        "output_dir": str(output),
        "layer_indices": args.layer_indices,
        "color_naming": maybe_load(output / "color_naming" / "summary.json"),
        "image_embeddings": maybe_load(output / "embeddings" / "image_embedding_summary.json"),
        "text_embeddings": maybe_load(output / "embeddings" / "text_embedding_summary.json"),
        "text_cielab_alignment": maybe_load(output / "rdm" / "text_cielab_alignment_summary.json"),
        "query_embeddings": maybe_load(output / "embeddings" / "query_embedding_summary.json"),
        "answer_position": maybe_load(output / "generation_side" / "answer_position_summary.json"),
        "cee_ground_truth": maybe_load(output / "encoding_side" / "cee_ground_truth_summary.json"),
        "lm_head_cielab_alignment": maybe_load(output / "rdm" / "lm_head_cielab_alignment_summary.json"),
        "similarity": maybe_load(output / "similarity" / "summary.json"),
        "relation": maybe_load(output / "relation" / "summary.json"),
        "negative_control": maybe_load(output / "negative_control" / "summary.json"),
    }
    write_json(output / "summary.json", summary)
    print(f"\nWrote final pipeline summary to: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
