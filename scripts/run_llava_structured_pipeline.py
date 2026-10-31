#!/usr/bin/env python3
"""Run the streamlined human-perceptual color alignment pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from structured_color_utils import write_json
from layer_selection import resolve_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract image/text/query/CEe/CEg embeddings and compare them with human color geometry.")
    parser.add_argument("--input-csv", type=Path, default=Path("data/colorsurvey_rgb_aggregated_new.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/perceptual_alignment"))
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--layer-indices", default="all", help="all (default): LM input state and every LM layer; or comma-separated indices")
    parser.add_argument("--samples-per-color", type=int, default=100)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-permutation-rounds", type=int, default=200)
    parser.add_argument("--sample-permutation-scope", choices=("core", "all", "none"), default="core")
    parser.add_argument("--category-permutation-rounds", type=int, default=10000)
    parser.add_argument("--skip-cee", action="store_true")
    parser.add_argument("--skip-ceg", action="store_true")
    parser.add_argument("--reuse-dataset", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--embedding-distances", default="cosine", help="Primary: cosine; pearson can be requested explicitly")
    parser.add_argument("--appendix-metrics", action="store_true", help="Optional non-generation CKA/probes")
    parser.add_argument("--cee-reference-difference", action="store_true", help="Test CEe-visual versus CEe-textual only for this comparative claim")
    parser.add_argument("--rsa-coefficients", default="spearman,kendall_tau_b")
    parser.add_argument("--bootstrap-rounds", type=int, default=2000, help="Legacy compatibility only; current Delta inference uses category-permutation-rounds")
    parser.add_argument("--cee-template", default="The color of this image is {color}")
    parser.add_argument("--mismatch-repeats", type=int, default=5)
    parser.add_argument("--mismatch-seed", type=int, default=None)
    return parser.parse_args()


def run_step(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===")
    subprocess.run(command, check=True)


def load_if_present(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> None:
    args = parse_args()
    args.layer_indices = ",".join(map(str, resolve_layers(args.layer_indices, args.model_id)))
    print(f"Language-model hidden-state indices: {args.layer_indices}", flush=True)
    py = sys.executable
    scripts = Path(__file__).resolve().parent
    output = args.output_dir
    ground_truth = output / "data" / "ground_truth.csv"
    dataset_root = output / "data"
    model_args = ["--model-id", args.model_id, "--device", args.device, "--dtype", args.dtype, "--seed", str(args.seed)]

    if not args.reuse_dataset:
        run_step(
            "strictly filtered basic-color stimulus set",
            [
                py,
                str(scripts / "build_perceptual_patch_dataset.py"),
                "--input-csv",
                str(args.input_csv),
                "--output-dir",
                str(output),
                "--samples-per-color",
                str(args.samples_per_color),
                "--patch-size",
                str(args.patch_size),
                "--seed",
                str(args.seed),
            ],
        )

    run_step(
        "image and query embeddings (one shared forward pass)",
        [
            py,
            str(scripts / "image_embedding_extraction.py"),
            "--ground-truth",
            str(ground_truth),
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(output),
            "--layer-indices",
            args.layer_indices,
            *model_args,
        ],
    )
    run_step(
        "text color-word embeddings",
        [py, str(scripts / "text_embedding_extraction.py"), "--output-dir", str(output), "--layer-indices", args.layer_indices, *model_args],
    )
    if not args.skip_cee:
        run_step(
            "CEe embeddings",
            [
                py,
                str(scripts / "context_embedding_encoding_analysis.py"),
                "--ground-truth",
                str(ground_truth),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(output),
                "--layer-indices",
                args.layer_indices,
                "--template", args.cee_template,
                "--mismatch-repeats", str(args.mismatch_repeats),
                *(["--mismatch-seed", str(args.mismatch_seed)] if args.mismatch_seed is not None else []),
                *model_args,
            ],
        )
    if not args.skip_ceg:
        run_step(
            "CEg decision and generated-token embeddings",
            [
                py,
                str(scripts / "generation_context_embedding_analysis.py"),
                "--ground-truth",
                str(ground_truth),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(output),
                "--layer-indices",
                args.layer_indices,
                *model_args,
            ],
        )

    plot_flag = ["--no-plots"] if args.no_plots else []
    run_step(
        "unified perceptual alignment",
        [
            py,
            str(scripts / "perceptual_alignment_analysis.py"),
            "--ground-truth",
            str(ground_truth),
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(output),
            "--sample-permutation-rounds",
            str(args.sample_permutation_rounds),
            "--sample-permutation-scope", args.sample_permutation_scope,
            "--category-permutation-rounds",
            str(args.category_permutation_rounds),
            "--seed",
            str(args.seed),
            "--embedding-distances", args.embedding_distances,
            "--rsa-coefficients", args.rsa_coefficients,
            "--bootstrap-rounds", str(args.bootstrap_rounds),
            *(["--appendix-metrics"] if args.appendix_metrics else []),
            *(["--cee-reference-difference"] if args.cee_reference_difference else []),
            *(["--exclude-cee"] if args.skip_cee else []),
            *(["--exclude-ceg"] if args.skip_ceg else []),
            *(["--exclude-mismatched"] if args.mismatch_repeats == 0 else []),
            *plot_flag,
        ],
    )

    summary = {
        "research_target": "Alignment of model representations with human perceptual color geometry",
        "model_id": args.model_id,
        "samples_per_color_requested": args.samples_per_color,
        "strict_human_label_filter": "agreement_rate > 0.75 and majority_count >= 2",
        "black_white_special_cases": False,
        "accuracy_relation_analysis": False,
        "dataset": load_if_present(output / "data" / "metadata.json"),
        "image": load_if_present(output / "embeddings" / "image_embedding_summary.json"),
        "text": load_if_present(output / "embeddings" / "text_embedding_summary.json"),
        "cee": load_if_present(output / "encoding_side" / "cee_ground_truth_summary.json"),
        "ceg": load_if_present(output / "ceg_generation_side" / "summary.json"),
        "alignment": load_if_present(output / "analysis" / "summary.json"),
    }
    write_json(output / "summary.json", summary)
    print(f"\nWrote pipeline summary to: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
