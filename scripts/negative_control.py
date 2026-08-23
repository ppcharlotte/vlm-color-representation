#!/usr/bin/env python3
"""Negative controls for color-word similarity and generative correctness relations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from structured_color_utils import BASIC_COLORS, cosine_matrix, labels_to_indices, load_dataset, permutation_pvalue, read_csv_rows, spearman_corr, write_csv, write_json


SOURCE_TO_CONDITION = {
    "image_force_choice": "force_choice",
    "query_force_choice": "force_choice",
    "image_no_color_list": "no_color_list",
    "query_no_color_list": "no_color_list",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run negative controls for text-label alignment analyses.")
    parser.add_argument("--ground-truth", type=Path, default=Path("experiments/llava_structured_alignment/data/ground_truth.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("experiments/llava_structured_alignment/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/llava_structured_alignment"))
    parser.add_argument("--permutation-rounds", type=int, default=1000)
    parser.add_argument("--control-rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def behavior_correct(output: Path, condition: str) -> np.ndarray:
    rows = read_csv_rows(output / "color_naming" / f"{condition}_predictions.csv")
    rows = sorted(rows, key=lambda row: int(row["sample_index"]))
    return np.asarray([float(row["correct"]) for row in rows], dtype=float)


def source_layers(output: Path, source: str) -> tuple[np.ndarray, list[int]]:
    if source == "image_force_choice":
        data = np.load(output / "embeddings" / "image_embeddings_force_choice.npz", allow_pickle=True)
        return data["image_layer_embeddings"], [int(x) for x in data["layer_indices"].tolist()]
    if source == "image_no_color_list":
        data = np.load(output / "embeddings" / "image_embeddings_no_color_list.npz", allow_pickle=True)
        return data["image_layer_embeddings"], [int(x) for x in data["layer_indices"].tolist()]
    if source == "query_force_choice":
        data = np.load(output / "embeddings" / "query_embeddings_force_choice.npz", allow_pickle=True)
        return data["query_layer_embeddings"], [int(x) for x in data["layer_indices"].tolist()]
    data = np.load(output / "embeddings" / "query_embeddings_no_color_list.npz", allow_pickle=True)
    return data["query_layer_embeddings"], [int(x) for x in data["layer_indices"].tolist()]


def main() -> None:
    args = parse_args()
    output = args.output_dir
    dataset = load_dataset(args.ground_truth, args.dataset_root)
    labels = dataset.labels
    label_indices = labels_to_indices(labels, BASIC_COLORS)
    text_npz = np.load(output / "embeddings" / "text_embeddings.npz", allow_pickle=True)
    text_layers = text_npz["text_layer_embeddings"]
    rng = np.random.default_rng(args.seed)
    rows = []
    for source, condition in SOURCE_TO_CONDITION.items():
        layers, layer_indices = source_layers(output, source)
        correct = behavior_correct(output, condition)
        for array_idx, layer_idx in enumerate(layer_indices):
            sims = cosine_matrix(layers[array_idx], text_layers[array_idx])
            correct_sim = sims[np.arange(len(label_indices)), label_indices]
            wrong_mask = np.ones_like(sims, dtype=bool)
            wrong_mask[np.arange(len(label_indices)), label_indices] = False
            max_wrong_sim = np.max(np.where(wrong_mask, sims, -np.inf), axis=1)
            wrong_r, wrong_p, wrong_n = permutation_pvalue(max_wrong_sim, correct, spearman_corr, args.permutation_rounds, args.seed + array_idx)
            correct_r, correct_p, correct_n = permutation_pvalue(correct_sim, correct, spearman_corr, args.permutation_rounds, args.seed + 1000 + array_idx)
            permuted_rs = []
            for _ in range(args.control_rounds):
                perm = rng.permutation(len(BASIC_COLORS))
                perm_sims = sims[:, perm]
                perm_correct = perm_sims[np.arange(len(label_indices)), label_indices]
                r = spearman_corr(perm_correct, correct)
                if np.isfinite(r):
                    permuted_rs.append(r)
            permuted_rs = np.asarray(permuted_rs, dtype=float)
            rows.append(
                {
                    "source": source,
                    "condition": condition,
                    "layer_index": int(layer_idx),
                    "correct_similarity_r": correct_r,
                    "correct_similarity_p": correct_p,
                    "correct_similarity_n": correct_n,
                    "max_wrong_similarity_r": wrong_r,
                    "max_wrong_similarity_p": wrong_p,
                    "max_wrong_similarity_n": wrong_n,
                    "label_permuted_similarity_r_mean": float(np.mean(permuted_rs)) if len(permuted_rs) else float("nan"),
                    "label_permuted_similarity_r_std": float(np.std(permuted_rs)) if len(permuted_rs) else float("nan"),
                    "label_permuted_similarity_r_max": float(np.max(permuted_rs)) if len(permuted_rs) else float("nan"),
                    "control_rounds": args.control_rounds,
                }
            )
    out_dir = output / "negative_control"
    path = out_dir / "negative_control_results.csv"
    write_csv(
        path,
        [
            "source",
            "condition",
            "layer_index",
            "correct_similarity_r",
            "correct_similarity_p",
            "correct_similarity_n",
            "max_wrong_similarity_r",
            "max_wrong_similarity_p",
            "max_wrong_similarity_n",
            "label_permuted_similarity_r_mean",
            "label_permuted_similarity_r_std",
            "label_permuted_similarity_r_max",
            "control_rounds",
        ],
        rows,
    )
    best = max(rows, key=lambda row: row["correct_similarity_r"] if np.isfinite(row["correct_similarity_r"]) else -999)
    write_json(out_dir / "summary.json", {"negative_control_results_csv": str(path), "best_correct_similarity_relation": best})
    print(f"Wrote negative control outputs to: {out_dir}")


if __name__ == "__main__":
    main()
