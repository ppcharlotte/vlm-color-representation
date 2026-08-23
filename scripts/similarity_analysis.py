#!/usr/bin/env python3
"""Compute image/text/query similarity, RSA, CKA, Procrustes, and linear probes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from structured_color_utils import (
    BASIC_COLORS,
    centroids,
    cosine_matrix,
    labels_to_indices,
    linear_cka,
    linear_probe,
    load_dataset,
    permutation_pvalue,
    procrustes_error,
    rdm,
    save_line_plot,
    spearman_corr,
    upper_triangle_values,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze layer-wise representation similarity and linear decodability.")
    parser.add_argument("--ground-truth", type=Path, default=Path("experiments/llava_structured_alignment/data/ground_truth.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("experiments/llava_structured_alignment/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/llava_structured_alignment"))
    parser.add_argument("--permutation-rounds", type=int, default=1000)
    parser.add_argument("--linear-probe-test-size", type=float, default=0.3)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def analyze_source(prefix: str, source_layers: np.ndarray, text_layers: np.ndarray, layer_indices: list[int], labels: list[str], output: Path, args: argparse.Namespace) -> dict:
    label_indices = labels_to_indices(labels, BASIC_COLORS)
    layer_rows = []
    sample_rows = []
    class_rows = []
    probe_rows = []
    probe_class_rows = []
    for array_idx, layer_idx in enumerate(layer_indices):
        source = source_layers[array_idx]
        text = text_layers[array_idx]
        sims = cosine_matrix(source, text)
        correct_sim = sims[np.arange(len(labels)), label_indices]
        margins = []
        for sample_idx, true_idx in enumerate(label_indices):
            wrong = np.delete(sims[sample_idx], true_idx)
            margins.append(float(sims[sample_idx, true_idx] - wrong.mean()))
        margins = np.asarray(margins)
        source_centroids = centroids(source, labels, BASIC_COLORS)
        source_rdm = rdm(source_centroids)
        text_rdm = rdm(text)
        rsa_r, rsa_p, rsa_n = permutation_pvalue(upper_triangle_values(source_rdm), upper_triangle_values(text_rdm), spearman_corr, args.permutation_rounds, args.seed + array_idx * 10)
        cka = linear_cka(source_centroids, text)
        proc = procrustes_error(text, source_centroids)
        probe = linear_probe(source, labels, BASIC_COLORS, args.linear_probe_test_size, args.ridge_alpha, args.seed + 1000 + array_idx)
        for row in probe["class_rows"]:
            probe_class_rows.append({"source": prefix, "layer_index": int(layer_idx), **row})
        probe_rows.append({"source": prefix, "layer_index": int(layer_idx), "linear_probe_accuracy": probe["accuracy"], "linear_probe_train_n": probe["train_n"], "linear_probe_test_n": probe["test_n"]})
        layer_rows.append(
            {
                "source": prefix,
                "layer_index": int(layer_idx),
                "mean_correct_similarity": float(np.mean(correct_sim)),
                "mean_pair_margin": float(np.mean(margins)),
                "rsa_spearman": rsa_r,
                "rsa_spearman_p": rsa_p,
                "rsa_spearman_n": rsa_n,
                "linear_cka": cka,
                "procrustes_error": proc,
                "linear_probe_accuracy": probe["accuracy"],
            }
        )
        for sample_idx, true_idx in enumerate(label_indices):
            sample_rows.append(
                {
                    "source": prefix,
                    "layer_index": int(layer_idx),
                    "sample_index": sample_idx,
                    "true_color": BASIC_COLORS[true_idx],
                    "correct_similarity": float(correct_sim[sample_idx]),
                    "pair_margin": float(margins[sample_idx]),
                }
            )
        for color_idx, color in enumerate(BASIC_COLORS):
            mask = label_indices == color_idx
            local = np.ones(len(BASIC_COLORS), dtype=bool)
            local[color_idx] = False
            class_rows.append(
                {
                    "source": prefix,
                    "layer_index": int(layer_idx),
                    "color": color,
                    "support": int(np.sum(mask)),
                    "mean_correct_similarity": float(np.mean(correct_sim[mask])),
                    "mean_pair_margin": float(np.mean(margins[mask])),
                    "classwise_rsa_spearman": spearman_corr(source_rdm[color_idx, local], text_rdm[color_idx, local]),
                }
            )

    sim_dir = output / "similarity"
    probe_dir = output / "linear_probe"
    layer_path = sim_dir / f"{prefix}_layerwise.csv"
    sample_path = sim_dir / f"{prefix}_sample_metrics.csv"
    class_path = sim_dir / f"{prefix}_class_metrics.csv"
    probe_path = probe_dir / f"{prefix}_layerwise_probe.csv"
    probe_class_path = probe_dir / f"{prefix}_per_color_probe.csv"
    write_csv(layer_path, ["source", "layer_index", "mean_correct_similarity", "mean_pair_margin", "rsa_spearman", "rsa_spearman_p", "rsa_spearman_n", "linear_cka", "procrustes_error", "linear_probe_accuracy"], layer_rows)
    write_csv(sample_path, ["source", "layer_index", "sample_index", "true_color", "correct_similarity", "pair_margin"], sample_rows)
    write_csv(class_path, ["source", "layer_index", "color", "support", "mean_correct_similarity", "mean_pair_margin", "classwise_rsa_spearman"], class_rows)
    write_csv(probe_path, ["source", "layer_index", "linear_probe_accuracy", "linear_probe_train_n", "linear_probe_test_n"], probe_rows)
    write_csv(probe_class_path, ["source", "layer_index", "color", "support", "linear_probe_accuracy"], probe_class_rows)
    if not args.no_plots:
        for key, ylabel in [("mean_correct_similarity", "mean cosine"), ("mean_pair_margin", "mean margin"), ("rsa_spearman", "RSA Spearman"), ("linear_cka", "CKA"), ("linear_probe_accuracy", "probe accuracy")]:
            save_line_plot(output / "plots" / f"{prefix}_{key}.png", layer_rows, "layer_index", key, f"{prefix} {key}", ylabel)
    return {
        f"{prefix}_layerwise_csv": str(layer_path),
        f"{prefix}_sample_metrics_csv": str(sample_path),
        f"{prefix}_class_metrics_csv": str(class_path),
        f"{prefix}_probe_csv": str(probe_path),
        f"{prefix}_probe_class_csv": str(probe_class_path),
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir
    dataset = load_dataset(args.ground_truth, args.dataset_root)
    labels = dataset.labels
    text_npz = np.load(output / "embeddings" / "text_embeddings.npz", allow_pickle=True)
    text_layers = text_npz["text_layer_embeddings"]
    layer_indices = [int(x) for x in text_npz["layer_indices"].tolist()]

    fc_npz = np.load(output / "embeddings" / "image_embeddings_force_choice.npz", allow_pickle=True)
    nc_npz = np.load(output / "embeddings" / "image_embeddings_no_color_list.npz", allow_pickle=True)
    query_fc_path = output / "embeddings" / "query_embeddings_force_choice.npz"
    query_nc_path = output / "embeddings" / "query_embeddings_no_color_list.npz"
    query_fc = np.load(query_fc_path, allow_pickle=True)["query_layer_embeddings"] if query_fc_path.exists() else fc_npz["query_layer_embeddings"]
    query_nc = np.load(query_nc_path, allow_pickle=True)["query_layer_embeddings"] if query_nc_path.exists() else nc_npz["query_layer_embeddings"]

    summary = {}
    summary.update(analyze_source("image_force_choice", fc_npz["image_layer_embeddings"], text_layers, layer_indices, labels, output, args))
    summary.update(analyze_source("image_no_color_list", nc_npz["image_layer_embeddings"], text_layers, layer_indices, labels, output, args))
    summary.update(analyze_source("query_force_choice", query_fc, text_layers, layer_indices, labels, output, args))
    summary.update(analyze_source("query_no_color_list", query_nc, text_layers, layer_indices, labels, output, args))
    write_json(output / "similarity" / "summary.json", {"layer_indices": layer_indices, **summary})
    print(f"Wrote similarity and linear probe outputs to: {output}")


if __name__ == "__main__":
    main()
