#!/usr/bin/env python3
"""Extract LLaVA image embeddings and compare image RDMs with CIEDE2000 distances."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from structured_color_utils import (
    BASIC_COLORS,
    FORCE_CHOICE_PROMPT,
    NO_COLOR_LIST_PROMPT,
    LlavaHelper,
    ciede2000_matrix,
    centroids,
    labels_to_indices,
    linear_regression_summary,
    load_dataset,
    permutation_pvalue,
    rdm,
    rgb_to_lab,
    save_line_plot,
    save_matrix_csv,
    spearman_corr,
    upper_triangle_values,
    write_csv,
    write_json,
)


def parse_layers(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract image-token embeddings and compare them with CIEDE2000 RDMs.")
    parser.add_argument("--ground-truth", type=Path, default=Path("experiments/llava_structured_alignment/data/ground_truth.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("experiments/llava_structured_alignment/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/llava_structured_alignment"))
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--layer-indices", default="0,1,8,16,24,32")
    parser.add_argument("--pooling", choices=("mean", "first"), default="mean")
    parser.add_argument("--permutation-rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def analyze_cielab_alignment(name: str, embeddings_by_layer: np.ndarray, layer_indices: list[int], labels: list[str], cielab_sample_rdm: np.ndarray, cielab_centroid_rdm: np.ndarray, output: Path, args: argparse.Namespace) -> list[dict]:
    rows = []
    rdm_dir = output / "rdm" / f"{name}_sample_rdms"
    centroid_dir = output / "rdm" / f"{name}_centroid_rdms"
    rdm_dir.mkdir(parents=True, exist_ok=True)
    centroid_dir.mkdir(parents=True, exist_ok=True)
    label_indices = labels_to_indices(labels, BASIC_COLORS)
    for array_idx, layer_idx in enumerate(layer_indices):
        emb = embeddings_by_layer[array_idx]
        sample_rdm = rdm(emb)
        centroid_emb = centroids(emb, labels, BASIC_COLORS)
        centroid_rdm = rdm(centroid_emb)
        np.save(rdm_dir / f"layer_{layer_idx}_sample_rdm.npy", sample_rdm)
        save_matrix_csv(centroid_dir / f"layer_{layer_idx}_centroid_rdm.csv", BASIC_COLORS, centroid_rdm, "color")
        sr, sp, sn = permutation_pvalue(upper_triangle_values(sample_rdm), upper_triangle_values(cielab_sample_rdm), spearman_corr, args.permutation_rounds, args.seed + array_idx * 10)
        cr, cp, cn = permutation_pvalue(upper_triangle_values(centroid_rdm), upper_triangle_values(cielab_centroid_rdm), spearman_corr, args.permutation_rounds, args.seed + 100 + array_idx * 10)
        reg = linear_regression_summary(upper_triangle_values(sample_rdm), upper_triangle_values(cielab_sample_rdm))
        centroid_reg = linear_regression_summary(upper_triangle_values(centroid_rdm), upper_triangle_values(cielab_centroid_rdm))
        rows.append(
            {
                "representation": name,
                "layer_index": int(layer_idx),
                "sample_rsa_spearman": sr,
                "sample_rsa_p": sp,
                "sample_rsa_n": sn,
                "centroid_rsa_spearman": cr,
                "centroid_rsa_p": cp,
                "centroid_rsa_n": cn,
                "sample_distance_regression_r2": reg["r2"],
                "sample_distance_regression_slope": reg["slope"],
                "sample_distance_regression_n": reg["n"],
                "centroid_distance_regression_r2": centroid_reg["r2"],
                "centroid_distance_regression_slope": centroid_reg["slope"],
                "centroid_distance_regression_n": centroid_reg["n"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output = args.output_dir
    dataset = load_dataset(args.ground_truth, args.dataset_root)
    image_paths = [row["absolute_image_path"] for row in dataset.rows]
    labels = dataset.labels
    layer_indices = parse_layers(args.layer_indices)

    helper = LlavaHelper(args.model_id, args.device, args.dtype, args.seed)
    pre_fc, post_fc, image_fc, query_fc = helper.layerwise_image_and_query_embeddings(image_paths, FORCE_CHOICE_PROMPT, layer_indices, args.pooling)
    pre_nc, post_nc, image_nc, query_nc = helper.layerwise_image_and_query_embeddings(image_paths, NO_COLOR_LIST_PROMPT, layer_indices, args.pooling)

    embeddings_dir = output / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        embeddings_dir / "image_embeddings_force_choice.npz",
        pre_projector=pre_fc,
        post_projector=post_fc,
        image_layer_embeddings=image_fc,
        query_layer_embeddings=query_fc,
        layer_indices=np.asarray(layer_indices),
        labels=np.asarray(labels),
        image_paths=np.asarray(image_paths),
    )
    np.savez_compressed(
        embeddings_dir / "image_embeddings_no_color_list.npz",
        pre_projector=pre_nc,
        post_projector=post_nc,
        image_layer_embeddings=image_nc,
        query_layer_embeddings=query_nc,
        layer_indices=np.asarray(layer_indices),
        labels=np.asarray(labels),
        image_paths=np.asarray(image_paths),
    )

    rgb = np.asarray([[int(row["r"]), int(row["g"]), int(row["b"])] for row in dataset.rows], dtype=float)
    lab = rgb_to_lab(rgb)
    cielab_sample_rdm = ciede2000_matrix(lab)
    label_indices = labels_to_indices(labels, BASIC_COLORS)
    lab_centroids = np.vstack([lab[label_indices == idx].mean(axis=0) for idx in range(len(BASIC_COLORS))])
    cielab_centroid_rdm = ciede2000_matrix(lab_centroids)
    (output / "rdm").mkdir(parents=True, exist_ok=True)
    np.save(output / "rdm" / "cielab_sample_rdm.npy", cielab_sample_rdm)
    save_matrix_csv(output / "rdm" / "cielab_centroid_rdm.csv", BASIC_COLORS, cielab_centroid_rdm, "color")

    rows = []
    # Pre-projector dimensionality differs from LLM text embeddings, but it can still be compared to CIELAB.
    rows.extend(analyze_cielab_alignment("pre_projector", pre_fc[None, :, :], [-1], labels, cielab_sample_rdm, cielab_centroid_rdm, output, args))
    rows.extend(analyze_cielab_alignment("post_projector", post_fc[None, :, :], [0], labels, cielab_sample_rdm, cielab_centroid_rdm, output, args))
    rows.extend(analyze_cielab_alignment("image_force_choice", image_fc, layer_indices, labels, cielab_sample_rdm, cielab_centroid_rdm, output, args))
    rows.extend(analyze_cielab_alignment("image_no_color_list", image_nc, layer_indices, labels, cielab_sample_rdm, cielab_centroid_rdm, output, args))
    results_path = output / "rdm" / "image_cielab_alignment.csv"
    write_csv(
        results_path,
        [
            "representation",
            "layer_index",
            "sample_rsa_spearman",
            "sample_rsa_p",
            "sample_rsa_n",
            "centroid_rsa_spearman",
            "centroid_rsa_p",
            "centroid_rsa_n",
            "sample_distance_regression_r2",
            "sample_distance_regression_slope",
            "sample_distance_regression_n",
            "centroid_distance_regression_r2",
            "centroid_distance_regression_slope",
            "centroid_distance_regression_n",
        ],
        rows,
    )
    if not args.no_plots:
        for rep in ["image_force_choice", "image_no_color_list"]:
            rep_rows = [row for row in rows if row["representation"] == rep]
            save_line_plot(output / "plots" / f"{rep}_cielab_sample_rsa.png", rep_rows, "layer_index", "sample_rsa_spearman", f"{rep} vs CIEDE2000 Sample RSA", "Spearman r")
            save_line_plot(output / "plots" / f"{rep}_cielab_centroid_rsa.png", rep_rows, "layer_index", "centroid_rsa_spearman", f"{rep} vs CIEDE2000 Centroid RSA", "Spearman r")
    write_json(
        output / "embeddings" / "image_embedding_summary.json",
        {
            "model_id": args.model_id,
            "ground_truth": str(args.ground_truth),
            "n_samples": len(dataset.rows),
            "layer_indices": layer_indices,
            "force_choice_npz": str(embeddings_dir / "image_embeddings_force_choice.npz"),
            "no_color_list_npz": str(embeddings_dir / "image_embeddings_no_color_list.npz"),
            "image_cielab_alignment_csv": str(results_path),
        },
    )
    print(f"Wrote image embeddings and CIELAB alignment outputs to: {output}")


if __name__ == "__main__":
    main()
