#!/usr/bin/env python3
"""Compare LLaVA color-word RDMs with perceptual CIEDE2000 color geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from structured_color_utils import (
    BASIC_COLORS,
    ciede2000_matrix,
    labels_to_indices,
    load_dataset,
    pearson_corr,
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


def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    n = len(a)
    if n < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    concordant = discordant = ties_a = ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = np.sign(a[i] - a[j])
            db = np.sign(b[i] - b[j])
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif da == db:
                concordant += 1
            else:
                discordant += 1
    denom = np.sqrt((concordant + discordant + ties_a) * (concordant + discordant + ties_b))
    return float((concordant - discordant) / denom) if denom else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute RSA between color-word embeddings and CIEDE2000 perceptual color distances.")
    parser.add_argument("--ground-truth", type=Path, default=Path("experiments/llava_structured_alignment/data/ground_truth.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("experiments/llava_structured_alignment/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/llava_structured_alignment"))
    parser.add_argument("--permutation-rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def perceptual_centroid_rdm(args: argparse.Namespace, output: Path) -> np.ndarray:
    existing = output / "rdm" / "cielab_centroid_rdm.csv"
    if existing.exists():
        rows = []
        with existing.open(newline="", encoding="utf-8") as handle:
            import csv

            for row in csv.DictReader(handle):
                rows.append([float(row[color]) for color in BASIC_COLORS])
        return np.asarray(rows, dtype=float)

    dataset = load_dataset(args.ground_truth, args.dataset_root)
    labels = dataset.labels
    label_indices = labels_to_indices(labels, BASIC_COLORS)
    lab = rgb_to_lab(dataset.rgb)
    lab_centroids = np.vstack([lab[label_indices == idx].mean(axis=0) for idx in range(len(BASIC_COLORS))])
    matrix = ciede2000_matrix(lab_centroids)
    save_matrix_csv(existing, BASIC_COLORS, matrix, "color")
    return matrix


def main() -> None:
    args = parse_args()
    output = args.output_dir
    embeddings_path = output / "embeddings" / "text_embeddings.npz"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Text embeddings not found: {embeddings_path}. Run text_embedding_extraction.py first.")

    text_npz = np.load(embeddings_path, allow_pickle=True)
    text_layers = text_npz["text_layer_embeddings"]
    layer_indices = [int(x) for x in text_npz["layer_indices"].tolist()]
    cielab_rdm = perceptual_centroid_rdm(args, output)
    perceptual_vec = upper_triangle_values(cielab_rdm)

    rows = []
    rdm_dir = output / "rdm" / "text_rdms"
    for array_idx, layer_idx in enumerate(layer_indices):
        text_rdm = rdm(text_layers[array_idx])
        save_matrix_csv(rdm_dir / f"layer_{layer_idx}_text_rdm.csv", BASIC_COLORS, text_rdm, "color")
        text_vec = upper_triangle_values(text_rdm)
        sr, sp, sn = permutation_pvalue(text_vec, perceptual_vec, spearman_corr, args.permutation_rounds, args.seed + array_idx * 10)
        kr, kp, _ = permutation_pvalue(text_vec, perceptual_vec, kendall_tau_b, args.permutation_rounds, args.seed + 100 + array_idx * 10)
        pr, pp, _ = permutation_pvalue(text_vec, perceptual_vec, pearson_corr, args.permutation_rounds, args.seed + 200 + array_idx * 10)
        rows.append(
            {
                "layer_index": int(layer_idx),
                "rsa_spearman": sr,
                "rsa_spearman_p": sp,
                "rsa_kendall_tau_b": kr,
                "rsa_kendall_tau_b_p": kp,
                "rsa_pearson": pr,
                "rsa_pearson_p": pp,
                "n_pairs": sn,
            }
        )

    result_path = output / "rdm" / "text_cielab_alignment.csv"
    write_csv(
        result_path,
        ["layer_index", "rsa_spearman", "rsa_spearman_p", "rsa_kendall_tau_b", "rsa_kendall_tau_b_p", "rsa_pearson", "rsa_pearson_p", "n_pairs"],
        rows,
    )
    if not args.no_plots:
        save_line_plot(output / "plots" / "text_cielab_rsa_spearman.png", rows, "layer_index", "rsa_spearman", "Text Color Words vs CIEDE2000 RSA", "Spearman r")
        save_line_plot(output / "plots" / "text_cielab_rsa_kendall.png", rows, "layer_index", "rsa_kendall_tau_b", "Text Color Words vs CIEDE2000 Kendall", "Kendall tau-b")
        save_line_plot(output / "plots" / "text_cielab_rsa_pearson.png", rows, "layer_index", "rsa_pearson", "Text Color Words vs CIEDE2000 Pearson", "Pearson r")

    write_json(
        output / "rdm" / "text_cielab_alignment_summary.json",
        {
            "text_embeddings_npz": str(embeddings_path),
            "cielab_centroid_rdm_csv": str(output / "rdm" / "cielab_centroid_rdm.csv"),
            "text_cielab_alignment_csv": str(result_path),
            "layer_indices": layer_indices,
            "permutation_rounds": args.permutation_rounds,
        },
    )
    print(f"Wrote text-CIELAB alignment results to: {result_path}")


if __name__ == "__main__":
    main()
