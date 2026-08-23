#!/usr/bin/env python3
"""Compare LM-head output color vectors with CIEDE2000 perceptual color geometry."""

from __future__ import annotations

import argparse
import csv
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
    save_matrix_csv,
    spearman_corr,
    try_import_matplotlib,
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
    parser = argparse.ArgumentParser(description="Compute RSA between LM-head color vectors and CIEDE2000 color distances.")
    parser.add_argument("--ground-truth", type=Path, default=Path("experiments/llava_structured_alignment/data/ground_truth.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("experiments/llava_structured_alignment/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/llava_structured_alignment"))
    parser.add_argument("--permutation-rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def load_or_compute_cielab_centroid_rdm(args: argparse.Namespace, output: Path) -> np.ndarray:
    path = output / "rdm" / "cielab_centroid_rdm.csv"
    if path.exists():
        rows = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append([float(row[color]) for color in BASIC_COLORS])
        return np.asarray(rows, dtype=float)

    dataset = load_dataset(args.ground_truth, args.dataset_root)
    labels = dataset.labels
    label_indices = labels_to_indices(labels, BASIC_COLORS)
    lab = rgb_to_lab(dataset.rgb)
    lab_centroids = np.vstack([lab[label_indices == idx].mean(axis=0) for idx in range(len(BASIC_COLORS))])
    matrix = ciede2000_matrix(lab_centroids)
    save_matrix_csv(path, BASIC_COLORS, matrix, "color")
    return matrix


def save_metric_plot(path: Path, row: dict[str, float]) -> None:
    plt = try_import_matplotlib()
    if plt is None:
        return
    metrics = ["rsa_spearman", "rsa_kendall_tau_b", "rsa_pearson"]
    labels = ["Spearman", "Kendall", "Pearson"]
    values = [float(row[m]) for m in metrics]
    fig, ax = plt.subplots(figsize=(5.8, 4.0), dpi=160)
    ax.bar(labels, values, color="#4C78A8")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("RSA correlation")
    ax.set_title("LM-head color vectors vs CIEDE2000 RSA")
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output_dir
    lm_head_path = output / "embeddings" / "lm_head_color_vectors.npy"
    if not lm_head_path.exists():
        raise FileNotFoundError(
            f"LM-head color vectors not found: {lm_head_path}. Run answer_position_embedding_analysis.py first."
        )

    lm_head_vectors = np.load(lm_head_path)
    if lm_head_vectors.shape[0] != len(BASIC_COLORS):
        raise ValueError(f"Expected {len(BASIC_COLORS)} LM-head vectors, got shape {lm_head_vectors.shape}")

    lm_head_rdm = rdm(lm_head_vectors)
    lm_head_rdm_path = output / "rdm" / "lm_head_color_rdm.csv"
    save_matrix_csv(lm_head_rdm_path, BASIC_COLORS, lm_head_rdm, "color")

    cielab_rdm = load_or_compute_cielab_centroid_rdm(args, output)
    lm_vec = upper_triangle_values(lm_head_rdm)
    cielab_vec = upper_triangle_values(cielab_rdm)

    sr, sp, sn = permutation_pvalue(lm_vec, cielab_vec, spearman_corr, args.permutation_rounds, args.seed)
    kr, kp, _ = permutation_pvalue(lm_vec, cielab_vec, kendall_tau_b, args.permutation_rounds, args.seed + 100)
    pr, pp, _ = permutation_pvalue(lm_vec, cielab_vec, pearson_corr, args.permutation_rounds, args.seed + 200)
    row = {
        "representation": "lm_head_color_vectors",
        "rsa_spearman": sr,
        "rsa_spearman_p": sp,
        "rsa_kendall_tau_b": kr,
        "rsa_kendall_tau_b_p": kp,
        "rsa_pearson": pr,
        "rsa_pearson_p": pp,
        "n_pairs": sn,
    }

    result_path = output / "rdm" / "lm_head_cielab_alignment.csv"
    write_csv(
        result_path,
        [
            "representation",
            "rsa_spearman",
            "rsa_spearman_p",
            "rsa_kendall_tau_b",
            "rsa_kendall_tau_b_p",
            "rsa_pearson",
            "rsa_pearson_p",
            "n_pairs",
        ],
        [row],
    )

    if not args.no_plots:
        save_metric_plot(output / "plots" / "lm_head_cielab_rsa.png", row)

    write_json(
        output / "rdm" / "lm_head_cielab_alignment_summary.json",
        {
            "lm_head_color_vectors_npy": str(lm_head_path),
            "lm_head_color_rdm_csv": str(lm_head_rdm_path),
            "cielab_centroid_rdm_csv": str(output / "rdm" / "cielab_centroid_rdm.csv"),
            "lm_head_cielab_alignment_csv": str(result_path),
            "permutation_rounds": args.permutation_rounds,
        },
    )
    print(f"Wrote LM-head-CIELAB alignment results to: {result_path}")


if __name__ == "__main__":
    main()
