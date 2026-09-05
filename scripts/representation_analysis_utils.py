"""Shared representation-analysis utilities for color embedding experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from structured_color_utils import (
    BASIC_COLORS,
    centroids,
    ciede2000_matrix,
    cosine_matrix,
    labels_to_indices,
    linear_cka,
    linear_probe,
    linear_regression_summary,
    rdm,
    rgb_to_lab,
    save_line_plot,
    save_matrix_csv,
    spearman_corr,
    permutation_pvalue,
    procrustes_error,
    upper_triangle_values,
    write_csv,
)


def load_or_compute_cielab(args, output: Path, dataset) -> tuple[np.ndarray, np.ndarray]:
    sample_path = output / "rdm" / "cielab_sample_rdm.npy"
    centroid_path = output / "rdm" / "cielab_centroid_rdm.csv"
    if sample_path.exists() and centroid_path.exists():
        sample = np.load(sample_path)
        import csv

        with centroid_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        centroid = np.asarray([[float(row[color]) for color in BASIC_COLORS] for row in rows], dtype=float)
        return sample, centroid

    rgb = np.asarray([[int(row["r"]), int(row["g"]), int(row["b"])] for row in dataset.rows], dtype=float)
    lab = rgb_to_lab(rgb)
    sample = ciede2000_matrix(lab)
    label_indices = labels_to_indices(dataset.labels, BASIC_COLORS)
    lab_centroids = np.vstack([lab[label_indices == idx].mean(axis=0) for idx in range(len(BASIC_COLORS))])
    centroid = ciede2000_matrix(lab_centroids)
    (output / "rdm").mkdir(parents=True, exist_ok=True)
    np.save(sample_path, sample)
    save_matrix_csv(centroid_path, BASIC_COLORS, centroid, "color")
    return sample, centroid


def analyze_against_color_prototypes(
    prefix: str,
    source_layers: np.ndarray,
    prototype_layers: np.ndarray,
    layer_indices: list[int],
    labels: list[str],
    output: Path,
    args,
) -> dict:
    label_indices = labels_to_indices(labels, BASIC_COLORS)
    layer_rows = []
    sample_rows = []
    class_rows = []
    probe_rows = []
    probe_class_rows = []
    for array_idx, layer_idx in enumerate(layer_indices):
        source = source_layers[array_idx]
        prototypes = prototype_layers[array_idx]
        sims = cosine_matrix(source, prototypes)
        correct_sim = sims[np.arange(len(labels)), label_indices]
        margins = []
        for sample_idx, true_idx in enumerate(label_indices):
            wrong = np.delete(sims[sample_idx], true_idx)
            margins.append(float(sims[sample_idx, true_idx] - wrong.mean()))
        margins = np.asarray(margins)
        source_centroids = centroids(source, labels, BASIC_COLORS)
        source_rdm = rdm(source_centroids)
        prototype_rdm = rdm(prototypes)
        rsa_r, rsa_p, rsa_n = permutation_pvalue(
            upper_triangle_values(source_rdm),
            upper_triangle_values(prototype_rdm),
            spearman_corr,
            args.permutation_rounds,
            args.seed + array_idx * 10,
        )
        cka = linear_cka(source_centroids, prototypes)
        proc = procrustes_error(prototypes, source_centroids)
        probe = linear_probe(source, labels, BASIC_COLORS, args.linear_probe_test_size, args.ridge_alpha, args.seed + 1000 + array_idx)
        for row in probe["class_rows"]:
            probe_class_rows.append({"source": prefix, "layer_index": int(layer_idx), **row})
        probe_rows.append(
            {
                "source": prefix,
                "layer_index": int(layer_idx),
                "linear_probe_accuracy": probe["accuracy"],
                "linear_probe_train_n": probe["train_n"],
                "linear_probe_test_n": probe["test_n"],
            }
        )
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
                    "classwise_rsa_spearman": spearman_corr(source_rdm[color_idx, local], prototype_rdm[color_idx, local]),
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
        for key, ylabel in [
            ("mean_correct_similarity", "mean cosine"),
            ("mean_pair_margin", "mean margin"),
            ("rsa_spearman", "RSA Spearman"),
            ("linear_cka", "CKA"),
            ("linear_probe_accuracy", "probe accuracy"),
        ]:
            save_line_plot(output / "plots" / f"{prefix}_{key}.png", layer_rows, "layer_index", key, f"{prefix} {key}", ylabel)
    return {
        f"{prefix}_layerwise_csv": str(layer_path),
        f"{prefix}_sample_metrics_csv": str(sample_path),
        f"{prefix}_class_metrics_csv": str(class_path),
        f"{prefix}_probe_csv": str(probe_path),
        f"{prefix}_probe_class_csv": str(probe_class_path),
    }


def analyze_against_embedding_target(
    condition: str,
    source_layers: np.ndarray,
    target_layers: np.ndarray,
    layer_indices: list[int],
    labels: list[str],
    output: Path,
    args,
) -> list[dict]:
    rows = []
    for array_idx, layer_idx in enumerate(layer_indices):
        source = source_layers[array_idx]
        target = target_layers[array_idx]
        source_rdm = rdm(source)
        target_rdm = rdm(target)
        source_centroids = centroids(source, labels, BASIC_COLORS)
        target_centroids = centroids(target, labels, BASIC_COLORS)
        source_centroid_rdm = rdm(source_centroids)
        target_centroid_rdm = rdm(target_centroids)
        sample_r, sample_p, sample_n = permutation_pvalue(
            upper_triangle_values(source_rdm),
            upper_triangle_values(target_rdm),
            spearman_corr,
            args.permutation_rounds,
            args.seed + array_idx * 10,
        )
        centroid_r, centroid_p, centroid_n = permutation_pvalue(
            upper_triangle_values(source_centroid_rdm),
            upper_triangle_values(target_centroid_rdm),
            spearman_corr,
            args.permutation_rounds,
            args.seed + 100 + array_idx * 10,
        )
        paired = np.diag(cosine_matrix(source, target))
        rows.append(
            {
                "condition": condition,
                "layer_index": int(layer_idx),
                "mean_paired_sample_cosine": float(np.mean(paired)),
                "std_paired_sample_cosine": float(np.std(paired)),
                "sample_rsa_spearman": sample_r,
                "sample_rsa_p": sample_p,
                "sample_rsa_n": sample_n,
                "centroid_rsa_spearman": centroid_r,
                "centroid_rsa_p": centroid_p,
                "centroid_rsa_n": centroid_n,
                "linear_cka_centroids": linear_cka(source_centroids, target_centroids),
                "procrustes_error_target_to_source_centroids": procrustes_error(target_centroids, source_centroids),
            }
        )
    return rows
