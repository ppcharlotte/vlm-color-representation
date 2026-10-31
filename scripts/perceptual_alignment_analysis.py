#!/usr/bin/env python3
"""Unified alignment analysis for image, query, text, CEe, and CEg embeddings."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from structured_color_utils import (
    BASIC_COLORS,
    ciede2000_matrix,
    centroids,
    linear_cka,
    linear_probe,
    load_dataset,
    rdm,
    rdm_permutation_pvalue,
    rgb_to_lab,
    save_line_plot,
    save_matrix_csv,
    spearman_corr,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare every retained representation with human perceptual color geometry.")
    parser.add_argument("--ground-truth", type=Path, default=Path("experiments/perceptual_alignment/data/ground_truth.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("experiments/perceptual_alignment/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/perceptual_alignment"))
    parser.add_argument("--sample-permutation-rounds", type=int, default=200)
    parser.add_argument("--sample-permutation-scope", choices=("core", "all", "none"), default="core", help="core: sample inference only for primary cosine/Spearman model-perceptual comparisons; all: legacy exhaustive tests; none: sample effect sizes only")
    parser.add_argument("--category-permutation-rounds", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--linear-probe-test-size", type=float, default=0.3)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--embedding-distances", default="cosine", help="Primary: cosine; pearson can be requested explicitly")
    parser.add_argument("--appendix-metrics", action="store_true", help="Optional non-generation CKA/probes")
    parser.add_argument("--cee-reference-difference", action="store_true", help="Test CEe-visual versus CEe-textual only for this comparative claim")
    parser.add_argument("--rsa-coefficients", default="spearman,kendall_tau_b")
    parser.add_argument("--bootstrap-rounds", type=int, default=2000, help="Legacy compatibility only; current Delta inference uses category-permutation-rounds")
    parser.add_argument("--exclude-cee", action="store_true", help="Ignore CEe caches, including matched-text caches.")
    parser.add_argument("--exclude-ceg", action="store_true", help="Ignore CEg caches.")
    parser.add_argument("--exclude-mismatched", action="store_true", help="Ignore mismatched-color caches.")
    return parser.parse_args()


def fdr_bh(values: list[float]) -> list[float]:
    result = np.full(len(values), np.nan, dtype=float)
    finite = np.asarray([index for index, value in enumerate(values) if np.isfinite(value)], dtype=int)
    if not len(finite):
        return result.tolist()
    p = np.asarray([values[index] for index in finite], dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result[finite] = restored
    return result.tolist()


def human_geometry(dataset, indices: np.ndarray | None = None):
    if indices is None:
        indices = np.arange(len(dataset.rows))
    lab = rgb_to_lab(dataset.rgb[indices])
    labels = [dataset.labels[int(index)] for index in indices]
    active_colors = [color for color in BASIC_COLORS if color in set(labels)]
    sample_rdm = ciede2000_matrix(lab)
    category_lab = centroids(lab, labels, active_colors)
    return labels, active_colors, sample_rdm, ciede2000_matrix(category_lab)


def analyze_layers(
    source: str,
    layers: np.ndarray,
    layer_indices: list[int],
    labels: list[str],
    active_colors: list[str],
    perceptual_sample_rdm: np.ndarray,
    perceptual_category_rdm: np.ndarray,
    text_layers: np.ndarray,
    args: argparse.Namespace,
) -> list[dict]:
    rows = []
    for array_idx, layer_idx in enumerate(layer_indices):
        values = layers[array_idx]
        source_sample_rdm = rdm(values)
        source_categories = centroids(values, labels, active_colors)
        source_category_rdm = rdm(source_categories)
        sample_r, sample_p, sample_n = rdm_permutation_pvalue(
            source_sample_rdm,
            perceptual_sample_rdm,
            spearman_corr,
            args.sample_permutation_rounds,
            args.seed + array_idx * 31,
        )
        human_r, human_p, human_n = rdm_permutation_pvalue(
            source_category_rdm,
            perceptual_category_rdm,
            spearman_corr,
            args.category_permutation_rounds,
            args.seed + 1000 + array_idx * 31,
        )
        text_rdm = rdm(text_layers[array_idx])
        text_r, text_p, text_n = rdm_permutation_pvalue(
            source_category_rdm,
            text_rdm,
            spearman_corr,
            args.category_permutation_rounds,
            args.seed + 2000 + array_idx * 31,
        )
        probe = linear_probe(values, labels, active_colors, args.linear_probe_test_size, args.ridge_alpha, args.seed + array_idx)
        rows.append(
            {
                "source": source,
                "layer_index": int(layer_idx),
                "n_samples": int(values.shape[0]),
                "sample_ciede2000_rsa": sample_r,
                "sample_ciede2000_p": sample_p,
                "sample_ciede2000_n_pairs": sample_n,
                "human_category_rsa": human_r,
                "human_category_p": human_p,
                "human_category_n_pairs": human_n,
                "text_category_rsa": text_r,
                "text_category_p": text_p,
                "text_category_n_pairs": text_n,
                "text_category_linear_cka": linear_cka(source_categories, text_layers[array_idx]),
                "linear_probe_accuracy": probe["accuracy"],
                "linear_probe_train_n": probe["train_n"],
                "linear_probe_test_n": probe["test_n"],
            }
        )
    return rows


def add_if_present(sources: list[tuple], path: Path, key: str, source: str, indices_key: str | None = None) -> None:
    if not path.exists():
        return
    data = np.load(path, allow_pickle=True)
    indices = np.asarray(data[indices_key], dtype=int) if indices_key and indices_key in data.files else None
    sources.append((source, data[key], [int(x) for x in data["layer_indices"].tolist()], indices))


def save_rsa_comparison_plot(path: Path, rows: list[dict], source: str) -> None:
    """Plot perceptual and textual RSA curves together for one representation."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    usable = sorted((row for row in rows if int(row["layer_index"]) >= 0), key=lambda row: int(row["layer_index"]))
    if not usable:
        return
    layers = [int(row["layer_index"]) for row in usable]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    for metric, label in (
        ("sample_ciede2000_rsa", "sample ↔ CIEDE2000"),
        ("human_category_rsa", "class ↔ human geometry"),
        ("text_category_rsa", "class ↔ text geometry"),
    ):
        values = np.asarray([float(row[metric]) for row in usable], dtype=float)
        if np.isfinite(values).any():
            ax.plot(layers, values, marker="o", linewidth=1.8, label=label)
    ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("layer_index")
    ax.set_ylabel("Spearman r")
    ax.set_title(f"{source}: representational alignment")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def safe_layer_name(layer_index: int) -> str:
    return f"minus_{abs(layer_index)}" if layer_index < 0 else str(layer_index)


def save_rdm_heatmap(path: Path, matrix: np.ndarray, labels: list[str], title: str, correlation: bool = False) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=170)
    image = ax.imshow(matrix, cmap="coolwarm" if correlation else "viridis", vmin=-1 if correlation else None, vmax=1 if correlation else None)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Spearman r" if correlation else "dissimilarity")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def category_spaces_for_sources(dataset, sources: list[tuple]) -> list[dict]:
    spaces = []
    for source, layers, layer_indices, indices in sources:
        if indices is None:
            active_indices = np.arange(len(dataset.rows))
        else:
            active_indices = np.asarray(indices, dtype=int)
        labels = [dataset.labels[int(index)] for index in active_indices]
        colors = [color for color in BASIC_COLORS if color in set(labels)]
        for array_idx, layer_idx in enumerate(layer_indices):
            category_embeddings = centroids(layers[array_idx], labels, colors)
            spaces.append(
                {
                    "source": source,
                    "layer_index": int(layer_idx),
                    "colors": colors,
                    "embeddings": category_embeddings,
                    "rdm": rdm(category_embeddings),
                }
            )
    return spaces


def analyze_pairwise_category_spaces(
    spaces: list[dict],
    text_layers: np.ndarray,
    text_layer_indices: list[int],
    human_embeddings: np.ndarray,
    human_rdm: np.ndarray,
    human_colors: list[str],
    output: Path,
    args: argparse.Namespace,
) -> tuple[list[dict], Path]:
    text_by_layer = {int(layer): text_layers[idx] for idx, layer in enumerate(text_layer_indices)}
    source_layers = sorted({int(space["layer_index"]) for space in spaces})
    pair_rows = []
    pair_counter = 0
    rdm_dir = output / "rdm" / "category"
    plot_dir = output / "plots" / "category_rdms"
    rdm_dir.mkdir(parents=True, exist_ok=True)

    save_matrix_csv(rdm_dir / "human_ciede2000.csv", human_colors, human_rdm, "color")
    if not args.no_plots:
        save_rdm_heatmap(plot_dir / "human_ciede2000.png", human_rdm, human_colors, "Human CIEDE2000 category RDM")

    for layer_idx, text_values in text_by_layer.items():
        text_rdm = rdm(text_values)
        save_matrix_csv(rdm_dir / f"text_layer_{safe_layer_name(layer_idx)}.csv", BASIC_COLORS, text_rdm, "color")
        if not args.no_plots:
            save_rdm_heatmap(
                plot_dir / f"text_layer_{safe_layer_name(layer_idx)}.png",
                text_rdm,
                BASIC_COLORS,
                f"Text color-word RDM, layer {layer_idx}",
            )

    for space in spaces:
        layer_name = safe_layer_name(int(space["layer_index"]))
        save_matrix_csv(rdm_dir / f"{space['source']}_layer_{layer_name}.csv", space["colors"], space["rdm"], "color")
        if not args.no_plots:
            save_rdm_heatmap(
                plot_dir / f"{space['source']}_layer_{layer_name}.png",
                space["rdm"],
                space["colors"],
                f"{space['source']} category RDM, layer {space['layer_index']}",
            )

    for layer_idx in source_layers:
        layer_spaces = [space for space in spaces if int(space["layer_index"]) == layer_idx]
        text_layer = layer_idx if layer_idx in text_by_layer else text_layer_indices[0]
        layer_spaces.extend(
            [
                {
                    "source": "text_color_words",
                    "layer_index": layer_idx,
                    "colors": BASIC_COLORS,
                    "embeddings": text_by_layer[text_layer],
                    "rdm": rdm(text_by_layer[text_layer]),
                },
                {
                    "source": "human_ciede2000",
                    "layer_index": layer_idx,
                    "colors": human_colors,
                    "embeddings": human_embeddings,
                    "rdm": human_rdm,
                },
            ]
        )
        for left, right in itertools.combinations(layer_spaces, 2):
            shared = [color for color in BASIC_COLORS if color in set(left["colors"]) and color in set(right["colors"])]
            if len(shared) < 3:
                continue
            left_indices = [left["colors"].index(color) for color in shared]
            right_indices = [right["colors"].index(color) for color in shared]
            left_rdm = left["rdm"][np.ix_(left_indices, left_indices)]
            right_rdm = right["rdm"][np.ix_(right_indices, right_indices)]
            rsa_value, rsa_p, n_pairs = rdm_permutation_pvalue(
                left_rdm,
                right_rdm,
                spearman_corr,
                args.category_permutation_rounds,
                args.seed + 10000 + pair_counter * 13,
            )
            left_embeddings = left["embeddings"][left_indices]
            right_embeddings = right["embeddings"][right_indices]
            pair_rows.append(
                {
                    "layer_index": int(layer_idx),
                    "source_a": left["source"],
                    "source_b": right["source"],
                    "n_shared_colors": len(shared),
                    "shared_colors": "|".join(shared),
                    "rdm_rsa_spearman": rsa_value,
                    "rdm_rsa_p": rsa_p,
                    "rdm_rsa_n_pairs": n_pairs,
                    "linear_cka": linear_cka(left_embeddings, right_embeddings),
                }
            )
            pair_counter += 1

        if not args.no_plots:
            names = [space["source"] for space in layer_spaces]
            matrix = np.eye(len(names), dtype=float)
            relevant = [row for row in pair_rows if int(row["layer_index"]) == layer_idx]
            for row in relevant:
                i = names.index(row["source_a"])
                j = names.index(row["source_b"])
                matrix[i, j] = matrix[j, i] = float(row["rdm_rsa_spearman"])
            save_rdm_heatmap(
                output / "plots" / "rdm_pairwise" / f"layer_{safe_layer_name(layer_idx)}_rsa.png",
                matrix,
                names,
                f"Pairwise category-RDM RSA, layer {layer_idx}",
                correlation=True,
            )

    adjusted = fdr_bh([float(row["rdm_rsa_p"]) for row in pair_rows])
    for row, q_value in zip(pair_rows, adjusted):
        row["rdm_rsa_q_fdr"] = q_value
    pair_path = output / "analysis" / "category_rdm_pairwise.csv"
    write_csv(pair_path, list(pair_rows[0]) if pair_rows else [], pair_rows)
    return pair_rows, pair_path


def legacy_main() -> None:
    """Historical v1 implementation, retained for interpreting archived outputs."""
    args = parse_args()
    output = args.output_dir
    dataset = load_dataset(args.ground_truth, args.dataset_root)
    text_data = np.load(output / "embeddings" / "text_embeddings.npz", allow_pickle=True)
    text_layers = text_data["text_layer_embeddings"]
    text_layer_indices = [int(x) for x in text_data["layer_indices"].tolist()]

    sources: list[tuple[str, np.ndarray, list[int], np.ndarray | None]] = []
    image_path = output / "embeddings" / "image_query_embeddings.npz"
    image_data = np.load(image_path, allow_pickle=True)
    image_layer_indices = [int(x) for x in image_data["layer_indices"].tolist()]
    if image_layer_indices != text_layer_indices:
        raise ValueError(f"Image layers {image_layer_indices} do not match text layers {text_layer_indices}.")
    if 0 not in text_layer_indices:
        raise ValueError("Layer 0 must be extracted so projector-stage representations have a fixed text reference.")
    sources.append(("image_tokens", image_data["image_layer_embeddings"], image_layer_indices, None))
    sources.append(("query_tokens", image_data["query_layer_embeddings"], image_layer_indices, None))
    sources.append(("pre_projector", image_data["pre_projector"][None], [0], None))
    sources.append(("post_projector", image_data["post_projector"][None], [0], None))
    add_if_present(sources, output / "embeddings" / "cee_ground_truth_embeddings.npz", "cee_layer_embeddings", "cee", None)
    add_if_present(
        sources,
        output / "ceg_generation_side" / "embeddings" / "ceg_decision_embeddings.npz",
        "ceg_decision_layer_embeddings",
        "ceg_decision",
        "sample_indices",
    )
    add_if_present(
        sources,
        output / "ceg_generation_side" / "embeddings" / "ceg_token_from_generate_embeddings.npz",
        "ceg_token_from_generate_layer_embeddings",
        "ceg_token",
        "sample_indices",
    )

    global_labels, global_colors, global_sample_rdm, global_category_rdm = human_geometry(dataset)
    global_lab = rgb_to_lab(dataset.rgb)
    global_human_embeddings = centroids(global_lab, global_labels, global_colors)
    rdm_dir = output / "rdm"
    rdm_dir.mkdir(parents=True, exist_ok=True)
    np.save(rdm_dir / "ciede2000_sample_rdm.npy", global_sample_rdm)
    save_matrix_csv(rdm_dir / "human_ciede2000_category_rdm.csv", global_colors, global_category_rdm, "color")

    rows = []
    for source, layers, layer_indices, indices in sources:
        if indices is None:
            labels, active_colors, sample_rdm, category_rdm = global_labels, global_colors, global_sample_rdm, global_category_rdm
        else:
            labels, active_colors, sample_rdm, _subset_category_rdm = human_geometry(dataset, indices)
            global_color_indices = [global_colors.index(color) for color in active_colors]
            category_rdm = global_category_rdm[np.ix_(global_color_indices, global_color_indices)]
        color_indices = [BASIC_COLORS.index(color) for color in active_colors]
        if layers.shape[0] == len(text_layer_indices) and layer_indices == text_layer_indices:
            source_text_layers = text_layers[:, color_indices, :]
        else:
            # Projector stages are anchored to the layer-0 comparison group and
            # always use layer-0 text geometry, independent of CLI layer order.
            layer_zero_idx = text_layer_indices.index(0)
            source_text_layers = np.repeat(text_layers[layer_zero_idx : layer_zero_idx + 1, color_indices, :], layers.shape[0], axis=0)
        rows.extend(analyze_layers(source, layers, layer_indices, labels, active_colors, sample_rdm, category_rdm, source_text_layers, args))

    text_rows = []
    for array_idx, layer_idx in enumerate(text_layer_indices):
        active_text = text_layers[array_idx, [BASIC_COLORS.index(color) for color in global_colors], :]
        text_rdm = rdm(active_text)
        value, p_value, n_pairs = rdm_permutation_pvalue(
            text_rdm,
            global_category_rdm,
            spearman_corr,
            args.category_permutation_rounds,
            args.seed + 5000 + array_idx * 31,
        )
        text_rows.append(
            {
                "source": "text_color_words",
                "layer_index": int(layer_idx),
                "n_samples": len(global_colors),
                "sample_ciede2000_rsa": float("nan"),
                "sample_ciede2000_p": float("nan"),
                "sample_ciede2000_n_pairs": 0,
                "human_category_rsa": value,
                "human_category_p": p_value,
                "human_category_n_pairs": n_pairs,
                "text_category_rsa": 1.0,
                "text_category_p": float("nan"),
                "text_category_n_pairs": n_pairs,
                "text_category_linear_cka": 1.0,
                "linear_probe_accuracy": float("nan"),
                "linear_probe_train_n": 0,
                "linear_probe_test_n": 0,
            }
        )
    rows.extend(text_rows)

    category_spaces = category_spaces_for_sources(dataset, sources)
    pair_rows, pairwise_path = analyze_pairwise_category_spaces(
        category_spaces,
        text_layers,
        text_layer_indices,
        global_human_embeddings,
        global_category_rdm,
        global_colors,
        output,
        args,
    )

    for p_key, q_key in (
        ("sample_ciede2000_p", "sample_ciede2000_q_fdr"),
        ("human_category_p", "human_category_q_fdr"),
        ("text_category_p", "text_category_q_fdr"),
    ):
        adjusted = fdr_bh([float(row[p_key]) for row in rows])
        for row, value in zip(rows, adjusted):
            row[q_key] = value

    result_path = output / "analysis" / "perceptual_alignment.csv"
    fields = list(rows[0]) if rows else []
    write_csv(result_path, fields, rows)
    probe_rows = [
        {
            "source": row["source"],
            "layer_index": row["layer_index"],
            "n_samples": row["n_samples"],
            "linear_probe_accuracy": row["linear_probe_accuracy"],
            "linear_probe_train_n": row["linear_probe_train_n"],
            "linear_probe_test_n": row["linear_probe_test_n"],
        }
        for row in rows
        if np.isfinite(float(row["linear_probe_accuracy"]))
    ]
    probe_path = output / "analysis" / "linear_probe.csv"
    write_csv(probe_path, list(probe_rows[0]) if probe_rows else [], probe_rows)
    if not args.no_plots:
        for source in sorted({row["source"] for row in rows}):
            source_rows = [row for row in rows if row["source"] == source and int(row["layer_index"]) >= 0]
            if source_rows:
                plot_specs = (
                    ("sample_ciede2000_rsa", "Sample-level CIEDE2000 RSA", "Spearman r"),
                    ("human_category_rsa", "Class-level human-geometry RSA", "Spearman r"),
                    ("text_category_rsa", "Class-level text-geometry RSA", "Spearman r"),
                    ("text_category_linear_cka", "Class-level text CKA", "linear CKA"),
                    ("linear_probe_accuracy", "Basic-color linear probe", "accuracy"),
                )
                for metric, title, ylabel in plot_specs:
                    finite_rows = [row for row in source_rows if np.isfinite(float(row[metric]))]
                    if finite_rows:
                        save_line_plot(
                            output / "plots" / f"{source}_{metric}.png",
                            finite_rows,
                            "layer_index",
                            metric,
                            f"{source}: {title}",
                            ylabel,
                        )
                save_rsa_comparison_plot(output / "plots" / f"{source}_rsa_comparison.png", source_rows, source)
    write_json(
        output / "analysis" / "summary.json",
        {
            "alignment_csv": str(result_path),
            "category_rdm_pairwise_csv": str(pairwise_path),
            "linear_probe_csv": str(probe_path),
            "category_rdm_directory": str(output / "rdm" / "category"),
            "sources": sorted({row["source"] for row in rows}),
            "sample_permutation_rounds": args.sample_permutation_rounds,
            "category_permutation_rounds": args.category_permutation_rounds,
            "multiple_testing": "Benjamini-Hochberg FDR within each p-value family",
            "active_basic_colors": global_colors,
            "hard_labels_used_for_category_centroids_and_linear_probe": True,
            "strict_filter": "agreement_rate > 0.75 and majority_count >= 2",
            "pairwise_category_comparisons": len(pair_rows),
        },
    )
    print(f"Wrote unified perceptual alignment results to: {result_path}")


def main() -> None:
    from extended_alignment_analysis import run_extended_analysis
    run_extended_analysis(parse_args())


if __name__ == "__main__":
    main()
