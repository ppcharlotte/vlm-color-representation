#!/usr/bin/env python3
"""Run color-patch alignment experiments for VLM/CLIP embeddings.

This script implements the pure-color-patch pilot described in
research_notes/color_alignment_method_plan_zh.md:

- zero-shot text-prototype color classification
- instance-level pair alignment margin
- relational alignment: RSA, Kendall tau, CKA, Procrustes error
- class-wise alignment and class-wise accuracy
- confusion matrix and visual separability controls
- image-only linear probe with a small numpy ridge classifier
- Procrustes interventions: correct, random orthogonal, label-permuted

The default input is the generated color patch dataset:
  data/color_patches_aggregated_100/ground_truth.csv

Embeddings can be extracted with OpenCLIP, or loaded from a previously saved
NPZ file using --embeddings-npz.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


BASIC_COLORS = (
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
)

DEFAULT_PROMPTS = (
    "{}",
    "the color {}",
    "a {} color patch",
    "this image is {}",
)


@dataclass
class Dataset:
    image_paths: list[str]
    labels: list[str]
    rows: list[dict[str, str]]


@dataclass
class Embeddings:
    image_embeddings: np.ndarray
    text_embeddings: np.ndarray
    colors: list[str]
    prompts: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VLM/CLIP alignment analysis on pure color patch data."
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/color_patches_aggregated_100/ground_truth.csv"),
        help="Ground-truth CSV produced by build_color_patches_from_aggregated.py.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("."),
        help="Root used to resolve relative image_path values. Default: current directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/color_patch_alignment"),
        help="Directory for result CSV/JSON/NPZ outputs.",
    )
    parser.add_argument(
        "--backend",
        choices=("open_clip",),
        default="open_clip",
        help="Embedding backend. Default: open_clip.",
    )
    parser.add_argument("--model", default="ViT-B-32", help="OpenCLIP model name.")
    parser.add_argument("--pretrained", default="openai", help="OpenCLIP pretrained tag.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument("--batch-size", type=int, default=64, help="Image batch size.")
    parser.add_argument(
        "--prompt",
        dest="prompts",
        action="append",
        help="Prompt template containing {}. Can be repeated. Defaults to a prompt ensemble.",
    )
    parser.add_argument(
        "--embeddings-npz",
        type=Path,
        help="Load existing embeddings NPZ instead of extracting model embeddings.",
    )
    parser.add_argument(
        "--save-embeddings-name",
        default="embeddings.npz",
        help="Filename for saved embeddings inside output dir.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--linear-probe-test-size",
        type=float,
        default=0.3,
        help="Test fraction for image-only linear probe. Default: 0.3",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge regularization for numpy linear probe. Default: 1.0",
    )
    parser.add_argument(
        "--permutation-rounds",
        type=int,
        default=1000,
        help="Permutation rounds for correlation p-values. Default: 1000",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable PNG visualization output.",
    )
    return parser.parse_args()


def ensure_prompt_templates(prompts: list[str] | None) -> list[str]:
    templates = prompts if prompts else list(DEFAULT_PROMPTS)
    for prompt in templates:
        if "{}" not in prompt:
            raise ValueError(f"Prompt template must contain {{}}: {prompt!r}")
    return templates


def load_dataset(path: Path, dataset_root: Path) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth CSV not found: {path}")

    rows: list[dict[str, str]] = []
    image_paths: list[str] = []
    labels: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "basic_color"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Ground-truth CSV missing columns: {sorted(missing)}")

        for row in reader:
            label = row["basic_color"].strip().lower()
            if label not in BASIC_COLORS:
                continue
            raw_path = Path(row["image_path"])
            image_path = raw_path if raw_path.is_absolute() else dataset_root / raw_path
            rows.append(row)
            image_paths.append(str(image_path))
            labels.append(label)

    if not rows:
        raise ValueError(f"No usable rows found in {path}")
    return Dataset(image_paths=image_paths, labels=labels, rows=rows)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return l2_normalize(a) @ l2_normalize(b).T


def choose_device(device: str):
    import torch

    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_open_clip_embeddings(
    dataset: Dataset,
    colors: list[str],
    prompts: list[str],
    model_name: str,
    pretrained: str,
    device_name: str,
    batch_size: int,
) -> Embeddings:
    try:
        import torch
        import open_clip
    except Exception as exc:
        raise RuntimeError(
            "OpenCLIP embedding extraction requires torch and open_clip_torch. "
            "Install them in your project environment, or pass --embeddings-npz "
            "to analyze precomputed embeddings."
        ) from exc

    device = choose_device(device_name)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    image_features: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(dataset.image_paths), batch_size):
            batch_paths = dataset.image_paths[start : start + batch_size]
            images = []
            for image_path in batch_paths:
                with Image.open(image_path) as image:
                    images.append(preprocess(image.convert("RGB")))
            image_tensor = torch.stack(images).to(device)
            features = model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            image_features.append(features.detach().cpu().numpy())

        text_features_by_color: list[np.ndarray] = []
        all_prompt_texts: list[str] = []
        for color in colors:
            color_prompts = [template.format(color) for template in prompts]
            all_prompt_texts.extend(color_prompts)
            tokens = tokenizer(color_prompts).to(device)
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            prototype = features.mean(dim=0, keepdim=True)
            prototype = prototype / prototype.norm(dim=-1, keepdim=True)
            text_features_by_color.append(prototype.detach().cpu().numpy()[0])

    return Embeddings(
        image_embeddings=np.vstack(image_features).astype(np.float32),
        text_embeddings=np.vstack(text_features_by_color).astype(np.float32),
        colors=colors,
        prompts=prompts,
    )


def save_embeddings(path: Path, embeddings: Embeddings, dataset: Dataset) -> None:
    np.savez_compressed(
        path,
        image_embeddings=embeddings.image_embeddings,
        text_embeddings=embeddings.text_embeddings,
        colors=np.array(embeddings.colors),
        labels=np.array(dataset.labels),
        image_paths=np.array(dataset.image_paths),
        prompts=np.array(embeddings.prompts),
    )


def load_embeddings(path: Path) -> tuple[Embeddings, list[str], list[str]]:
    data = np.load(path, allow_pickle=False)
    colors = [str(x) for x in data["colors"].tolist()]
    prompts = [str(x) for x in data["prompts"].tolist()] if "prompts" in data else []
    labels = [str(x) for x in data["labels"].tolist()]
    image_paths = [str(x) for x in data["image_paths"].tolist()]
    return (
        Embeddings(
            image_embeddings=np.asarray(data["image_embeddings"], dtype=np.float32),
            text_embeddings=np.asarray(data["text_embeddings"], dtype=np.float32),
            colors=colors,
            prompts=prompts,
        ),
        labels,
        image_paths,
    )


def labels_to_indices(labels: list[str], colors: list[str]) -> np.ndarray:
    color_to_index = {color: i for i, color in enumerate(colors)}
    return np.array([color_to_index[label] for label in labels], dtype=int)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(path: Path, row_labels: list[str], col_labels: list[str], matrix: np.ndarray) -> None:
    rows = []
    for label, values in zip(row_labels, matrix):
        row = {"label": label}
        row.update({col: float(value) for col, value in zip(col_labels, values)})
        rows.append(row)
    write_csv(path, ["label", *col_labels], rows)


def rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else float("nan")


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    return pearson_corr(rankdata_average(a), rankdata_average(b))


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    concordant = 0
    discordant = 0
    n = len(a)
    for i in range(n - 1):
        da = a[i] - a[i + 1 :]
        db = b[i] - b[i + 1 :]
        sign = np.sign(da * db)
        concordant += int(np.sum(sign > 0))
        discordant += int(np.sum(sign < 0))
    denom = concordant + discordant
    return float((concordant - discordant) / denom) if denom else float("nan")


def rdm(embeddings: np.ndarray) -> np.ndarray:
    return 1.0 - cosine_matrix(embeddings, embeddings)


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[indices]


def linear_cka(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    gram_a = a @ a.T
    gram_b = b @ b.T
    numerator = np.sum(gram_a * gram_b)
    denominator = math.sqrt(float(np.sum(gram_a * gram_a) * np.sum(gram_b * gram_b)))
    return float(numerator / denominator) if denominator else float("nan")


def orthogonal_procrustes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    cross_cov = source.T @ target
    u, _, vt = np.linalg.svd(cross_cov, full_matrices=False)
    return u @ vt


def procrustes_error(source: np.ndarray, target: np.ndarray) -> float:
    w = orthogonal_procrustes(source, target)
    aligned = source @ w
    denom = np.linalg.norm(target)
    return float(np.linalg.norm(aligned - target) / denom) if denom else float("nan")


def mantel_permutation_pvalue(
    visual_rdm: np.ndarray,
    text_rdm: np.ndarray,
    rounds: int,
    seed: int,
) -> tuple[float, float]:
    observed = spearman_corr(upper_triangle_values(visual_rdm), upper_triangle_values(text_rdm))
    rng = np.random.default_rng(seed)
    null_values = []
    for _ in range(rounds):
        permutation = rng.permutation(visual_rdm.shape[0])
        permuted = visual_rdm[permutation][:, permutation]
        null_values.append(
            spearman_corr(upper_triangle_values(permuted), upper_triangle_values(text_rdm))
        )
    null = np.asarray(null_values)
    p_value = (np.sum(np.abs(null) >= abs(observed)) + 1) / (rounds + 1)
    return float(observed), float(p_value)


def vector_permutation_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    corr_fn,
    rounds: int,
    seed: int,
) -> tuple[float, float]:
    """Two-sided permutation p-value for a vector correlation."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if len(a) < 3:
        return float("nan"), float("nan")

    observed = corr_fn(a, b)
    rng = np.random.default_rng(seed)
    null_values = []
    for _ in range(rounds):
        null_values.append(corr_fn(a, rng.permutation(b)))
    null = np.asarray(null_values)
    p_value = (np.sum(np.abs(null) >= abs(observed)) + 1) / (rounds + 1)
    return float(observed), float(p_value)


def rdm_permutation_pvalue(
    visual_rdm: np.ndarray,
    text_rdm: np.ndarray,
    corr_fn,
    rounds: int,
    seed: int,
) -> tuple[float, float]:
    """Mantel-style permutation p-value for RDM correlations."""
    observed = corr_fn(upper_triangle_values(visual_rdm), upper_triangle_values(text_rdm))
    rng = np.random.default_rng(seed)
    null_values = []
    for _ in range(rounds):
        permutation = rng.permutation(visual_rdm.shape[0])
        permuted = visual_rdm[permutation][:, permutation]
        null_values.append(corr_fn(upper_triangle_values(permuted), upper_triangle_values(text_rdm)))
    null = np.asarray(null_values)
    p_value = (np.sum(np.abs(null) >= abs(observed)) + 1) / (rounds + 1)
    return float(observed), float(p_value)


def try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        warnings.warn(f"matplotlib is unavailable; skipping plots: {exc}")
        return None


def save_heatmap(path: Path, matrix: np.ndarray, row_labels: list[str], col_labels: list[str], title: str, cmap: str = "viridis") -> None:
    plt = try_import_matplotlib()
    if plt is None:
        return
    fig_width = max(7, 0.55 * len(col_labels) + 2)
    fig_height = max(6, 0.5 * len(row_labels) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=160)
    image = ax.imshow(matrix, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if matrix.shape[0] <= 12 and matrix.shape[1] <= 12:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                label = f"{int(value)}" if float(value).is_integer() else f"{value:.2f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=7, color="white" if value > np.nanmax(matrix) * 0.55 else "black")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_bar_chart(path: Path, labels: list[str], values: list[float], title: str, ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    plt = try_import_matplotlib()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    ax.bar(labels, values, color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_scatter(path: Path, x: np.ndarray, y: np.ndarray, labels: list[str], title: str, xlabel: str, ylabel: str, subtitle: str = "") -> None:
    plt = try_import_matplotlib()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=160)
    ax.scatter(x, y, s=42, color="#F58518")
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_histogram(path: Path, values: np.ndarray, title: str, xlabel: str) -> None:
    plt = try_import_matplotlib()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    ax.hist(values, bins=30, color="#54A24B", edgecolor="white")
    ax.axvline(float(np.mean(values)), color="#E45756", linestyle="--", linewidth=1.5, label="mean")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_intervention_plot(path: Path, intervention_rows: list[dict]) -> None:
    labels = [row["condition"] for row in intervention_rows]
    values = [float(row["accuracy"]) for row in intervention_rows]
    save_bar_chart(path, labels, values, "Procrustes Intervention Accuracy", "accuracy", (0, 1))


def centroids_by_color(image_embeddings: np.ndarray, label_indices: np.ndarray, colors: list[str]) -> np.ndarray:
    centroids = []
    for index, color in enumerate(colors):
        mask = label_indices == index
        if not np.any(mask):
            raise ValueError(f"No samples for color {color}")
        centroid = image_embeddings[mask].mean(axis=0)
        centroids.append(centroid)
    return l2_normalize(np.vstack(centroids))


def zero_shot_predictions(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    labels: list[str],
    colors: list[str],
) -> tuple[list[dict], dict, np.ndarray, np.ndarray]:
    label_indices = labels_to_indices(labels, colors)
    similarities = cosine_matrix(image_embeddings, text_embeddings)
    pred_indices = np.argmax(similarities, axis=1)
    margins = []
    rows = []

    for i, (true_index, pred_index) in enumerate(zip(label_indices, pred_indices)):
        correct_similarity = similarities[i, true_index]
        wrong_mask = np.ones(similarities.shape[1], dtype=bool)
        wrong_mask[true_index] = False
        wrong_mean = similarities[i, wrong_mask].mean()
        margin = float(correct_similarity - wrong_mean)
        margins.append(margin)
        rows.append(
            {
                "sample_index": i,
                "true_color": colors[true_index],
                "pred_color": colors[pred_index],
                "correct": int(true_index == pred_index),
                "pair_alignment_margin": margin,
                "correct_similarity": float(correct_similarity),
                "pred_similarity": float(similarities[i, pred_index]),
            }
        )

    accuracy = float(np.mean(pred_indices == label_indices))
    summary = {
        "zero_shot_accuracy": accuracy,
        "pair_alignment_mean": float(np.mean(margins)),
        "pair_alignment_std": float(np.std(margins)),
    }
    return rows, summary, similarities, pred_indices


def class_metrics(
    labels: list[str],
    pred_indices: np.ndarray,
    margins: np.ndarray,
    colors: list[str],
) -> list[dict]:
    label_indices = labels_to_indices(labels, colors)
    rows = []
    for index, color in enumerate(colors):
        mask = label_indices == index
        support = int(np.sum(mask))
        correct = int(np.sum(pred_indices[mask] == index))
        rows.append(
            {
                "color": color,
                "support": support,
                "accuracy": correct / support if support else float("nan"),
                "mean_pair_alignment_margin": float(np.mean(margins[mask])) if support else float("nan"),
            }
        )
    return rows


def confusion_matrix(labels: list[str], pred_indices: np.ndarray, colors: list[str]) -> np.ndarray:
    label_indices = labels_to_indices(labels, colors)
    matrix = np.zeros((len(colors), len(colors)), dtype=int)
    for true_index, pred_index in zip(label_indices, pred_indices):
        matrix[true_index, pred_index] += 1
    return matrix


def classwise_alignment_rows(visual_rdm: np.ndarray, text_rdm: np.ndarray, class_rows: list[dict], colors: list[str]) -> list[dict]:
    accuracy_by_color = {row["color"]: row["accuracy"] for row in class_rows}
    margin_by_color = {row["color"]: row["mean_pair_alignment_margin"] for row in class_rows}
    rows = []
    for i, color in enumerate(colors):
        mask = np.ones(len(colors), dtype=bool)
        mask[i] = False
        local_spearman = spearman_corr(visual_rdm[i, mask], text_rdm[i, mask])
        local_pearson = pearson_corr(visual_rdm[i, mask], text_rdm[i, mask])
        rows.append(
            {
                "color": color,
                "classwise_rsa_spearman": local_spearman,
                "classwise_rsa_pearson": local_pearson,
                "accuracy": accuracy_by_color.get(color),
                "mean_pair_alignment_margin": margin_by_color.get(color),
            }
        )
    return rows


def visual_separability(image_embeddings: np.ndarray, label_indices: np.ndarray, colors: list[str]) -> dict:
    normalized = l2_normalize(image_embeddings)
    distance = 1.0 - normalized @ normalized.T
    silhouettes = []
    intra_distances = []
    inter_centroid_distances = []
    centroids = centroids_by_color(image_embeddings, label_indices, colors)
    centroid_distance = rdm(centroids)

    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            inter_centroid_distances.append(centroid_distance[i, j])

    for i, label in enumerate(label_indices):
        same_mask = label_indices == label
        other_mask = ~same_mask
        same_indices = np.where(same_mask)[0]
        if len(same_indices) > 1:
            a = float(np.mean(distance[i, same_indices[same_indices != i]]))
            intra_distances.append(a)
        else:
            a = 0.0
        b = float(
            min(
                np.mean(distance[i, label_indices == other_label])
                for other_label in range(len(colors))
                if other_label != label and np.any(label_indices == other_label)
            )
        )
        silhouettes.append((b - a) / max(a, b) if max(a, b) else 0.0)

    mean_intra = float(np.mean(intra_distances)) if intra_distances else float("nan")
    mean_inter = float(np.mean(inter_centroid_distances)) if inter_centroid_distances else float("nan")
    return {
        "visual_silhouette_cosine": float(np.mean(silhouettes)),
        "visual_mean_intra_class_distance": mean_intra,
        "visual_mean_inter_centroid_distance": mean_inter,
        "visual_inter_intra_ratio": mean_inter / mean_intra if mean_intra else float("nan"),
    }


def stratified_split(label_indices: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    train = []
    test = []
    for label in sorted(set(label_indices.tolist())):
        indices = np.where(label_indices == label)[0].tolist()
        rng.shuffle(indices)
        n_test = max(1, int(round(len(indices) * test_size)))
        test.extend(indices[:n_test])
        train.extend(indices[n_test:])
    return np.array(sorted(train), dtype=int), np.array(sorted(test), dtype=int)


def linear_probe(
    image_embeddings: np.ndarray,
    label_indices: np.ndarray,
    colors: list[str],
    test_size: float,
    alpha: float,
    seed: int,
) -> dict:
    x = l2_normalize(image_embeddings)
    y = label_indices
    train_idx, test_idx = stratified_split(y, test_size, seed)
    x_train = np.hstack([x[train_idx], np.ones((len(train_idx), 1))])
    x_test = np.hstack([x[test_idx], np.ones((len(test_idx), 1))])
    y_train = np.zeros((len(train_idx), len(colors)), dtype=float)
    y_train[np.arange(len(train_idx)), y[train_idx]] = 1.0

    reg = alpha * np.eye(x_train.shape[1])
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x_train.T @ x_train + reg, x_train.T @ y_train)
    scores = x_test @ weights
    pred = np.argmax(scores, axis=1)
    accuracy = float(np.mean(pred == y[test_idx]))

    rows = []
    for index, color in enumerate(colors):
        mask = y[test_idx] == index
        support = int(np.sum(mask))
        rows.append(
            {
                "color": color,
                "linear_probe_test_support": support,
                "linear_probe_accuracy": float(np.mean(pred[mask] == index)) if support else float("nan"),
            }
        )
    return {
        "linear_probe_accuracy": accuracy,
        "linear_probe_train_n": int(len(train_idx)),
        "linear_probe_test_n": int(len(test_idx)),
        "linear_probe_class_rows": rows,
    }


def random_orthogonal(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return q * signs


def intervention_results(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    visual_centroids: np.ndarray,
    labels: list[str],
    colors: list[str],
    seed: int,
) -> list[dict]:
    label_indices = labels_to_indices(labels, colors)
    rows = []

    conditions: list[tuple[str, np.ndarray]] = [("original", text_embeddings)]

    w_correct = orthogonal_procrustes(text_embeddings, visual_centroids)
    conditions.append(("correct_procrustes_text_to_visual", text_embeddings @ w_correct))

    q_random = random_orthogonal(text_embeddings.shape[1], seed)
    conditions.append(("random_orthogonal_text", text_embeddings @ q_random))

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(colors))
    permuted_text = text_embeddings[permutation]
    w_permuted = orthogonal_procrustes(permuted_text, visual_centroids)
    conditions.append(("label_permuted_procrustes", text_embeddings @ w_permuted))

    for condition, prototypes in conditions:
        sims = cosine_matrix(image_embeddings, prototypes)
        pred = np.argmax(sims, axis=1)
        rows.append(
            {
                "condition": condition,
                "accuracy": float(np.mean(pred == label_indices)),
                "mean_pair_alignment_margin": float(
                    np.mean(
                        [
                            sims[i, true_index]
                            - np.mean(np.delete(sims[i], true_index))
                            for i, true_index in enumerate(label_indices)
                        ]
                    )
                ),
            }
        )
    return rows


def run_analysis(dataset: Dataset, embeddings: Embeddings, args: argparse.Namespace) -> dict:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = embeddings.colors
    image_embeddings = l2_normalize(embeddings.image_embeddings)
    text_embeddings = l2_normalize(embeddings.text_embeddings)
    label_indices = labels_to_indices(dataset.labels, colors)

    prediction_rows, zs_summary, similarities, pred_indices = zero_shot_predictions(
        image_embeddings, text_embeddings, dataset.labels, colors
    )
    margins = np.array([row["pair_alignment_margin"] for row in prediction_rows], dtype=float)

    for row, source in zip(prediction_rows, dataset.rows):
        row["image_path"] = source.get("image_path")
        row["hex"] = source.get("hex")
        row["total_votes"] = source.get("total_votes")
        row["agreement_rate"] = source.get("agreement_rate")

    write_csv(
        output_dir / "sample_predictions.csv",
        [
            "sample_index",
            "image_path",
            "hex",
            "true_color",
            "pred_color",
            "correct",
            "pair_alignment_margin",
            "correct_similarity",
            "pred_similarity",
            "total_votes",
            "agreement_rate",
        ],
        prediction_rows,
    )

    class_rows = class_metrics(dataset.labels, pred_indices, margins, colors)
    visual_centroids = centroids_by_color(image_embeddings, label_indices, colors)
    visual_rdm = rdm(visual_centroids)
    text_rdm = rdm(text_embeddings)
    visual_rdm_values = upper_triangle_values(visual_rdm)
    text_rdm_values = upper_triangle_values(text_rdm)

    class_alignment = classwise_alignment_rows(visual_rdm, text_rdm, class_rows, colors)
    class_row_by_color = {row["color"]: row for row in class_rows}
    for row in class_alignment:
        row.update(class_row_by_color[row["color"]])

    class_accuracy_values = np.array([row["accuracy"] for row in class_alignment], dtype=float)
    class_rsa_spearman_values = np.array(
        [row["classwise_rsa_spearman"] for row in class_alignment], dtype=float
    )
    class_rsa_pearson_values = np.array(
        [row["classwise_rsa_pearson"] for row in class_alignment], dtype=float
    )
    class_margin_values = np.array(
        [row["mean_pair_alignment_margin"] for row in class_alignment], dtype=float
    )

    write_csv(
        output_dir / "class_metrics.csv",
        [
            "color",
            "support",
            "accuracy",
            "mean_pair_alignment_margin",
            "classwise_rsa_spearman",
            "classwise_rsa_pearson",
        ],
        class_alignment,
    )

    conf = confusion_matrix(dataset.labels, pred_indices, colors)
    write_matrix_csv(output_dir / "confusion_matrix.csv", colors, colors, conf)
    write_matrix_csv(output_dir / "rdm_visual_centroids.csv", colors, colors, visual_rdm)
    write_matrix_csv(output_dir / "rdm_text_prototypes.csv", colors, colors, text_rdm)
    write_matrix_csv(output_dir / "similarity_image_text.csv", [str(i) for i in range(len(dataset.labels))], colors, similarities)

    rsa_spearman, rsa_spearman_p = rdm_permutation_pvalue(
        visual_rdm, text_rdm, spearman_corr, args.permutation_rounds, args.seed
    )
    rsa_kendall, rsa_kendall_p = rdm_permutation_pvalue(
        visual_rdm, text_rdm, kendall_tau, args.permutation_rounds, args.seed + 1
    )
    rsa_pearson, rsa_pearson_p = rdm_permutation_pvalue(
        visual_rdm, text_rdm, pearson_corr, args.permutation_rounds, args.seed + 2
    )
    class_spearman_acc_r, class_spearman_acc_p = vector_permutation_pvalue(
        class_rsa_spearman_values, class_accuracy_values, spearman_corr, args.permutation_rounds, args.seed + 3
    )
    class_pearson_acc_r, class_pearson_acc_p = vector_permutation_pvalue(
        class_rsa_pearson_values, class_accuracy_values, pearson_corr, args.permutation_rounds, args.seed + 4
    )
    class_margin_acc_r, class_margin_acc_p = vector_permutation_pvalue(
        class_margin_values, class_accuracy_values, spearman_corr, args.permutation_rounds, args.seed + 5
    )
    sample_margin_correct_r, sample_margin_correct_p = vector_permutation_pvalue(
        margins, np.array([row["correct"] for row in prediction_rows], dtype=float), spearman_corr, args.permutation_rounds, args.seed + 6
    )

    correlation_rows = [
        {"analysis": "global_rdm", "x": "visual_rdm", "y": "text_rdm", "method": "spearman_mantel", "r": rsa_spearman, "p_value_permutation_two_sided": rsa_spearman_p, "n": len(visual_rdm_values)},
        {"analysis": "global_rdm", "x": "visual_rdm", "y": "text_rdm", "method": "kendall_mantel", "r": rsa_kendall, "p_value_permutation_two_sided": rsa_kendall_p, "n": len(visual_rdm_values)},
        {"analysis": "global_rdm", "x": "visual_rdm", "y": "text_rdm", "method": "pearson_mantel", "r": rsa_pearson, "p_value_permutation_two_sided": rsa_pearson_p, "n": len(visual_rdm_values)},
        {"analysis": "class_level", "x": "classwise_rsa_spearman", "y": "class_accuracy", "method": "spearman", "r": class_spearman_acc_r, "p_value_permutation_two_sided": class_spearman_acc_p, "n": len(colors)},
        {"analysis": "class_level", "x": "classwise_rsa_pearson", "y": "class_accuracy", "method": "pearson", "r": class_pearson_acc_r, "p_value_permutation_two_sided": class_pearson_acc_p, "n": len(colors)},
        {"analysis": "class_level", "x": "mean_pair_alignment_margin", "y": "class_accuracy", "method": "spearman", "r": class_margin_acc_r, "p_value_permutation_two_sided": class_margin_acc_p, "n": len(colors)},
        {"analysis": "sample_level", "x": "pair_alignment_margin", "y": "correct", "method": "spearman", "r": sample_margin_correct_r, "p_value_permutation_two_sided": sample_margin_correct_p, "n": len(prediction_rows)},
    ]
    write_csv(
        output_dir / "correlations.csv",
        ["analysis", "x", "y", "method", "r", "p_value_permutation_two_sided", "n"],
        correlation_rows,
    )

    relational_summary = {
        "rsa_spearman": rsa_spearman,
        "rsa_spearman_permutation_p": rsa_spearman_p,
        "rsa_kendall_tau": rsa_kendall,
        "rsa_kendall_tau_permutation_p": rsa_kendall_p,
        "rsa_pearson": rsa_pearson,
        "rsa_pearson_permutation_p": rsa_pearson_p,
        "classwise_rsa_spearman_vs_accuracy_r": class_spearman_acc_r,
        "classwise_rsa_spearman_vs_accuracy_p": class_spearman_acc_p,
        "classwise_rsa_pearson_vs_accuracy_r": class_pearson_acc_r,
        "classwise_rsa_pearson_vs_accuracy_p": class_pearson_acc_p,
        "class_pair_margin_vs_accuracy_spearman_r": class_margin_acc_r,
        "class_pair_margin_vs_accuracy_spearman_p": class_margin_acc_p,
        "sample_pair_margin_vs_correct_spearman_r": sample_margin_correct_r,
        "sample_pair_margin_vs_correct_spearman_p": sample_margin_correct_p,
        "linear_cka_centroids_vs_text": linear_cka(visual_centroids, text_embeddings),
        "procrustes_error_text_to_visual_centroids": procrustes_error(text_embeddings, visual_centroids),
    }

    separability = visual_separability(image_embeddings, label_indices, colors)
    probe = linear_probe(
        image_embeddings,
        label_indices,
        colors,
        args.linear_probe_test_size,
        args.ridge_alpha,
        args.seed,
    )
    write_csv(
        output_dir / "linear_probe_class_metrics.csv",
        ["color", "linear_probe_test_support", "linear_probe_accuracy"],
        probe["linear_probe_class_rows"],
    )

    interventions = intervention_results(
        image_embeddings, text_embeddings, visual_centroids, dataset.labels, colors, args.seed
    )
    write_csv(
        output_dir / "procrustes_interventions.csv",
        ["condition", "accuracy", "mean_pair_alignment_margin"],
        interventions,
    )

    plot_files: list[str] = []
    if not args.no_plots:
        plots_dir = output_dir / "plots"
        plot_specs = [
            (plots_dir / "confusion_matrix.png", lambda p: save_heatmap(p, conf, colors, colors, "Zero-shot Confusion Matrix", "Blues")),
            (plots_dir / "rdm_visual_centroids.png", lambda p: save_heatmap(p, visual_rdm, colors, colors, "Visual Color Centroid RDM", "magma")),
            (plots_dir / "rdm_text_prototypes.png", lambda p: save_heatmap(p, text_rdm, colors, colors, "Text Color Prototype RDM", "magma")),
            (plots_dir / "class_accuracy.png", lambda p: save_bar_chart(p, colors, class_accuracy_values.tolist(), "Zero-shot Accuracy by Color", "accuracy", (0, 1))),
            (plots_dir / "pair_alignment_margin_hist.png", lambda p: save_histogram(p, margins, "Pair Alignment Margin Distribution", "correct similarity - mean wrong similarity")),
            (plots_dir / "classwise_rsa_vs_accuracy.png", lambda p: save_scatter(
                p,
                class_rsa_spearman_values,
                class_accuracy_values,
                colors,
                "Class-wise Alignment vs Accuracy",
                "class-wise RSA Spearman",
                "zero-shot accuracy",
                f"r={class_spearman_acc_r:.3f}, p={class_spearman_acc_p:.4f}",
            )),
            (plots_dir / "pair_margin_vs_accuracy.png", lambda p: save_scatter(
                p,
                class_margin_values,
                class_accuracy_values,
                colors,
                "Pair Margin vs Accuracy",
                "mean pair alignment margin",
                "zero-shot accuracy",
                f"r={class_margin_acc_r:.3f}, p={class_margin_acc_p:.4f}",
            )),
            (plots_dir / "procrustes_interventions.png", lambda p: save_intervention_plot(p, interventions)),
        ]
        for plot_path, plotter in plot_specs:
            plotter(plot_path)
            if plot_path.exists():
                plot_files.append(str(plot_path))

    summary = {
        "ground_truth_csv": str(args.ground_truth),
        "n_samples": len(dataset.labels),
        "n_colors": len(colors),
        "colors": colors,
        "prompts": embeddings.prompts,
        "backend": args.backend,
        "model": args.model,
        "pretrained": args.pretrained,
        **zs_summary,
        **relational_summary,
        **separability,
        "linear_probe_accuracy": probe["linear_probe_accuracy"],
        "linear_probe_train_n": probe["linear_probe_train_n"],
        "linear_probe_test_n": probe["linear_probe_test_n"],
        "plot_files": plot_files,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    prompts = ensure_prompt_templates(args.prompts)

    if args.embeddings_npz:
        embeddings, labels, image_paths = load_embeddings(args.embeddings_npz)
        dataset = Dataset(
            image_paths=image_paths,
            labels=labels,
            rows=[{"image_path": p, "basic_color": y} for p, y in zip(image_paths, labels)],
        )
    else:
        dataset = load_dataset(args.ground_truth, args.dataset_root)
        embeddings = extract_open_clip_embeddings(
            dataset=dataset,
            colors=list(BASIC_COLORS),
            prompts=prompts,
            model_name=args.model,
            pretrained=args.pretrained,
            device_name=args.device,
            batch_size=args.batch_size,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_embeddings(args.output_dir / args.save_embeddings_name, embeddings, dataset)

    summary = run_analysis(dataset, embeddings, args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote experiment outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
