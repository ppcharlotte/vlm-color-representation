#!/usr/bin/env python3
"""Evaluate generative VLM color naming on pure color patches.

The script asks a vision-language generation model to answer one of the
11 basic color names for each image patch, then evaluates normalized accuracy,
macro-F1, per-color accuracy, and a confusion matrix.

Default dataset:
  data/color_patches_aggregated_100/ground_truth.csv

Supported backend in this script:
  transformers-qwen-vl: Qwen2/2.5-VL style HuggingFace models.
  transformers-llava: LLaVA HuggingFace models such as llava-hf/llava-1.5-7b-hf.

Example:
  python3 -B scripts/run_generative_color_naming_experiment.py \
    --backend transformers-qwen-vl \
    --model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --limit 20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


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

ALIASES = {
    "grey": "gray",
}

DEFAULT_PROMPT = (
    "Look at the image. Answer with exactly one basic color name from this list: "
    "black, blue, brown, gray, green, orange, pink, purple, red, white, yellow. "
    "Do not explain."
)


@dataclass
class Sample:
    index: int
    image_path: str
    true_color: str
    metadata: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generative VLM color naming on color patch images."
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/color_patches_aggregated_100/ground_truth.csv"),
        help="Ground-truth CSV with image_path and basic_color columns.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("."),
        help="Root for resolving relative image_path values. Default: current directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/generative_color_naming"),
        help="Directory for result CSV/JSON outputs.",
    )
    parser.add_argument(
        "--backend",
        choices=("transformers-qwen-vl", "transformers-llava"),
        default="transformers-qwen-vl",
        help="Generative backend. Default: transformers-qwen-vl.",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HuggingFace model id. Default: Qwen/Qwen2.5-VL-3B-Instruct.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Model dtype. Default: auto.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Instruction prompt sent with every image.")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Max generated tokens. Default: 8")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature. Default: 0")
    parser.add_argument("--limit", type=int, help="Optional sample limit for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed where supported.")
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples already present in sample_predictions.csv.",
    )
    return parser.parse_args()


def load_samples(ground_truth: Path, dataset_root: Path, limit: int | None) -> list[Sample]:
    if not ground_truth.exists():
        raise FileNotFoundError(f"Ground-truth CSV not found: {ground_truth}")

    samples: list[Sample] = []
    with ground_truth.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "basic_color"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Ground-truth CSV missing columns: {sorted(missing)}")

        for idx, row in enumerate(reader):
            true_color = row["basic_color"].strip().lower()
            if true_color not in BASIC_COLORS:
                continue
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = dataset_root / image_path
            samples.append(
                Sample(
                    index=idx,
                    image_path=str(image_path),
                    true_color=true_color,
                    metadata=row,
                )
            )
            if limit is not None and len(samples) >= limit:
                break

    if not samples:
        raise ValueError("No samples loaded from ground-truth CSV.")
    return samples


def normalize_color_answer(text: str) -> tuple[str | None, str]:
    """Map a generated answer to one of the 11 basic colors when unambiguous."""
    cleaned = text.strip().lower()
    cleaned = cleaned.replace("_", " ").replace("-", " ").replace("/", " ")
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
    tokens = re.findall(r"[a-z]+", cleaned)
    mapped = [ALIASES.get(token, token) for token in tokens]
    matches = [token for token in mapped if token in BASIC_COLORS]
    unique = sorted(set(matches))

    if len(unique) == 1:
        return unique[0], "single_basic_color_match"
    if len(unique) > 1:
        return None, "ambiguous_multiple_basic_colors"
    return None, "no_basic_color_match"


def choose_torch_device(device_name: str):
    import torch

    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_torch_dtype(dtype_name: str):
    import torch

    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return "auto"



class LlavaGenerator:
    def __init__(self, model_id: str, device_name: str, dtype_name: str, seed: int) -> None:
        try:
            import torch
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except Exception as exc:
            raise RuntimeError(
                "transformers-llava backend requires torch and transformers with LLaVA support. "
                "Install with: pip install torch torchvision transformers accelerate pillow"
            ) from exc

        self.torch = torch
        self.device = choose_torch_device(device_name)
        torch.manual_seed(seed)

        dtype = choose_torch_dtype(dtype_name)
        model_kwargs = {"torch_dtype": dtype} if dtype != "auto" else {"torch_dtype": "auto"}
        if self.device.type == "cuda":
            model_kwargs["device_map"] = "auto"

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=True,
            **model_kwargs,
        )
        if self.device.type != "cuda":
            self.model.to(self.device)
        self.model.eval()

    def generate(self, image_path: str, prompt: str, max_new_tokens: int, temperature: float) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if hasattr(self.processor, "apply_chat_template"):
            try:
                text = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception:
                text = f"USER: <image>\n{prompt}\nASSISTANT:"
        else:
            text = f"USER: <image>\n{prompt}\nASSISTANT:"

        inputs = self.processor(text=text, images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        do_sample = temperature > 0

        with self.torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
            )

        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_len:]
        outputs = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return outputs[0].strip()


class QwenVLGenerator:
    def __init__(self, model_id: str, device_name: str, dtype_name: str, seed: int) -> None:
        try:
            import torch
            from transformers import AutoProcessor
        except Exception as exc:
            raise RuntimeError(
                "transformers-qwen-vl backend requires torch and transformers. "
                "Install with: pip install torch torchvision transformers accelerate pillow"
            ) from exc

        self.torch = torch
        self.device = choose_torch_device(device_name)
        torch.manual_seed(seed)

        dtype = choose_torch_dtype(dtype_name)
        model_kwargs = {"torch_dtype": dtype} if dtype != "auto" else {"torch_dtype": "auto"}
        if self.device.type == "cuda":
            model_kwargs["device_map"] = "auto"

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model_classes = []
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration

            model_classes.append(Qwen2_5_VLForConditionalGeneration)
        except Exception:
            pass
        try:
            from transformers import Qwen2VLForConditionalGeneration

            model_classes.append(Qwen2VLForConditionalGeneration)
        except Exception:
            pass
        try:
            from transformers import AutoModelForVision2Seq

            model_classes.append(AutoModelForVision2Seq)
        except Exception:
            pass
        if not model_classes:
            raise RuntimeError(
                "Could not find a supported VLM generation class in transformers. "
                "Try upgrading transformers: pip install -U transformers"
            )

        last_error = None
        for model_class in model_classes:
            try:
                self.model = model_class.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    **model_kwargs,
                )
                break
            except Exception as exc:
                last_error = exc
        else:
            raise RuntimeError(f"Failed to load {model_id!r} with available VLM classes") from last_error
        if self.device.type != "cuda":
            self.model.to(self.device)
        self.model.eval()

    def generate(self, image_path: str, prompt: str, max_new_tokens: int, temperature: float) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        do_sample = temperature > 0

        with self.torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
            )

        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_len:]
        outputs = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return outputs[0].strip()


def build_generator(args: argparse.Namespace):
    if args.backend == "transformers-qwen-vl":
        return QwenVLGenerator(args.model_id, args.device, args.dtype, args.seed)
    if args.backend == "transformers-llava":
        return LlavaGenerator(args.model_id, args.device, args.dtype, args.seed)
    raise ValueError(f"Unsupported backend: {args.backend}")


def load_completed_predictions(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            completed.add(int(row["sample_index"]))
    return completed


def append_prediction(path: Path, row: dict[str, str | int | float | None], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def confusion_matrix(rows: list[dict[str, str]]) -> list[list[int]]:
    color_to_index = {color: idx for idx, color in enumerate(BASIC_COLORS)}
    matrix = [[0 for _ in BASIC_COLORS] for _ in BASIC_COLORS]
    for row in rows:
        true_color = row["true_color"]
        pred_color = row["normalized_pred_color"]
        if true_color in color_to_index and pred_color in color_to_index:
            matrix[color_to_index[true_color]][color_to_index[pred_color]] += 1
    return matrix


def write_confusion(path: Path, matrix: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_color", *BASIC_COLORS])
        for color, values in zip(BASIC_COLORS, matrix):
            writer.writerow([color, *values])


def compute_metrics(rows: list[dict[str, str]]) -> tuple[dict, list[dict]]:
    total = len(rows)
    parsed = [row for row in rows if row["normalized_pred_color"]]
    correct = [row for row in parsed if row["true_color"] == row["normalized_pred_color"]]

    per_color_rows = []
    f1_values = []
    for color in BASIC_COLORS:
        support = sum(row["true_color"] == color for row in rows)
        predicted = sum(row["normalized_pred_color"] == color for row in rows)
        true_positive = sum(
            row["true_color"] == color and row["normalized_pred_color"] == color for row in rows
        )
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            f1_values.append(f1)
        per_color_rows.append(
            {
                "color": color,
                "support": support,
                "predicted": predicted,
                "true_positive": true_positive,
                "accuracy_recall": recall,
                "precision": precision,
                "f1": f1,
            }
        )

    metrics = {
        "n_samples": total,
        "n_parseable": len(parsed),
        "parse_rate": len(parsed) / total if total else 0.0,
        "normalized_accuracy_all": len(correct) / total if total else 0.0,
        "normalized_accuracy_parseable_only": len(correct) / len(parsed) if parsed else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
    }
    return metrics, per_color_rows


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else float("nan")


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    return pearson_corr(rankdata_average(a), rankdata_average(b))


def permutation_pvalue(a: np.ndarray, b: np.ndarray, corr_fn, rounds: int, seed: int) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan"), float("nan")
    observed = corr_fn(a, b)
    rng = np.random.default_rng(seed)
    null = np.asarray([corr_fn(a, rng.permutation(b)) for _ in range(rounds)])
    p_value = (np.sum(np.abs(null) >= abs(observed)) + 1) / (rounds + 1)
    return float(observed), float(p_value)


def parse_float_or_nan(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def compute_correlations(rows: list[dict[str, str]], rounds: int, seed: int) -> list[dict]:
    correct = np.array([parse_float_or_nan(row.get("correct")) for row in rows], dtype=float)
    candidates = [
        ("agreement_rate", "correct", "spearman"),
        ("label_entropy_bits", "correct", "spearman"),
        ("total_votes", "correct", "spearman"),
        ("num_basic_labels", "correct", "spearman"),
        ("latency_sec", "correct", "spearman"),
    ]
    output = []
    for offset, (x_name, y_name, method) in enumerate(candidates):
        x = np.array([parse_float_or_nan(row.get(x_name)) for row in rows], dtype=float)
        valid = np.isfinite(x) & np.isfinite(correct)
        if np.sum(valid) < 3:
            continue
        r, p_value = permutation_pvalue(x[valid], correct[valid], spearman_corr, rounds, seed + offset)
        output.append(
            {
                "analysis": "sample_level",
                "x": x_name,
                "y": y_name,
                "method": method,
                "r": r,
                "p_value_permutation_two_sided": p_value,
                "n": int(np.sum(valid)),
            }
        )
    return output


def try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        warnings.warn(f"matplotlib is unavailable; skipping plots: {exc}")
        return None


def save_heatmap(path: Path, matrix: list[list[int]], title: str) -> None:
    plt = try_import_matplotlib()
    if plt is None:
        return
    values = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=160)
    image = ax.imshow(values, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(BASIC_COLORS)))
    ax.set_yticks(np.arange(len(BASIC_COLORS)))
    ax.set_xticklabels(BASIC_COLORS, rotation=45, ha="right")
    ax.set_yticklabels(BASIC_COLORS)
    ax.set_xlabel("predicted color")
    ax.set_ylabel("true color")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    max_value = float(np.nanmax(values)) if values.size else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if value:
                ax.text(j, i, str(int(value)), ha="center", va="center", fontsize=7, color="white" if value > max_value * 0.55 else "black")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_metric_bars(path: Path, per_color_rows: list[dict], metric: str, title: str, ylabel: str) -> None:
    plt = try_import_matplotlib()
    if plt is None:
        return
    labels = [row["color"] for row in per_color_rows]
    values = [float(row[metric]) for row in per_color_rows]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    ax.bar(labels, values, color="#4C78A8")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_parse_status_plot(path: Path, rows: list[dict[str, str]]) -> None:
    plt = try_import_matplotlib()
    if plt is None:
        return
    counts = Counter(row.get("parse_status", "") for row in rows)
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    ax.bar(labels, values, color="#F58518")
    ax.set_title("Generated Answer Parse Status")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_correlation_scatter(path: Path, rows: list[dict[str, str]], x_name: str, correlation_row: dict | None) -> None:
    plt = try_import_matplotlib()
    if plt is None:
        return
    x = np.array([parse_float_or_nan(row.get(x_name)) for row in rows], dtype=float)
    y = np.array([parse_float_or_nan(row.get("correct")) for row in rows], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=160)
    ax.scatter(x[valid], y[valid], s=16, alpha=0.45, color="#54A24B")
    subtitle = ""
    if correlation_row:
        subtitle = f"r={float(correlation_row['r']):.3f}, p={float(correlation_row['p_value_permutation_two_sided']):.4f}"
    ax.set_title(f"{x_name} vs Correct" + (f"\n{subtitle}" if subtitle else ""))
    ax.set_xlabel(x_name)
    ax.set_ylabel("correct")
    ax.set_yticks([0, 1])
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_plots(output_dir: Path, rows: list[dict[str, str]], per_color_rows: list[dict], matrix: list[list[int]], correlation_rows: list[dict], no_plots: bool) -> list[str]:
    if no_plots:
        return []
    plots_dir = output_dir / "plots"
    plot_files = []
    plot_specs = [
        (plots_dir / "confusion_matrix.png", lambda p: save_heatmap(p, matrix, "Generative VLM Confusion Matrix")),
        (plots_dir / "class_recall.png", lambda p: save_metric_bars(p, per_color_rows, "accuracy_recall", "Recall by True Color", "recall / accuracy")),
        (plots_dir / "class_precision.png", lambda p: save_metric_bars(p, per_color_rows, "precision", "Precision by Predicted Color", "precision")),
        (plots_dir / "class_f1.png", lambda p: save_metric_bars(p, per_color_rows, "f1", "F1 by Color", "F1")),
        (plots_dir / "parse_status.png", lambda p: save_parse_status_plot(p, rows)),
    ]
    corr_by_x = {row["x"]: row for row in correlation_rows}
    for x_name in ("agreement_rate", "label_entropy_bits", "total_votes"):
        plot_specs.append(
            (
                plots_dir / f"{x_name}_vs_correct.png",
                lambda p, name=x_name: save_correlation_scatter(p, rows, name, corr_by_x.get(name)),
            )
        )
    for plot_path, plotter in plot_specs:
        plotter(plot_path)
        if plot_path.exists():
            plot_files.append(str(plot_path))
    return plot_files


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.ground_truth, args.dataset_root, args.limit)
    predictions_path = args.output_dir / "sample_predictions.csv"
    completed = load_completed_predictions(predictions_path) if args.resume else set()

    fieldnames = [
        "sample_index",
        "image_path",
        "hex",
        "true_color",
        "raw_response",
        "normalized_pred_color",
        "parse_status",
        "correct",
        "latency_sec",
        "total_votes",
        "majority_count",
        "agreement_rate",
        "label_entropy_bits",
        "num_basic_labels",
        "basic_label_set",
    ]

    generator = build_generator(args)
    for sample in samples:
        if sample.index in completed:
            continue
        start = time.time()
        raw = generator.generate(sample.image_path, args.prompt, args.max_new_tokens, args.temperature)
        latency = time.time() - start
        normalized, status = normalize_color_answer(raw)
        row = {
            "sample_index": sample.index,
            "image_path": sample.image_path,
            "hex": sample.metadata.get("hex", ""),
            "true_color": sample.true_color,
            "raw_response": raw,
            "normalized_pred_color": normalized or "",
            "parse_status": status,
            "correct": int(normalized == sample.true_color),
            "latency_sec": round(latency, 4),
            "total_votes": sample.metadata.get("total_votes", ""),
            "majority_count": sample.metadata.get("majority_count", ""),
            "agreement_rate": sample.metadata.get("agreement_rate", ""),
            "label_entropy_bits": sample.metadata.get("label_entropy_bits", ""),
            "num_basic_labels": sample.metadata.get("num_basic_labels", ""),
            "basic_label_set": sample.metadata.get("basic_label_set", ""),
        }
        append_prediction(predictions_path, row, fieldnames)
        print(
            f"[{sample.index}] true={sample.true_color} pred={normalized} "
            f"raw={raw!r} correct={row['correct']}"
        )

    rows = read_prediction_rows(predictions_path)
    metrics, per_color_rows = compute_metrics(rows)
    correlation_rows = compute_correlations(rows, args.permutation_rounds, args.seed)
    matrix = confusion_matrix(rows)
    plot_files = save_plots(args.output_dir, rows, per_color_rows, matrix, correlation_rows, args.no_plots)

    for row in correlation_rows:
        key_base = f"{row['x']}_vs_{row['y']}_{row['method']}"
        metrics[f"{key_base}_r"] = row["r"]
        metrics[f"{key_base}_p"] = row["p_value_permutation_two_sided"]

    metrics.update(
        {
            "backend": args.backend,
            "model_id": args.model_id,
            "ground_truth_csv": str(args.ground_truth),
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "permutation_rounds": args.permutation_rounds,
            "plot_files": plot_files,
        }
    )

    write_csv(
        args.output_dir / "class_metrics.csv",
        ["color", "support", "predicted", "true_positive", "accuracy_recall", "precision", "f1"],
        per_color_rows,
    )
    write_csv(
        args.output_dir / "correlations.csv",
        ["analysis", "x", "y", "method", "r", "p_value_permutation_two_sided", "n"],
        correlation_rows,
    )
    write_confusion(args.output_dir / "confusion_matrix.csv", matrix)
    (args.output_dir / "summary.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Wrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
