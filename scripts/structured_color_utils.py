#!/usr/bin/env python3
"""Shared utilities for the modular LLaVA color representation pipeline."""

from __future__ import annotations

import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


BASIC_COLORS = [
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
]

FORCE_CHOICE_PROMPT = (
    "What color is this image? Answer only with exactly one color name from this list: "
    "black, blue, brown, gray, green, orange, pink, purple, red, white, yellow."
)
NO_COLOR_LIST_PROMPT = "What color is this image? Answer only with exactly one basic color name."

TEXT_CONTEXTS = [
    "The item is {color}.",
    "a {color} color patch.",
    "the image is {color}.",
]


@dataclass
class Dataset:
    rows: list[dict[str, str]]

    @property
    def labels(self) -> list[str]:
        return [normalize_color(row.get("basic_color") or row.get("true_color") or row.get("majority_basic_color")) for row in self.rows]

    @property
    def image_paths(self) -> list[str]:
        return [row["image_path"] for row in self.rows]


def normalize_color(value: object) -> str:
    text = str(value or "").strip().lower()
    if text == "grey":
        return "gray"
    return text


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def labels_to_indices(labels: list[str], colors: list[str] = BASIC_COLORS) -> np.ndarray:
    mapping = {color: idx for idx, color in enumerate(colors)}
    return np.array([mapping[normalize_color(label)] for label in labels], dtype=int)


def sample_color_rows(input_csv: Path, samples_per_color: int, seed: int) -> list[dict[str, str]]:
    rows = read_csv_rows(input_csv)
    buckets: dict[str, list[dict[str, str]]] = {color: [] for color in BASIC_COLORS}
    for row in rows:
        label = normalize_color(row.get("majority_basic_color"))
        if label in buckets:
            buckets[label].append(row)
    rng = random.Random(seed)
    selected = []
    for color in BASIC_COLORS:
        bucket = list(buckets[color])
        rng.shuffle(bucket)
        for row in bucket[:samples_per_color]:
            out = dict(row)
            out["basic_color"] = color
            selected.append(out)
    selected.sort(key=lambda row: (BASIC_COLORS.index(row["basic_color"]), int(row["r"]), int(row["g"]), int(row["b"])))
    for idx, row in enumerate(selected):
        row["sample_index"] = str(idx)
    return selected


def generate_patch_dataset(selected_rows: list[dict[str, str]], output_dir: Path, patch_size: int) -> Dataset:
    patches_dir = output_dir / "patches"
    out_rows = []
    counters = {color: 0 for color in BASIC_COLORS}
    for row in selected_rows:
        color = normalize_color(row["basic_color"])
        r, g, b = int(row["r"]), int(row["g"]), int(row["b"])
        local_idx = counters[color]
        counters[color] += 1
        rel_path = Path("patches") / color / f"{color}_{local_idx:03d}_{r:03d}_{g:03d}_{b:03d}.png"
        image_path = output_dir / rel_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (patch_size, patch_size), (r, g, b)).save(image_path)
        out = dict(row)
        out.update(
            {
                "sample_index": str(len(out_rows)),
                "basic_color": color,
                "true_color": color,
                "image_path": str(rel_path),
                "absolute_image_path": str(image_path),
            }
        )
        out_rows.append(out)
    return Dataset(out_rows)


def load_dataset(path: Path, dataset_root: Path | None = None) -> Dataset:
    rows = read_csv_rows(path)
    root = dataset_root if dataset_root is not None else path.parent
    for row in rows:
        row["basic_color"] = normalize_color(row.get("basic_color") or row.get("true_color") or row.get("majority_basic_color"))
        image_path = row.get("image_path") or row.get("absolute_image_path")
        if image_path:
            p = Path(image_path)
            row["absolute_image_path"] = str(p if p.is_absolute() else root / p)
    return Dataset(rows)


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


class LlavaHelper:
    def __init__(self, model_id: str, device: str = "auto", dtype: str = "auto", seed: int = 42) -> None:
        try:
            import torch
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except Exception as exc:
            raise RuntimeError("Install torch, torchvision, transformers, accelerate, and Pillow for LLaVA support.") from exc

        self.torch = torch
        self.device = choose_torch_device(device)
        torch.manual_seed(seed)
        model_dtype = choose_torch_dtype(dtype)
        kwargs = {"torch_dtype": model_dtype} if model_dtype != "auto" else {"torch_dtype": "auto"}
        if self.device.type == "cuda":
            kwargs["device_map"] = "auto"
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, trust_remote_code=True, **kwargs)
        if self.device.type != "cuda":
            self.model.to(self.device)
        self.model.eval()
        self.vision_feature_layer = getattr(self.model.config, "vision_feature_layer", -2)
        self.vision_feature_select_strategy = getattr(self.model.config, "vision_feature_select_strategy", "default")
        self.vision_tower = self._find_vision_tower()
        self.projector = self._find_projector()

    def _resolve_path(self, root, dotted_path: str):
        current = root
        for part in dotted_path.split("."):
            if not hasattr(current, part):
                return None
            current = getattr(current, part)
        return current

    def _find_vision_tower(self):
        if hasattr(self.model, "get_vision_tower"):
            tower = self.model.get_vision_tower()
            if tower is not None:
                return tower
        for path in ("vision_tower", "model.vision_tower", "model.vision_model", "vision_model"):
            tower = self._resolve_path(self.model, path)
            if tower is not None:
                return tower
        raise AttributeError("Could not find LLaVA vision tower")

    def _find_projector(self):
        for path in ("multi_modal_projector", "model.multi_modal_projector", "mm_projector", "model.mm_projector", "projector", "model.projector"):
            projector = self._resolve_path(self.model, path)
            if projector is not None:
                return projector
        raise AttributeError("Could not find LLaVA multimodal projector")

    def language_model(self):
        for path in ("language_model", "model.language_model", "model"):
            module = self._resolve_path(self.model, path)
            if module is not None and callable(getattr(module, "forward", None)):
                return module
        raise AttributeError("Could not find language model")

    def embedding_layer(self):
        emb = self.model.get_input_embeddings() if hasattr(self.model, "get_input_embeddings") else None
        return emb if emb is not None else self.model.language_model.get_input_embeddings()

    def tokenizer(self):
        return getattr(self.processor, "tokenizer", self.processor)

    def image_token_id(self) -> int:
        token_id = getattr(self.model.config, "image_token_index", None)
        if token_id is not None:
            return int(token_id)
        tokenizer = self.tokenizer()
        token_id = tokenizer.convert_tokens_to_ids("<image>")
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError("Could not resolve <image> token id")
        return int(token_id)

    def chat_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        if hasattr(self.processor, "apply_chat_template"):
            try:
                return self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            except Exception:
                pass
        return f"USER: <image>\n{prompt}\nASSISTANT:"

    def generate(self, image_path: str, prompt: str, max_new_tokens: int = 8, temperature: float = 0.0) -> str:
        image = Image.open(image_path).convert("RGB")
        text = self.chat_prompt(prompt)
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
        return self.processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def pixel_values(self, image_paths: list[str]):
        images = [Image.open(path).convert("RGB") for path in image_paths]
        text = ["USER: <image>\nASSISTANT:" for _ in images]
        inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True)
        return inputs["pixel_values"].to(self.device)

    def vision_features(self, image_paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
        pixel_values = self.pixel_values(image_paths)
        with self.torch.no_grad():
            outputs = self.vision_tower(pixel_values, output_hidden_states=True)
            selected = outputs.hidden_states[self.vision_feature_layer]
            if self.vision_feature_select_strategy != "full":
                selected = selected[:, 1:]
            projected = self.projector(selected)
        if hasattr(projected, "last_hidden_state"):
            projected = projected.last_hidden_state
        return selected.detach().float().cpu().numpy(), projected.detach().float().cpu().numpy()

    def expanded_language_inputs(self, image_path: str, prompt: str):
        embedding_layer = self.embedding_layer()
        tokenizer = self.tokenizer()
        image_token_id = self.image_token_id()
        _pre, projected = self.vision_features([image_path])
        image_features_t = self.torch.from_numpy(projected).to(self.device, dtype=embedding_layer.weight.dtype)
        encoded = tokenizer(self.chat_prompt(prompt), return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        positions = (input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten()
        if len(positions) != 1:
            raise ValueError(f"Prompt must contain exactly one image token; found {len(positions)}")
        pos = int(positions[0].item())
        token_embeds = embedding_layer(input_ids)
        inputs_embeds = self.torch.cat([token_embeds[:, :pos, :], image_features_t, token_embeds[:, pos + 1 :, :]], dim=1)
        attention_mask = self.torch.ones(inputs_embeds.shape[:2], dtype=self.torch.long, device=self.device)
        query_mask = self.torch.zeros(inputs_embeds.shape[1], dtype=self.torch.bool, device=self.device)
        special_ids = {x for x in [getattr(tokenizer, "bos_token_id", None), getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "pad_token_id", None)] if x is not None}
        n_image = image_features_t.shape[1]
        for original_idx, token_id in enumerate(input_ids[0].tolist()):
            if original_idx == pos or int(token_id) in special_ids:
                continue
            expanded_idx = original_idx if original_idx < pos else original_idx + n_image - 1
            query_mask[expanded_idx] = True
        return inputs_embeds, attention_mask, query_mask, (pos, pos + n_image)

    def layerwise_image_and_query_embeddings(self, image_paths: list[str], prompt: str, layer_indices: list[int], pooling: str = "mean") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        language_model = self.language_model()
        image_layers = [[] for _ in layer_indices]
        query_layers = [[] for _ in layer_indices]
        pre_projector = []
        post_projector = []
        with self.torch.no_grad():
            for image_path in image_paths:
                pre, projected = self.vision_features([image_path])
                pre_projector.append(pre[0].mean(axis=0) if pooling == "mean" else pre[0, 0])
                post_projector.append(projected[0].mean(axis=0) if pooling == "mean" else projected[0, 0])
                inputs_embeds, attention_mask, query_mask, (image_start, image_end) = self.expanded_language_inputs(image_path, prompt)
                outputs = language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                for out_idx, layer_idx in enumerate(layer_indices):
                    hidden = outputs.hidden_states[layer_idx][0]
                    image_tokens = hidden[image_start:image_end]
                    query_tokens = hidden[query_mask]
                    image_pooled = image_tokens.mean(dim=0) if pooling == "mean" else image_tokens[0]
                    query_pooled = query_tokens.mean(dim=0) if query_tokens.shape[0] else hidden.mean(dim=0)
                    image_layers[out_idx].append(image_pooled.detach().float().cpu().numpy())
                    query_layers[out_idx].append(query_pooled.detach().float().cpu().numpy())
        return (
            np.asarray(pre_projector, dtype=np.float32),
            np.asarray(post_projector, dtype=np.float32),
            np.stack([np.vstack(v) for v in image_layers]).astype(np.float32),
            np.stack([np.vstack(v) for v in query_layers]).astype(np.float32),
        )

    def color_token_mask(self, input_ids, color: str, text: str | None = None, offsets=None):
        tokenizer = self.tokenizer()
        sequence = input_ids[0]

        if text is not None and offsets is not None:
            match = re.search(rf"(?<![A-Za-z]){re.escape(color)}(?![A-Za-z])", text, flags=re.IGNORECASE)
            if match is not None:
                token_offsets = offsets[0] if getattr(offsets, "ndim", 0) == 3 else offsets
                mask = self.torch.zeros_like(sequence, dtype=self.torch.bool)
                start_char, end_char = match.span()
                for token_idx, pair in enumerate(token_offsets.tolist()):
                    token_start, token_end = int(pair[0]), int(pair[1])
                    if token_end > start_char and token_start < end_char:
                        mask[token_idx] = True
                if bool(mask.any().item()):
                    return mask

        candidates = []
        for variant in (color, " " + color):
            ids = tokenizer(variant, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(input_ids.device)
            if ids.numel() and not any(self.torch.equal(ids, existing) for existing in candidates):
                candidates.append(ids)
        for color_ids in candidates:
            for start in range(0, max(0, sequence.shape[0] - color_ids.shape[0] + 1)):
                if self.torch.equal(sequence[start : start + color_ids.shape[0]], color_ids):
                    mask = self.torch.zeros_like(sequence, dtype=self.torch.bool)
                    mask[start : start + color_ids.shape[0]] = True
                    return mask
        raise ValueError(f"Could not locate color token '{color}' in tokenized text: {text or tokenizer.decode(sequence)}")

    def layerwise_text_embeddings(self, colors: list[str], contexts: list[str], layer_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        language_model = self.language_model()
        tokenizer = self.tokenizer()
        aggregated = [[] for _ in layer_indices]
        context_specific = [[[] for _ in colors] for _ in layer_indices]
        with self.torch.no_grad():
            for color_idx, color in enumerate(colors):
                by_layer = [[] for _ in layer_indices]
                for context in contexts:
                    text = context.format(color=color)
                    try:
                        encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
                        offsets = encoded.pop("offset_mapping", None)
                    except Exception:
                        encoded = tokenizer(text, return_tensors="pt")
                        offsets = None
                    input_ids = encoded["input_ids"].to(self.device)
                    attention_mask = encoded.get("attention_mask")
                    attention_mask = attention_mask.to(self.device) if attention_mask is not None else None
                    color_mask = self.color_token_mask(input_ids, color, text=text, offsets=offsets)
                    outputs = language_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False, return_dict=True)
                    for out_idx, layer_idx in enumerate(layer_indices):
                        hidden = outputs.hidden_states[layer_idx][0]
                        selected = hidden[color_mask]
                        vector = selected.mean(dim=0)
                        vector = vector / vector.norm(dim=-1, keepdim=True)
                        by_layer[out_idx].append(vector)
                        context_specific[out_idx][color_idx].append(vector.detach().float().cpu().numpy())
                for out_idx in range(len(layer_indices)):
                    proto = self.torch.stack(by_layer[out_idx], dim=0).mean(dim=0)
                    proto = proto / proto.norm(dim=-1, keepdim=True)
                    aggregated[out_idx].append(proto.detach().float().cpu().numpy())
        context_array = []
        for out_idx in range(len(layer_indices)):
            color_arrays = []
            for color_idx in range(len(colors)):
                color_arrays.append(np.vstack(context_specific[out_idx][color_idx]))
            context_array.append(np.stack(color_arrays))
        return np.stack([np.vstack(v) for v in aggregated]).astype(np.float32), np.stack(context_array).astype(np.float32)


def normalize_color_answer(text: str) -> tuple[str | None, str]:
    lowered = (text or "").strip().lower()
    cleaned = re.sub(r"[^a-zA-Z\s/-]", " ", lowered).replace("grey", "gray")
    words = re.findall(r"[a-z]+", cleaned)
    for word in words:
        if word in BASIC_COLORS:
            return word, "basic_color_found"
    for color in BASIC_COLORS:
        if color in cleaned:
            return color, "basic_color_substring"
    return None, "no_basic_color_found"


def confusion_matrix(labels: list[str], preds: list[str | None], colors: list[str] = BASIC_COLORS) -> np.ndarray:
    mapping = {color: idx for idx, color in enumerate(colors)}
    matrix = np.zeros((len(colors), len(colors)), dtype=int)
    for label, pred in zip(labels, preds):
        label = normalize_color(label)
        pred = normalize_color(pred)
        if label in mapping and pred in mapping:
            matrix[mapping[label], mapping[pred]] += 1
    return matrix


def class_metrics(rows: list[dict[str, str]], pred_key: str = "normalized_pred_color") -> list[dict[str, float | str | int]]:
    out = []
    for color in BASIC_COLORS:
        true_rows = [row for row in rows if normalize_color(row.get("true_color") or row.get("basic_color")) == color]
        pred_rows = [row for row in rows if normalize_color(row.get(pred_key)) == color]
        tp = sum(normalize_color(row.get(pred_key)) == color for row in true_rows)
        support = len(true_rows)
        predicted = len(pred_rows)
        recall = tp / support if support else float("nan")
        precision = tp / predicted if predicted else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if precision + recall and np.isfinite(precision + recall) else float("nan")
        out.append({"color": color, "support": support, "predicted": predicted, "true_positive": tp, "accuracy_recall": recall, "precision": precision, "f1": f1})
    return out


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(denom, 1e-12)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return l2_normalize(a) @ l2_normalize(b).T


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
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    return pearson_corr(rankdata_average(a), rankdata_average(b))


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def permutation_pvalue(a: np.ndarray, b: np.ndarray, corr_fn=spearman_corr, rounds: int = 1000, seed: int = 42) -> tuple[float, float, int]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 3:
        return float("nan"), float("nan"), int(len(a))
    observed = corr_fn(a, b)
    if not np.isfinite(observed):
        return observed, float("nan"), int(len(a))
    rng = np.random.default_rng(seed)
    null = np.asarray([corr_fn(a, rng.permutation(b)) for _ in range(rounds)])
    p_value = (np.sum(np.abs(null) >= abs(observed)) + 1) / (rounds + 1)
    return float(observed), float(p_value), int(len(a))


def rdm(embeddings: np.ndarray, metric: str = "cosine") -> np.ndarray:
    if metric == "euclidean":
        x = np.asarray(embeddings, dtype=float)
        diff = x[:, None, :] - x[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))
    return 1.0 - cosine_matrix(embeddings, embeddings)


def centroids(embeddings: np.ndarray, labels: list[str], colors: list[str] = BASIC_COLORS) -> np.ndarray:
    label_indices = labels_to_indices(labels, colors)
    return np.vstack([np.asarray(embeddings)[label_indices == idx].mean(axis=0) for idx in range(len(colors))])


def linear_cka(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float) - np.asarray(a, dtype=float).mean(axis=0, keepdims=True)
    b = np.asarray(b, dtype=float) - np.asarray(b, dtype=float).mean(axis=0, keepdims=True)
    ga = a @ a.T
    gb = b @ b.T
    denom = math.sqrt(float(np.sum(ga * ga) * np.sum(gb * gb)))
    return float(np.sum(ga * gb) / denom) if denom else float("nan")


def procrustes_error(source: np.ndarray, target: np.ndarray) -> float:
    denom = np.linalg.norm(target)
    if not denom:
        return float("nan")
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    u_source, s_source, _ = np.linalg.svd(source, full_matrices=False)
    u_target, s_target, _ = np.linalg.svd(target, full_matrices=False)
    small = (s_source[:, None] * (u_source.T @ u_target)) * s_target[None, :]
    singular_values = np.linalg.svd(small, compute_uv=False)
    residual_sq = np.linalg.norm(source) ** 2 + np.linalg.norm(target) ** 2 - 2.0 * float(np.sum(singular_values))
    return float(math.sqrt(max(residual_sq, 0.0)) / denom)


def linear_probe(embeddings: np.ndarray, labels: list[str], colors: list[str] = BASIC_COLORS, test_size: float = 0.3, alpha: float = 1.0, seed: int = 42) -> dict:
    x = l2_normalize(embeddings)
    y = labels_to_indices(labels, colors)
    rng = random.Random(seed)
    train = []
    test = []
    for label in sorted(set(y.tolist())):
        indices = np.where(y == label)[0].tolist()
        rng.shuffle(indices)
        n_test = max(1, int(round(len(indices) * test_size)))
        test.extend(indices[:n_test])
        train.extend(indices[n_test:])
    train_idx = np.array(sorted(train), dtype=int)
    test_idx = np.array(sorted(test), dtype=int)
    x_train = np.hstack([x[train_idx], np.ones((len(train_idx), 1))])
    x_test = np.hstack([x[test_idx], np.ones((len(test_idx), 1))])
    y_train = np.zeros((len(train_idx), len(colors)), dtype=float)
    y_train[np.arange(len(train_idx)), y[train_idx]] = 1.0
    kernel = x_train @ x_train.T
    weights = np.linalg.solve(kernel + alpha * np.eye(len(train_idx)), y_train)
    scores = (x_test @ x_train.T) @ weights
    pred = np.argmax(scores, axis=1)
    class_rows = []
    for idx, color in enumerate(colors):
        mask = y[test_idx] == idx
        class_rows.append({"color": color, "support": int(np.sum(mask)), "linear_probe_accuracy": float(np.mean(pred[mask] == idx)) if np.any(mask) else float("nan")})
    return {
        "accuracy": float(np.mean(pred == y[test_idx])),
        "train_n": int(len(train_idx)),
        "test_n": int(len(test_idx)),
        "class_rows": class_rows,
        "scores": scores,
        "test_indices": test_idx,
        "pred_indices": pred,
    }


def srgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=float) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrix = np.array(
        [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
    )
    return linear @ matrix.T


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    white = np.array([0.95047, 1.00000, 1.08883])
    t = np.asarray(xyz, dtype=float) / white
    delta = 6 / 29
    f = np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4 / 29)
    l = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.stack([l, a, b], axis=1)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return xyz_to_lab(srgb_to_xyz(rgb))


def ciede2000_pair(lab1: np.ndarray, lab2: np.ndarray) -> float:
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2
    c1 = math.sqrt(a1 * a1 + b1 * b1)
    c2 = math.sqrt(a2 * a2 + b2 * b2)
    c_bar = (c1 + c2) / 2
    g = 0.5 * (1 - math.sqrt(c_bar**7 / (c_bar**7 + 25**7))) if c_bar else 0.0
    ap1 = (1 + g) * a1
    ap2 = (1 + g) * a2
    cp1 = math.sqrt(ap1 * ap1 + b1 * b1)
    cp2 = math.sqrt(ap2 * ap2 + b2 * b2)
    hp1 = math.degrees(math.atan2(b1, ap1)) % 360 if cp1 else 0.0
    hp2 = math.degrees(math.atan2(b2, ap2)) % 360 if cp2 else 0.0
    dl = l2 - l1
    dc = cp2 - cp1
    if cp1 * cp2 == 0:
        dh = 0.0
    else:
        dh_angle = hp2 - hp1
        if dh_angle > 180:
            dh_angle -= 360
        elif dh_angle < -180:
            dh_angle += 360
        dh = 2 * math.sqrt(cp1 * cp2) * math.sin(math.radians(dh_angle / 2))
    lp = (l1 + l2) / 2
    cp = (cp1 + cp2) / 2
    if cp1 * cp2 == 0:
        hp = hp1 + hp2
    elif abs(hp1 - hp2) <= 180:
        hp = (hp1 + hp2) / 2
    elif hp1 + hp2 < 360:
        hp = (hp1 + hp2 + 360) / 2
    else:
        hp = (hp1 + hp2 - 360) / 2
    t = 1 - 0.17 * math.cos(math.radians(hp - 30)) + 0.24 * math.cos(math.radians(2 * hp)) + 0.32 * math.cos(math.radians(3 * hp + 6)) - 0.20 * math.cos(math.radians(4 * hp - 63))
    delta_theta = 30 * math.exp(-((hp - 275) / 25) ** 2)
    rc = 2 * math.sqrt(cp**7 / (cp**7 + 25**7)) if cp else 0.0
    sl = 1 + (0.015 * (lp - 50) ** 2) / math.sqrt(20 + (lp - 50) ** 2)
    sc = 1 + 0.045 * cp
    sh = 1 + 0.015 * cp * t
    rt = -math.sin(math.radians(2 * delta_theta)) * rc
    return math.sqrt((dl / sl) ** 2 + (dc / sc) ** 2 + (dh / sh) ** 2 + rt * (dc / sc) * (dh / sh))


def ciede2000_matrix(lab: np.ndarray) -> np.ndarray:
    n = len(lab)
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = ciede2000_pair(lab[i], lab[j])
            out[i, j] = d
            out[j, i] = d
    return out


def save_matrix_csv(path: Path, labels: list[str], matrix: np.ndarray, label_name: str = "label") -> None:
    rows = []
    for label, values in zip(labels, matrix):
        row = {label_name: label}
        row.update({str(col): float(value) for col, value in zip(labels, values)})
        rows.append(row)
    write_csv(path, [label_name, *[str(label) for label in labels]], rows)


def try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def save_confusion_plot(path: Path, matrix: np.ndarray, colors: list[str] = BASIC_COLORS, title: str = "Confusion Matrix") -> None:
    plt = try_import_matplotlib()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(colors)))
    ax.set_yticks(range(len(colors)))
    ax.set_xticklabels(colors, rotation=45, ha="right")
    ax.set_yticklabels(colors)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_line_plot(path: Path, rows: list[dict], x_key: str, y_key: str, title: str, ylabel: str) -> None:
    plt = try_import_matplotlib()
    if plt is None or not rows:
        return
    x = [float(row[x_key]) for row in rows]
    y = [float(row[y_key]) for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    ax.plot(x, y, marker="o", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_error_cases_plot(path: Path, rows: list[dict[str, str]], dataset_root: Path, max_cols: int = 8) -> None:
    errors = [row for row in rows if str(row.get("correct")) in {"0", "False", "false"}]
    plt = try_import_matplotlib()
    if plt is None or not errors:
        return
    n = len(errors)
    cols = min(max_cols, max(1, n))
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2.0, rows_n * 2.2), dpi=140)
    axes = np.asarray(axes).reshape(-1)
    for ax, row in zip(axes, errors):
        p = Path(row.get("image_path") or row.get("absolute_image_path"))
        if not p.is_absolute():
            p = dataset_root / p
        ax.imshow(Image.open(p).convert("RGB"))
        ax.set_title(f"T:{row.get('true_color')}\nP:{row.get('normalized_pred_color')}", fontsize=7)
        ax.axis("off")
    for ax in axes[len(errors) :]:
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def linear_regression_summary(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.std(x) == 0:
        return {"n": int(len(x)), "r2": float("nan"), "slope": float("nan"), "intercept": float("nan")}
    design = np.column_stack([x, np.ones(len(x))])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"n": int(len(x)), "r2": 1 - ss_res / ss_tot if ss_tot else float("nan"), "slope": float(slope), "intercept": float(intercept)}


def logistic_regression_1d(x: np.ndarray, y: np.ndarray, steps: int = 4000, lr: float = 0.05) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 5 or np.std(x) == 0:
        return {"n": int(len(x)), "coef": float("nan"), "intercept": float("nan"), "accuracy": float("nan")}
    xz = (x - x.mean()) / x.std()
    coef = 0.0
    intercept = 0.0
    for _ in range(steps):
        logits = coef * xz + intercept
        probs = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        coef -= lr * float(np.mean((probs - y) * xz))
        intercept -= lr * float(np.mean(probs - y))
    probs = 1 / (1 + np.exp(-np.clip(coef * xz + intercept, -30, 30)))
    return {"n": int(len(x)), "coef": float(coef), "intercept": float(intercept), "accuracy": float(np.mean((probs >= 0.5) == y))}

