#!/usr/bin/env python3
"""Analyze LLaVA projector-space color alignment on pure color patches.

This experiment tests whether LLaVA's projected image tokens, after the
multimodal projector maps CLIP vision features into the LLM hidden dimension,
are geometrically aligned with color-word embeddings from the LLM input space.

It is the generative-VLM analogue of the CLIP embedding alignment experiment:

- image representation: mean pooled projector output image tokens
- text representation: LLM input embeddings for color prompts
- cosine-prototype classification is kept as a diagnostic only
- alignment metrics: pair margin, RSA, Kendall, Pearson, CKA, Procrustes
- layer-wise alignment vs final generative color correctness with permutation p-values
- optional visualization PNGs

Default model:
  llava-hf/llava-1.5-7b-hf
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import warnings
from collections import Counter
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
    image_projected_embeddings: np.ndarray
    text_embeddings: np.ndarray
    colors: list[str]
    prompts: list[str]
    image_pooling: str
    text_representation: str
    image_layer_embeddings: np.ndarray | None = None
    text_layer_embeddings: np.ndarray | None = None
    layer_indices: list[int] | None = None
    query_layer_embeddings: np.ndarray | None = None
    query_no_color_list_layer_embeddings: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLaVA projector-space color alignment analysis.")
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
        help="Root for resolving relative image_path values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/llava_projector_alignment"),
        help="Directory for output files.",
    )
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf", help="HuggingFace LLaVA model id.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Model dtype.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Image batch size. Default: 1")
    parser.add_argument(
        "--prompt",
        dest="prompts",
        action="append",
        help="Color prompt template containing {}. Can be repeated.",
    )
    parser.add_argument(
        "--image-pooling",
        choices=("mean", "first"),
        default="mean",
        help="How to pool projected image tokens. Default: mean",
    )
    parser.add_argument(
        "--vision-feature-layer",
        type=int,
        help="Override LLaVA vision_feature_layer. Defaults to model config or -2.",
    )
    parser.add_argument(
        "--vision-feature-select-strategy",
        default=None,
        help="Override model vision_feature_select_strategy. Common values: default, full.",
    )
    parser.add_argument(
        "--embeddings-npz",
        type=Path,
        help="Analyze precomputed projector/text embeddings instead of loading LLaVA.",
    )
    parser.add_argument("--save-embeddings-name", default="llava_projector_embeddings.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutation-rounds", type=int, default=1000)
    parser.add_argument(
        "--run-layerwise",
        action="store_true",
        help="Also analyze image-token/text hidden-state alignment at every LLM layer.",
    )
    parser.add_argument(
        "--layer-indices",
        help="Comma-separated layer indices to analyze, e.g. 0,1,8,16,32. Default: all layers.",
    )
    parser.add_argument(
        "--layerwise-image-prompt",
        default="USER: <image>\nASSISTANT:",
        help="Prompt used when inserting projected image tokens into the LLM for layer-wise image hidden states.",
    )
    parser.add_argument(
        "--run-query-analysis",
        action="store_true",
        help="Also extract query-token hidden states and analyze query color decodability/alignment.",
    )
    parser.add_argument(
        "--query-prompt",
        default="USER: <image>\nWhat color is this patch? Answer with exactly one basic color name from this list: black, blue, brown, gray, green, orange, pink, purple, red, white, yellow.\nASSISTANT:",
        help="Prompt used for query-token layer-wise analysis. Must contain exactly one <image> token.",
    )
    parser.add_argument(
        "--run-query-no-color-list-analysis",
        action="store_true",
        help="Also run query-token analysis with a prompt that does not list the 11 color options.",
    )
    parser.add_argument(
        "--query-no-color-list-prompt",
        default="USER: <image>\nWhat color is this patch?\nASSISTANT:",
        help="No-color-list query prompt. Must contain exactly one <image> token.",
    )
    parser.add_argument(
        "--generative-results-dir",
        type=Path,
        help="Optional output directory from run_generative_color_naming_experiment.py for behavior-first reporting.",
    )
    parser.add_argument(
        "--plot-projector-diagnostics",
        action="store_true",
        help="Keep old projector-space diagnostic plots. Default is off because projector output is treated as layer 0/baseline.",
    )
    parser.add_argument(
        "--linear-probe-test-size",
        type=float,
        default=0.3,
        help="Held-out fraction for the final selected layer image-only linear probe.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength for the final selected layer linear probe.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def parse_layer_indices(raw: str | None, num_layers: int) -> list[int]:
    if not raw:
        return list(range(num_layers))
    indices = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0:
            idx = num_layers + idx
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index {part} resolved to {idx}, outside 0..{num_layers - 1}")
        indices.append(idx)
    return sorted(dict.fromkeys(indices))


def prompt_templates(prompts: list[str] | None) -> list[str]:
    templates = prompts if prompts else list(DEFAULT_PROMPTS)
    for template in templates:
        if "{}" not in template:
            raise ValueError(f"Prompt template must contain {{}}: {template!r}")
    return templates


def load_dataset(path: Path, root: Path) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth CSV not found: {path}")
    rows, image_paths, labels = [], [], []
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
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = root / image_path
            rows.append(row)
            image_paths.append(str(image_path))
            labels.append(label)
    if not rows:
        raise ValueError("No samples loaded.")
    return Dataset(image_paths=image_paths, labels=labels, rows=rows)


def choose_device(device_name: str):
    import torch

    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_dtype(dtype_name: str):
    import torch

    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return "auto"


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return l2_normalize(a) @ l2_normalize(b).T


def pool_tokens(tokens: np.ndarray, method: str) -> np.ndarray:
    if method == "first":
        return tokens[:, 0, :]
    return tokens.mean(axis=1)


class LlavaProjectorExtractor:
    def __init__(self, model_id: str, device_name: str, dtype_name: str, args: argparse.Namespace) -> None:
        try:
            import torch
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except Exception as exc:
            raise RuntimeError(
                "This script requires torch and transformers with LLaVA support. "
                "Install with: pip install -r requirements-generative.txt"
            ) from exc

        self.torch = torch
        self.device = choose_device(device_name)
        dtype = choose_dtype(dtype_name)
        kwargs = {"torch_dtype": dtype} if dtype != "auto" else {"torch_dtype": "auto"}
        if self.device.type == "cuda":
            kwargs["device_map"] = "auto"
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, trust_remote_code=True, **kwargs)
        if self.device.type != "cuda":
            self.model.to(self.device)
        self.model.eval()
        self.vision_feature_layer = (
            args.vision_feature_layer
            if args.vision_feature_layer is not None
            else getattr(self.model.config, "vision_feature_layer", -2)
        )
        self.vision_feature_select_strategy = (
            args.vision_feature_select_strategy
            if args.vision_feature_select_strategy is not None
            else getattr(self.model.config, "vision_feature_select_strategy", "default")
        )
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
        candidate_paths = (
            "vision_tower",
            "model.vision_tower",
            "model.vision_model",
            "vision_model",
        )
        for path in candidate_paths:
            tower = self._resolve_path(self.model, path)
            if tower is not None:
                return tower
        available = ", ".join(name for name in dir(self.model) if "vision" in name.lower())
        raise AttributeError(
            "Could not find LLaVA vision tower on this model. "
            f"Tried {candidate_paths}. Vision-like attributes: {available}"
        )

    def _find_projector(self):
        candidate_paths = (
            "multi_modal_projector",
            "model.multi_modal_projector",
            "mm_projector",
            "model.mm_projector",
            "projector",
            "model.projector",
        )
        for path in candidate_paths:
            projector = self._resolve_path(self.model, path)
            if projector is not None:
                return projector
        available = ", ".join(
            name for name in dir(self.model) if "project" in name.lower() or "mm" in name.lower()
        )
        raise AttributeError(
            "Could not find LLaVA multimodal projector on this model. "
            f"Tried {candidate_paths}. Projector-like attributes: {available}"
        )

    def _pixel_values(self, image_paths: list[str]):
        images = [Image.open(path).convert("RGB") for path in image_paths]
        # LLaVA processors usually require a text placeholder to build pixel_values reliably.
        text = ["USER: <image>\nASSISTANT:" for _ in images]
        inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True)
        return inputs["pixel_values"].to(self.device)

    def get_projected_image_features(self, image_paths: list[str]) -> np.ndarray:
        pixel_values = self._pixel_values(image_paths)
        with self.torch.no_grad():
            # Use the explicit vision_tower -> projector path. Some transformers
            # versions return BaseModelOutputWithPooling from get_image_features(),
            # which is a structured object rather than the projected tensor.
            vision_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
            if not hasattr(vision_outputs, "hidden_states") or vision_outputs.hidden_states is None:
                raise RuntimeError("vision_tower did not return hidden_states; cannot extract projector features")

            selected = vision_outputs.hidden_states[self.vision_feature_layer]
            if self.vision_feature_select_strategy == "default":
                selected = selected[:, 1:]
            elif self.vision_feature_select_strategy == "full":
                pass
            else:
                # Keep behavior close to HF LLaVA defaults for unknown strategy names.
                selected = selected[:, 1:]

            features = self.projector(selected)

        if isinstance(features, (list, tuple)):
            features = features[0]
        if hasattr(features, "last_hidden_state"):
            features = features.last_hidden_state
        return features.detach().float().cpu().numpy()

    def get_text_embedding_matrix(self):
        if hasattr(self.model, "get_input_embeddings"):
            emb = self.model.get_input_embeddings()
            if emb is not None:
                return emb
        return self.model.language_model.get_input_embeddings()

    def encode_color_texts(self, colors: list[str], prompts: list[str]) -> np.ndarray:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        embedding_layer = self.get_text_embedding_matrix()
        prototypes = []
        with self.torch.no_grad():
            for color in colors:
                prompt_vectors = []
                for template in prompts:
                    text = template.format(color)
                    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
                    input_ids = encoded["input_ids"].to(self.device)
                    token_embeddings = embedding_layer(input_ids)
                    vector = token_embeddings.mean(dim=1)
                    vector = vector / vector.norm(dim=-1, keepdim=True)
                    prompt_vectors.append(vector)
                prototype = self.torch.cat(prompt_vectors, dim=0).mean(dim=0, keepdim=True)
                prototype = prototype / prototype.norm(dim=-1, keepdim=True)
                prototypes.append(prototype.detach().float().cpu().numpy()[0])
        return np.vstack(prototypes).astype(np.float32)

    def _find_language_model(self):
        candidate_paths = ("language_model", "model.language_model", "model")
        for path in candidate_paths:
            module = self._resolve_path(self.model, path)
            if module is not None and callable(getattr(module, "forward", None)):
                return module
        raise AttributeError("Could not find underlying LLM/language_model for layer-wise hidden states")

    def _image_token_id(self) -> int:
        token_id = getattr(self.model.config, "image_token_index", None)
        if token_id is not None:
            return int(token_id)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        token_id = tokenizer.convert_tokens_to_ids("<image>")
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError("Could not resolve <image> token id")
        return int(token_id)

    def _color_token_mask(self, input_ids, color: str):
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        color_ids = tokenizer(color, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(input_ids.device)
        sequence = input_ids[0]
        for start in range(0, max(0, sequence.shape[0] - color_ids.shape[0] + 1)):
            if self.torch.equal(sequence[start : start + color_ids.shape[0]], color_ids):
                mask = self.torch.zeros_like(sequence, dtype=self.torch.bool)
                mask[start : start + color_ids.shape[0]] = True
                return mask
        special_ids = set(x for x in [getattr(tokenizer, "bos_token_id", None), getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "pad_token_id", None)] if x is not None)
        mask = self.torch.ones_like(sequence, dtype=self.torch.bool)
        for special_id in special_ids:
            mask &= sequence != special_id
        return mask

    def _special_token_ids(self) -> set[int]:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        values = [
            getattr(tokenizer, "bos_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
        ]
        return {int(x) for x in values if x is not None}

    def _expanded_language_inputs(self, image_path: str, prompt: str):
        embedding_layer = self.get_text_embedding_matrix()
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        image_token_id = self._image_token_id()
        image_features = self.get_projected_image_features([image_path])
        image_features_t = self.torch.from_numpy(image_features).to(self.device, dtype=embedding_layer.weight.dtype)
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        positions = (input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten()
        if len(positions) != 1:
            raise ValueError(f"Prompt must contain exactly one <image> token; found {len(positions)} in {prompt!r}")
        pos = int(positions[0].item())
        token_embeds = embedding_layer(input_ids)
        inputs_embeds = self.torch.cat(
            [token_embeds[:, :pos, :], image_features_t, token_embeds[:, pos + 1 :, :]], dim=1
        )
        attention_mask = self.torch.ones(inputs_embeds.shape[:2], dtype=self.torch.long, device=self.device)

        query_mask = self.torch.zeros(inputs_embeds.shape[1], dtype=self.torch.bool, device=self.device)
        special_ids = self._special_token_ids()
        sequence = input_ids[0]
        n_image = image_features_t.shape[1]
        for original_idx, token_id in enumerate(sequence.tolist()):
            if original_idx == pos or int(token_id) in special_ids:
                continue
            expanded_idx = original_idx if original_idx < pos else original_idx + n_image - 1
            query_mask[expanded_idx] = True

        return inputs_embeds, attention_mask, query_mask, (pos, pos + n_image)

    def get_layerwise_image_embeddings(self, image_paths: list[str], pooling: str, image_prompt: str, layer_indices_raw: str | None):
        language_model = self._find_language_model()
        embedding_layer = self.get_text_embedding_matrix()
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        image_token_id = self._image_token_id()
        layer_vectors: list[list[np.ndarray]] | None = None
        selected_layer_indices: list[int] | None = None

        with self.torch.no_grad():
            for image_path in image_paths:
                image_features = self.get_projected_image_features([image_path])
                image_features_t = self.torch.from_numpy(image_features).to(self.device, dtype=embedding_layer.weight.dtype)
                encoded = tokenizer(image_prompt, return_tensors="pt")
                input_ids = encoded["input_ids"].to(self.device)
                positions = (input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten()
                if len(positions) != 1:
                    raise ValueError(
                        f"Layer-wise image prompt must contain exactly one <image> token; found {len(positions)} in {image_prompt!r}"
                    )
                pos = int(positions[0].item())
                token_embeds = embedding_layer(input_ids)
                inputs_embeds = self.torch.cat(
                    [token_embeds[:, :pos, :], image_features_t, token_embeds[:, pos + 1 :, :]], dim=1
                )
                attention_mask = self.torch.ones(inputs_embeds.shape[:2], dtype=self.torch.long, device=self.device)
                outputs = language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                hidden_states = outputs.hidden_states
                if selected_layer_indices is None:
                    selected_layer_indices = parse_layer_indices(layer_indices_raw, len(hidden_states))
                    layer_vectors = [[] for _ in selected_layer_indices]
                image_start = pos
                image_end = pos + image_features_t.shape[1]
                for out_idx, layer_idx in enumerate(selected_layer_indices):
                    image_tokens = hidden_states[layer_idx][:, image_start:image_end, :]
                    if pooling == "first":
                        pooled = image_tokens[:, 0, :]
                    else:
                        pooled = image_tokens.mean(dim=1)
                    layer_vectors[out_idx].append(pooled.detach().float().cpu().numpy()[0])

        assert layer_vectors is not None and selected_layer_indices is not None
        return np.stack([np.vstack(vectors) for vectors in layer_vectors]).astype(np.float32), selected_layer_indices

    def get_layerwise_query_embeddings(self, image_paths: list[str], query_prompt: str, layer_indices: list[int]):
        language_model = self._find_language_model()
        layer_vectors: list[list[np.ndarray]] = [[] for _ in layer_indices]
        with self.torch.no_grad():
            for image_path in image_paths:
                inputs_embeds, attention_mask, query_mask, _ = self._expanded_language_inputs(image_path, query_prompt)
                outputs = language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                for out_idx, layer_idx in enumerate(layer_indices):
                    hidden = outputs.hidden_states[layer_idx][0]
                    selected = hidden[query_mask]
                    if selected.shape[0] == 0:
                        selected = hidden
                    pooled = selected.mean(dim=0, keepdim=True)
                    layer_vectors[out_idx].append(pooled.detach().float().cpu().numpy()[0])
        return np.stack([np.vstack(vectors) for vectors in layer_vectors]).astype(np.float32)

    def get_layerwise_text_embeddings(self, colors: list[str], prompts: list[str], layer_indices: list[int]):
        language_model = self._find_language_model()
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        layer_color_vectors = [[] for _ in layer_indices]
        with self.torch.no_grad():
            for color in colors:
                prompt_vectors_by_layer = [[] for _ in layer_indices]
                for template in prompts:
                    text = template.format(color)
                    encoded = tokenizer(text, return_tensors="pt")
                    input_ids = encoded["input_ids"].to(self.device)
                    attention_mask = encoded.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    color_mask = self._color_token_mask(input_ids, color)
                    outputs = language_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                    for out_idx, layer_idx in enumerate(layer_indices):
                        hidden = outputs.hidden_states[layer_idx][0]
                        selected = hidden[color_mask]
                        if selected.shape[0] == 0:
                            selected = hidden
                        vector = selected.mean(dim=0, keepdim=True)
                        vector = vector / vector.norm(dim=-1, keepdim=True)
                        prompt_vectors_by_layer[out_idx].append(vector)
                for out_idx in range(len(layer_indices)):
                    proto = self.torch.cat(prompt_vectors_by_layer[out_idx], dim=0).mean(dim=0, keepdim=True)
                    proto = proto / proto.norm(dim=-1, keepdim=True)
                    layer_color_vectors[out_idx].append(proto.detach().float().cpu().numpy()[0])
        return np.stack([np.vstack(vectors) for vectors in layer_color_vectors]).astype(np.float32)


def extract_embeddings(dataset: Dataset, args: argparse.Namespace, prompts: list[str]) -> Embeddings:
    extractor = LlavaProjectorExtractor(args.model_id, args.device, args.dtype, args)
    image_batches = []
    for start in range(0, len(dataset.image_paths), args.batch_size):
        batch_paths = dataset.image_paths[start : start + args.batch_size]
        projected_tokens = extractor.get_projected_image_features(batch_paths)
        pooled = pool_tokens(projected_tokens, args.image_pooling)
        image_batches.append(pooled)
    image_embeddings = np.vstack(image_batches).astype(np.float32)
    text_embeddings = extractor.encode_color_texts(list(BASIC_COLORS), prompts)

    image_layer_embeddings = None
    text_layer_embeddings = None
    query_layer_embeddings = None
    query_no_color_list_layer_embeddings = None
    layer_indices = None
    if (args.run_query_analysis or args.run_query_no_color_list_analysis) and not args.run_layerwise:
        raise ValueError("Query analyses require --run-layerwise so query/text states use the same selected layers.")
    if args.run_layerwise:
        image_layer_embeddings, layer_indices = extractor.get_layerwise_image_embeddings(
            dataset.image_paths, args.image_pooling, args.layerwise_image_prompt, args.layer_indices
        )
        text_layer_embeddings = extractor.get_layerwise_text_embeddings(list(BASIC_COLORS), prompts, layer_indices)
        if args.run_query_analysis:
            query_layer_embeddings = extractor.get_layerwise_query_embeddings(
                dataset.image_paths, args.query_prompt, layer_indices
            )
        if args.run_query_no_color_list_analysis:
            query_no_color_list_layer_embeddings = extractor.get_layerwise_query_embeddings(
                dataset.image_paths, args.query_no_color_list_prompt, layer_indices
            )

    return Embeddings(
        image_projected_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        colors=list(BASIC_COLORS),
        prompts=prompts,
        image_pooling=args.image_pooling,
        text_representation="llm_input_token_embedding_prompt_mean",
        image_layer_embeddings=image_layer_embeddings,
        text_layer_embeddings=text_layer_embeddings,
        layer_indices=layer_indices,
        query_layer_embeddings=query_layer_embeddings,
        query_no_color_list_layer_embeddings=query_no_color_list_layer_embeddings,
    )


def save_embeddings(path: Path, embeddings: Embeddings, dataset: Dataset) -> None:
    np.savez_compressed(
        path,
        image_projected_embeddings=embeddings.image_projected_embeddings,
        text_embeddings=embeddings.text_embeddings,
        colors=np.array(embeddings.colors),
        labels=np.array(dataset.labels),
        image_paths=np.array(dataset.image_paths),
        prompts=np.array(embeddings.prompts),
        image_pooling=np.array([embeddings.image_pooling]),
        text_representation=np.array([embeddings.text_representation]),
        image_layer_embeddings=embeddings.image_layer_embeddings if embeddings.image_layer_embeddings is not None else np.array([]),
        text_layer_embeddings=embeddings.text_layer_embeddings if embeddings.text_layer_embeddings is not None else np.array([]),
        layer_indices=np.array(embeddings.layer_indices or [], dtype=int),
        query_layer_embeddings=embeddings.query_layer_embeddings if embeddings.query_layer_embeddings is not None else np.array([]),
        query_no_color_list_layer_embeddings=embeddings.query_no_color_list_layer_embeddings if embeddings.query_no_color_list_layer_embeddings is not None else np.array([]),
    )


def load_embeddings(path: Path) -> tuple[Embeddings, Dataset]:
    data = np.load(path, allow_pickle=False)
    colors = [str(x) for x in data["colors"].tolist()]
    labels = [str(x) for x in data["labels"].tolist()]
    image_paths = [str(x) for x in data["image_paths"].tolist()]
    prompts = [str(x) for x in data["prompts"].tolist()]
    image_pooling = str(data["image_pooling"][0]) if "image_pooling" in data else "mean"
    text_representation = str(data["text_representation"][0]) if "text_representation" in data else "unknown"
    image_layer_embeddings = None
    text_layer_embeddings = None
    query_layer_embeddings = None
    query_no_color_list_layer_embeddings = None
    layer_indices = None
    if "image_layer_embeddings" in data and data["image_layer_embeddings"].size:
        image_layer_embeddings = np.asarray(data["image_layer_embeddings"], dtype=np.float32)
    if "text_layer_embeddings" in data and data["text_layer_embeddings"].size:
        text_layer_embeddings = np.asarray(data["text_layer_embeddings"], dtype=np.float32)
    if "layer_indices" in data and data["layer_indices"].size:
        layer_indices = [int(x) for x in data["layer_indices"].tolist()]
    if "query_layer_embeddings" in data and data["query_layer_embeddings"].size:
        query_layer_embeddings = np.asarray(data["query_layer_embeddings"], dtype=np.float32)
    if "query_no_color_list_layer_embeddings" in data and data["query_no_color_list_layer_embeddings"].size:
        query_no_color_list_layer_embeddings = np.asarray(data["query_no_color_list_layer_embeddings"], dtype=np.float32)
    embeddings = Embeddings(
        image_projected_embeddings=np.asarray(data["image_projected_embeddings"], dtype=np.float32),
        text_embeddings=np.asarray(data["text_embeddings"], dtype=np.float32),
        colors=colors,
        prompts=prompts,
        image_pooling=image_pooling,
        text_representation=text_representation,
        image_layer_embeddings=image_layer_embeddings,
        text_layer_embeddings=text_layer_embeddings,
        layer_indices=layer_indices,
        query_layer_embeddings=query_layer_embeddings,
        query_no_color_list_layer_embeddings=query_no_color_list_layer_embeddings,
    )
    dataset = Dataset(
        image_paths=image_paths,
        labels=labels,
        rows=[{"image_path": p, "basic_color": y} for p, y in zip(image_paths, labels)],
    )
    return embeddings, dataset


def labels_to_indices(labels: list[str], colors: list[str]) -> np.ndarray:
    mapping = {color: idx for idx, color in enumerate(colors)}
    return np.array([mapping[label] for label in labels], dtype=int)


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


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = np.asarray(a)[valid], np.asarray(b)[valid]
    concordant = 0
    discordant = 0
    for i in range(len(a) - 1):
        sign = np.sign((a[i] - a[i + 1 :]) * (b[i] - b[i + 1 :]))
        concordant += int(np.sum(sign > 0))
        discordant += int(np.sum(sign < 0))
    denom = concordant + discordant
    return float((concordant - discordant) / denom) if denom else float("nan")


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def rdm(embeddings: np.ndarray) -> np.ndarray:
    return 1.0 - cosine_matrix(embeddings, embeddings)


def permutation_pvalue(a: np.ndarray, b: np.ndarray, corr_fn, rounds: int, seed: int) -> tuple[float, float]:
    observed = corr_fn(a, b)
    if not np.isfinite(observed):
        return observed, float("nan")
    rng = np.random.default_rng(seed)
    null = np.asarray([corr_fn(a, rng.permutation(b)) for _ in range(rounds)])
    p_value = (np.sum(np.abs(null) >= abs(observed)) + 1) / (rounds + 1)
    return float(observed), float(p_value)


def rdm_permutation_pvalue(a_rdm: np.ndarray, b_rdm: np.ndarray, corr_fn, rounds: int, seed: int) -> tuple[float, float]:
    observed = corr_fn(upper_triangle_values(a_rdm), upper_triangle_values(b_rdm))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(rounds):
        p = rng.permutation(a_rdm.shape[0])
        permuted = a_rdm[p][:, p]
        null.append(corr_fn(upper_triangle_values(permuted), upper_triangle_values(b_rdm)))
    null = np.asarray(null)
    p_value = (np.sum(np.abs(null) >= abs(observed)) + 1) / (rounds + 1)
    return float(observed), float(p_value)


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
    # The non-zero singular values of source.T @ target can be computed from
    # thin SVD factors of the much smaller n-by-d matrices. This keeps the
    # Procrustes residual tractable for LLaVA hidden sizes.
    u_source, s_source, _ = np.linalg.svd(source, full_matrices=False)
    u_target, s_target, _ = np.linalg.svd(target, full_matrices=False)
    small = (s_source[:, None] * (u_source.T @ u_target)) * s_target[None, :]
    singular_values = np.linalg.svd(small, compute_uv=False)
    residual_sq = np.linalg.norm(source) ** 2 + np.linalg.norm(target) ** 2 - 2.0 * float(np.sum(singular_values))
    return float(math.sqrt(max(residual_sq, 0.0)) / denom)


def centroids(embeddings: np.ndarray, label_indices: np.ndarray, colors: list[str]) -> np.ndarray:
    result = []
    for idx, color in enumerate(colors):
        mask = label_indices == idx
        if not np.any(mask):
            raise ValueError(f"No samples for {color}")
        result.append(embeddings[mask].mean(axis=0))
    return l2_normalize(np.vstack(result))


def zero_shot(image_embeddings: np.ndarray, text_embeddings: np.ndarray, labels: list[str], colors: list[str]):
    label_indices = labels_to_indices(labels, colors)
    similarities = cosine_matrix(image_embeddings, text_embeddings)
    pred = np.argmax(similarities, axis=1)
    rows = []
    margins = []
    for i, true_idx in enumerate(label_indices):
        wrong = np.delete(similarities[i], true_idx)
        margin = float(similarities[i, true_idx] - wrong.mean())
        margins.append(margin)
        rows.append(
            {
                "sample_index": i,
                "true_color": colors[true_idx],
                "pred_color": colors[pred[i]],
                "correct": int(pred[i] == true_idx),
                "pair_alignment_margin": margin,
                "correct_similarity": float(similarities[i, true_idx]),
                "pred_similarity": float(similarities[i, pred[i]]),
            }
        )
    return rows, similarities, pred, np.asarray(margins), float(np.mean(pred == label_indices))


def confusion(labels: list[str], pred: np.ndarray, colors: list[str]) -> np.ndarray:
    label_indices = labels_to_indices(labels, colors)
    matrix = np.zeros((len(colors), len(colors)), dtype=int)
    for true_idx, pred_idx in zip(label_indices, pred):
        matrix[true_idx, pred_idx] += 1
    return matrix


def class_rows(labels: list[str], pred: np.ndarray, margins: np.ndarray, colors: list[str], visual_rdm: np.ndarray, text_rdm: np.ndarray) -> list[dict]:
    label_indices = labels_to_indices(labels, colors)
    output = []
    for idx, color in enumerate(colors):
        mask = label_indices == idx
        local_mask = np.ones(len(colors), dtype=bool)
        local_mask[idx] = False
        support = int(np.sum(mask))
        output.append(
            {
                "color": color,
                "support": support,
                "accuracy": float(np.mean(pred[mask] == idx)) if support else float("nan"),
                "mean_pair_alignment_margin": float(np.mean(margins[mask])) if support else float("nan"),
                "classwise_rsa_spearman": spearman_corr(visual_rdm[idx, local_mask], text_rdm[idx, local_mask]),
                "classwise_rsa_pearson": pearson_corr(visual_rdm[idx, local_mask], text_rdm[idx, local_mask]),
            }
        )
    return output


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


def linear_probe_image_only(
    image_embeddings: np.ndarray,
    label_indices: np.ndarray,
    colors: list[str],
    test_size: float,
    alpha: float,
    seed: int,
) -> dict:
    """Image-only dual ridge probe for high-dimensional LLaVA hidden states."""
    x = l2_normalize(image_embeddings)
    y = label_indices
    train_idx, test_idx = stratified_split(y, test_size, seed)
    x_train = np.hstack([x[train_idx], np.ones((len(train_idx), 1))])
    x_test = np.hstack([x[test_idx], np.ones((len(test_idx), 1))])
    y_train = np.zeros((len(train_idx), len(colors)), dtype=float)
    y_train[np.arange(len(train_idx)), y[train_idx]] = 1.0

    kernel = x_train @ x_train.T
    reg = alpha * np.eye(len(train_idx))
    dual_weights = np.linalg.solve(kernel + reg, y_train)
    scores = (x_test @ x_train.T) @ dual_weights
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
    prediction_rows = []
    for sample_index, pred_index in zip(test_idx.tolist(), pred.tolist()):
        true_index = int(y[sample_index])
        prediction_rows.append(
            {
                "sample_index": int(sample_index),
                "true_color": colors[true_index],
                "pred_color": colors[pred_index],
                "correct": int(pred_index == true_index),
            }
        )
    return {
        "linear_probe_accuracy": accuracy,
        "linear_probe_train_n": int(len(train_idx)),
        "linear_probe_test_n": int(len(test_idx)),
        "linear_probe_class_rows": rows,
        "linear_probe_prediction_rows": prediction_rows,
    }


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(path: Path, row_labels: list[str], col_labels: list[str], matrix: np.ndarray) -> None:
    rows = []
    for label, values in zip(row_labels, matrix):
        row = {"label": label}
        row.update({col: float(v) for col, v in zip(col_labels, values)})
        rows.append(row)
    write_csv(path, ["label", *col_labels], rows)


def try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        warnings.warn(f"matplotlib unavailable; skipping plots: {exc}")
        return None


def save_heatmap(path: Path, matrix: np.ndarray, row_labels: list[str], col_labels: list[str], title: str, cmap: str) -> None:
    plt = try_import_matplotlib()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=160)
    image = ax.imshow(matrix, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if matrix.shape[0] <= 12 and matrix.shape[1] <= 12:
        max_value = float(np.nanmax(matrix)) if matrix.size else 0.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                if v != 0:
                    label = str(int(v)) if float(v).is_integer() else f"{v:.2f}"
                    ax.text(j, i, label, ha="center", va="center", fontsize=7, color="white" if v > max_value * 0.55 else "black")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_scatter(path: Path, x: np.ndarray, y: np.ndarray, labels: list[str], title: str, xlabel: str, ylabel: str, subtitle: str) -> None:
    plt = try_import_matplotlib()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=160)
    ax.scatter(x, y, s=44, color="#F58518")
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


def make_plots(output_dir: Path, colors: list[str], conf: np.ndarray, visual_rdm: np.ndarray, text_rdm: np.ndarray, class_data: list[dict], corr_lookup: dict[str, dict], no_plots: bool, plot_projector_diagnostics: bool) -> list[str]:
    if no_plots or not plot_projector_diagnostics:
        return []
    plots = output_dir / "plots"
    paths = []
    specs = [
        (plots / "projector_confusion_matrix.png", lambda p: save_heatmap(p, conf, colors, colors, "Projector-space Similarity Confusion", "Blues")),
        (plots / "projector_visual_rdm.png", lambda p: save_heatmap(p, visual_rdm, colors, colors, "Projected Image Centroid RDM", "magma")),
        (plots / "llm_color_text_rdm.png", lambda p: save_heatmap(p, text_rdm, colors, colors, "LLM Color Text Embedding RDM", "magma")),
    ]
    class_acc = np.array([r["accuracy"] for r in class_data], dtype=float)
    class_rsa = np.array([r["classwise_rsa_spearman"] for r in class_data], dtype=float)
    class_margin = np.array([r["mean_pair_alignment_margin"] for r in class_data], dtype=float)
    row = corr_lookup.get("classwise_rsa_spearman_vs_accuracy")
    subtitle = f"r={row['r']:.3f}, p={row['p_value_permutation_two_sided']:.4f}" if row else ""
    specs.append((plots / "classwise_rsa_vs_accuracy.png", lambda p: save_scatter(p, class_rsa, class_acc, colors, "Class-wise Projector Alignment vs Accuracy", "class-wise RSA Spearman", "accuracy", subtitle)))
    row = corr_lookup.get("mean_pair_alignment_margin_vs_accuracy")
    subtitle = f"r={row['r']:.3f}, p={row['p_value_permutation_two_sided']:.4f}" if row else ""
    specs.append((plots / "pair_margin_vs_accuracy.png", lambda p: save_scatter(p, class_margin, class_acc, colors, "Pair Margin vs Accuracy", "mean pair alignment margin", "accuracy", subtitle)))
    for path, plotter in specs:
        plotter(path)
        if path.exists():
            paths.append(str(path))
    return paths


def analyze_single_space(image_embeddings: np.ndarray, text_embeddings: np.ndarray, labels: list[str], colors: list[str], rounds: int, seed: int) -> dict:
    image_embeddings = l2_normalize(image_embeddings)
    text_embeddings = l2_normalize(text_embeddings)
    label_indices = labels_to_indices(labels, colors)
    pred_rows, similarities, pred, margins, accuracy = zero_shot(image_embeddings, text_embeddings, labels, colors)
    correct_similarities = similarities[np.arange(len(label_indices)), label_indices]
    image_centroids = centroids(image_embeddings, label_indices, colors)
    visual_rdm = rdm(image_centroids)
    text_rdm = rdm(text_embeddings)
    class_data = class_rows(labels, pred, margins, colors, visual_rdm, text_rdm)
    rsa_s, rsa_s_p = rdm_permutation_pvalue(visual_rdm, text_rdm, spearman_corr, rounds, seed)
    rsa_k, rsa_k_p = rdm_permutation_pvalue(visual_rdm, text_rdm, kendall_tau, rounds, seed + 1)
    rsa_p, rsa_p_p = rdm_permutation_pvalue(visual_rdm, text_rdm, pearson_corr, rounds, seed + 2)
    class_acc = np.array([r["accuracy"] for r in class_data], dtype=float)
    class_rsa = np.array([r["classwise_rsa_spearman"] for r in class_data], dtype=float)
    class_margin = np.array([r["mean_pair_alignment_margin"] for r in class_data], dtype=float)
    cr, cp = permutation_pvalue(class_rsa, class_acc, spearman_corr, rounds, seed + 3)
    mr, mp = permutation_pvalue(class_margin, class_acc, spearman_corr, rounds, seed + 4)
    sr, sp = permutation_pvalue(margins, np.array([r["correct"] for r in pred_rows], dtype=float), spearman_corr, rounds, seed + 5)
    class_correct_similarity = []
    for class_idx in range(len(colors)):
        mask = label_indices == class_idx
        class_correct_similarity.append(float(np.mean(correct_similarities[mask])) if np.any(mask) else float("nan"))
    return {
        "cosine_prototype_accuracy": accuracy,
        "pair_alignment_mean": float(np.mean(margins)),
        "pair_alignment_std": float(np.std(margins)),
        "mean_correct_similarity": float(np.mean(correct_similarities)),
        "rsa_spearman": rsa_s,
        "rsa_spearman_p": rsa_s_p,
        "rsa_kendall_tau": rsa_k,
        "rsa_kendall_tau_p": rsa_k_p,
        "rsa_pearson": rsa_p,
        "rsa_pearson_p": rsa_p_p,
        "linear_cka": linear_cka(image_centroids, text_embeddings),
        "procrustes_error": procrustes_error(text_embeddings, image_centroids),
        "classwise_rsa_spearman_vs_cosine_prototype_accuracy_r": cr,
        "classwise_rsa_spearman_vs_cosine_prototype_accuracy_p": cp,
        "class_pair_margin_vs_cosine_prototype_accuracy_spearman_r": mr,
        "class_pair_margin_vs_cosine_prototype_accuracy_spearman_p": mp,
        "sample_pair_margin_vs_cosine_prototype_correct_spearman_r": sr,
        "sample_pair_margin_vs_cosine_prototype_correct_spearman_p": sp,
        "classwise_rsa_spearman_vs_accuracy_r": cr,
        "classwise_rsa_spearman_vs_accuracy_p": cp,
        "class_pair_margin_vs_accuracy_spearman_r": mr,
        "class_pair_margin_vs_accuracy_spearman_p": mp,
        "sample_pair_margin_vs_correct_spearman_r": sr,
        "sample_pair_margin_vs_correct_spearman_p": sp,
        "_sample_pair_alignment_margins": margins,
        "_sample_correct_similarities": correct_similarities,
        "_class_pair_alignment_margins": class_margin,
        "_class_correct_similarities": np.asarray(class_correct_similarity, dtype=float),
        "_classwise_rsa_spearman": class_rsa,
        "_class_cosine_prototype_accuracy": class_acc,
    }


def parse_correct_value(value: object) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes"}:
        return 1.0
    if text in {"0", "0.0", "false", "no"}:
        return 0.0
    try:
        parsed = float(text)
    except ValueError:
        return float("nan")
    return parsed if np.isfinite(parsed) else float("nan")


def load_generative_outcomes(args: argparse.Namespace, dataset: Dataset, colors: list[str]) -> dict | None:
    """Load final VLM color-name correctness and align it to this dataset order."""
    if args.generative_results_dir is None:
        return None
    predictions_path = args.generative_results_dir / "sample_predictions.csv"
    if not predictions_path.exists():
        return None

    rows = read_csv_rows(predictions_path)
    if not rows:
        return None
    columns = set(rows[0].keys())
    if "normalized_pred_color" not in columns or "raw_response" not in columns:
        warnings.warn(
            f"Skipping generative outcome alignment because {predictions_path} does not look like "
            "run_generative_color_naming_experiment.py output. Pass the generative experiment directory to "
            "--generative-results-dir."
        )
        return None

    by_index = {}
    for row in rows:
        raw_index = row.get("sample_index")
        try:
            sample_index = int(float(raw_index))
        except (TypeError, ValueError):
            continue
        by_index[sample_index] = row

    label_indices = labels_to_indices(dataset.labels, colors)
    sample_correct = np.full(len(dataset.labels), np.nan, dtype=float)
    for sample_index, true_idx in enumerate(label_indices):
        row = by_index.get(sample_index)
        if row is None:
            continue
        correct = parse_correct_value(row.get("correct"))
        if not np.isfinite(correct):
            pred_color = (row.get("normalized_pred_color") or row.get("pred_color") or "").strip().lower()
            if pred_color:
                correct = 1.0 if pred_color == colors[true_idx] else 0.0
        sample_correct[sample_index] = correct

    class_accuracy = []
    class_support = []
    for color_idx, _color in enumerate(colors):
        mask = (label_indices == color_idx) & np.isfinite(sample_correct)
        class_support.append(int(np.sum(mask)))
        class_accuracy.append(float(np.mean(sample_correct[mask])) if np.any(mask) else float("nan"))

    return {
        "sample_correct": sample_correct,
        "class_accuracy": np.asarray(class_accuracy, dtype=float),
        "class_support": np.asarray(class_support, dtype=int),
        "n_sample_predictions": int(np.sum(np.isfinite(sample_correct))),
        "path": str(predictions_path),
    }


def finite_permutation_correlation(x: np.ndarray, y: np.ndarray, rounds: int, seed: int) -> tuple[float, float, int]:
    if x is None or y is None:
        return float("nan"), float("nan"), 0
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return float("nan"), float("nan"), int(np.sum(mask))
    r_value, p_value = permutation_pvalue(x[mask], y[mask], spearman_corr, rounds, seed)
    return r_value, p_value, int(np.sum(mask))


def analyze_layerwise_vs_generative(
    rows: list[dict],
    generative_outcomes: dict | None,
    args: argparse.Namespace,
    output: Path,
    prefix: str,
    title_prefix: str,
    seed_offset: int,
) -> tuple[dict, list[str]]:
    if generative_outcomes is None or not rows:
        return {}, []

    sample_correct = np.asarray(generative_outcomes["sample_correct"], dtype=float)
    class_accuracy = np.asarray(generative_outcomes["class_accuracy"], dtype=float)
    result_rows = []
    for row_idx, row in enumerate(rows):
        layer_seed = args.seed + seed_offset + row_idx * 100
        sample_margin_r, sample_margin_p, sample_margin_n = finite_permutation_correlation(
            row.get("_sample_pair_alignment_margins"), sample_correct, args.permutation_rounds, layer_seed + 1
        )
        sample_similarity_r, sample_similarity_p, sample_similarity_n = finite_permutation_correlation(
            row.get("_sample_correct_similarities"), sample_correct, args.permutation_rounds, layer_seed + 2
        )
        class_margin_r, class_margin_p, class_margin_n = finite_permutation_correlation(
            row.get("_class_pair_alignment_margins"), class_accuracy, args.permutation_rounds, layer_seed + 3
        )
        class_similarity_r, class_similarity_p, class_similarity_n = finite_permutation_correlation(
            row.get("_class_correct_similarities"), class_accuracy, args.permutation_rounds, layer_seed + 4
        )
        class_rsa_r, class_rsa_p, class_rsa_n = finite_permutation_correlation(
            row.get("_classwise_rsa_spearman"), class_accuracy, args.permutation_rounds, layer_seed + 5
        )
        class_cosine_r, class_cosine_p, class_cosine_n = finite_permutation_correlation(
            row.get("_class_cosine_prototype_accuracy"), class_accuracy, args.permutation_rounds, layer_seed + 6
        )
        result_rows.append(
            {
                "layer_index": row["layer_index"],
                "generative_sample_n": sample_margin_n,
                "generative_class_n": class_margin_n,
                "sample_pair_margin_vs_generative_correct_r": sample_margin_r,
                "sample_pair_margin_vs_generative_correct_p": sample_margin_p,
                "sample_correct_similarity_vs_generative_correct_r": sample_similarity_r,
                "sample_correct_similarity_vs_generative_correct_p": sample_similarity_p,
                "class_pair_margin_vs_generative_accuracy_r": class_margin_r,
                "class_pair_margin_vs_generative_accuracy_p": class_margin_p,
                "class_correct_similarity_vs_generative_accuracy_r": class_similarity_r,
                "class_correct_similarity_vs_generative_accuracy_p": class_similarity_p,
                "classwise_rsa_vs_generative_accuracy_r": class_rsa_r,
                "classwise_rsa_vs_generative_accuracy_p": class_rsa_p,
                "class_cosine_prototype_accuracy_vs_generative_accuracy_r": class_cosine_r,
                "class_cosine_prototype_accuracy_vs_generative_accuracy_p": class_cosine_p,
                "class_cosine_prototype_accuracy_vs_generative_accuracy_n": class_cosine_n,
            }
        )

    csv_path = output / f"{prefix}_layerwise_vs_generative.csv"
    fieldnames = [
        "layer_index",
        "generative_sample_n",
        "generative_class_n",
        "sample_pair_margin_vs_generative_correct_r",
        "sample_pair_margin_vs_generative_correct_p",
        "sample_correct_similarity_vs_generative_correct_r",
        "sample_correct_similarity_vs_generative_correct_p",
        "class_pair_margin_vs_generative_accuracy_r",
        "class_pair_margin_vs_generative_accuracy_p",
        "class_correct_similarity_vs_generative_accuracy_r",
        "class_correct_similarity_vs_generative_accuracy_p",
        "classwise_rsa_vs_generative_accuracy_r",
        "classwise_rsa_vs_generative_accuracy_p",
        "class_cosine_prototype_accuracy_vs_generative_accuracy_r",
        "class_cosine_prototype_accuracy_vs_generative_accuracy_p",
        "class_cosine_prototype_accuracy_vs_generative_accuracy_n",
    ]
    write_csv(csv_path, fieldnames, result_rows)

    plot_files = []
    if not args.no_plots:
        specs = [
            (
                f"{prefix}_layerwise_sample_pair_margin_vs_generative_correct.png",
                "sample_pair_margin_vs_generative_correct_r",
                f"{title_prefix} Pair Margin vs Generative Correctness",
                "Spearman r",
            ),
            (
                f"{prefix}_layerwise_sample_correct_similarity_vs_generative_correct.png",
                "sample_correct_similarity_vs_generative_correct_r",
                f"{title_prefix} Correct-Color Similarity vs Generative Correctness",
                "Spearman r",
            ),
            (
                f"{prefix}_layerwise_class_rsa_vs_generative_accuracy.png",
                "classwise_rsa_vs_generative_accuracy_r",
                f"{title_prefix} Class-wise RSA vs Generative Accuracy",
                "Spearman r",
            ),
        ]
        for filename, y_key, title, ylabel in specs:
            plot_path = output / "plots" / filename
            save_layerwise_plot(plot_path, result_rows, y_key, title, ylabel)
            if plot_path.exists():
                plot_files.append(str(plot_path))

    best_sample_margin = max(
        result_rows,
        key=lambda item: item["sample_pair_margin_vs_generative_correct_r"] if np.isfinite(item["sample_pair_margin_vs_generative_correct_r"]) else -999,
    )
    best_sample_similarity = max(
        result_rows,
        key=lambda item: item["sample_correct_similarity_vs_generative_correct_r"] if np.isfinite(item["sample_correct_similarity_vs_generative_correct_r"]) else -999,
    )
    best_class_rsa = max(
        result_rows,
        key=lambda item: item["classwise_rsa_vs_generative_accuracy_r"] if np.isfinite(item["classwise_rsa_vs_generative_accuracy_r"]) else -999,
    )
    summary = {
        f"{prefix}_layerwise_vs_generative_csv": str(csv_path),
        f"{prefix}_best_sample_pair_margin_vs_generative_layer": best_sample_margin.get("layer_index"),
        f"{prefix}_best_sample_pair_margin_vs_generative_r": best_sample_margin.get("sample_pair_margin_vs_generative_correct_r"),
        f"{prefix}_best_sample_pair_margin_vs_generative_p": best_sample_margin.get("sample_pair_margin_vs_generative_correct_p"),
        f"{prefix}_best_sample_correct_similarity_vs_generative_layer": best_sample_similarity.get("layer_index"),
        f"{prefix}_best_sample_correct_similarity_vs_generative_r": best_sample_similarity.get("sample_correct_similarity_vs_generative_correct_r"),
        f"{prefix}_best_sample_correct_similarity_vs_generative_p": best_sample_similarity.get("sample_correct_similarity_vs_generative_correct_p"),
        f"{prefix}_best_classwise_rsa_vs_generative_layer": best_class_rsa.get("layer_index"),
        f"{prefix}_best_classwise_rsa_vs_generative_r": best_class_rsa.get("classwise_rsa_vs_generative_accuracy_r"),
        f"{prefix}_best_classwise_rsa_vs_generative_p": best_class_rsa.get("classwise_rsa_vs_generative_accuracy_p"),
        f"{prefix}_layerwise_vs_generative_plot_files": plot_files,
    }
    return summary, plot_files


def save_layerwise_plot(path: Path, layer_rows: list[dict], y_key: str, title: str, ylabel: str) -> None:
    plt = try_import_matplotlib()
    if plt is None:
        return
    x = [int(row["layer_index"]) for row in layer_rows]
    y = [float(row[y_key]) for row in layer_rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    ax.plot(x, y, marker="o", linewidth=1.8, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("LLM layer index (0 = input embedding)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def analyze_layerwise(dataset: Dataset, embeddings: Embeddings, args: argparse.Namespace, output: Path, generative_outcomes: dict | None = None) -> dict:
    if embeddings.image_layer_embeddings is None or embeddings.text_layer_embeddings is None:
        return {}
    layer_indices = embeddings.layer_indices or list(range(embeddings.image_layer_embeddings.shape[0]))
    label_indices = labels_to_indices(dataset.labels, embeddings.colors)
    rows = []
    layer_probes = []
    layer_probe_class_rows = []
    for array_idx, layer_idx in enumerate(layer_indices):
        metrics = analyze_single_space(
            embeddings.image_layer_embeddings[array_idx],
            embeddings.text_layer_embeddings[array_idx],
            dataset.labels,
            embeddings.colors,
            args.permutation_rounds,
            args.seed + 1000 + array_idx * 10,
        )
        probe = linear_probe_image_only(
            embeddings.image_layer_embeddings[array_idx],
            label_indices,
            embeddings.colors,
            args.linear_probe_test_size,
            args.ridge_alpha,
            args.seed + 8000 + array_idx * 10,
        )
        layer_probes.append(probe)
        for class_row in probe["linear_probe_class_rows"]:
            layer_probe_class_rows.append({"layer_index": int(layer_idx), **class_row})
        rows.append(
            {
                "layer_index": int(layer_idx),
                **metrics,
                "linear_probe_accuracy": probe["linear_probe_accuracy"],
                "linear_probe_train_n": probe["linear_probe_train_n"],
                "linear_probe_test_n": probe["linear_probe_test_n"],
            }
        )
    fieldnames = [
        "layer_index",
        "cosine_prototype_accuracy",
        "linear_probe_accuracy",
        "linear_probe_train_n",
        "linear_probe_test_n",
        "pair_alignment_mean",
        "pair_alignment_std",
        "mean_correct_similarity",
        "rsa_spearman",
        "rsa_spearman_p",
        "rsa_kendall_tau",
        "rsa_kendall_tau_p",
        "rsa_pearson",
        "rsa_pearson_p",
        "linear_cka",
        "procrustes_error",
        "classwise_rsa_spearman_vs_cosine_prototype_accuracy_r",
        "classwise_rsa_spearman_vs_cosine_prototype_accuracy_p",
        "class_pair_margin_vs_cosine_prototype_accuracy_spearman_r",
        "class_pair_margin_vs_cosine_prototype_accuracy_spearman_p",
        "sample_pair_margin_vs_cosine_prototype_correct_spearman_r",
        "sample_pair_margin_vs_cosine_prototype_correct_spearman_p",
    ]
    write_csv(output / "layerwise_metrics.csv", fieldnames, rows)
    write_csv(output / "image_layerwise_metrics.csv", fieldnames, rows)
    write_csv(
        output / "layerwise_linear_probe_class_metrics.csv",
        ["layer_index", "color", "linear_probe_test_support", "linear_probe_accuracy"],
        layer_probe_class_rows,
    )
    write_csv(
        output / "image_layerwise_linear_probe_class_metrics.csv",
        ["layer_index", "color", "linear_probe_test_support", "linear_probe_accuracy"],
        layer_probe_class_rows,
    )

    layer_labels = [str(row["layer_index"]) for row in rows]
    cosine_accuracy_values = np.array([row["cosine_prototype_accuracy"] for row in rows], dtype=float)
    linear_probe_accuracy_values = np.array([row["linear_probe_accuracy"] for row in rows], dtype=float)
    layerwise_correlation_specs = [
        ("pair_alignment_mean_vs_linear_probe_accuracy", "pair_alignment_mean", "linear_probe_accuracy", "spearman", args.seed + 7011),
        ("rsa_spearman_vs_linear_probe_accuracy", "rsa_spearman", "linear_probe_accuracy", "spearman", args.seed + 7012),
        ("linear_cka_vs_linear_probe_accuracy", "linear_cka", "linear_probe_accuracy", "spearman", args.seed + 7013),
        ("procrustes_error_vs_linear_probe_accuracy", "procrustes_error", "linear_probe_accuracy", "spearman", args.seed + 7014),
        ("cosine_prototype_accuracy_vs_linear_probe_accuracy", "cosine_prototype_accuracy", "linear_probe_accuracy", "spearman", args.seed + 7015),
    ]
    layerwise_correlation_rows = []
    for name, x_key, y_key, method, corr_seed in layerwise_correlation_specs:
        x_values = np.array([row[x_key] for row in rows], dtype=float)
        y_values = np.array([row[y_key] for row in rows], dtype=float)
        r_value, p_value = permutation_pvalue(x_values, y_values, spearman_corr, args.permutation_rounds, corr_seed)
        layerwise_correlation_rows.append(
            {
                "analysis": "layer_level",
                "relationship": name,
                "x": x_key,
                "y": y_key,
                "method": method,
                "r": r_value,
                "p_value_permutation_two_sided": p_value,
                "n_layers": len(rows),
            }
        )
    write_csv(
        output / "layerwise_correlations.csv",
        ["analysis", "relationship", "x", "y", "method", "r", "p_value_permutation_two_sided", "n_layers"],
        layerwise_correlation_rows,
    )
    write_csv(
        output / "image_layerwise_correlations.csv",
        ["analysis", "relationship", "x", "y", "method", "r", "p_value_permutation_two_sided", "n_layers"],
        layerwise_correlation_rows,
    )

    generative_summary, generative_plot_files = analyze_layerwise_vs_generative(
        rows,
        generative_outcomes,
        args,
        output,
        prefix="image",
        title_prefix="Image-token",
        seed_offset=15000,
    )

    final_layer_array_index = len(layer_indices) - 1
    final_layer_index = int(layer_indices[final_layer_array_index])
    final_probe = layer_probes[final_layer_array_index]
    write_csv(
        output / "final_layer_linear_probe_class_metrics.csv",
        ["color", "linear_probe_test_support", "linear_probe_accuracy"],
        final_probe["linear_probe_class_rows"],
    )
    write_csv(
        output / "final_layer_linear_probe_predictions.csv",
        ["sample_index", "true_color", "pred_color", "correct"],
        final_probe["linear_probe_prediction_rows"],
    )

    plot_files = []
    if not args.no_plots:
        specs = [
            ("layerwise_linear_probe_accuracy.png", "linear_probe_accuracy", "Layer-wise Image-only Linear Probe Accuracy", "linear probe accuracy"),
            ("layerwise_cosine_prototype_accuracy.png", "cosine_prototype_accuracy", "Layer-wise Cosine Prototype Accuracy (Diagnostic)", "cosine prototype accuracy"),
            ("layerwise_pair_alignment.png", "pair_alignment_mean", "Layer-wise Pair Alignment Margin", "mean pair margin"),
            ("layerwise_correct_color_similarity.png", "mean_correct_similarity", "Layer-wise Correct-Color Cosine Similarity", "mean cosine similarity"),
            ("layerwise_rsa_spearman.png", "rsa_spearman", "Layer-wise RSA Spearman", "RSA Spearman"),
            ("layerwise_cka.png", "linear_cka", "Layer-wise Linear CKA", "CKA"),
            ("layerwise_procrustes_error.png", "procrustes_error", "Layer-wise Procrustes Error", "error"),
        ]
        for filename, key, title, ylabel in specs:
            plot_path = output / "plots" / filename
            save_layerwise_plot(plot_path, rows, key, title, ylabel)
            if plot_path.exists():
                plot_files.append(str(plot_path))
        row = next((r for r in layerwise_correlation_rows if r["relationship"] == "cosine_prototype_accuracy_vs_linear_probe_accuracy"), None)
        subtitle = f"Spearman r={row['r']:.3f}, p={row['p_value_permutation_two_sided']:.4f}" if row else ""
        plot_path = output / "plots" / "layerwise_cosine_vs_linear_probe_accuracy.png"
        save_scatter(
            plot_path,
            cosine_accuracy_values,
            linear_probe_accuracy_values,
            layer_labels,
            "Layer-wise Cosine Accuracy vs Linear Probe Accuracy",
            "cosine prototype accuracy",
            "image-only linear probe accuracy",
            subtitle,
        )
        if plot_path.exists():
            plot_files.append(str(plot_path))
    plot_files.extend(generative_plot_files)
    best_cosine_accuracy = max(rows, key=lambda r: r["cosine_prototype_accuracy"]) if rows else {}
    best_probe_accuracy = max(rows, key=lambda r: r["linear_probe_accuracy"]) if rows else {}
    best_rsa = max(rows, key=lambda r: r["rsa_spearman"] if np.isfinite(r["rsa_spearman"]) else -999) if rows else {}
    return {
        "layerwise_n_layers": len(rows),
        "layerwise_metrics_csv": str(output / "layerwise_metrics.csv"),
        "image_layerwise_metrics_csv": str(output / "image_layerwise_metrics.csv"),
        "layerwise_correlations_csv": str(output / "layerwise_correlations.csv"),
        "image_layerwise_correlations_csv": str(output / "image_layerwise_correlations.csv"),
        "layerwise_linear_probe_class_metrics_csv": str(output / "layerwise_linear_probe_class_metrics.csv"),
        "image_layerwise_linear_probe_class_metrics_csv": str(output / "image_layerwise_linear_probe_class_metrics.csv"),
        "layerwise_best_cosine_prototype_accuracy_layer": best_cosine_accuracy.get("layer_index"),
        "layerwise_best_cosine_prototype_accuracy": best_cosine_accuracy.get("cosine_prototype_accuracy"),
        "layerwise_best_linear_probe_layer": best_probe_accuracy.get("layer_index"),
        "layerwise_best_linear_probe_accuracy": best_probe_accuracy.get("linear_probe_accuracy"),
        "layerwise_best_rsa_spearman_layer": best_rsa.get("layer_index"),
        "layerwise_best_rsa_spearman": best_rsa.get("rsa_spearman"),
        "final_layer_linear_probe_layer_index": final_layer_index,
        "final_layer_linear_probe_accuracy": final_probe["linear_probe_accuracy"],
        "final_layer_linear_probe_train_n": final_probe["linear_probe_train_n"],
        "final_layer_linear_probe_test_n": final_probe["linear_probe_test_n"],
        "final_layer_linear_probe_class_metrics_csv": str(output / "final_layer_linear_probe_class_metrics.csv"),
        "layerwise_plot_files": plot_files,
        **generative_summary,
    }


def analyze_query_layerwise(
    dataset: Dataset,
    embeddings: Embeddings,
    args: argparse.Namespace,
    output: Path,
    generative_outcomes: dict | None = None,
    query_embeddings: np.ndarray | None = None,
    prefix: str = "query",
    title_prefix: str = "Query-token",
) -> dict:
    if query_embeddings is None:
        query_embeddings = embeddings.query_layer_embeddings
    if query_embeddings is None or embeddings.text_layer_embeddings is None:
        return {}
    layer_indices = embeddings.layer_indices or list(range(query_embeddings.shape[0]))
    label_indices = labels_to_indices(dataset.labels, embeddings.colors)
    rows = []
    class_rows_all = []
    for array_idx, layer_idx in enumerate(layer_indices):
        metrics = analyze_single_space(
            query_embeddings[array_idx],
            embeddings.text_layer_embeddings[array_idx],
            dataset.labels,
            embeddings.colors,
            args.permutation_rounds,
            args.seed + 9000 + array_idx * 10,
        )
        probe = linear_probe_image_only(
            query_embeddings[array_idx],
            label_indices,
            embeddings.colors,
            args.linear_probe_test_size,
            args.ridge_alpha,
            args.seed + 10000 + array_idx * 10,
        )
        for class_row in probe["linear_probe_class_rows"]:
            class_rows_all.append({"layer_index": int(layer_idx), **class_row})
        rows.append(
            {
                "layer_index": int(layer_idx),
                **metrics,
                "query_text_cosine_prototype_accuracy": metrics["cosine_prototype_accuracy"],
                "query_linear_probe_accuracy": probe["linear_probe_accuracy"],
                "linear_probe_train_n": probe["linear_probe_train_n"],
                "linear_probe_test_n": probe["linear_probe_test_n"],
            }
        )

    fieldnames = [
        "layer_index",
        "query_text_cosine_prototype_accuracy",
        "query_linear_probe_accuracy",
        "linear_probe_train_n",
        "linear_probe_test_n",
        "pair_alignment_mean",
        "pair_alignment_std",
        "mean_correct_similarity",
        "rsa_spearman",
        "rsa_spearman_p",
        "rsa_kendall_tau",
        "rsa_kendall_tau_p",
        "rsa_pearson",
        "rsa_pearson_p",
        "linear_cka",
        "procrustes_error",
        "classwise_rsa_spearman_vs_cosine_prototype_accuracy_r",
        "classwise_rsa_spearman_vs_cosine_prototype_accuracy_p",
        "class_pair_margin_vs_cosine_prototype_accuracy_spearman_r",
        "class_pair_margin_vs_cosine_prototype_accuracy_spearman_p",
        "sample_pair_margin_vs_cosine_prototype_correct_spearman_r",
        "sample_pair_margin_vs_cosine_prototype_correct_spearman_p",
    ]
    metrics_path = output / f"{prefix}_layerwise_metrics.csv"
    correlations_path = output / f"{prefix}_layerwise_correlations.csv"
    class_metrics_path = output / f"{prefix}_layerwise_linear_probe_class_metrics.csv"
    write_csv(metrics_path, fieldnames, rows)
    write_csv(
        class_metrics_path,
        ["layer_index", "color", "linear_probe_test_support", "linear_probe_accuracy"],
        class_rows_all,
    )

    query_corr_specs = [
        ("query_pair_alignment_mean_vs_linear_probe_accuracy", "pair_alignment_mean", "query_linear_probe_accuracy", args.seed + 11001),
        ("query_rsa_spearman_vs_linear_probe_accuracy", "rsa_spearman", "query_linear_probe_accuracy", args.seed + 11002),
        ("query_linear_cka_vs_linear_probe_accuracy", "linear_cka", "query_linear_probe_accuracy", args.seed + 11003),
        ("query_cosine_prototype_accuracy_vs_linear_probe_accuracy", "query_text_cosine_prototype_accuracy", "query_linear_probe_accuracy", args.seed + 11004),
    ]
    corr_rows = []
    for name, x_key, y_key, corr_seed in query_corr_specs:
        x_values = np.array([row[x_key] for row in rows], dtype=float)
        y_values = np.array([row[y_key] for row in rows], dtype=float)
        r_value, p_value = permutation_pvalue(x_values, y_values, spearman_corr, args.permutation_rounds, corr_seed)
        corr_rows.append(
            {
                "analysis": "query_layer_level",
                "relationship": name,
                "x": x_key,
                "y": y_key,
                "method": "spearman",
                "r": r_value,
                "p_value_permutation_two_sided": p_value,
                "n_layers": len(rows),
            }
        )
    write_csv(
        correlations_path,
        ["analysis", "relationship", "x", "y", "method", "r", "p_value_permutation_two_sided", "n_layers"],
        corr_rows,
    )

    generative_summary, generative_plot_files = analyze_layerwise_vs_generative(
        rows,
        generative_outcomes,
        args,
        output,
        prefix=prefix,
        title_prefix=title_prefix,
        seed_offset=17000,
    )

    plot_files = []
    if not args.no_plots:
        specs = [
            (f"{prefix}_layerwise_text_cosine_prototype_accuracy.png", "query_text_cosine_prototype_accuracy", f"{title_prefix}/Text Cosine Prototype Accuracy (Diagnostic)", "cosine prototype accuracy"),
            (f"{prefix}_layerwise_linear_probe_accuracy.png", "query_linear_probe_accuracy", f"{title_prefix} Image-conditioned Linear Probe Accuracy", "linear probe accuracy"),
            (f"{prefix}_layerwise_pair_alignment.png", "pair_alignment_mean", f"{title_prefix} Pair Alignment Margin", "mean pair margin"),
            (f"{prefix}_layerwise_correct_color_similarity.png", "mean_correct_similarity", f"{title_prefix} Correct-Color Cosine Similarity", "mean cosine similarity"),
            (f"{prefix}_layerwise_rsa_spearman.png", "rsa_spearman", f"{title_prefix} vs Text RSA Spearman", "RSA Spearman"),
            (f"{prefix}_layerwise_cka.png", "linear_cka", f"{title_prefix} vs Text Linear CKA", "CKA"),
            (f"{prefix}_layerwise_procrustes_error.png", "procrustes_error", f"{title_prefix} vs Text Procrustes Error", "error"),
        ]
        for filename, key, title, ylabel in specs:
            plot_path = output / "plots" / filename
            save_layerwise_plot(plot_path, rows, key, title, ylabel)
            if plot_path.exists():
                plot_files.append(str(plot_path))
    plot_files.extend(generative_plot_files)

    best_query_probe = max(rows, key=lambda r: r["query_linear_probe_accuracy"]) if rows else {}
    best_query_rsa = max(rows, key=lambda r: r["rsa_spearman"] if np.isfinite(r["rsa_spearman"]) else -999) if rows else {}
    best_query_cka = max(rows, key=lambda r: r["linear_cka"] if np.isfinite(r["linear_cka"]) else -999) if rows else {}
    return {
        f"{prefix}_layerwise_metrics_csv": str(metrics_path),
        f"{prefix}_layerwise_correlations_csv": str(correlations_path),
        f"{prefix}_layerwise_linear_probe_class_metrics_csv": str(class_metrics_path),
        f"{prefix}_best_linear_probe_layer": best_query_probe.get("layer_index"),
        f"{prefix}_best_linear_probe_accuracy": best_query_probe.get("query_linear_probe_accuracy"),
        f"{prefix}_best_rsa_spearman_layer": best_query_rsa.get("layer_index"),
        f"{prefix}_best_rsa_spearman": best_query_rsa.get("rsa_spearman"),
        f"{prefix}_best_cka_layer": best_query_cka.get("layer_index"),
        f"{prefix}_best_cka": best_query_cka.get("linear_cka"),
        f"{prefix}_layerwise_plot_files": plot_files,
        **generative_summary,
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def copy_csv_if_present(source: Path, target: Path) -> str | None:
    if not source.exists():
        return None
    rows = read_csv_rows(source)
    if not rows:
        return None
    write_csv(target, list(rows[0].keys()), rows)
    return str(target)


def analyze_generative_behavior(args: argparse.Namespace, output: Path) -> dict:
    if args.generative_results_dir is None:
        return {}
    source = args.generative_results_dir
    if not source.exists():
        raise FileNotFoundError(f"Generative results directory not found: {source}")

    summary_path = source / "summary.json"
    generative_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    behavior_summary = {
        "generative_results_dir": str(source),
        "generative_accuracy": generative_summary.get("normalized_accuracy_all"),
        "generative_accuracy_parseable_only": generative_summary.get("normalized_accuracy_parseable_only"),
        "generative_macro_f1": generative_summary.get("macro_f1"),
        "generative_parse_rate": generative_summary.get("parse_rate"),
        "generative_prompt": generative_summary.get("prompt"),
    }
    (output / "vlm_output_summary.json").write_text(
        json.dumps({**behavior_summary, "source_summary": generative_summary}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    class_csv = copy_csv_if_present(source / "class_metrics.csv", output / "vlm_output_class_accuracy.csv")
    conf_csv = copy_csv_if_present(source / "confusion_matrix.csv", output / "vlm_output_confusion_matrix.csv")
    if class_csv:
        behavior_summary["vlm_output_class_accuracy_csv"] = class_csv
    if conf_csv:
        behavior_summary["vlm_output_confusion_matrix_csv"] = conf_csv
    return behavior_summary


def analyze(dataset: Dataset, embeddings: Embeddings, args: argparse.Namespace) -> dict:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    colors = embeddings.colors
    image_embeddings = l2_normalize(embeddings.image_projected_embeddings)
    text_embeddings = l2_normalize(embeddings.text_embeddings)
    label_indices = labels_to_indices(dataset.labels, colors)

    pred_rows, similarities, pred, margins, accuracy = zero_shot(image_embeddings, text_embeddings, dataset.labels, colors)
    for row, source in zip(pred_rows, dataset.rows):
        row.update({"image_path": source.get("image_path"), "hex": source.get("hex")})
    write_csv(output / "sample_predictions.csv", ["sample_index", "image_path", "hex", "true_color", "pred_color", "correct", "pair_alignment_margin", "correct_similarity", "pred_similarity"], pred_rows)

    image_centroids = centroids(image_embeddings, label_indices, colors)
    visual_rdm = rdm(image_centroids)
    text_rdm = rdm(text_embeddings)
    visual_values = upper_triangle_values(visual_rdm)
    text_values = upper_triangle_values(text_rdm)

    class_data = class_rows(dataset.labels, pred, margins, colors, visual_rdm, text_rdm)
    write_csv(output / "class_metrics.csv", ["color", "support", "accuracy", "mean_pair_alignment_margin", "classwise_rsa_spearman", "classwise_rsa_pearson"], class_data)
    conf = confusion(dataset.labels, pred, colors)
    write_matrix(output / "confusion_matrix.csv", colors, colors, conf)
    write_matrix(output / "rdm_projected_image_centroids.csv", colors, colors, visual_rdm)
    write_matrix(output / "rdm_llm_color_text_embeddings.csv", colors, colors, text_rdm)

    rsa_s, rsa_s_p = rdm_permutation_pvalue(visual_rdm, text_rdm, spearman_corr, args.permutation_rounds, args.seed)
    rsa_k, rsa_k_p = rdm_permutation_pvalue(visual_rdm, text_rdm, kendall_tau, args.permutation_rounds, args.seed + 1)
    rsa_p, rsa_p_p = rdm_permutation_pvalue(visual_rdm, text_rdm, pearson_corr, args.permutation_rounds, args.seed + 2)

    class_acc = np.array([r["accuracy"] for r in class_data], dtype=float)
    class_rsa = np.array([r["classwise_rsa_spearman"] for r in class_data], dtype=float)
    class_margin = np.array([r["mean_pair_alignment_margin"] for r in class_data], dtype=float)
    cr, cp = permutation_pvalue(class_rsa, class_acc, spearman_corr, args.permutation_rounds, args.seed + 3)
    mr, mp = permutation_pvalue(class_margin, class_acc, spearman_corr, args.permutation_rounds, args.seed + 4)
    sr, sp = permutation_pvalue(margins, np.array([r["correct"] for r in pred_rows], dtype=float), spearman_corr, args.permutation_rounds, args.seed + 5)

    correlation_rows = [
        {"analysis": "global_rdm", "x": "projected_image_centroid_rdm", "y": "llm_color_text_rdm", "method": "spearman_mantel", "r": rsa_s, "p_value_permutation_two_sided": rsa_s_p, "n": len(visual_values)},
        {"analysis": "global_rdm", "x": "projected_image_centroid_rdm", "y": "llm_color_text_rdm", "method": "kendall_mantel", "r": rsa_k, "p_value_permutation_two_sided": rsa_k_p, "n": len(visual_values)},
        {"analysis": "global_rdm", "x": "projected_image_centroid_rdm", "y": "llm_color_text_rdm", "method": "pearson_mantel", "r": rsa_p, "p_value_permutation_two_sided": rsa_p_p, "n": len(visual_values)},
        {"analysis": "class_level", "x": "classwise_rsa_spearman", "y": "accuracy", "method": "spearman", "r": cr, "p_value_permutation_two_sided": cp, "n": len(colors), "key": "classwise_rsa_spearman_vs_accuracy"},
        {"analysis": "class_level", "x": "mean_pair_alignment_margin", "y": "accuracy", "method": "spearman", "r": mr, "p_value_permutation_two_sided": mp, "n": len(colors), "key": "mean_pair_alignment_margin_vs_accuracy"},
        {"analysis": "sample_level", "x": "pair_alignment_margin", "y": "correct", "method": "spearman", "r": sr, "p_value_permutation_two_sided": sp, "n": len(pred_rows)},
    ]
    write_csv(output / "correlations.csv", ["analysis", "x", "y", "method", "r", "p_value_permutation_two_sided", "n"], correlation_rows)
    corr_lookup = {row.get("key", ""): row for row in correlation_rows if row.get("key")}

    plot_files = make_plots(
        output,
        colors,
        conf,
        visual_rdm,
        text_rdm,
        class_data,
        corr_lookup,
        args.no_plots,
        args.plot_projector_diagnostics,
    )
    generative_behavior_summary = analyze_generative_behavior(args, output)
    generative_outcomes = load_generative_outcomes(args, dataset, colors)

    layerwise_summary = analyze_layerwise(dataset, embeddings, args, output, generative_outcomes=generative_outcomes)
    plot_files.extend(layerwise_summary.get("layerwise_plot_files", []))
    query_summary = analyze_query_layerwise(dataset, embeddings, args, output, generative_outcomes=generative_outcomes)
    plot_files.extend(query_summary.get("query_layerwise_plot_files", []))
    query_no_color_list_summary = analyze_query_layerwise(
        dataset,
        embeddings,
        args,
        output,
        generative_outcomes=generative_outcomes,
        query_embeddings=embeddings.query_no_color_list_layer_embeddings,
        prefix="query_no_color_list",
        title_prefix="Query-token No-color-list",
    )
    plot_files.extend(query_no_color_list_summary.get("query_no_color_list_layerwise_plot_files", []))
    summary = {
        "model_id": args.model_id,
        "ground_truth_csv": str(args.ground_truth),
        "n_samples": len(dataset.labels),
        "n_colors": len(colors),
        "image_representation": "mean/pooled LLaVA multimodal projector output tokens",
        "image_pooling": embeddings.image_pooling,
        "text_representation": embeddings.text_representation,
        "prompts": embeddings.prompts,
        "projector_space_zero_shot_accuracy": accuracy,
        "pair_alignment_mean": float(np.mean(margins)),
        "pair_alignment_std": float(np.std(margins)),
        "rsa_spearman": rsa_s,
        "rsa_spearman_permutation_p": rsa_s_p,
        "rsa_kendall_tau": rsa_k,
        "rsa_kendall_tau_permutation_p": rsa_k_p,
        "rsa_pearson": rsa_p,
        "rsa_pearson_permutation_p": rsa_p_p,
        "classwise_rsa_spearman_vs_accuracy_r": cr,
        "classwise_rsa_spearman_vs_accuracy_p": cp,
        "class_pair_margin_vs_accuracy_spearman_r": mr,
        "class_pair_margin_vs_accuracy_spearman_p": mp,
        "sample_pair_margin_vs_correct_spearman_r": sr,
        "sample_pair_margin_vs_correct_spearman_p": sp,
        "linear_cka_projected_centroids_vs_text": linear_cka(image_centroids, text_embeddings),
        "procrustes_error_text_to_projected_centroids": procrustes_error(text_embeddings, image_centroids),
        "generative_outcome_sample_predictions_csv": generative_outcomes.get("path") if generative_outcomes else None,
        "generative_outcome_n_sample_predictions": generative_outcomes.get("n_sample_predictions") if generative_outcomes else None,
        "plot_files": plot_files,
        **layerwise_summary,
        **query_summary,
        **query_no_color_list_summary,
        **generative_behavior_summary,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    prompts = prompt_templates(args.prompts)
    if args.embeddings_npz:
        embeddings, dataset = load_embeddings(args.embeddings_npz)
    else:
        dataset = load_dataset(args.ground_truth, args.dataset_root)
        embeddings = extract_embeddings(dataset, args, prompts)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_embeddings(args.output_dir / args.save_embeddings_name, embeddings, dataset)
    summary = analyze(dataset, embeddings, args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote LLaVA projector alignment outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
