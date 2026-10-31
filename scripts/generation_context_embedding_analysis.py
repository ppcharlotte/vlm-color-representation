#!/usr/bin/env python3
"""Extract generation-side CEg decision and generated-token embeddings."""

from __future__ import annotations

import argparse
from layer_selection import resolve_layers
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from structured_color_utils import BASIC_COLORS, LlavaHelper, load_dataset, normalize_color, normalize_color_answer, write_csv, write_json


DEFAULT_CEG_PROMPT = "What color is this image? Answer with a short sentence in this exact format: The color is <color>."


def parse_layers(text: str, model_id=None) -> list[int]:
    return resolve_layers(text, model_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract generation-side color representations.")
    parser.add_argument("--ground-truth", type=Path, default=Path("experiments/perceptual_alignment/data/ground_truth.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("experiments/perceptual_alignment/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/perceptual_alignment"))
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--layer-indices", default="all", help="all LM hidden states (default), or comma-separated indices")
    parser.add_argument("--prompt", default=DEFAULT_CEG_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-extraction", action="store_true", help="Reuse existing CEg embeddings when present.")
    return parser.parse_args()


def tokenizer_ids(tokenizer, text: str) -> list[int]:
    return [int(value) for value in tokenizer(text, add_special_tokens=False)["input_ids"]]


def find_subsequence(sequence: list[int], pattern: list[int]) -> int | None:
    for start in range(max(0, len(sequence) - len(pattern) + 1)):
        if pattern and sequence[start : start + len(pattern)] == pattern:
            return start
    return None


def generated_color_span(tokenizer, generated_ids: list[int], color: str) -> tuple[int | None, int | None, str]:
    variants = [color, " " + color, "\n" + color, color.capitalize(), " " + color.capitalize()]
    if color == "gray":
        variants.extend(["grey", " grey", "Grey", " Grey"])
    seen = set()
    for variant in variants:
        ids = tuple(tokenizer_ids(tokenizer, variant))
        if not ids or ids in seen:
            continue
        seen.add(ids)
        start = find_subsequence(generated_ids, list(ids))
        if start is not None:
            return start, start + len(ids), "token_id_match"
    return None, None, "no_token_id_match"


def generation_image_states(helper, inputs, hidden_steps, layer_indices):
    """Raw image means from the actual generate prefill; no extra prompt forward."""
    mask = helper.image_token_mask(inputs["input_ids"]).detach().cpu().numpy().reshape(-1).astype(bool)
    length = hidden_steps[0][0].shape[1]
    positions = prefill_image_positions(mask, length)
    return np.stack([hidden_steps[0][layer][0][positions]
                     .mean(dim=0).detach().float().cpu().numpy() for layer in layer_indices])


def prefill_image_positions(mask, length):
    """Handle processor-expanded image IDs and legacy single-placeholder prefill."""
    mask = np.asarray(mask, bool)
    if len(mask) != length:
        positions = np.flatnonzero(mask)
        if len(positions) != 1:
            raise ValueError("Generation prefill image mask cannot be matched to hidden states")
        start = int(positions[0])
        image_length = length - len(mask) + 1
        if image_length < 1:
            raise ValueError("Invalid expanded generation image length")
        mask = np.zeros(length, dtype=bool)
        mask[start:start + image_length] = True
    if not mask.any():
        raise ValueError("No image tokens in generation prefill")
    return np.flatnonzero(mask).tolist()


def generate_one(helper: LlavaHelper, image_path: str, prompt: str, max_new_tokens: int, temperature: float, layer_indices=None):
    image = Image.open(image_path).convert("RGB")
    inputs = helper.prepare_inputs(helper.chat_prompt(prompt), image)
    do_sample = temperature > 0
    with helper.torch.no_grad():
        outputs = helper.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
    input_len = int(inputs["input_ids"].shape[1])
    generated_sequence = outputs.sequences[:, input_len:] if outputs.sequences.shape[1] > input_len else outputs.sequences
    generated_ids = [int(value) for value in generated_sequence[0].detach().cpu().tolist()]
    generated_text = helper.processor.batch_decode(
        generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    images = generation_image_states(helper, inputs, outputs.hidden_states, layer_indices) if layer_indices is not None else None
    return generated_text, generated_ids, outputs.hidden_states, images


def normalize_vector(tensor):
    tensor = tensor.float()
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def extract_decision_layers(helper: LlavaHelper, hidden_steps, start: int | None, end: int | None, layer_indices: list[int]):
    if start is None or end is None:
        return None, "no_color_token_span"
    if start < 0 or end <= start or end > len(hidden_steps):
        return None, "decision_hidden_unavailable"
    vectors = []
    for layer_idx in layer_indices:
        pieces = [hidden_steps[token_idx][layer_idx][:, -1, :] for token_idx in range(start, end)]
        vectors.append(normalize_vector(helper.torch.cat(pieces, dim=0).mean(dim=0)).detach().cpu().numpy())
    return np.vstack(vectors).astype(np.float32), "predict_color_step"


def extract_token_layers(helper: LlavaHelper, hidden_steps, start: int | None, end: int | None, layer_indices: list[int]):
    if start is None or end is None:
        return None, "no_color_token_span"
    if start < 0 or end <= start or end >= len(hidden_steps):
        return None, "no_next_generation_step_after_color_token"
    vectors = []
    for layer_idx in layer_indices:
        pieces = [hidden_steps[token_idx + 1][layer_idx][:, -1, :] for token_idx in range(start, end)]
        vectors.append(normalize_vector(helper.torch.cat(pieces, dim=0).mean(dim=0)).detach().cpu().numpy())
    return np.vstack(vectors).astype(np.float32), "generated_token_next_step"


def extract_ceg(helper: LlavaHelper, dataset, prompt: str, layer_indices: list[int], max_new_tokens: int, temperature: float):
    tokenizer = helper.tokenizer()
    decisions, tokens, valid_rows, all_rows, image_states = [], [], [], [], []
    for row in dataset.rows:
        sample_index = int(row["sample_index"])
        image_path = row["absolute_image_path"]
        context_color = normalize_color(row["basic_color"])
        generated_text, generated_ids, hidden_steps, image_state = generate_one(helper, image_path, prompt, max_new_tokens, temperature, layer_indices)
        image_states.append(image_state)
        predicted_color, parse_status = normalize_color_answer(generated_text)
        start = end = None
        span_status = "no_color"
        decision_source = token_source = ""
        decision = token = None
        included = predicted_color in BASIC_COLORS
        if included:
            start, end, span_status = generated_color_span(tokenizer, generated_ids, predicted_color)
            decision, decision_source = extract_decision_layers(helper, hidden_steps, start, end, layer_indices)
            token, token_source = extract_token_layers(helper, hidden_steps, start, end, layer_indices)
            included = decision is not None and token is not None

        record = {
            "sample_index": sample_index,
            "image_path": row.get("image_path", ""),
            "absolute_image_path": image_path,
            "r": row.get("r", ""),
            "g": row.get("g", ""),
            "b": row.get("b", ""),
            "human_category": context_color,
            "generated_text": generated_text,
            "generated_token_ids": " ".join(map(str, generated_ids)),
            "normalized_pred_color": predicted_color or "",
            "parse_status": parse_status,
            "generated_color_token_start": "" if start is None else start,
            "generated_color_token_end": "" if end is None else end,
            "generated_color_span_status": span_status,
            "decision_embedding_source": decision_source,
            "token_embedding_source": token_source,
            "included_in_embedding_analysis": int(included),
        }
        all_rows.append(record)
        if included:
            valid_rows.append(record)
            decisions.append(decision)
            tokens.append(token)
    if not valid_rows:
        raise RuntimeError("No generated color tokens had both CEg representations.")
    return (
        np.stack(decisions).transpose(1, 0, 2).astype(np.float32),
        np.stack(tokens).transpose(1, 0, 2).astype(np.float32),
        all_rows,
        valid_rows,
        np.stack(image_states).transpose(1, 0, 2).astype(np.float32),
    )


def read_prediction_rows(path: Path) -> tuple[list[dict], list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows, [row for row in rows if int(row.get("included_in_embedding_analysis", "0")) == 1]


def main() -> None:
    args = parse_args()
    output = args.output_dir
    ceg_dir = output / "ceg_generation_side"
    embeddings_dir = ceg_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.ground_truth, args.dataset_root)
    layer_indices = parse_layers(args.layer_indices, args.model_id)
    decision_path = embeddings_dir / "ceg_decision_embeddings.npz"
    token_path = embeddings_dir / "ceg_token_from_generate_embeddings.npz"
    predictions_path = ceg_dir / "predictions.csv"
    image_path = embeddings_dir / "generation_image_embeddings.npz"

    reuse = args.skip_extraction and decision_path.exists() and token_path.exists() and predictions_path.exists()
    if reuse:
        for cache_path in (decision_path, token_path):
            with np.load(cache_path, allow_pickle=False) as saved:
                if saved["layer_indices"].tolist() != layer_indices:
                    raise ValueError("Cached CEg layer range differs from requested layers; rerun without --skip-extraction.")
        if not image_path.exists():
            raise ValueError("CEg cache lacks generation visual sanity states; rerun without --skip-extraction.")
        all_rows, valid_rows = read_prediction_rows(predictions_path)
    else:
        helper = LlavaHelper(args.model_id, args.device, args.dtype, args.seed)
        decision_layers, token_layers, all_rows, valid_rows, image_layers = extract_ceg(
            helper, dataset, args.prompt, layer_indices, args.max_new_tokens, args.temperature
        )
        fields = list(all_rows[0]) if all_rows else []
        write_csv(predictions_path, fields, all_rows)
        metadata = {
            "model_id": np.asarray(args.model_id),
            "image_paths": np.asarray([row["absolute_image_path"] for row in valid_rows]),
            "layer_indices": np.asarray(layer_indices),
            "human_categories": np.asarray([row["human_category"] for row in valid_rows]),
            "predicted_labels": np.asarray([row["normalized_pred_color"] for row in valid_rows]),
            "sample_indices": np.asarray([int(row["sample_index"]) for row in valid_rows]),
            "prompt": args.prompt,
            "seed": np.asarray(args.seed),
        }
        np.savez_compressed(decision_path, ceg_decision_layer_embeddings=decision_layers, **metadata)
        np.savez_compressed(token_path, ceg_token_from_generate_layer_embeddings=token_layers, **metadata)
        np.savez_compressed(image_path, image_layer_embeddings=image_layers, layer_indices=layer_indices,
            image_paths=np.asarray([row["absolute_image_path"] for row in dataset.rows]),
            model_id=np.asarray(args.model_id), seed=np.asarray(args.seed), pooling=np.asarray("mean"),
            prompt=np.asarray(args.prompt), extraction=np.asarray("actual_generate_prefill"))

    write_json(
        ceg_dir / "summary.json",
        {
            "model_id": args.model_id,
            "prompt": args.prompt,
            "layer_indices": layer_indices,
            "n_samples": len(dataset.rows),
            "n_included_for_embedding_analysis": len(valid_rows),
            "included_rate_for_embedding_analysis": len(valid_rows) / len(dataset.rows) if dataset.rows else float("nan"),
            "predictions_csv": str(predictions_path),
            "ceg_decision_npz": str(decision_path),
            "ceg_token_npz": str(token_path),
            "generation_image_npz": str(image_path),
            "G_pre": "L2-normalized mean of states immediately before each generated color token",
            "G_post": "L2-normalized mean of generated color-token states from following generation steps",
            "analysis_script": "perceptual_alignment_analysis.py",
        },
    )
    print(f"Wrote CEg embeddings to: {embeddings_dir}")


if __name__ == "__main__":
    main()
