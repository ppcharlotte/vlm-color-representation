#!/usr/bin/env python3
"""Extract CEg representations for model-generated color words.

This script studies generation-side color representations from free generation.
It saves two related representations:

1. CEg-decision: the hidden state used to predict the generated color token.
2. CEg-token: the hidden state of the generated color token after it is placed
   in the assistant answer context and processed by the LLM.

The generated color comes from the model prediction, not from the ground truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from answer_position_embedding_analysis import analyze_against_color_prototypes, analyze_against_image
from structured_color_utils import (
    BASIC_COLORS,
    LlavaHelper,
    centroids,
    ciede2000_matrix,
    labels_to_indices,
    linear_regression_summary,
    load_dataset,
    normalize_color,
    normalize_color_answer,
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

DEFAULT_CEG_PROMPT = "What color is this image? Answer with a short sentence in this exact format: The color is <color>."


def parse_layers(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and analyze free-generation CEg color representations.")
    parser.add_argument("--ground-truth", type=Path, default=Path("experiments/llava_structured_alignment/data/ground_truth.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("experiments/llava_structured_alignment/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/llava_structured_alignment"))
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--layer-indices", default="0,1,8,16,24,32")
    parser.add_argument("--prompt", default=DEFAULT_CEG_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--permutation-rounds", type=int, default=1000)
    parser.add_argument("--linear-probe-test-size", type=float, default=0.3)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-extraction", action="store_true", help="Reuse existing CEg embeddings if available.")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def tokenizer_ids(tokenizer, text: str) -> list[int]:
    return [int(x) for x in tokenizer(text, add_special_tokens=False)["input_ids"]]


def find_subsequence(sequence: list[int], pattern: list[int]) -> int | None:
    if not pattern or len(pattern) > len(sequence):
        return None
    for start in range(len(sequence) - len(pattern) + 1):
        if sequence[start : start + len(pattern)] == pattern:
            return start
    return None


def generated_color_span(tokenizer, generated_ids: list[int], color: str) -> tuple[int | None, int | None, str]:
    variants = [color, " " + color, "\n" + color, color.capitalize(), " " + color.capitalize()]
    if color == "gray":
        variants.extend(["grey", " grey", "Grey", " Grey"])
    seen: set[tuple[int, ...]] = set()
    for variant in variants:
        ids = tuple(tokenizer_ids(tokenizer, variant))
        if not ids or ids in seen:
            continue
        seen.add(ids)
        start = find_subsequence(generated_ids, list(ids))
        if start is not None:
            return start, start + len(ids), "token_id_match"
    return None, None, "no_token_id_match"


def generate_one(helper: LlavaHelper, image_path: str, prompt: str, max_new_tokens: int, temperature: float):
    image = Image.open(image_path).convert("RGB")
    text = helper.chat_prompt(prompt)
    inputs = helper.processor(text=text, images=image, return_tensors="pt")
    inputs = {key: value.to(helper.device) for key, value in inputs.items()}
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
    generated_ids = [int(x) for x in outputs.sequences[0, input_len:].detach().cpu().tolist()]
    generated_text = helper.processor.batch_decode(
        outputs.sequences[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return generated_text, generated_ids, outputs.hidden_states


def normalize_vector(tensor):
    tensor = tensor.float()
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def extract_decision_layers(helper: LlavaHelper, hidden_steps, start: int, end: int, layer_indices: list[int]) -> np.ndarray | None:
    if start is None or end is None or end > len(hidden_steps):
        return None
    layer_vectors = []
    for layer_idx in layer_indices:
        pieces = []
        for step in range(start, end):
            step_hidden = hidden_steps[step][layer_idx]
            pieces.append(step_hidden[:, -1, :])
        vector = normalize_vector(helper.torch.cat(pieces, dim=0).mean(dim=0))
        layer_vectors.append(vector.detach().cpu().numpy())
    return np.vstack(layer_vectors).astype(np.float32)


def chat_prompt_with_assistant(helper: LlavaHelper, prompt: str, answer: str) -> str:
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]
    if hasattr(helper.processor, "apply_chat_template"):
        try:
            return helper.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        except Exception:
            pass
    return f"USER: <image>\n{prompt}\nASSISTANT: {answer}"


def expanded_assistant_inputs_with_color_mask(helper: LlavaHelper, image_path: str, prompt: str, answer: str, color: str):
    embedding_layer = helper.embedding_layer()
    tokenizer = helper.tokenizer()
    image_token_id = helper.image_token_id()

    _pre, projected = helper.vision_features([image_path])
    image_features_t = helper.torch.from_numpy(projected).to(helper.device, dtype=embedding_layer.weight.dtype)

    text = chat_prompt_with_assistant(helper, prompt, answer)
    try:
        encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        offsets = encoded.pop("offset_mapping", None)
    except Exception:
        encoded = tokenizer(text, return_tensors="pt")
        offsets = None

    input_ids = encoded["input_ids"].to(helper.device)
    positions = (input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten()
    if len(positions) != 1:
        raise ValueError(f"Prompt must contain exactly one image token; found {len(positions)}")
    pos = int(positions[0].item())

    original_color_mask = helper.color_token_mask(input_ids, color, text=text, offsets=offsets).to(helper.device)
    token_embeds = embedding_layer(input_ids)
    inputs_embeds = helper.torch.cat([token_embeds[:, :pos, :], image_features_t, token_embeds[:, pos + 1 :, :]], dim=1)
    attention_mask = helper.torch.ones(inputs_embeds.shape[:2], dtype=helper.torch.long, device=helper.device)

    n_image = int(image_features_t.shape[1])
    expanded_color_mask = helper.torch.zeros(inputs_embeds.shape[1], dtype=helper.torch.bool, device=helper.device)
    for original_idx in original_color_mask.nonzero(as_tuple=False).flatten().tolist():
        original_idx = int(original_idx)
        if original_idx == pos:
            continue
        expanded_idx = original_idx if original_idx < pos else original_idx + n_image - 1
        expanded_color_mask[expanded_idx] = True
    if not bool(expanded_color_mask.any().item()):
        raise ValueError(f"Could not locate generated color token '{color}' in assistant answer: {answer}")
    return inputs_embeds, attention_mask, expanded_color_mask, text


def extract_token_layers(helper: LlavaHelper, image_path: str, prompt: str, answer: str, color: str, layer_indices: list[int]) -> np.ndarray:
    language_model = helper.language_model()
    inputs_embeds, attention_mask, color_mask, _expanded_text = expanded_assistant_inputs_with_color_mask(helper, image_path, prompt, answer, color)
    layer_vectors = []
    with helper.torch.no_grad():
        outputs = language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        for layer_idx in layer_indices:
            hidden = outputs.hidden_states[layer_idx][0]
            active_mask = color_mask.to(device=hidden.device, dtype=helper.torch.bool).flatten()
            if active_mask.shape[0] != hidden.shape[0]:
                raise ValueError(f"Color mask length mismatch for generated answer '{answer}': mask={active_mask.shape[0]}, hidden={hidden.shape[0]}")
            selected = hidden[active_mask]
            vector = normalize_vector(selected.mean(dim=0))
            layer_vectors.append(vector.detach().cpu().numpy())
    return np.vstack(layer_vectors).astype(np.float32)


def extract_ceg(helper: LlavaHelper, dataset, prompt: str, layer_indices: list[int], max_new_tokens: int, temperature: float):
    tokenizer = helper.tokenizer()
    decision_vectors = []
    token_vectors = []
    valid_rows = []
    all_rows = []
    for row in dataset.rows:
        sample_index = int(row["sample_index"])
        image_path = row["absolute_image_path"]
        true_color = normalize_color(row["basic_color"])
        generated_text, generated_ids, hidden_steps = generate_one(helper, image_path, prompt, max_new_tokens, temperature)
        predicted_color, parse_status = normalize_color_answer(generated_text)
        span_start = span_end = None
        span_status = "no_color"
        decision = None
        token = None
        is_valid = predicted_color in BASIC_COLORS
        if is_valid:
            span_start, span_end, span_status = generated_color_span(tokenizer, generated_ids, predicted_color)
            decision = extract_decision_layers(helper, hidden_steps, span_start, span_end, layer_indices)
            if decision is None:
                is_valid = False
                span_status = f"{span_status};decision_hidden_unavailable"
        if is_valid:
            try:
                token = extract_token_layers(helper, image_path, prompt, generated_text, predicted_color, layer_indices)
            except Exception as exc:
                is_valid = False
                span_status = f"{span_status};token_forward_failed:{type(exc).__name__}"
        record = {
            "sample_index": sample_index,
            "image_path": row.get("image_path", ""),
            "absolute_image_path": image_path,
            "r": row.get("r", ""),
            "g": row.get("g", ""),
            "b": row.get("b", ""),
            "true_color": true_color,
            "generated_text": generated_text,
            "generated_token_ids": " ".join(map(str, generated_ids)),
            "normalized_pred_color": predicted_color or "",
            "parse_status": parse_status,
            "generated_color_token_start": "" if span_start is None else span_start,
            "generated_color_token_end": "" if span_end is None else span_end,
            "generated_color_span_status": span_status,
            "correct": int(predicted_color == true_color),
            "included_in_embedding_analysis": int(is_valid),
        }
        all_rows.append(record)
        if is_valid:
            valid_rows.append(record)
            decision_vectors.append(decision)
            token_vectors.append(token)
    if not valid_rows:
        raise RuntimeError("No generated color tokens could be extracted for CEg analysis.")
    return np.stack(decision_vectors).transpose(1, 0, 2).astype(np.float32), np.stack(token_vectors).transpose(1, 0, 2).astype(np.float32), all_rows, valid_rows


def subset_dataset_rows(dataset, valid_rows: list[dict]) -> list[dict]:
    by_index = {int(row["sample_index"]): row for row in dataset.rows}
    return [by_index[int(row["sample_index"])] for row in valid_rows]


def cielab_rdms(rows_subset: list[dict], labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray([[int(row["r"]), int(row["g"]), int(row["b"])] for row in rows_subset], dtype=float)
    lab = rgb_to_lab(rgb)
    sample = ciede2000_matrix(lab)
    label_indices = labels_to_indices(labels, BASIC_COLORS)
    cent = np.vstack([lab[label_indices == idx].mean(axis=0) for idx in range(len(BASIC_COLORS))])
    return sample, ciede2000_matrix(cent)


def analyze_against_cielab(prefix: str, source_layers: np.ndarray, layer_indices: list[int], labels: list[str], sample_cielab: np.ndarray, centroid_cielab: np.ndarray, ceg_dir: Path, args: argparse.Namespace) -> tuple[Path, list[dict]]:
    rows = []
    rdm_dir = ceg_dir / "rdm" / f"{prefix}_sample_rdms"
    centroid_dir = ceg_dir / "rdm" / f"{prefix}_centroid_rdms"
    rdm_dir.mkdir(parents=True, exist_ok=True)
    centroid_dir.mkdir(parents=True, exist_ok=True)
    for array_idx, layer_idx in enumerate(layer_indices):
        source = source_layers[array_idx]
        source_rdm = rdm(source)
        source_centroids = centroids(source, labels, BASIC_COLORS)
        source_centroid_rdm = rdm(source_centroids)
        np.save(rdm_dir / f"layer_{layer_idx}_sample_rdm.npy", source_rdm)
        save_matrix_csv(centroid_dir / f"layer_{layer_idx}_centroid_rdm.csv", BASIC_COLORS, source_centroid_rdm, "color")
        sample_r, sample_p, sample_n = permutation_pvalue(upper_triangle_values(source_rdm), upper_triangle_values(sample_cielab), spearman_corr, args.permutation_rounds, args.seed + array_idx * 10)
        centroid_r, centroid_p, centroid_n = permutation_pvalue(upper_triangle_values(source_centroid_rdm), upper_triangle_values(centroid_cielab), spearman_corr, args.permutation_rounds, args.seed + 100 + array_idx * 10)
        reg = linear_regression_summary(upper_triangle_values(source_rdm), upper_triangle_values(sample_cielab))
        centroid_reg = linear_regression_summary(upper_triangle_values(source_centroid_rdm), upper_triangle_values(centroid_cielab))
        rows.append(
            {
                "source": prefix,
                "layer_index": int(layer_idx),
                "sample_rsa_spearman": sample_r,
                "sample_rsa_p": sample_p,
                "sample_rsa_n": sample_n,
                "centroid_rsa_spearman": centroid_r,
                "centroid_rsa_p": centroid_p,
                "centroid_rsa_n": centroid_n,
                "sample_distance_regression_r2": reg["r2"],
                "sample_distance_regression_slope": reg["slope"],
                "sample_distance_regression_n": reg["n"],
                "centroid_distance_regression_r2": centroid_reg["r2"],
                "centroid_distance_regression_slope": centroid_reg["slope"],
                "centroid_distance_regression_n": centroid_reg["n"],
            }
        )
    path = ceg_dir / "rdm" / f"{prefix}_cielab_alignment.csv"
    write_csv(
        path,
        [
            "source",
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
        save_line_plot(ceg_dir / "plots" / f"{prefix}_cielab_sample_rsa.png", rows, "layer_index", "sample_rsa_spearman", f"{prefix} vs CIEDE2000 sample RSA", "Spearman r")
        save_line_plot(ceg_dir / "plots" / f"{prefix}_cielab_centroid_rsa.png", rows, "layer_index", "centroid_rsa_spearman", f"{prefix} vs CIEDE2000 centroid RSA", "Spearman r")
    return path, rows


def write_image_alignment(prefix: str, source_layers: np.ndarray, image_layers: np.ndarray, layer_indices: list[int], labels: list[str], ceg_dir: Path, args: argparse.Namespace) -> Path:
    rows = analyze_against_image(prefix, source_layers, image_layers, layer_indices, labels, ceg_dir, args)
    path = ceg_dir / "rdm" / f"{prefix}_image_alignment.csv"
    write_csv(
        path,
        [
            "condition",
            "layer_index",
            "mean_paired_sample_cosine",
            "std_paired_sample_cosine",
            "sample_rsa_spearman",
            "sample_rsa_p",
            "sample_rsa_n",
            "centroid_rsa_spearman",
            "centroid_rsa_p",
            "centroid_rsa_n",
            "linear_cka_centroids",
            "procrustes_error_image_to_answer_centroids",
        ],
        rows,
    )
    if not args.no_plots:
        save_line_plot(ceg_dir / "plots" / f"{prefix}_image_sample_rsa.png", rows, "layer_index", "sample_rsa_spearman", f"{prefix} vs image sample RSA", "Spearman r")
        save_line_plot(ceg_dir / "plots" / f"{prefix}_image_centroid_rsa.png", rows, "layer_index", "centroid_rsa_spearman", f"{prefix} vs image centroid RSA", "Spearman r")
    return path


def subset_layer_array(path: Path, key: str, valid_rows: list[dict]) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.load(path, allow_pickle=True)[key]
    indices = [int(row["sample_index"]) for row in valid_rows]
    return arr[:, indices, :]


def main() -> None:
    args = parse_args()
    output = args.output_dir
    ceg_dir = output / "ceg_generation_side"
    embeddings_dir = ceg_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.ground_truth, args.dataset_root)
    layer_indices = parse_layers(args.layer_indices)

    decision_path = embeddings_dir / "ceg_decision_embeddings.npz"
    token_path = embeddings_dir / "ceg_token_embeddings.npz"
    predictions_path = ceg_dir / "predictions.csv"

    if args.skip_extraction and decision_path.exists() and token_path.exists() and predictions_path.exists():
        decision_npz = np.load(decision_path, allow_pickle=True)
        token_npz = np.load(token_path, allow_pickle=True)
        decision_layers = decision_npz["ceg_decision_layer_embeddings"]
        token_layers = token_npz["ceg_token_layer_embeddings"]
        import csv

        with predictions_path.open(newline="", encoding="utf-8") as handle:
            all_rows = list(csv.DictReader(handle))
        valid_rows = [row for row in all_rows if int(row.get("included_in_embedding_analysis", "0")) == 1]
    else:
        helper = LlavaHelper(args.model_id, args.device, args.dtype, args.seed)
        decision_layers, token_layers, all_rows, valid_rows = extract_ceg(helper, dataset, args.prompt, layer_indices, args.max_new_tokens, args.temperature)
        write_csv(
            predictions_path,
            [
                "sample_index",
                "image_path",
                "absolute_image_path",
                "r",
                "g",
                "b",
                "true_color",
                "generated_text",
                "generated_token_ids",
                "normalized_pred_color",
                "parse_status",
                "generated_color_token_start",
                "generated_color_token_end",
                "generated_color_span_status",
                "correct",
                "included_in_embedding_analysis",
            ],
            all_rows,
        )
        np.savez_compressed(
            decision_path,
            ceg_decision_layer_embeddings=decision_layers,
            layer_indices=np.asarray(layer_indices),
            true_labels=np.asarray([row["true_color"] for row in valid_rows]),
            predicted_labels=np.asarray([row["normalized_pred_color"] for row in valid_rows]),
            sample_indices=np.asarray([int(row["sample_index"]) for row in valid_rows]),
            prompt=args.prompt,
        )
        np.savez_compressed(
            token_path,
            ceg_token_layer_embeddings=token_layers,
            layer_indices=np.asarray(layer_indices),
            true_labels=np.asarray([row["true_color"] for row in valid_rows]),
            predicted_labels=np.asarray([row["normalized_pred_color"] for row in valid_rows]),
            sample_indices=np.asarray([int(row["sample_index"]) for row in valid_rows]),
            prompt=args.prompt,
        )

    true_labels = [normalize_color(row["true_color"]) for row in valid_rows]
    predicted_labels = [normalize_color(row["normalized_pred_color"]) for row in valid_rows]
    rows_subset = subset_dataset_rows(dataset, valid_rows)
    sample_cielab, centroid_cielab = cielab_rdms(rows_subset, true_labels)
    save_matrix_csv(ceg_dir / "rdm" / "cielab_centroid_rdm_for_parseable.csv", BASIC_COLORS, centroid_cielab, "color")
    np.save(ceg_dir / "rdm" / "cielab_sample_rdm_for_parseable.npy", sample_cielab)

    summary = {
        "model_id": args.model_id,
        "ground_truth": str(args.ground_truth),
        "prompt": args.prompt,
        "layer_indices": layer_indices,
        "n_samples": len(dataset.rows),
        "n_parseable_for_embedding_analysis": len(valid_rows),
        "parse_rate_for_embedding_analysis": len(valid_rows) / len(dataset.rows) if dataset.rows else float("nan"),
        "accuracy_among_all_generated": float(np.mean([int(row["correct"]) for row in all_rows])) if all_rows else float("nan"),
        "accuracy_among_included": float(np.mean([int(row["correct"]) for row in valid_rows])) if valid_rows else float("nan"),
        "predictions_csv": str(predictions_path),
        "ceg_decision_npz": str(decision_path),
        "ceg_token_npz": str(token_path),
    }

    text_path = output / "embeddings" / "text_embeddings.npz"
    if not text_path.exists():
        raise FileNotFoundError(f"Text embeddings not found: {text_path}. Run text_embedding_extraction.py first.")
    text_npz = np.load(text_path, allow_pickle=True)
    text_layers = text_npz["text_layer_embeddings"]
    text_layer_indices = [int(x) for x in text_npz["layer_indices"].tolist()]
    if text_layer_indices != layer_indices:
        raise ValueError(f"CEg layer indices {layer_indices} do not match text embedding layers {text_layer_indices}.")

    for prefix, layers in (("ceg_decision", decision_layers), ("ceg_token", token_layers)):
        cielab_path, _rows = analyze_against_cielab(prefix, layers, layer_indices, true_labels, sample_cielab, centroid_cielab, ceg_dir, args)
        summary[f"{prefix}_cielab_alignment_csv"] = str(cielab_path)
        summary.update(analyze_against_color_prototypes(f"{prefix}_vs_text_true_label", layers, text_layers, layer_indices, true_labels, ceg_dir, args))
        summary.update(analyze_against_color_prototypes(f"{prefix}_vs_text_predicted_label", layers, text_layers, layer_indices, predicted_labels, ceg_dir, args))

    image_path = output / "embeddings" / "image_embeddings_no_color_list.npz"
    image_layers = subset_layer_array(image_path, "image_layer_embeddings", valid_rows)
    if image_layers is not None:
        summary["ceg_decision_image_alignment_csv"] = str(write_image_alignment("ceg_decision", decision_layers, image_layers, layer_indices, true_labels, ceg_dir, args))
        summary["ceg_token_image_alignment_csv"] = str(write_image_alignment("ceg_token", token_layers, image_layers, layer_indices, true_labels, ceg_dir, args))
    else:
        summary["image_alignment_warning"] = f"Image embeddings not found: {image_path}"

    cee_path = output / "embeddings" / "cee_ground_truth_embeddings.npz"
    cee_layers = subset_layer_array(cee_path, "cee_layer_embeddings", valid_rows)
    if cee_layers is not None:
        summary["ceg_decision_cee_alignment_csv"] = str(write_image_alignment("ceg_decision_vs_cee", decision_layers, cee_layers, layer_indices, true_labels, ceg_dir, args))
        summary["ceg_token_cee_alignment_csv"] = str(write_image_alignment("ceg_token_vs_cee", token_layers, cee_layers, layer_indices, true_labels, ceg_dir, args))
    else:
        summary["cee_alignment_warning"] = f"CEe embeddings not found: {cee_path}"

    answer_path = output / "embeddings" / "answer_position_embeddings_no_color_list.npz"
    answer_layers = subset_layer_array(answer_path, "answer_position_layer_embeddings", valid_rows)
    if answer_layers is not None:
        summary["ceg_decision_answer_alignment_csv"] = str(write_image_alignment("ceg_decision_vs_pre_answer", decision_layers, answer_layers, layer_indices, true_labels, ceg_dir, args))
        summary["ceg_token_answer_alignment_csv"] = str(write_image_alignment("ceg_token_vs_pre_answer", token_layers, answer_layers, layer_indices, true_labels, ceg_dir, args))
    else:
        summary["answer_alignment_warning"] = f"Answer-position embeddings not found: {answer_path}"

    write_json(ceg_dir / "summary.json", summary)
    print(f"Wrote CEg generation-side outputs to: {ceg_dir}")


if __name__ == "__main__":
    main()
