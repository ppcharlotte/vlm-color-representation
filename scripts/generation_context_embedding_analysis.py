#!/usr/bin/env python3
"""Extract and analyze generation-side color embeddings from LLaVA.

This script keeps two generation-side representations:

1. CEg-decision: the hidden state used to predict the generated color token.
2. CEg-token-from-generate: the generated color token hidden state taken from
   the original generate() pass, using the following generation step.

No re-forwarded assistant answer is used here.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from representation_analysis_utils import analyze_against_color_prototypes, analyze_against_embedding_target
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
    parser = argparse.ArgumentParser(description="Extract and analyze generation-side color representations.")
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


def extract_decision_layers(helper: LlavaHelper, hidden_steps, start: int | None, end: int | None, layer_indices: list[int]) -> tuple[np.ndarray | None, str]:
    if start is None or end is None:
        return None, "no_color_token_span"
    if start < 0 or end <= start:
        return None, "invalid_color_token_span"
    if end > len(hidden_steps):
        return None, "decision_hidden_unavailable"

    layer_vectors = []
    for layer_idx in layer_indices:
        pieces = []
        for generated_token_idx in range(start, end):
            step_hidden = hidden_steps[generated_token_idx][layer_idx]
            pieces.append(step_hidden[:, -1, :])
        vector = normalize_vector(helper.torch.cat(pieces, dim=0).mean(dim=0))
        layer_vectors.append(vector.detach().cpu().numpy())
    return np.vstack(layer_vectors).astype(np.float32), "predict_color_step"


def extract_token_from_generate_layers(helper: LlavaHelper, hidden_steps, start: int | None, end: int | None, layer_indices: list[int]) -> tuple[np.ndarray | None, str]:
    if start is None or end is None:
        return None, "no_color_token_span"
    if start < 0 or end <= start:
        return None, "invalid_color_token_span"
    if end >= len(hidden_steps):
        return None, "no_next_generation_step_after_color_token"

    layer_vectors = []
    for layer_idx in layer_indices:
        pieces = []
        for generated_token_idx in range(start, end):
            next_step_idx = generated_token_idx + 1
            step_hidden = hidden_steps[next_step_idx][layer_idx]
            pieces.append(step_hidden[:, -1, :])
        vector = normalize_vector(helper.torch.cat(pieces, dim=0).mean(dim=0))
        layer_vectors.append(vector.detach().cpu().numpy())
    return np.vstack(layer_vectors).astype(np.float32), "generated_token_next_step"


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
        decision_source = ""
        token_source = ""
        decision = None
        token = None
        is_valid = predicted_color in BASIC_COLORS

        if is_valid:
            span_start, span_end, span_status = generated_color_span(tokenizer, generated_ids, predicted_color)
            decision, decision_source = extract_decision_layers(helper, hidden_steps, span_start, span_end, layer_indices)
            token, token_source = extract_token_from_generate_layers(helper, hidden_steps, span_start, span_end, layer_indices)
            if decision is None or token is None:
                is_valid = False
                span_status = f"{span_status};decision={decision_source};token={token_source}"

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
            "decision_embedding_source": decision_source,
            "token_from_generate_embedding_source": token_source,
            "correct": int(predicted_color == true_color),
            "included_in_embedding_analysis": int(is_valid),
        }
        all_rows.append(record)
        if is_valid:
            valid_rows.append(record)
            decision_vectors.append(decision)
            token_vectors.append(token)

    if not valid_rows:
        raise RuntimeError("No generated color tokens had both decision and token-from-generate embeddings.")
    decision_layers = np.stack(decision_vectors).transpose(1, 0, 2).astype(np.float32)
    token_layers = np.stack(token_vectors).transpose(1, 0, 2).astype(np.float32)
    return decision_layers, token_layers, all_rows, valid_rows


def read_prediction_rows(path: Path) -> tuple[list[dict], list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    valid_rows = [row for row in all_rows if int(row.get("included_in_embedding_analysis", "0")) == 1]
    return all_rows, valid_rows


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


def write_alignment(prefix: str, source_layers: np.ndarray, target_layers: np.ndarray, layer_indices: list[int], labels: list[str], ceg_dir: Path, args: argparse.Namespace, target_label: str) -> Path:
    rows = analyze_against_embedding_target(prefix, source_layers, target_layers, layer_indices, labels, ceg_dir, args)
    file_label = target_label.lower().replace(" ", "_").replace("-", "_")
    path = ceg_dir / "rdm" / f"{prefix}_{file_label}_alignment.csv"
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
            "procrustes_error_target_to_source_centroids",
        ],
        rows,
    )
    if not args.no_plots:
        save_line_plot(ceg_dir / "plots" / f"{prefix}_{file_label}_sample_rsa.png", rows, "layer_index", "sample_rsa_spearman", f"{prefix} vs {target_label} sample RSA", "Spearman r")
        save_line_plot(ceg_dir / "plots" / f"{prefix}_{file_label}_centroid_rsa.png", rows, "layer_index", "centroid_rsa_spearman", f"{prefix} vs {target_label} centroid RSA", "Spearman r")
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
    (ceg_dir / "rdm").mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.ground_truth, args.dataset_root)
    layer_indices = parse_layers(args.layer_indices)

    decision_path = embeddings_dir / "ceg_decision_embeddings.npz"
    token_path = embeddings_dir / "ceg_token_from_generate_embeddings.npz"
    predictions_path = ceg_dir / "predictions.csv"

    if args.skip_extraction and decision_path.exists() and token_path.exists() and predictions_path.exists():
        decision_npz = np.load(decision_path, allow_pickle=True)
        token_npz = np.load(token_path, allow_pickle=True)
        decision_layers = decision_npz["ceg_decision_layer_embeddings"]
        token_layers = token_npz["ceg_token_from_generate_layer_embeddings"]
        all_rows, valid_rows = read_prediction_rows(predictions_path)
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
                "decision_embedding_source",
                "token_from_generate_embedding_source",
                "correct",
                "included_in_embedding_analysis",
            ],
            all_rows,
        )
        common_meta = {
            "layer_indices": np.asarray(layer_indices),
            "true_labels": np.asarray([row["true_color"] for row in valid_rows]),
            "predicted_labels": np.asarray([row["normalized_pred_color"] for row in valid_rows]),
            "sample_indices": np.asarray([int(row["sample_index"]) for row in valid_rows]),
            "prompt": args.prompt,
        }
        np.savez_compressed(decision_path, ceg_decision_layer_embeddings=decision_layers, **common_meta)
        np.savez_compressed(token_path, ceg_token_from_generate_layer_embeddings=token_layers, **common_meta)

    true_labels = [normalize_color(row["true_color"]) for row in valid_rows]
    predicted_labels = [normalize_color(row["normalized_pred_color"]) for row in valid_rows]
    rows_subset = subset_dataset_rows(dataset, valid_rows)
    sample_cielab, centroid_cielab = cielab_rdms(rows_subset, true_labels)
    save_matrix_csv(ceg_dir / "rdm" / "cielab_centroid_rdm_for_included_samples.csv", BASIC_COLORS, centroid_cielab, "color")
    np.save(ceg_dir / "rdm" / "cielab_sample_rdm_for_included_samples.npy", sample_cielab)

    summary = {
        "model_id": args.model_id,
        "ground_truth": str(args.ground_truth),
        "prompt": args.prompt,
        "layer_indices": layer_indices,
        "n_samples": len(dataset.rows),
        "n_included_for_embedding_analysis": len(valid_rows),
        "included_rate_for_embedding_analysis": len(valid_rows) / len(dataset.rows) if dataset.rows else float("nan"),
        "accuracy_among_all_generated": float(np.mean([int(row["correct"]) for row in all_rows])) if all_rows else float("nan"),
        "accuracy_among_included": float(np.mean([int(row["correct"]) for row in valid_rows])) if valid_rows else float("nan"),
        "predictions_csv": str(predictions_path),
        "ceg_decision_npz": str(decision_path),
        "ceg_token_from_generate_npz": str(token_path),
        "embedding_definitions": {
            "ceg_decision": "hidden state used by the LM head to predict the generated color token",
            "ceg_token_from_generate": "hidden state of the generated color token from the original generate pass, taken at the following generation step",
        },
    }

    text_path = output / "embeddings" / "text_embeddings.npz"
    if not text_path.exists():
        raise FileNotFoundError(f"Text embeddings not found: {text_path}. Run text_embedding_extraction.py first.")
    text_npz = np.load(text_path, allow_pickle=True)
    text_layers = text_npz["text_layer_embeddings"]
    text_layer_indices = [int(x) for x in text_npz["layer_indices"].tolist()]
    if text_layer_indices != layer_indices:
        raise ValueError(f"CEg layer indices {layer_indices} do not match text embedding layers {text_layer_indices}.")

    image_path = output / "embeddings" / "image_embeddings_no_color_list.npz"
    image_layers = subset_layer_array(image_path, "image_layer_embeddings", valid_rows)

    for prefix, layers in (("ceg_decision", decision_layers), ("ceg_token_from_generate", token_layers)):
        cielab_path, _rows = analyze_against_cielab(prefix, layers, layer_indices, true_labels, sample_cielab, centroid_cielab, ceg_dir, args)
        summary[f"{prefix}_cielab_alignment_csv"] = str(cielab_path)
        summary.update(analyze_against_color_prototypes(f"{prefix}_vs_text_true_label", layers, text_layers, layer_indices, true_labels, ceg_dir, args))
        summary.update(analyze_against_color_prototypes(f"{prefix}_vs_text_predicted_label", layers, text_layers, layer_indices, predicted_labels, ceg_dir, args))
        if image_layers is not None:
            summary[f"{prefix}_image_alignment_csv"] = str(write_alignment(prefix, layers, image_layers, layer_indices, true_labels, ceg_dir, args, "image"))

    if image_layers is None:
        summary["image_alignment_warning"] = f"Image embeddings not found: {image_path}"

    write_json(ceg_dir / "summary.json", summary)
    print(f"Wrote CEg generation-side outputs to: {ceg_dir}")


if __name__ == "__main__":
    main()
