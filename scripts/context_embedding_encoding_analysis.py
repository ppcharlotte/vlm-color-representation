#!/usr/bin/env python3
"""Extract CEe embeddings for ground-truth color words in image context.

CEe means Context Embedding at the encoding phase. For each color patch, this
script feeds the image together with a declarative sentence such as
"The color of this image is blue" and extracts the hidden state of the color
word itself. This keeps the color word controlled by the ground-truth label and
separates encoding-side color-word representations from generation-side states.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from representation_analysis_utils import (
    analyze_against_color_prototypes,
    analyze_against_embedding_target,
    load_or_compute_cielab,
)
from structured_color_utils import (
    BASIC_COLORS,
    LlavaHelper,
    centroids,
    linear_regression_summary,
    load_dataset,
    permutation_pvalue,
    rdm,
    save_line_plot,
    save_matrix_csv,
    spearman_corr,
    upper_triangle_values,
    write_csv,
    write_json,
)


def parse_layers(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CEe ground-truth color-word embeddings and analyze their alignment.")
    parser.add_argument("--ground-truth", type=Path, default=Path("experiments/llava_structured_alignment/data/ground_truth.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("experiments/llava_structured_alignment/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/llava_structured_alignment"))
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--layer-indices", default="0,1,8,16,24,32")
    parser.add_argument("--template", default="The color of this image is {color}")
    parser.add_argument("--permutation-rounds", type=int, default=1000)
    parser.add_argument("--linear-probe-test-size", type=float, default=0.3)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-extraction", action="store_true", help="Reuse existing CEe embeddings if available.")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def expanded_language_inputs_with_color_mask(helper: LlavaHelper, image_path: str, statement: str, color: str):
    """Build image-expanded LLM inputs and locate the color word after expansion."""
    embedding_layer = helper.embedding_layer()
    tokenizer = helper.tokenizer()
    image_token_id = helper.image_token_id()

    _pre, projected = helper.vision_features([image_path])
    image_features_t = helper.torch.from_numpy(projected).to(helper.device, dtype=embedding_layer.weight.dtype)

    text = helper.chat_prompt(statement)
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

    n_image = image_features_t.shape[1]
    expanded_color_mask = helper.torch.zeros(inputs_embeds.shape[1], dtype=helper.torch.bool, device=helper.device)
    for original_idx in original_color_mask.nonzero(as_tuple=False).flatten().tolist():
        original_idx = int(original_idx)
        if original_idx == pos:
            continue
        expanded_idx = original_idx if original_idx < pos else original_idx + n_image - 1
        expanded_color_mask[expanded_idx] = True
    if not bool(expanded_color_mask.any().item()):
        raise ValueError(f"Could not map color token '{color}' to expanded prompt: {text}")
    return inputs_embeds, attention_mask, expanded_color_mask, (pos, pos + n_image), text


def extract_cee_layers(helper: LlavaHelper, image_paths: list[str], labels: list[str], layer_indices: list[int], template: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    language_model = helper.language_model()
    layer_values = [[] for _ in layer_indices]
    image_layer_values = [[] for _ in layer_indices]
    prompt_rows = []
    with helper.torch.no_grad():
        for sample_idx, (image_path, color) in enumerate(zip(image_paths, labels)):
            statement = template.format(color=color)
            inputs_embeds, attention_mask, color_mask, image_span, expanded_text = expanded_language_inputs_with_color_mask(helper, image_path, statement, color)
            outputs = language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            for out_idx, layer_idx in enumerate(layer_indices):
                hidden = outputs.hidden_states[layer_idx][0]
                active_color_mask = color_mask.to(device=hidden.device, dtype=helper.torch.bool).flatten()
                if active_color_mask.shape[0] != hidden.shape[0]:
                    raise ValueError(
                        f"Color-token mask length does not match hidden states for sample={sample_idx}, "
                        f"color={color}, layer={layer_idx}: mask={active_color_mask.shape[0]}, hidden={hidden.shape[0]}"
                    )
                selected = hidden[active_color_mask]
                if selected.shape[0] == 0:
                    raise ValueError(f"No color-token hidden states selected for sample={sample_idx}, color={color}, layer={layer_idx}")
                vector = selected.mean(dim=0)
                vector = vector / vector.norm(dim=-1, keepdim=True)
                layer_values[out_idx].append(vector.detach().float().cpu().numpy())
                image_tokens = hidden[image_span[0] : image_span[1]]
                image_vector = image_tokens.mean(dim=0)
                image_vector = image_vector / image_vector.norm(dim=-1, keepdim=True)
                image_layer_values[out_idx].append(image_vector.detach().float().cpu().numpy())
            prompt_rows.append(
                {
                    "sample_index": sample_idx,
                    "true_color": color,
                    "image_path": image_path,
                    "statement": statement,
                    "expanded_prompt": expanded_text,
                    "n_color_tokens": int(color_mask.sum().item()),
                }
            )
    return (
        np.stack([np.vstack(values) for values in layer_values]).astype(np.float32),
        np.stack([np.vstack(values) for values in image_layer_values]).astype(np.float32),
        prompt_rows,
    )



def analyze_cee_against_cielab(condition: str, cee_layers: np.ndarray, layer_indices: list[int], labels: list[str], cielab_sample_rdm: np.ndarray, cielab_centroid_rdm: np.ndarray, output: Path, args: argparse.Namespace) -> list[dict]:
    rows = []
    rdm_dir = output / "rdm" / f"{condition}_sample_rdms"
    centroid_dir = output / "rdm" / f"{condition}_centroid_rdms"
    rdm_dir.mkdir(parents=True, exist_ok=True)
    centroid_dir.mkdir(parents=True, exist_ok=True)
    for array_idx, layer_idx in enumerate(layer_indices):
        cee = cee_layers[array_idx]
        cee_rdm = rdm(cee)
        cee_centroids = centroids(cee, labels, BASIC_COLORS)
        cee_centroid_rdm = rdm(cee_centroids)
        np.save(rdm_dir / f"layer_{layer_idx}_sample_rdm.npy", cee_rdm)
        save_matrix_csv(centroid_dir / f"layer_{layer_idx}_centroid_rdm.csv", BASIC_COLORS, cee_centroid_rdm, "color")
        sample_r, sample_p, sample_n = permutation_pvalue(upper_triangle_values(cee_rdm), upper_triangle_values(cielab_sample_rdm), spearman_corr, args.permutation_rounds, args.seed + array_idx * 10)
        centroid_r, centroid_p, centroid_n = permutation_pvalue(upper_triangle_values(cee_centroid_rdm), upper_triangle_values(cielab_centroid_rdm), spearman_corr, args.permutation_rounds, args.seed + 100 + array_idx * 10)
        reg = linear_regression_summary(upper_triangle_values(cee_rdm), upper_triangle_values(cielab_sample_rdm))
        centroid_reg = linear_regression_summary(upper_triangle_values(cee_centroid_rdm), upper_triangle_values(cielab_centroid_rdm))
        rows.append(
            {
                "condition": condition,
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
    return rows


def load_image_layers(output: Path, preferred: str = "no_color_list") -> np.ndarray | None:
    path = output / "embeddings" / f"image_embeddings_{preferred}.npz"
    if not path.exists():
        fallback = output / "embeddings" / "image_embeddings_force_choice.npz"
        path = fallback if fallback.exists() else path
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)["image_layer_embeddings"]


def main() -> None:
    args = parse_args()
    output = args.output_dir
    layer_indices = parse_layers(args.layer_indices)
    dataset = load_dataset(args.ground_truth, args.dataset_root)
    labels = dataset.labels
    image_paths = [row["absolute_image_path"] for row in dataset.rows]

    embeddings_dir = output / "embeddings"
    encoding_dir = output / "encoding_side"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    encoding_dir.mkdir(parents=True, exist_ok=True)

    cee_path = embeddings_dir / "cee_ground_truth_embeddings.npz"
    prompts_path = encoding_dir / "cee_ground_truth_prompts.csv"
    cee_layers = None
    cee_image_layers = None
    prompt_rows = []
    if args.skip_extraction and cee_path.exists():
        cee_npz = np.load(cee_path, allow_pickle=True)
        cee_layers = cee_npz["cee_layer_embeddings"]
        cee_image_layers = cee_npz["cee_image_layer_embeddings"] if "cee_image_layer_embeddings" in cee_npz.files else None
    if cee_layers is None or cee_image_layers is None:
        helper = LlavaHelper(args.model_id, args.device, args.dtype, args.seed)
        cee_layers, cee_image_layers, prompt_rows = extract_cee_layers(helper, image_paths, labels, layer_indices, args.template)
        np.savez_compressed(
            cee_path,
            cee_layer_embeddings=cee_layers,
            cee_image_layer_embeddings=cee_image_layers,
            layer_indices=np.asarray(layer_indices),
            labels=np.asarray(labels),
            image_paths=np.asarray(image_paths),
            template=args.template,
        )
        write_csv(prompts_path, ["sample_index", "true_color", "image_path", "statement", "expanded_prompt", "n_color_tokens"], prompt_rows)

    text_path = embeddings_dir / "text_embeddings.npz"
    if not text_path.exists():
        raise FileNotFoundError(f"Text embeddings not found: {text_path}. Run text_embedding_extraction.py first.")
    text_npz = np.load(text_path, allow_pickle=True)
    text_layers = text_npz["text_layer_embeddings"]
    text_layer_indices = [int(x) for x in text_npz["layer_indices"].tolist()]
    if text_layer_indices != layer_indices:
        raise ValueError(f"CEe layer indices {layer_indices} do not match text embedding layers {text_layer_indices}.")

    summary = {
        "model_id": args.model_id,
        "ground_truth": str(args.ground_truth),
        "cee_ground_truth_npz": str(cee_path),
        "cee_ground_truth_prompts_csv": str(prompts_path),
        "cee_image_embeddings_in_same_context": bool(cee_image_layers is not None),
        "template": args.template,
        "layer_indices": layer_indices,
        "n_samples": len(labels),
    }

    summary.update(analyze_against_color_prototypes("cee_ground_truth", cee_layers, text_layers, layer_indices, labels, output, args))

    cielab_sample_rdm, cielab_centroid_rdm = load_or_compute_cielab(args, output, dataset)
    cielab_rows = analyze_cee_against_cielab("cee_ground_truth", cee_layers, layer_indices, labels, cielab_sample_rdm, cielab_centroid_rdm, output, args)
    cielab_path = encoding_dir / "cee_cielab_alignment.csv"
    write_csv(
        cielab_path,
        [
            "condition",
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
        cielab_rows,
    )
    summary["cee_cielab_alignment_csv"] = str(cielab_path)

    image_layers = cee_image_layers if cee_image_layers is not None else load_image_layers(output, "no_color_list")
    if image_layers is not None:
        image_rows = analyze_against_embedding_target("cee_ground_truth", cee_layers, image_layers, layer_indices, labels, output, args)
        image_path = encoding_dir / "cee_image_alignment.csv"
        write_csv(
            image_path,
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
            image_rows,
        )
        summary["cee_image_alignment_csv"] = str(image_path)
    else:
        image_rows = []
        summary["cee_image_alignment_warning"] = "Image embeddings not found; skipped CEe-image alignment."

    if not args.no_plots:
        save_line_plot(output / "plots" / "cee_ground_truth_cielab_centroid_rsa.png", cielab_rows, "layer_index", "centroid_rsa_spearman", "CEe ground-truth vs CIEDE2000 centroid RSA", "Spearman r")
        save_line_plot(output / "plots" / "cee_ground_truth_cielab_sample_rsa.png", cielab_rows, "layer_index", "sample_rsa_spearman", "CEe ground-truth vs CIEDE2000 sample RSA", "Spearman r")
        if image_rows:
            save_line_plot(output / "plots" / "cee_ground_truth_image_centroid_rsa.png", image_rows, "layer_index", "centroid_rsa_spearman", "CEe ground-truth vs image centroid RSA", "Spearman r")
            save_line_plot(output / "plots" / "cee_ground_truth_image_sample_rsa.png", image_rows, "layer_index", "sample_rsa_spearman", "CEe ground-truth vs image sample RSA", "Spearman r")

    write_json(encoding_dir / "cee_ground_truth_summary.json", summary)
    print(f"Wrote CEe ground-truth embeddings and analyses to: {encoding_dir}")


if __name__ == "__main__":
    main()
