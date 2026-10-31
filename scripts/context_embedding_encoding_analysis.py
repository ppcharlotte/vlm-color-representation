#!/usr/bin/env python3
"""Extract encoding-side contextual color-word (CEe) embeddings."""

from __future__ import annotations

import argparse
from layer_selection import resolve_layers
from pathlib import Path

import numpy as np

from structured_color_utils import LlavaHelper, load_dataset, write_csv, write_json


def parse_layers(text: str, model_id=None) -> list[int]:
    return resolve_layers(text, model_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CEe color-word and same-context image-token embeddings.")
    parser.add_argument("--ground-truth", type=Path, default=Path("experiments/perceptual_alignment/data/ground_truth.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("experiments/perceptual_alignment/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/perceptual_alignment"))
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--layer-indices", default="all", help="all LM hidden states (default), or comma-separated indices")
    parser.add_argument("--template", default="The color of this image is {color}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mismatch-repeats", type=int, default=5)
    parser.add_argument("--mismatch-seed", type=int, default=None)
    parser.add_argument("--skip-extraction", action="store_true", help="Reuse existing CEe embeddings when present.")
    return parser.parse_args()


def extract_cee_layers(
    helper: LlavaHelper,
    image_paths: list[str],
    labels: list[str],
    layer_indices: list[int],
    template: str,
    return_raw_images: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    color_layers = [[] for _ in layer_indices]
    image_layers = [[] for _ in layer_indices]
    raw_image_layers = [[] for _ in layer_indices]
    prompt_rows = []
    text_layers = [[] for _ in layer_indices]
    text_cache = {}
    with helper.torch.no_grad():
        for sample_idx, (image_path, color) in enumerate(zip(image_paths, labels)):
            statement = template.format(color=color)
            if color not in text_cache:
                text_model, text_inputs, text_mask, text_prompt = helper.text_context_inputs_with_color_mask(statement, color)
                text_outputs = text_model(**text_inputs, output_hidden_states=True, use_cache=False, return_dict=True)
                vectors = []
                for layer_idx in layer_indices:
                    hidden = text_outputs.hidden_states[layer_idx][0]
                    vector = hidden[text_mask.to(device=hidden.device, dtype=helper.torch.bool)].mean(dim=0)
                    vector = vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    vectors.append(vector.detach().float().cpu().numpy())
                text_cache[color] = (vectors, text_prompt)
            for out_idx, vector in enumerate(text_cache[color][0]):
                text_layers[out_idx].append(vector)
            model, model_inputs, color_mask, image_mask, expanded_text = helper.context_inputs_with_color_mask(
                image_path, statement, color
            )
            outputs = model(**model_inputs, output_hidden_states=True, use_cache=False, return_dict=True)
            for out_idx, layer_idx in enumerate(layer_indices):
                hidden = outputs.hidden_states[layer_idx][0]
                active_color_mask = color_mask.to(device=hidden.device, dtype=helper.torch.bool).flatten()
                active_image_mask = image_mask.to(device=hidden.device, dtype=helper.torch.bool).flatten()
                if active_color_mask.shape[0] != hidden.shape[0] or active_image_mask.shape[0] != hidden.shape[0]:
                    raise ValueError(
                        f"Token-mask length mismatch for sample={sample_idx}, color={color}, layer={layer_idx}: "
                        f"hidden={hidden.shape[0]}, color_mask={active_color_mask.shape[0]}, image_mask={active_image_mask.shape[0]}"
                    )
                color_tokens = hidden[active_color_mask]
                image_tokens = hidden[active_image_mask]
                if not color_tokens.shape[0] or not image_tokens.shape[0]:
                    raise ValueError(f"Missing CEe color/image tokens for sample={sample_idx}, layer={layer_idx}")
                color_vector = color_tokens.mean(dim=0)
                color_vector = color_vector / color_vector.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                image_vector = image_tokens.mean(dim=0)
                raw_image_layers[out_idx].append(image_vector.detach().float().cpu().numpy())
                image_vector = image_vector / image_vector.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                color_layers[out_idx].append(color_vector.detach().float().cpu().numpy())
                image_layers[out_idx].append(image_vector.detach().float().cpu().numpy())
            prompt_rows.append(
                {
                    "sample_index": sample_idx,
                    "context_color": color,
                    "image_path": image_path,
                    "statement": statement,
                    "expanded_prompt": expanded_text,
                    "matched_text_prompt": text_cache[color][1],
                    "n_color_tokens": int(color_mask.sum().item()),
                }
            )
    result = (
        np.stack([np.vstack(values) for values in color_layers]).astype(np.float32),
        np.stack([np.vstack(values) for values in image_layers]).astype(np.float32),
        np.stack([np.vstack(values) for values in text_layers]).astype(np.float32),
        prompt_rows,
    )
    if return_raw_images:
        return (*result, np.stack([np.vstack(values) for values in raw_image_layers]).astype(np.float32))
    return result


def main() -> None:
    args = parse_args()
    layer_indices = parse_layers(args.layer_indices, args.model_id)
    dataset = load_dataset(args.ground_truth, args.dataset_root)
    labels = dataset.labels
    image_paths = [row["absolute_image_path"] for row in dataset.rows]
    embeddings_dir = args.output_dir / "embeddings"
    encoding_dir = args.output_dir / "encoding_side"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    encoding_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = embeddings_dir / "cee_ground_truth_embeddings.npz"
    prompts_path = encoding_dir / "cee_ground_truth_prompts.csv"
    helper = None

    if not (args.skip_extraction and embeddings_path.exists()):
        helper = LlavaHelper(args.model_id, args.device, args.dtype, args.seed)
        cee_layers, cee_image_layers, matched_text_layers, prompt_rows, raw_image_layers = extract_cee_layers(
            helper, image_paths, labels, layer_indices, args.template, return_raw_images=True
        )
        np.savez_compressed(
            embeddings_path,
            cee_layer_embeddings=cee_layers,
            cee_image_layer_embeddings=cee_image_layers,
            cee_raw_image_layer_embeddings=raw_image_layers,
            image_pooling=np.asarray("mean"),
            seed=np.asarray(args.seed),
            matched_text_layer_embeddings=matched_text_layers,
            matched_design_version=np.asarray(1),
            model_id=np.asarray(args.model_id),
            sample_indices=np.arange(len(dataset.rows)),
            layer_indices=np.asarray(layer_indices),
            labels=np.asarray(labels),
            image_paths=np.asarray(image_paths),
            template=args.template,
        )
        write_csv(
            prompts_path,
            ["sample_index", "context_color", "image_path", "statement", "expanded_prompt", "matched_text_prompt", "n_color_tokens"],
            prompt_rows,
        )

    else:
        with np.load(embeddings_path, allow_pickle=False) as saved:
            if "matched_text_layer_embeddings" not in saved or str(saved["template"]) != args.template or str(saved["model_id"]) != args.model_id or saved["layer_indices"].tolist() != layer_indices or saved["image_paths"].tolist() != image_paths or saved["labels"].tolist() != labels:
                raise ValueError("Cached CEe is legacy or incompatible; rerun without --skip-extraction.")

    from mismatched_color_control import mismatched_assignments
    mismatch_seed = args.seed + 100000 if args.mismatch_seed is None else args.mismatch_seed
    assignments = mismatched_assignments(labels, args.mismatch_repeats, mismatch_seed)
    mismatch_path = embeddings_dir / "cee_mismatched_embeddings.npz"
    mapping_path = encoding_dir / "cee_mismatched_pairings.csv"
    if args.mismatch_repeats:
        reuse = args.skip_extraction and mismatch_path.exists() and mapping_path.exists()
        if reuse:
            with np.load(mismatch_path, allow_pickle=False) as saved:
                if (str(saved["model_id"]) != args.model_id or str(saved["template"]) != args.template
                        or saved["image_paths"].tolist() != image_paths or saved["labels"].tolist() != labels
                        or saved["layer_indices"].tolist() != layer_indices
                        or not np.array_equal(saved["donor_indices"], assignments)):
                    raise ValueError("Mismatched cache does not match this design; rerun without --skip-extraction")
        else:
            if helper is None:
                helper = LlavaHelper(args.model_id, args.device, args.dtype, args.seed)
            wrong_layers, mapping_rows = [], []
            for repeat, donor_indices in enumerate(assignments):
                print(f"Extracting mismatched-color repeat {repeat + 1}/{args.mismatch_repeats}", flush=True)
                donor_paths = [image_paths[int(i)] for i in donor_indices]
                vectors, _, _, prompts = extract_cee_layers(helper, donor_paths, labels, layer_indices, args.template)
                wrong_layers.append(vectors)
                for i, donor in enumerate(donor_indices):
                    mapping_rows.append(dict(repeat=repeat, mismatch_seed=mismatch_seed,
                        sample_index=i, donor_index=int(donor), target_color=labels[i], donor_color=labels[int(donor)],
                        target_image_path=image_paths[i], donor_image_path=donor_paths[i],
                        statement=prompts[i]["statement"], expanded_prompt=prompts[i]["expanded_prompt"],
                        n_color_tokens=prompts[i]["n_color_tokens"]))
            np.savez_compressed(mismatch_path, mismatched_layer_embeddings=np.stack(wrong_layers),
                donor_indices=assignments, layer_indices=np.asarray(layer_indices), image_paths=np.asarray(image_paths),
                labels=np.asarray(labels), model_id=np.asarray(args.model_id), template=np.asarray(args.template),
                mismatch_seed=np.asarray(mismatch_seed), matched_design_version=np.asarray(1))
            write_csv(mapping_path, list(mapping_rows[0]), mapping_rows)

    write_json(
        encoding_dir / "cee_ground_truth_summary.json",
        {
            "model_id": args.model_id,
            "cee_embeddings_npz": str(embeddings_path),
            "prompts_csv": str(prompts_path),
            "layer_indices": layer_indices,
            "n_samples": len(dataset.rows),
            "template": args.template,
            "mismatch_repeats": args.mismatch_repeats,
            "mismatch_seed": mismatch_seed,
            "mismatch_embeddings": str(mismatch_path) if args.mismatch_repeats else None,
            "mismatch_pairings": str(mapping_path) if args.mismatch_repeats else None,
            "matched_text_baseline": "Same statement and chat formatting; image content omitted. Image-token positions necessarily differ.",
            "context_label_note": "The majority human color term conditions the CEe prompt; alignment statistics are computed separately.",
            "analysis_script": "perceptual_alignment_analysis.py",
        },
    )
    print(f"Wrote CEe embeddings to: {embeddings_path}")


if __name__ == "__main__":
    main()
