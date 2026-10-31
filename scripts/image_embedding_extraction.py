#!/usr/bin/env python3
"""Extract image representations once under a neutral color prompt."""

from __future__ import annotations

import argparse
from layer_selection import resolve_layers
from pathlib import Path

import numpy as np

from structured_color_utils import NO_COLOR_LIST_PROMPT, LlavaHelper, load_dataset, write_json


def parse_layers(text: str, model_id=None) -> list[int]:
    return resolve_layers(text, model_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract layer-wise image-token embeddings.")
    parser.add_argument("--ground-truth", type=Path, default=Path("experiments/perceptual_alignment/data/ground_truth.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("experiments/perceptual_alignment/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/perceptual_alignment"))
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--layer-indices", default="all", help="all LM hidden states (default), or comma-separated indices")
    parser.add_argument("--pooling", choices=("mean", "first"), default="mean")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.ground_truth, args.dataset_root)
    image_paths = [row["absolute_image_path"] for row in dataset.rows]
    layer_indices = parse_layers(args.layer_indices, args.model_id)
    helper = LlavaHelper(args.model_id, args.device, args.dtype, args.seed)
    pre, post, image_layers, query_layers = helper.layerwise_image_and_query_embeddings(
        image_paths, NO_COLOR_LIST_PROMPT, layer_indices, args.pooling
    )

    embeddings_dir = args.output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    path = embeddings_dir / "image_query_embeddings.npz"
    np.savez_compressed(
        path,
        pre_projector=pre,
        post_projector=post,
        image_layer_embeddings=image_layers,
        query_layer_embeddings=query_layers,
        layer_indices=np.asarray(layer_indices),
        image_paths=np.asarray(image_paths),
        human_probabilities=dataset.human_probabilities,
        prompt=NO_COLOR_LIST_PROMPT,
        is_onevision=int(helper.is_onevision),
        projector_capture_version=np.asarray(2),
        model_id=np.asarray(args.model_id),
        sample_indices=np.arange(len(dataset.rows)),
        rgb=dataset.rgb,
        pooling=np.asarray(args.pooling),
        seed=np.asarray(args.seed),
    )
    write_json(
        embeddings_dir / "image_embedding_summary.json",
        {
            "model_id": args.model_id,
            "n_samples": len(dataset.rows),
            "layer_indices": layer_indices,
            "prompt": NO_COLOR_LIST_PROMPT,
            "embeddings_npz": str(path),
            "note": "A single neutral prompt is used to avoid the duplicate force-choice forward pass.",
            "projector_stage_note": (
                "For OneVision, pre/post arrays are captured from the input and output of the model's actual "
                "multimodal projector during the same forward pass; layer 0 remains the post-projection "
                "language-model input after model-specific sequence construction."
                if helper.is_onevision
                else "For LLaVA-1.5, pre/post arrays are the selected vision-tower features and their projected features."
            ),
        },
    )
    print(f"Wrote image and query embeddings to: {path}")


if __name__ == "__main__":
    main()
