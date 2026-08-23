#!/usr/bin/env python3
"""Extract or materialize force-choice and no-color-list query embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from structured_color_utils import FORCE_CHOICE_PROMPT, NO_COLOR_LIST_PROMPT, LlavaHelper, load_dataset, rdm, save_matrix_csv, write_json


def parse_layers(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract or save query-token embeddings for both prompt conditions.")
    parser.add_argument("--ground-truth", type=Path, default=Path("experiments/llava_structured_alignment/data/ground_truth.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("experiments/llava_structured_alignment/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/llava_structured_alignment"))
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--layer-indices", default="0,1,8,16,24,32")
    parser.add_argument("--pooling", choices=("mean", "first"), default="mean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-reextract", action="store_true")
    return parser.parse_args()


def save_query(condition: str, query_layers: np.ndarray, layer_indices: list[int], labels: list[str], image_paths: list[str], output: Path) -> Path:
    embeddings_dir = output / "embeddings"
    path = embeddings_dir / f"query_embeddings_{condition}.npz"
    np.savez_compressed(
        path,
        query_layer_embeddings=query_layers,
        layer_indices=np.asarray(layer_indices),
        labels=np.asarray(labels),
        image_paths=np.asarray(image_paths),
    )
    rdm_dir = output / "rdm" / f"query_{condition}_rdms"
    rdm_dir.mkdir(parents=True, exist_ok=True)
    for array_idx, layer_idx in enumerate(layer_indices):
        np.save(rdm_dir / f"layer_{layer_idx}_sample_rdm.npy", rdm(query_layers[array_idx]))
    return path


def main() -> None:
    args = parse_args()
    output = args.output_dir
    embeddings_dir = output / "embeddings"
    layer_indices = parse_layers(args.layer_indices)
    dataset = load_dataset(args.ground_truth, args.dataset_root)
    labels = dataset.labels
    image_paths = [row["absolute_image_path"] for row in dataset.rows]

    existing_fc = embeddings_dir / "image_embeddings_force_choice.npz"
    existing_nc = embeddings_dir / "image_embeddings_no_color_list.npz"
    if existing_fc.exists() and existing_nc.exists() and not args.force_reextract:
        fc = np.load(existing_fc, allow_pickle=True)
        nc = np.load(existing_nc, allow_pickle=True)
        query_fc = fc["query_layer_embeddings"]
        query_nc = nc["query_layer_embeddings"]
        layer_indices = [int(x) for x in fc["layer_indices"].tolist()]
    else:
        helper = LlavaHelper(args.model_id, args.device, args.dtype, args.seed)
        _pre, _post, _image, query_fc = helper.layerwise_image_and_query_embeddings(image_paths, FORCE_CHOICE_PROMPT, layer_indices, args.pooling)
        _pre, _post, _image, query_nc = helper.layerwise_image_and_query_embeddings(image_paths, NO_COLOR_LIST_PROMPT, layer_indices, args.pooling)

    fc_path = save_query("force_choice", query_fc, layer_indices, labels, image_paths, output)
    nc_path = save_query("no_color_list", query_nc, layer_indices, labels, image_paths, output)
    write_json(
        embeddings_dir / "query_embedding_summary.json",
        {
            "layer_indices": layer_indices,
            "force_choice_npz": str(fc_path),
            "no_color_list_npz": str(nc_path),
            "force_choice_prompt": FORCE_CHOICE_PROMPT,
            "no_color_list_prompt": NO_COLOR_LIST_PROMPT,
        },
    )
    print(f"Wrote query embeddings to: {embeddings_dir}")


if __name__ == "__main__":
    main()
