#!/usr/bin/env python3
"""Extract layer-wise LLaVA LLM color-word embeddings and text RDMs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from structured_color_utils import BASIC_COLORS, TEXT_CONTEXTS, LlavaHelper, rdm, save_line_plot, save_matrix_csv, write_csv, write_json


def parse_layers(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract color-word embeddings from LLaVA's language model.")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/llava_structured_alignment"))
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--layer-indices", default="0,1,8,16,24,32")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir
    layer_indices = parse_layers(args.layer_indices)
    helper = LlavaHelper(args.model_id, args.device, args.dtype, args.seed)
    text_layers, context_layers = helper.layerwise_text_embeddings(BASIC_COLORS, TEXT_CONTEXTS, layer_indices)
    embeddings_dir = output / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    npz_path = embeddings_dir / "text_embeddings.npz"
    np.savez_compressed(
        npz_path,
        text_layer_embeddings=text_layers,
        context_text_layer_embeddings=context_layers,
        layer_indices=np.asarray(layer_indices),
        colors=np.asarray(BASIC_COLORS),
        contexts=np.asarray(TEXT_CONTEXTS),
    )

    rows = []
    rdm_dir = output / "rdm" / "text_rdms"
    for array_idx, layer_idx in enumerate(layer_indices):
        text_rdm = rdm(text_layers[array_idx])
        save_matrix_csv(rdm_dir / f"layer_{layer_idx}_text_rdm.csv", BASIC_COLORS, text_rdm, "color")
        rows.append({"layer_index": int(layer_idx), "text_rdm_csv": str(rdm_dir / f"layer_{layer_idx}_text_rdm.csv")})
    write_csv(output / "rdm" / "text_rdm_index.csv", ["layer_index", "text_rdm_csv"], rows)
    write_json(
        embeddings_dir / "text_embedding_summary.json",
        {
            "model_id": args.model_id,
            "layer_indices": layer_indices,
            "colors": BASIC_COLORS,
            "contexts": TEXT_CONTEXTS,
            "text_embeddings_npz": str(npz_path),
            "text_rdm_index_csv": str(output / "rdm" / "text_rdm_index.csv"),
        },
    )
    print(f"Wrote text embeddings to: {npz_path}")


if __name__ == "__main__":
    main()
