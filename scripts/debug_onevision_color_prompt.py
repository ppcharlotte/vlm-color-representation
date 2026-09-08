#!/usr/bin/env python3
"""Compare OneVision color naming through the project helper and HF pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageStat

from structured_color_utils import LlavaHelper, normalize_color_answer


DEFAULT_PROMPTS = [
    "What color is this image?",
    "What color is this image? Answer only with exactly one basic color name.",
    (
        "What is the basic color of this image? Answer only one color name from this list: "
        "black, blue, brown, gray, green, orange, pink, purple, red, white, or yellow."
    ),
    "This is a neutral grayscale image. What is its basic color? Answer only one word from: black, gray, white.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug OneVision color responses for one image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model-id", default="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt", action="append", help="Prompt to test. Can be repeated.")
    parser.add_argument("--skip-pipeline", action="store_true")
    return parser.parse_args()


def print_response(source: str, prompt: str, raw: str, raw_with_special: str | None = None) -> None:
    normalized, status = normalize_color_answer(raw)
    print(f"\n== {source} ==")
    print(f"prompt: {prompt}")
    print(f"raw: {raw!r}")
    if raw_with_special is not None:
        print(f"raw_with_special: {raw_with_special!r}")
    print(f"normalized: {normalized or ''} {status}")


def helper_outputs(args: argparse.Namespace, prompts: list[str]) -> None:
    helper = LlavaHelper(args.model_id, device=args.device, dtype=args.dtype, seed=42)
    for prompt in prompts:
        details = helper.generate_details(str(args.image), prompt, args.max_new_tokens, args.temperature)
        print_response("project_helper", prompt, details["text"], details["text_with_special_tokens"])


def pipeline_outputs(args: argparse.Namespace, prompts: list[str]) -> None:
    from transformers import pipeline

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }[args.dtype]
    pipe = pipeline(
        "image-text-to-text",
        model=args.model_id,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    image = Image.open(args.image).convert("RGB")
    for prompt in prompts:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        output = pipe(text=messages, max_new_tokens=args.max_new_tokens, do_sample=args.temperature > 0)
        raw = str(output)
        try:
            generated = output[0]["generated_text"][-1]["content"]
            raw = generated if isinstance(generated, str) else str(generated)
        except Exception:
            pass
        print_response("hf_pipeline", prompt, raw)


def main() -> None:
    args = parse_args()
    prompts = args.prompt or DEFAULT_PROMPTS
    image = Image.open(args.image).convert("RGB")
    print(f"image: {args.image}")
    print(f"size: {image.size}")
    print(f"mean_rgb: {[round(x, 3) for x in ImageStat.Stat(image).mean]}")

    helper_outputs(args, prompts)
    if not args.skip_pipeline:
        pipeline_outputs(args, prompts)


if __name__ == "__main__":
    main()
