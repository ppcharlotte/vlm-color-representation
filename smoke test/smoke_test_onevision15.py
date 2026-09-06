
#!/usr/bin/env python3
"""Smoke-test LLaVA-OneVision-1.5 before adapting the full color pipeline.

This script checks the minimum set of model interfaces needed by the current
structured color pipeline:

1. processor + chat template
2. image preprocessing fields and tensor shapes
3. generation and output parsing surface
4. generation hidden states
5. plain forward hidden states
6. likely vision tower / projector / language model module paths

It intentionally does not depend on the thesis repository modules.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import traceback
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ID = "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
DEFAULT_PROMPT = (
    "What color is this image? Answer only with exactly one color name from this list: "
    "black, blue, brown, gray, green, orange, pink, purple, red, white, yellow."
)

# Ordered from most specific to least specific. `AutoModel` is a last resort
# only: it returns the base model without an LM head, so `generate` on it is
# meaningless. We record which loader succeeded so this is visible in the summary.
AUTO_CLASS_CANDIDATES = (
    "AutoModelForImageTextToText",
    "AutoModelForVision2Seq",
    "AutoModelForCausalLM",
    "AutoModel",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test a LLaVA-OneVision-1.5 checkpoint.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--image", type=Path, default=None, help="Optional image path. If omitted, a red patch is generated.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--attn-implementation", default="sdpa", choices=("eager", "sdpa", "flash_attention_2"))
    parser.add_argument("--patch-default-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--fallback-to-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If loading with the requested attention implementation fails for every candidate class, retry with eager.",
    )
    parser.add_argument(
        "--patch-rope-default",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Re-register a 'default' entry in transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS. "
            "transformers 5.x removed it, which breaks 4.x-style remote code that does "
            "ROPE_INIT_FUNCTIONS[rope_type] with rope_type='default'. Disable to surface that incompatibility."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("onevision15_smoke_test"))
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Device / dtype selection and validation
# --------------------------------------------------------------------------- #


def choose_device(torch: Any, requested: str):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_dtype(torch: Any, requested: str):
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float32":
        return torch.float32
    return "auto"


def validate_runtime_combination(torch: Any, device: Any, dtype: Any, attn_implementation: str) -> None:
    """Fail early with a readable message instead of deep inside the model."""
    if attn_implementation == "flash_attention_2" and device.type != "cuda":
        raise SystemExit(
            f"flash_attention_2 requires a CUDA device, but the selected device is {device}. "
            "Use --attn-implementation sdpa or eager."
        )
    if device.type == "cpu" and dtype == torch.float16:
        raise SystemExit(
            "float16 on CPU is not supported by many PyTorch CPU kernels. "
            "Use --dtype bfloat16 or --dtype float32 on CPU."
        )


# --------------------------------------------------------------------------- #
# Summaries and probes
# --------------------------------------------------------------------------- #


def tensor_summary(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "shape"):
        return {
            "shape": list(value.shape),
            "dtype": str(getattr(value, "dtype", "")),
            "device": str(getattr(value, "device", "")),
        }
    return {"type": type(value).__name__}


def safe_getattr_path(root: Any, dotted_path: str) -> Any | None:
    current = root
    for part in dotted_path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def module_probe(model: Any) -> dict[str, str]:
    paths = [
        "vision_tower",
        "model.vision_tower",
        "vision_model",
        "model.vision_model",
        "visual",
        "model.visual",
        "visual.merger",
        "model.visual.merger",
        "multi_modal_projector",
        "model.multi_modal_projector",
        "mm_projector",
        "model.mm_projector",
        "projector",
        "model.projector",
        "language_model",
        "model.language_model",
        "lm_head",
        "model",
    ]
    found = {}
    for path in paths:
        obj = safe_getattr_path(model, path)
        if obj is not None:
            found[path] = type(obj).__name__
    return found


def _first_present(obj: Any, *names: str) -> Any:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def config_summary(config: Any) -> dict[str, Any]:
    text_config = getattr(config, "text_config", None)
    vision_config = getattr(config, "vision_config", None)
    return {
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "auto_map": getattr(config, "auto_map", None),
        # Classic LLaVA uses `image_token_index`; Qwen2-VL-style configs use `image_token_id`.
        "image_token_id": _first_present(config, "image_token_id", "image_token_index"),
        "vision_start_token_id": getattr(config, "vision_start_token_id", None),
        "vision_end_token_id": getattr(config, "vision_end_token_id", None),
        "vision_feature_layer": getattr(config, "vision_feature_layer", None),
        "vision_feature_select_strategy": getattr(config, "vision_feature_select_strategy", None),
        "attn_implementation": getattr(config, "_attn_implementation", None),
        "text_model_type": getattr(text_config, "model_type", None),
        "text_num_hidden_layers": _first_present(text_config, "num_hidden_layers") or getattr(config, "num_hidden_layers", None),
        "text_hidden_size": _first_present(text_config, "hidden_size") or getattr(config, "hidden_size", None),
        "vision_model_type": getattr(vision_config, "model_type", None),
        "vision_num_hidden_layers": _first_present(vision_config, "num_hidden_layers", "depth"),
        "vision_hidden_size": _first_present(vision_config, "hidden_size", "embed_dim"),
        "vision_image_size": getattr(vision_config, "image_size", None),
        "vision_patch_size": getattr(vision_config, "patch_size", None),
        "vision_spatial_merge_size": getattr(vision_config, "spatial_merge_size", None),
    }


def create_default_image(path: Path) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (560, 560), (220, 20, 20)).save(path)
    return path


# --------------------------------------------------------------------------- #
# Config patching
# --------------------------------------------------------------------------- #


def resolve_pad_token_id(transformers: Any, model_id: str, processor: Any, trust_remote_code: bool) -> int | None:
    try:
        generation_config = transformers.GenerationConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if generation_config.pad_token_id is not None:
            return int(generation_config.pad_token_id)
    except Exception:
        pass
    tokenizer = getattr(processor, "tokenizer", None)
    for attr in ("pad_token_id", "eos_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            return int(token_id)
    return None


def patch_config_token_ids(config: Any, pad_token_id: int | None) -> None:
    if pad_token_id is None:
        return
    config.pad_token_id = pad_token_id
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        text_config.pad_token_id = pad_token_id


def patch_config_attention(config: Any, attn_implementation: str) -> None:
    """Set the private attention field on the config and all sub-configs.

    `_attn_implementation` is a property in recent transformers versions and
    writing to it is the supported way to override the implementation. We also
    set `_attn_implementation_internal` for older versions where the property
    setter may not exist. There is no public `attn_implementation` attribute;
    setting one only pollutes the serialized config.
    """
    targets = [config]
    for attr in ("text_config", "vision_config"):
        sub_config = getattr(config, attr, None)
        if sub_config is not None:
            targets.append(sub_config)
    for target in targets:
        try:
            target._attn_implementation = attn_implementation
        except Exception:
            pass
        target._attn_implementation_internal = attn_implementation


# --------------------------------------------------------------------------- #
# transformers version shims
# --------------------------------------------------------------------------- #


def patch_transformers_rope_default(torch: Any) -> dict[str, str]:
    """Make `ROPE_INIT_FUNCTIONS["default"]` exist again.

    transformers 4.x exposed the plain RoPE initializer under the "default" key
    of `ROPE_INIT_FUNCTIONS`. transformers 5.x removed that key (each native
    model now carries its own `compute_default_rope_parameters` static method
    and only consults the dict for non-default rope types). Remote code written
    against 4.x still does `ROPE_INIT_FUNCTIONS[self.rope_type]` with
    rope_type="default" and raises KeyError under 5.x.

    This must run BEFORE the remote module is imported (i.e. before
    `dynamic_model_class` / `from_pretrained`). It mutates the dict in place, so
    a remote module that did `from transformers.modeling_rope_utils import
    ROPE_INIT_FUNCTIONS` sees the same object and picks the entry up.

    The local implementation mirrors 5.x `Qwen3RotaryEmbedding.compute_default_rope_parameters`.
    """
    try:
        from transformers import modeling_rope_utils
    except Exception as exc:
        return {"status": "modeling_rope_utils_import_failed", "error": f"{type(exc).__name__}: {exc}"}

    rope_fns = getattr(modeling_rope_utils, "ROPE_INIT_FUNCTIONS", None)
    if not isinstance(rope_fns, dict):
        return {"status": "ROPE_INIT_FUNCTIONS_not_found"}
    if "default" in rope_fns:
        return {"status": "default_already_available"}

    # Only present in some 4.x intermediate versions where the key was already
    # gone but the module-level function still existed. On 5.x this is None.
    builtin = getattr(modeling_rope_utils, "_compute_default_rope_parameters", None)
    if builtin is not None:
        rope_fns["default"] = builtin
        return {"status": "patched_with_transformers_builtin"}

    def compute_default_rope_parameters(config: Any, device=None, seq_len=None, layer_type=None, **kwargs):
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        # 5.x may group rope parameters per layer type for models with mixed attention layers.
        if layer_type is not None and isinstance(rope_parameters.get(layer_type), dict):
            rope_parameters = rope_parameters[layer_type]
        base = rope_parameters.get("rope_theta", getattr(config, "rope_theta", 10000.0))
        partial_rotary_factor = rope_parameters.get(
            "partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0)
        )
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        attention_factor = 1.0  # unused for plain RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        if device is not None:
            inv_freq = inv_freq.to(device)
        return inv_freq.to(device), attention_factor

    rope_fns["default"] = compute_default_rope_parameters
    return {"status": "patched_with_local_standard_rope"}


# --------------------------------------------------------------------------- #
# Remote-code helpers
# --------------------------------------------------------------------------- #


def dynamic_model_class(model_id: str, config: Any, trust_remote_code: bool):
    """Resolve the remote-code model class from `auto_map`, if there is one."""
    if not trust_remote_code:
        return None
    auto_map = getattr(config, "auto_map", {}) or {}
    class_ref = None
    for key in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM", "AutoModel"):
        if auto_map.get(key):
            class_ref = auto_map[key]
            break
    if not class_ref:
        return None
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except Exception:
        return None
    try:
        return get_class_from_dynamic_module(class_ref, model_id)
    except Exception:
        return None


def patch_remote_attention_defaults(model_cls: Any, attn_implementation: str) -> dict[str, Any]:
    """Point every `*_ATTENTION_CLASSES["default"]` in the remote module at the requested implementation.

    Rather than hard-coding guessed dictionary names, this discovers every
    module-level dict whose name ends with `_ATTENTION_CLASSES` and remaps its
    `default` entry by matching the requested implementation against the dict's
    own keys. The result records what was found so a silent no-op is visible in
    the summary.
    """
    report: dict[str, Any] = {"patched": {}, "discovered_attention_dicts": [], "module": None}
    if model_cls is None:
        return report
    module = sys.modules.get(model_cls.__module__)
    if module is None:
        return report
    report["module"] = model_cls.__module__

    alias_keys = {
        "eager": ("eager",),
        "sdpa": ("sdpa",),
        "flash_attention_2": ("flash_attention_2", "flash_attn_2", "flash"),
    }[attn_implementation]

    for name, value in vars(module).items():
        if not name.endswith("_ATTENTION_CLASSES") or not isinstance(value, dict):
            continue
        report["discovered_attention_dicts"].append({name: sorted(str(k) for k in value.keys())})
        target = next((value[key] for key in alias_keys if key in value), None)
        if target is None:
            continue
        value["default"] = target
        report["patched"][name] = f"default->{getattr(target, '__name__', repr(target))}"
    return report


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #


def _dtype_kwarg_name(transformers: Any) -> str:
    """`torch_dtype` was renamed to `dtype` in transformers 4.56; pick whichever this version accepts."""
    try:
        params = inspect.signature(transformers.PreTrainedModel.from_pretrained).parameters
        if "dtype" in params:
            return "dtype"
    except Exception:
        pass
    try:
        version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
        if version >= (4, 56):
            return "dtype"
    except Exception:
        pass
    return "torch_dtype"


def _load_candidates(transformers: Any, dynamic_cls: Any) -> list[tuple[str, Any]]:
    candidates: list[tuple[str, Any]] = []
    if dynamic_cls is not None:
        candidates.append(("dynamic_remote_class", dynamic_cls))
    for name in AUTO_CLASS_CANDIDATES:
        cls = getattr(transformers, name, None)
        if cls is not None:
            candidates.append((name, cls))
    return candidates


def _try_load(
    transformers: Any,
    candidates: list[tuple[str, Any]],
    model_id: str,
    config: Any,
    device: Any,
    dtype: Any,
    trust_remote_code: bool,
    attn_implementation: str,
    load_errors: dict[str, Any],
):
    load_kwargs: dict[str, Any] = {
        "config": config,
        "trust_remote_code": trust_remote_code,
        "attn_implementation": attn_implementation,
        _dtype_kwarg_name(transformers): dtype,
    }
    if device.type == "cuda":
        load_kwargs["device_map"] = "auto"
    for name, cls in candidates:
        try:
            model = cls.from_pretrained(model_id, **load_kwargs)
            return model, name
        except Exception as exc:
            load_errors[f"{attn_implementation}:{name}"] = {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
    return None, None


def load_model(
    transformers: Any,
    model_id: str,
    config: Any,
    device: Any,
    dtype: Any,
    trust_remote_code: bool,
    attn_implementation: str,
    patch_default_attention: bool,
    fallback_to_eager: bool,
):
    dynamic_cls = dynamic_model_class(model_id, config, trust_remote_code)
    remote_attention_patches: dict[str, Any] = {}
    if patch_default_attention:
        remote_attention_patches = patch_remote_attention_defaults(dynamic_cls, attn_implementation)

    candidates = _load_candidates(transformers, dynamic_cls)
    load_errors: dict[str, Any] = {}

    model, loader = _try_load(
        transformers, candidates, model_id, config, device, dtype, trust_remote_code, attn_implementation, load_errors
    )
    effective_attn = attn_implementation

    if model is None and fallback_to_eager and attn_implementation != "eager":
        patch_config_attention(config, "eager")
        if patch_default_attention:
            remote_attention_patches = patch_remote_attention_defaults(dynamic_cls, "eager")
        model, loader = _try_load(
            transformers, candidates, model_id, config, device, dtype, trust_remote_code, "eager", load_errors
        )
        effective_attn = "eager"

    if model is None:
        raise RuntimeError("Could not load model with any candidate class:\n" + json.dumps(load_errors, indent=2))

    if loader == "AutoModel":
        print(
            "WARNING: model was loaded with AutoModel, which has no LM head. "
            "generate() output below is not meaningful.",
            file=sys.stderr,
        )
    return model, loader, effective_attn, load_errors, remote_attention_patches


def model_dtype(model: Any):
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return None


def move_inputs(inputs: Any, device: Any, dtype: Any) -> dict[str, Any]:
    """Move all tensors to `device`; cast floating tensors to `dtype` (a real torch dtype, never "auto")."""
    moved = {}
    for key, value in dict(inputs).items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue
        is_float = getattr(value, "is_floating_point", lambda: False)()
        if is_float and dtype is not None:
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def hidden_summary(hidden_states: Any) -> dict[str, Any]:
    if hidden_states is None:
        return {"available": False}
    if not isinstance(hidden_states, (tuple, list)):
        return {"available": True, "type": type(hidden_states).__name__}
    out: dict[str, Any] = {"available": True, "num_steps_or_layers": len(hidden_states)}
    if hidden_states:
        first = hidden_states[0]
        out["first_item_type"] = type(first).__name__
        if isinstance(first, (tuple, list)):
            out["first_item_len"] = len(first)
            if first:
                out["first_tensor"] = tensor_summary(first[0])
                out["last_tensor"] = tensor_summary(first[-1])
            if len(hidden_states) > 1 and isinstance(hidden_states[-1], (tuple, list)) and hidden_states[-1]:
                out["last_step_last_tensor"] = tensor_summary(hidden_states[-1][-1])
        else:
            out["first_tensor"] = tensor_summary(first)
            out["last_tensor"] = tensor_summary(hidden_states[-1])
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()
    try:
        import torch
        import transformers
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Install torch, transformers, accelerate, and Pillow before running this smoke test.") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.image or create_default_image(args.output_dir / "red_patch.png")
    image = Image.open(image_path).convert("RGB")

    device = choose_device(torch, args.device)
    dtype = choose_dtype(torch, args.dtype)
    validate_runtime_combination(torch, device, dtype, args.attn_implementation)

    # Must happen before AutoProcessor/AutoConfig/from_pretrained: those can
    # import the remote modeling module, which reads ROPE_INIT_FUNCTIONS at call time
    # but may cache references at import time in some versions.
    rope_patch = (
        patch_transformers_rope_default(torch)
        if args.patch_rope_default
        else {"status": "skipped_by_flag"}
    )

    processor = transformers.AutoProcessor.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "padding_side"):
        processor.tokenizer.padding_side = "left"

    config = transformers.AutoConfig.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    pad_token_id = resolve_pad_token_id(transformers, args.model_id, processor, args.trust_remote_code)
    patch_config_token_ids(config, pad_token_id)
    patch_config_attention(config, args.attn_implementation)

    model, model_loader, effective_attn, load_errors, remote_attention_patches = load_model(
        transformers,
        args.model_id,
        config,
        device,
        dtype,
        args.trust_remote_code,
        args.attn_implementation,
        args.patch_default_attention,
        args.fallback_to_eager,
    )
    if device.type != "cuda":
        model.to(device)
    model.eval()

    target_device = getattr(model, "device", device)
    # Use the dtype the model actually ended up in (bf16 for "auto" on this
    # checkpoint), so pixel_values match the vision tower even when the remote
    # code does not cast them itself.
    effective_dtype = model_dtype(model)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
    raw_input_summary = {key: tensor_summary(value) for key, value in inputs.items()}
    inputs = move_inputs(inputs, target_device, effective_dtype)

    input_keys = list(inputs.keys())
    summary: dict[str, Any] = {
        "model_id": args.model_id,
        "model_loader": model_loader,
        "load_errors_before_success": load_errors,
        "transformers_version": getattr(transformers, "__version__", None),
        "torch_version": getattr(torch, "__version__", None),
        "device": str(target_device),
        "hf_device_map": getattr(model, "hf_device_map", None),
        "dtype_requested": args.dtype,
        "dtype_effective": str(effective_dtype),
        "attn_implementation_requested": args.attn_implementation,
        "attn_implementation_effective": effective_attn,
        "patch_default_attention": args.patch_default_attention,
        "remote_attention_patches": remote_attention_patches,
        "rope_default_patch": rope_patch,
        "image_path": str(image_path),
        "prompt_text": prompt_text,
        "patched_pad_token_id": pad_token_id,
        "config": config_summary(model.config),
        "processor_input_keys": input_keys,
        "processor_inputs_raw": raw_input_summary,
        "processor_inputs": {key: tensor_summary(value) for key, value in inputs.items()},
        # `image_grid_thw` present => Qwen2-VL-style flattened patches; absent => classic LLaVA-style pixel_values.
        "processor_style": "qwen2vl_like" if "image_grid_thw" in input_keys else "llava_like",
        "module_probe": module_probe(model),
    }

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    input_len = int(inputs["input_ids"].shape[-1])
    generated_ids = generated.sequences[:, input_len:]
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    summary["generated_text"] = decoded[0].strip() if decoded else ""
    summary["generated_token_ids"] = generated_ids[0].detach().cpu().tolist()
    summary["generate_hidden_states"] = hidden_summary(getattr(generated, "hidden_states", None))

    try:
        with torch.no_grad():
            forward_outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        summary["forward_hidden_states"] = hidden_summary(getattr(forward_outputs, "hidden_states", None))
        summary["forward_logits"] = tensor_summary(getattr(forward_outputs, "logits", None))
        image_hidden = getattr(forward_outputs, "image_hidden_states", None)
        summary["image_hidden_states"] = (
            {"available": False} if image_hidden is None else {"available": True, **(tensor_summary(image_hidden) or {})}
        )
    except Exception as exc:
        summary["forward_error"] = f"{type(exc).__name__}: {exc}"
        summary["forward_traceback"] = traceback.format_exc()

    output_json = args.output_dir / "summary.json"
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"\nSmoke-test summary written to: {output_json}")


if __name__ == "__main__":
    main()