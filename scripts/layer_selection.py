"""Resolve LM hidden-state indices, including the input state at index zero."""
from collections.abc import Mapping


def language_layer_count(config):
    def get(obj, key):
        return obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)

    # Prefer language configuration over any outer/vision depth.
    text = get(config, "text_config")
    if text is None:
        text = get(config, "llm_config")
    if text is None:
        text = config
    count = get(text, "num_hidden_layers")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("Cannot determine language-model num_hidden_layers from checkpoint config")
    return count


def resolve_layers(selection, model_id=None, config=None):
    if selection.strip().lower() == "all":
        if config is None:
            if model_id is None:
                raise ValueError("All-layer selection requires a model ID or config")
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return list(range(language_layer_count(config) + 1))
    layers = [int(part.strip()) for part in selection.split(",") if part.strip()]
    if not layers or any(i < 0 for i in layers) or len(set(layers)) != len(layers):
        raise ValueError("Layer indices must be unique nonnegative integers")
    return layers
