#!/usr/bin/env python3
"""Extract pre-answer hidden states and analyze generation-side color representations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from structured_color_utils import (
    BASIC_COLORS,
    FORCE_CHOICE_PROMPT,
    NO_COLOR_LIST_PROMPT,
    LlavaHelper,
    centroids,
    ciede2000_matrix,
    cosine_matrix,
    labels_to_indices,
    linear_cka,
    linear_probe,
    linear_regression_summary,
    load_dataset,
    pearson_corr,
    permutation_pvalue,
    procrustes_error,
    rdm,
    rgb_to_lab,
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
    parser = argparse.ArgumentParser(description="Extract pre-answer hidden states and analyze answer-position color representations.")
    parser.add_argument("--ground-truth", type=Path, default=Path("experiments/llava_structured_alignment/data/ground_truth.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("experiments/llava_structured_alignment/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/llava_structured_alignment"))
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--layer-indices", default="0,1,8,16,24,32")
    parser.add_argument("--permutation-rounds", type=int, default=1000)
    parser.add_argument("--linear-probe-test-size", type=float, default=0.3)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-extraction", action="store_true", help="Reuse existing answer-position embeddings if available.")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def extract_answer_position_layers(helper: LlavaHelper, image_paths: list[str], prompt: str, layer_indices: list[int]) -> np.ndarray:
    language_model = helper.language_model()
    layer_values = [[] for _ in layer_indices]
    with helper.torch.no_grad():
        for image_path in image_paths:
            inputs_embeds, attention_mask, query_mask, _image_span = helper.expanded_language_inputs(image_path, prompt)
            query_positions = query_mask.nonzero(as_tuple=False).flatten()
            if len(query_positions) == 0:
                raise ValueError("No query tokens found for answer-position extraction")
            answer_pos = int(query_positions[-1].item())
            outputs = language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            for out_idx, layer_idx in enumerate(layer_indices):
                hidden = outputs.hidden_states[layer_idx][0, answer_pos]
                layer_values[out_idx].append(hidden.detach().float().cpu().numpy())
    return np.stack([np.vstack(values) for values in layer_values]).astype(np.float32)


def get_lm_head_weight(helper: LlavaHelper):
    module = None
    if hasattr(helper.model, "get_output_embeddings"):
        module = helper.model.get_output_embeddings()
    if module is None:
        language_model = helper.language_model()
        if hasattr(language_model, "get_output_embeddings"):
            module = language_model.get_output_embeddings()
    if module is None:
        for path in ("lm_head", "language_model.lm_head", "model.lm_head", "model.language_model.lm_head"):
            module = helper._resolve_path(helper.model, path)
            if module is not None:
                break
    if module is None or not hasattr(module, "weight"):
        raise AttributeError("Could not find LM head weight")
    return module.weight.detach()


def output_color_vectors(helper: LlavaHelper, output: Path) -> np.ndarray:
    tokenizer = helper.tokenizer()
    weight = get_lm_head_weight(helper)
    rows = []
    vectors = []
    for color in BASIC_COLORS:
        candidates = [" " + color, color]
        encoded = []
        for candidate in candidates:
            ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
            if ids:
                encoded.append((candidate, [int(x) for x in ids]))
        if not encoded:
            raise ValueError(f"Could not tokenize color word: {color}")
        single = [item for item in encoded if len(item[1]) == 1]
        candidate, ids = single[0] if single else min(encoded, key=lambda item: len(item[1]))
        token_weight = weight[helper.torch.tensor(ids, device=weight.device)]
        vector = token_weight.float().mean(dim=0)
        vectors.append(vector.cpu().numpy())
        tokens = [tokenizer.convert_ids_to_tokens(idx) for idx in ids]
        rows.append({"color": color, "chosen_text": candidate, "token_ids": " ".join(map(str, ids)), "tokens": " ".join(map(str, tokens)), "n_tokens": len(ids)})
    path = output / "embeddings" / "lm_head_color_tokens.csv"
    write_csv(path, ["color", "chosen_text", "token_ids", "tokens", "n_tokens"], rows)
    return np.vstack(vectors).astype(np.float32)


def load_or_compute_cielab(args: argparse.Namespace, output: Path, dataset) -> tuple[np.ndarray, np.ndarray]:
    sample_path = output / "rdm" / "cielab_sample_rdm.npy"
    centroid_path = output / "rdm" / "cielab_centroid_rdm.csv"
    if sample_path.exists() and centroid_path.exists():
        sample = np.load(sample_path)
        rows = []
        import csv

        with centroid_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append([float(row[color]) for color in BASIC_COLORS])
        return sample, np.asarray(rows, dtype=float)

    rgb = np.asarray([[int(row["r"]), int(row["g"]), int(row["b"])] for row in dataset.rows], dtype=float)
    lab = rgb_to_lab(rgb)
    sample = ciede2000_matrix(lab)
    label_indices = labels_to_indices(dataset.labels, BASIC_COLORS)
    lab_centroids = np.vstack([lab[label_indices == idx].mean(axis=0) for idx in range(len(BASIC_COLORS))])
    centroid = ciede2000_matrix(lab_centroids)
    (output / "rdm").mkdir(parents=True, exist_ok=True)
    np.save(sample_path, sample)
    save_matrix_csv(centroid_path, BASIC_COLORS, centroid, "color")
    return sample, centroid


def analyze_against_color_prototypes(
    prefix: str,
    source_layers: np.ndarray,
    prototype_layers: np.ndarray,
    layer_indices: list[int],
    labels: list[str],
    output: Path,
    args: argparse.Namespace,
) -> dict:
    label_indices = labels_to_indices(labels, BASIC_COLORS)
    layer_rows = []
    sample_rows = []
    class_rows = []
    probe_rows = []
    probe_class_rows = []
    for array_idx, layer_idx in enumerate(layer_indices):
        source = source_layers[array_idx]
        prototypes = prototype_layers[array_idx]
        sims = cosine_matrix(source, prototypes)
        correct_sim = sims[np.arange(len(labels)), label_indices]
        margins = []
        for sample_idx, true_idx in enumerate(label_indices):
            wrong = np.delete(sims[sample_idx], true_idx)
            margins.append(float(sims[sample_idx, true_idx] - wrong.mean()))
        margins = np.asarray(margins)
        source_centroids = centroids(source, labels, BASIC_COLORS)
        source_rdm = rdm(source_centroids)
        prototype_rdm = rdm(prototypes)
        rsa_r, rsa_p, rsa_n = permutation_pvalue(upper_triangle_values(source_rdm), upper_triangle_values(prototype_rdm), spearman_corr, args.permutation_rounds, args.seed + array_idx * 10)
        cka = linear_cka(source_centroids, prototypes)
        proc = procrustes_error(prototypes, source_centroids)
        probe = linear_probe(source, labels, BASIC_COLORS, args.linear_probe_test_size, args.ridge_alpha, args.seed + 1000 + array_idx)
        for row in probe["class_rows"]:
            probe_class_rows.append({"source": prefix, "layer_index": int(layer_idx), **row})
        probe_rows.append({"source": prefix, "layer_index": int(layer_idx), "linear_probe_accuracy": probe["accuracy"], "linear_probe_train_n": probe["train_n"], "linear_probe_test_n": probe["test_n"]})
        layer_rows.append(
            {
                "source": prefix,
                "layer_index": int(layer_idx),
                "mean_correct_similarity": float(np.mean(correct_sim)),
                "mean_pair_margin": float(np.mean(margins)),
                "rsa_spearman": rsa_r,
                "rsa_spearman_p": rsa_p,
                "rsa_spearman_n": rsa_n,
                "linear_cka": cka,
                "procrustes_error": proc,
                "linear_probe_accuracy": probe["accuracy"],
            }
        )
        for sample_idx, true_idx in enumerate(label_indices):
            sample_rows.append(
                {
                    "source": prefix,
                    "layer_index": int(layer_idx),
                    "sample_index": sample_idx,
                    "true_color": BASIC_COLORS[true_idx],
                    "correct_similarity": float(correct_sim[sample_idx]),
                    "pair_margin": float(margins[sample_idx]),
                }
            )
        for color_idx, color in enumerate(BASIC_COLORS):
            mask = label_indices == color_idx
            local = np.ones(len(BASIC_COLORS), dtype=bool)
            local[color_idx] = False
            class_rows.append(
                {
                    "source": prefix,
                    "layer_index": int(layer_idx),
                    "color": color,
                    "support": int(np.sum(mask)),
                    "mean_correct_similarity": float(np.mean(correct_sim[mask])),
                    "mean_pair_margin": float(np.mean(margins[mask])),
                    "classwise_rsa_spearman": spearman_corr(source_rdm[color_idx, local], prototype_rdm[color_idx, local]),
                }
            )

    sim_dir = output / "similarity"
    probe_dir = output / "linear_probe"
    layer_path = sim_dir / f"{prefix}_layerwise.csv"
    sample_path = sim_dir / f"{prefix}_sample_metrics.csv"
    class_path = sim_dir / f"{prefix}_class_metrics.csv"
    probe_path = probe_dir / f"{prefix}_layerwise_probe.csv"
    probe_class_path = probe_dir / f"{prefix}_per_color_probe.csv"
    write_csv(layer_path, ["source", "layer_index", "mean_correct_similarity", "mean_pair_margin", "rsa_spearman", "rsa_spearman_p", "rsa_spearman_n", "linear_cka", "procrustes_error", "linear_probe_accuracy"], layer_rows)
    write_csv(sample_path, ["source", "layer_index", "sample_index", "true_color", "correct_similarity", "pair_margin"], sample_rows)
    write_csv(class_path, ["source", "layer_index", "color", "support", "mean_correct_similarity", "mean_pair_margin", "classwise_rsa_spearman"], class_rows)
    write_csv(probe_path, ["source", "layer_index", "linear_probe_accuracy", "linear_probe_train_n", "linear_probe_test_n"], probe_rows)
    write_csv(probe_class_path, ["source", "layer_index", "color", "support", "linear_probe_accuracy"], probe_class_rows)
    if not args.no_plots:
        for key, ylabel in [("mean_correct_similarity", "mean cosine"), ("mean_pair_margin", "mean margin"), ("rsa_spearman", "RSA Spearman"), ("linear_cka", "CKA"), ("linear_probe_accuracy", "probe accuracy")]:
            save_line_plot(output / "plots" / f"{prefix}_{key}.png", layer_rows, "layer_index", key, f"{prefix} {key}", ylabel)
    return {
        f"{prefix}_layerwise_csv": str(layer_path),
        f"{prefix}_sample_metrics_csv": str(sample_path),
        f"{prefix}_class_metrics_csv": str(class_path),
        f"{prefix}_probe_csv": str(probe_path),
        f"{prefix}_probe_class_csv": str(probe_class_path),
    }


def analyze_against_image(condition: str, answer_layers: np.ndarray, image_layers: np.ndarray, layer_indices: list[int], labels: list[str], output: Path, args: argparse.Namespace) -> list[dict]:
    rows = []
    for array_idx, layer_idx in enumerate(layer_indices):
        answer = answer_layers[array_idx]
        image = image_layers[array_idx]
        answer_rdm = rdm(answer)
        image_rdm = rdm(image)
        answer_centroids = centroids(answer, labels, BASIC_COLORS)
        image_centroids = centroids(image, labels, BASIC_COLORS)
        answer_centroid_rdm = rdm(answer_centroids)
        image_centroid_rdm = rdm(image_centroids)
        sample_r, sample_p, sample_n = permutation_pvalue(upper_triangle_values(answer_rdm), upper_triangle_values(image_rdm), spearman_corr, args.permutation_rounds, args.seed + array_idx * 10)
        centroid_r, centroid_p, centroid_n = permutation_pvalue(upper_triangle_values(answer_centroid_rdm), upper_triangle_values(image_centroid_rdm), spearman_corr, args.permutation_rounds, args.seed + 100 + array_idx * 10)
        paired = np.diag(cosine_matrix(answer, image))
        rows.append(
            {
                "condition": condition,
                "layer_index": int(layer_idx),
                "mean_paired_sample_cosine": float(np.mean(paired)),
                "std_paired_sample_cosine": float(np.std(paired)),
                "sample_rsa_spearman": sample_r,
                "sample_rsa_p": sample_p,
                "sample_rsa_n": sample_n,
                "centroid_rsa_spearman": centroid_r,
                "centroid_rsa_p": centroid_p,
                "centroid_rsa_n": centroid_n,
                "linear_cka_centroids": linear_cka(answer_centroids, image_centroids),
                "procrustes_error_image_to_answer_centroids": procrustes_error(image_centroids, answer_centroids),
            }
        )
    return rows


def analyze_against_cielab(condition: str, answer_layers: np.ndarray, layer_indices: list[int], labels: list[str], cielab_sample_rdm: np.ndarray, cielab_centroid_rdm: np.ndarray, output: Path, args: argparse.Namespace) -> list[dict]:
    rows = []
    rdm_dir = output / "rdm" / f"answer_{condition}_sample_rdms"
    centroid_dir = output / "rdm" / f"answer_{condition}_centroid_rdms"
    rdm_dir.mkdir(parents=True, exist_ok=True)
    centroid_dir.mkdir(parents=True, exist_ok=True)
    for array_idx, layer_idx in enumerate(layer_indices):
        answer = answer_layers[array_idx]
        answer_rdm = rdm(answer)
        answer_centroids = centroids(answer, labels, BASIC_COLORS)
        answer_centroid_rdm = rdm(answer_centroids)
        np.save(rdm_dir / f"layer_{layer_idx}_sample_rdm.npy", answer_rdm)
        save_matrix_csv(centroid_dir / f"layer_{layer_idx}_centroid_rdm.csv", BASIC_COLORS, answer_centroid_rdm, "color")
        sample_r, sample_p, sample_n = permutation_pvalue(upper_triangle_values(answer_rdm), upper_triangle_values(cielab_sample_rdm), spearman_corr, args.permutation_rounds, args.seed + array_idx * 10)
        centroid_r, centroid_p, centroid_n = permutation_pvalue(upper_triangle_values(answer_centroid_rdm), upper_triangle_values(cielab_centroid_rdm), spearman_corr, args.permutation_rounds, args.seed + 100 + array_idx * 10)
        reg = linear_regression_summary(upper_triangle_values(answer_rdm), upper_triangle_values(cielab_sample_rdm))
        centroid_reg = linear_regression_summary(upper_triangle_values(answer_centroid_rdm), upper_triangle_values(cielab_centroid_rdm))
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


def load_answer_embeddings(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)["answer_position_layer_embeddings"]


def main() -> None:
    args = parse_args()
    output = args.output_dir
    layer_indices = parse_layers(args.layer_indices)
    dataset = load_dataset(args.ground_truth, args.dataset_root)
    labels = dataset.labels
    image_paths = [row["absolute_image_path"] for row in dataset.rows]
    embeddings_dir = output / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    fc_path = embeddings_dir / "answer_position_embeddings_force_choice.npz"
    nc_path = embeddings_dir / "answer_position_embeddings_no_color_list.npz"
    answer_fc = load_answer_embeddings(fc_path) if args.skip_extraction else None
    answer_nc = load_answer_embeddings(nc_path) if args.skip_extraction else None

    helper = None
    lm_head_vectors = None
    if answer_fc is None or answer_nc is None or not (embeddings_dir / "lm_head_color_vectors.npy").exists():
        helper = LlavaHelper(args.model_id, args.device, args.dtype, args.seed)
    if answer_fc is None:
        answer_fc = extract_answer_position_layers(helper, image_paths, FORCE_CHOICE_PROMPT, layer_indices)
        np.savez_compressed(fc_path, answer_position_layer_embeddings=answer_fc, layer_indices=np.asarray(layer_indices), labels=np.asarray(labels), image_paths=np.asarray(image_paths), prompt=FORCE_CHOICE_PROMPT)
    if answer_nc is None:
        answer_nc = extract_answer_position_layers(helper, image_paths, NO_COLOR_LIST_PROMPT, layer_indices)
        np.savez_compressed(nc_path, answer_position_layer_embeddings=answer_nc, layer_indices=np.asarray(layer_indices), labels=np.asarray(labels), image_paths=np.asarray(image_paths), prompt=NO_COLOR_LIST_PROMPT)

    lm_head_path = embeddings_dir / "lm_head_color_vectors.npy"
    if lm_head_path.exists():
        lm_head_vectors = np.load(lm_head_path)
    else:
        lm_head_vectors = output_color_vectors(helper, output)
        np.save(lm_head_path, lm_head_vectors)

    text_npz_path = embeddings_dir / "text_embeddings.npz"
    if not text_npz_path.exists():
        raise FileNotFoundError(f"Text embeddings not found: {text_npz_path}. Run text_embedding_extraction.py first.")
    text_npz = np.load(text_npz_path, allow_pickle=True)
    text_layers = text_npz["text_layer_embeddings"]
    layer_indices = [int(x) for x in text_npz["layer_indices"].tolist()]
    if answer_fc.shape[0] != len(layer_indices) or answer_nc.shape[0] != len(layer_indices):
        raise ValueError("Answer-position embeddings and text embeddings have different layer counts. Re-run with matching --layer-indices.")

    image_fc_path = embeddings_dir / "image_embeddings_force_choice.npz"
    image_nc_path = embeddings_dir / "image_embeddings_no_color_list.npz"
    if not image_fc_path.exists() or not image_nc_path.exists():
        raise FileNotFoundError("Image embeddings not found. Run image_embedding_extraction.py first.")
    image_fc = np.load(image_fc_path, allow_pickle=True)["image_layer_embeddings"]
    image_nc = np.load(image_nc_path, allow_pickle=True)["image_layer_embeddings"]

    summary = {
        "force_choice_answer_npz": str(fc_path),
        "no_color_list_answer_npz": str(nc_path),
        "lm_head_color_vectors_npy": str(lm_head_path),
        "layer_indices": layer_indices,
    }

    summary.update(analyze_against_color_prototypes("answer_force_choice", answer_fc, text_layers, layer_indices, labels, output, args))
    summary.update(analyze_against_color_prototypes("answer_no_color_list", answer_nc, text_layers, layer_indices, labels, output, args))
    lm_head_layers = np.repeat(lm_head_vectors[None, :, :], len(layer_indices), axis=0)
    summary.update(analyze_against_color_prototypes("answer_lm_head_force_choice", answer_fc, lm_head_layers, layer_indices, labels, output, args))
    summary.update(analyze_against_color_prototypes("answer_lm_head_no_color_list", answer_nc, lm_head_layers, layer_indices, labels, output, args))

    image_rows = []
    image_rows.extend(analyze_against_image("force_choice", answer_fc, image_fc, layer_indices, labels, output, args))
    image_rows.extend(analyze_against_image("no_color_list", answer_nc, image_nc, layer_indices, labels, output, args))
    image_alignment_path = output / "generation_side" / "answer_image_alignment.csv"
    write_csv(
        image_alignment_path,
        ["condition", "layer_index", "mean_paired_sample_cosine", "std_paired_sample_cosine", "sample_rsa_spearman", "sample_rsa_p", "sample_rsa_n", "centroid_rsa_spearman", "centroid_rsa_p", "centroid_rsa_n", "linear_cka_centroids", "procrustes_error_image_to_answer_centroids"],
        image_rows,
    )

    cielab_sample_rdm, cielab_centroid_rdm = load_or_compute_cielab(args, output, dataset)
    cielab_rows = []
    cielab_rows.extend(analyze_against_cielab("force_choice", answer_fc, layer_indices, labels, cielab_sample_rdm, cielab_centroid_rdm, output, args))
    cielab_rows.extend(analyze_against_cielab("no_color_list", answer_nc, layer_indices, labels, cielab_sample_rdm, cielab_centroid_rdm, output, args))
    cielab_alignment_path = output / "generation_side" / "answer_cielab_alignment.csv"
    write_csv(
        cielab_alignment_path,
        ["condition", "layer_index", "sample_rsa_spearman", "sample_rsa_p", "sample_rsa_n", "centroid_rsa_spearman", "centroid_rsa_p", "centroid_rsa_n", "sample_distance_regression_r2", "sample_distance_regression_slope", "sample_distance_regression_n", "centroid_distance_regression_r2", "centroid_distance_regression_slope", "centroid_distance_regression_n"],
        cielab_rows,
    )

    if not args.no_plots:
        for condition in ("force_choice", "no_color_list"):
            cond_image_rows = [row for row in image_rows if row["condition"] == condition]
            cond_cielab_rows = [row for row in cielab_rows if row["condition"] == condition]
            save_line_plot(output / "plots" / f"answer_{condition}_image_centroid_rsa.png", cond_image_rows, "layer_index", "centroid_rsa_spearman", f"answer_{condition} vs image centroid RSA", "Spearman r")
            save_line_plot(output / "plots" / f"answer_{condition}_cielab_centroid_rsa.png", cond_cielab_rows, "layer_index", "centroid_rsa_spearman", f"answer_{condition} vs CIEDE2000 centroid RSA", "Spearman r")
            save_line_plot(output / "plots" / f"answer_{condition}_cielab_sample_rsa.png", cond_cielab_rows, "layer_index", "sample_rsa_spearman", f"answer_{condition} vs CIEDE2000 sample RSA", "Spearman r")

    summary.update(
        {
            "answer_image_alignment_csv": str(image_alignment_path),
            "answer_cielab_alignment_csv": str(cielab_alignment_path),
        }
    )
    write_json(output / "generation_side" / "answer_position_summary.json", summary)
    print(f"Wrote answer-position embeddings and analyses to: {output / 'generation_side'}")


if __name__ == "__main__":
    main()
