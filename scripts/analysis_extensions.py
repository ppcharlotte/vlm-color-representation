"""Template robustness, paired category contrasts and visual equivalence for v2."""
from itertools import combinations
import numpy as np

from alignment_statistics import compare_rdms, paired_category_swap, stable_seed
from structured_color_utils import BASIC_COLORS, TEXT_CONTEXTS, centroids, ciede2000_matrix, rdm


def template_subsets(contexts):
    if list(contexts) != TEXT_CONTEXTS:
        raise ValueError("Template balancing requires the original ordered 30 templates")
    # Each set has 5 generic descriptors + 10 concrete object contexts.
    return {"balanced_A": list(range(0, 10, 2)) + list(range(10, 30, 2)),
            "balanced_B": list(range(1, 10, 2)) + list(range(11, 30, 2))}


def template_robustness(output, lab, labels, layers, model_id, seed, rounds, coefficients):
    path = output / "embeddings/text_embeddings.npz"
    with np.load(path, allow_pickle=False) as data:
        if "context_text_layer_embeddings" not in data or "contexts" not in data:
            return [], [], "missing_per_template_cache_rerun_text_extraction"
        contexts = data["contexts"].tolist()
        subsets = template_subsets(contexts)
        values, full = data["context_text_layer_embeddings"], data["text_layer_embeddings"]
        if values.shape[:3] != (len(layers), 11, 30) or values.shape[3] != full.shape[-1]:
            raise ValueError("Per-template embeddings must be [layer,11 colors,30 contexts,features]")
        membership = [dict(subset=subset, template_index=i, template=contexts[i],
                           template_type="generic" if i < 10 else "object_context")
                      for subset, indices in subsets.items() for i in indices]
        human = ciede2000_matrix(centroids(lab, labels, BASIC_COLORS))
        rows = []
        for pos, layer in enumerate(layers):
            geometries = {"full_30": rdm(full[pos], "cosine"), "perceptual": human}
            for subset, indices in subsets.items():
                # Same average-then-L2-normalize rule as full text extraction.
                prototype = values[pos][:, indices, :].mean(axis=1)
                prototype /= np.maximum(np.linalg.norm(prototype, axis=-1, keepdims=True), 1e-12)
                geometries[subset] = rdm(prototype, "cosine")
            comparisons = [(a, b, "subset_vs_subset") for a, b in combinations(subsets, 2)]
            comparisons += [(a, "full_30", "subset_vs_full") for a in subsets]
            comparisons += [(a, "perceptual", "subset_vs_perceptual") for a in subsets]
            for a, b, kind in comparisons:
                for coefficient in coefficients:
                    pseed = stable_seed(seed, "templates", a, b, coefficient, layer)
                    value, p, pairs, status = compare_rdms(geometries[a], geometries[b], coefficient, rounds, pseed)
                    rows.append(dict(model_id=model_id, model=model_id, run_id=str(output.resolve()), seed=seed,
                        layer_index=layer, subset_a=a, subset_b=b, source_a=a, source_b=b,
                        comparison_type=kind, level="category", embedding_distance="cosine",
                        rsa_coefficient=coefficient, alignment=value, p_value=p, n_pairs=pairs,
                        permutation_rounds=rounds, permutation_seed=pseed, status=status,
                        test="two_sided_RDM_label_permutation"))
    return rows, membership, "computed"


def paired_contrasts(sources, lab, labels, layers, model_id, run_id, seed, distances,
                     coefficients, rounds, cee_reference_difference=False):
    """All three sources use the same sample intersection before centroid construction."""
    by_name = {s.name: s for s in sources}
    designs = []
    if "cee" in by_name and "matched_text" in by_name:
        for ref in ("image_tokens", "text_color_words"):
            designs.append(("cee_image_vs_no_image", "cee", "matched_text", ref))
    if cee_reference_difference and all(s in by_name for s in ("cee", "image_tokens", "text_color_words")):
        # Correlation is symmetric: keep shared CEe RDM fixed; swap V/T prototypes.
        designs.append(("cee_visual_vs_textual", "image_tokens", "text_color_words", "cee"))
    if all(s in by_name for s in ("ceg_decision", "ceg_token")):
        for ref in ("image_tokens", "text_color_words"):
            designs.append(("generation_post_vs_pre", "ceg_token", "ceg_decision", ref))
    rows = []
    for kind, a, b, ref in designs:
        if any(name not in by_name for name in (a, b, ref) if name != "perceptual"):
            continue
        selected = [by_name[name] for name in (a, b, ref) if name != "perceptual"]
        ids = selected[0].indices
        for source in selected[1:]:
            ids = np.intersect1d(ids, source.indices)
        active_labels = [labels[int(i)] for i in ids]
        colors = [c for c in BASIC_COLORS if c in active_labels]
        if len(colors) < 3:
            continue

        def prototype(name, layer):
            if name == "perceptual":
                return centroids(lab[ids], active_labels, colors)
            source = by_name[name]
            values = source.at(layer)
            if source.category_only:
                return values[[BASIC_COLORS.index(c) for c in colors]]
            positions = {int(i): pos for pos, i in enumerate(source.indices)}
            return centroids(values[[positions[int(i)] for i in ids]], active_labels, colors)

        for layer in layers:
            pa, pb, pr = (prototype(name, layer) for name in (a, b, ref))
            for distance in distances:
                reference = ciede2000_matrix(pr) if ref == "perceptual" else rdm(pr, distance)
                for coefficient in coefficients:
                    pseed = stable_seed(seed, "paired_swap", kind, ref, distance, coefficient, layer)
                    result = paired_category_swap(pa, pb, reference, distance, coefficient, rounds, pseed)
                    result.update(model=model_id, model_id=model_id, run_id=run_id, seed=seed,
                        layer_index=layer, level="category", source_a=a, source_b=b, reference=ref,
                        contrast=kind, embedding_distance=distance, rsa_coefficient=coefficient,
                        n_samples=len(ids), n_colors=len(colors), n_pairs=len(colors)*(len(colors)-1)//2,
                        shared_colors="|".join(colors), sample_indices="|".join(map(str, ids)))
                    result["p_delta"] = result["p_value"]
                    if kind == "cee_image_vs_no_image":
                        result.update(rsa_image=result["rsa_a"], rsa_no_image=result["rsa_b"])
                    elif kind == "cee_visual_vs_textual":
                        result.update(rsa_cee_visual=result["rsa_a"], rsa_cee_textual=result["rsa_b"])
                    else:
                        result.update(rsa_post=result["rsa_a"], rsa_pre=result["rsa_b"])
                    rows.append(result)
    return rows


def generation_wide(rows):
    groups = {}
    for row in rows:
        if row["contrast"] != "generation_post_vs_pre":
            continue
        key = (row["layer_index"], row["embedding_distance"], row["rsa_coefficient"])
        base = groups.setdefault(key, {k: row[k] for k in (
            "model", "model_id", "run_id", "seed", "stimulus_seed", "layer_index", "level", "embedding_distance",
            "rsa_coefficient", "source_a", "source_b", "n_samples", "n_colors", "shared_colors",
            "permutation_rounds", "requested_permutation_rounds", "test", "null_hypothesis")})
        suffix = "visual" if row["reference"] == "image_tokens" else "textual"
        for target, source in (("rsa_pre", "rsa_pre"), ("rsa_post", "rsa_post"),
                               ("delta", "delta_rsa"), ("p_delta", "p_delta"),
                               ("q_delta", "q_delta_fdr"), ("status", "status"),
                               ("permutation_seed", "permutation_seed")):
            base[f"{target}_{suffix}"] = row[source]
    return list(groups.values())


def visual_equivalence(output, dataset, layers, model_id, seed, include_cee=True, include_ceg=True):
    """Compare RAW mean-pooled image states, preserving norms for difference checks."""
    paths = [row["absolute_image_path"] for row in dataset.rows]
    conditions = {}
    unavailable = []
    specifications = [("visual_only", output / "embeddings/image_query_embeddings.npz", "image_layer_embeddings", "pooling")]
    if include_cee:
        specifications.append(("cee", output / "embeddings/cee_ground_truth_embeddings.npz", "cee_raw_image_layer_embeddings", "image_pooling"))
    if include_ceg:
        specifications.append(("generation", output / "ceg_generation_side/embeddings/generation_image_embeddings.npz", "image_layer_embeddings", "pooling"))
    for name, path, key, pooling_key in specifications:
        if not path.exists():
            unavailable.append(name)
            continue
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                unavailable.append(name)
                continue
            if data["layer_indices"].tolist() != layers or data["image_paths"].tolist() != paths:
                raise ValueError(f"Visual sanity cache support mismatch: {name}")
            if "model_id" in data and str(data["model_id"]) != model_id:
                raise ValueError(f"Visual sanity model mismatch: {name}")
            if pooling_key not in data or str(data[pooling_key]) != "mean":
                unavailable.append(name + "_missing_or_nonmean_pooling")
                continue
            conditions[name] = data[key]
    rows = []
    for a, b in combinations(conditions, 2):
        av, bv = conditions[a], conditions[b]
        if av.shape != bv.shape or av.ndim != 3 or av.shape[:2] != (len(layers), len(paths)):
            raise ValueError("Visual sanity arrays differ in shape")
        for pos, layer in enumerate(layers):
            first, second = av[pos].astype(float), bv[pos].astype(float)
            norms = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
            cosine = np.divide(np.sum(first * second, axis=1), norms,
                               out=np.full(len(first), np.nan), where=norms > 0)
            difference = np.abs(first - second)
            rows.append(dict(model=model_id, model_id=model_id, seed=seed, run_id=str(output.resolve()),
                layer_index=layer, condition_a=a, condition_b=b, n_samples=len(first),
                mean_cosine_similarity=float(cosine.mean()), min_cosine_similarity=float(cosine.min()),
                max_abs_difference=float(difference.max()), mean_abs_difference=float(difference.mean()),
                pooling="raw_mean", status="computed" if np.isfinite(cosine).all() else "zero_or_nonfinite_vector"))
    return rows, unavailable
