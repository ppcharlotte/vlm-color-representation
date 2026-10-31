"""Canonical long-format, layerwise comparisons for the colour pipeline (v2)."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import json
import warnings
import numpy as np

from alignment_statistics import compare_rdms, rdm_alignment, cka_permutation, paired_category_swap, distance_regression, stable_seed
from analysis_extensions import template_robustness, paired_contrasts, generation_wide, visual_equivalence
from structured_color_utils import (
    BASIC_COLORS, centroids, ciede2000_matrix, linear_probe, load_dataset,
    rdm, rgb_to_lab, save_matrix_csv, write_csv, write_json,
)


@dataclass
class Source:
    name: str
    values: np.ndarray
    layers: list[int]
    indices: np.ndarray
    fixed: bool = False
    category_only: bool = False
    mismatch_repeat: int | None = None

    def at(self, layer):
        return self.values[0 if self.fixed else self.layers.index(layer)]


def validate_source(source, n_samples, expected_layers):
    if source.values.ndim != 3 or source.values.shape[0] != len(source.layers):
        raise ValueError(f"Invalid layer array: {source.name}")
    if not source.fixed and source.layers != expected_layers:
        raise ValueError(f"Layer mismatch: {source.name}")
    if len(np.unique(source.indices)) != len(source.indices) or np.any(source.indices < 0) or np.any(source.indices >= n_samples):
        raise ValueError(f"Invalid sample indices: {source.name}")
    if not source.category_only and source.values.shape[1] != len(source.indices):
        raise ValueError(f"Sample count mismatch: {source.name}")
    if source.category_only and source.values.shape[1] != len(BASIC_COLORS):
        raise ValueError(f"Category count mismatch: {source.name}")
    if not np.isfinite(source.values).all():
        raise ValueError(f"Non-finite embeddings: {source.name}")


def load_sources(output, dataset, include_cee=True, include_ceg=True, include_mismatched=True):
    n = len(dataset.rows)
    if [int(row["sample_index"]) for row in dataset.rows] != list(range(n)):
        raise ValueError("Dataset sample_index must equal row position (0..N-1)")
    full = np.arange(n)
    with np.load(output / "embeddings/text_embeddings.npz", allow_pickle=False) as text:
        model_id = str(text["model_id"]) if "model_id" in text else None
        layers = text["layer_indices"].astype(int).tolist()
        if len(layers) != len(set(layers)) or 0 not in layers:
            raise ValueError("Unique layer indices including 0 are required")
        if text["colors"].tolist() != BASIC_COLORS:
            raise ValueError("Text color order does not match BASIC_COLORS")
        sources = [Source("text_color_words", text["text_layer_embeddings"], layers, full, category_only=True)]
    with np.load(output / "embeddings/image_query_embeddings.npz", allow_pickle=False) as image:
        if model_id and "model_id" in image and str(image["model_id"]) != model_id:
            raise ValueError("Text and image embeddings come from different models")
        if image["layer_indices"].astype(int).tolist() != layers:
            raise ValueError("Image/text layer mismatch")
        expected_paths = [row["absolute_image_path"] for row in dataset.rows]
        if image["image_paths"].tolist() != expected_paths:
            raise ValueError("Image embedding paths/order do not match this dataset")
        if "rgb" in image and not np.array_equal(image["rgb"], dataset.rgb):
            raise ValueError("Image RGB metadata differs from current dataset")
        if "is_onevision" in image and bool(int(image["is_onevision"])) and "projector_capture_version" not in image:
            raise ValueError("OneVision cache has unverified projector provenance; rerun extraction")
        for name, key in (("image_tokens", "image_layer_embeddings"),):
            sources.append(Source(name, image[key], layers, full))
        for name in ("pre_projector", "post_projector"):
            sources.append(Source(name, image[name][None], [0], full, fixed=True))
    cee_path = output / "embeddings/cee_ground_truth_embeddings.npz"
    if include_cee and cee_path.exists():
        with np.load(cee_path, allow_pickle=False) as cee:
            if model_id and "model_id" in cee and str(cee["model_id"]) != model_id:
                raise ValueError("CEe model differs from text model")
            if cee["image_paths"].tolist() != expected_paths or cee["labels"].tolist() != dataset.labels:
                raise ValueError("CEe dataset support differs from image embeddings")
            cee_layers = cee["layer_indices"].astype(int).tolist()
            sources.append(Source("cee", cee["cee_layer_embeddings"], cee_layers, full))
            if "cee_image_layer_embeddings" in cee:
                sources.append(Source("cee_image_tokens", cee["cee_image_layer_embeddings"], cee_layers, full))
            if "matched_text_layer_embeddings" in cee and "matched_design_version" in cee:
                sources.append(Source("matched_text", cee["matched_text_layer_embeddings"], cee_layers, full))
    if include_ceg:
        for name, filename, key in (
            ("ceg_decision", "ceg_decision_embeddings.npz", "ceg_decision_layer_embeddings"),
            ("ceg_token", "ceg_token_from_generate_embeddings.npz", "ceg_token_from_generate_layer_embeddings"),
        ):
            path = output / "ceg_generation_side/embeddings" / filename
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    ids = data["sample_indices"].astype(int)
                    if model_id and "model_id" in data and str(data["model_id"]) != model_id:
                        raise ValueError("CEg model differs from text model")
                    if "image_paths" in data and data["image_paths"].tolist() != [expected_paths[int(i)] for i in ids]:
                        raise ValueError("CEg image paths differ from dataset")
                    if data["human_categories"].tolist() != [dataset.labels[int(i)] for i in ids]:
                        raise ValueError("CEg labels do not match the dataset")
                    sources.append(Source(name, data[key], data["layer_indices"].astype(int).tolist(), ids))
    mismatch_path = output / "embeddings/cee_mismatched_embeddings.npz"
    if include_cee and include_mismatched and mismatch_path.exists():
        if not cee_path.exists():
            raise ValueError("Mismatched condition requires the matching CEe baseline")
        with np.load(mismatch_path, allow_pickle=False) as data, np.load(cee_path, allow_pickle=False) as cee:
            if (str(data["model_id"]) != str(cee["model_id"]) or str(data["template"]) != str(cee["template"])
                    or data["image_paths"].tolist() != expected_paths or data["labels"].tolist() != dataset.labels):
                raise ValueError("Mismatched condition uses a different model, prompt or dataset")
            mappings = data["donor_indices"]
            values = data["mismatched_layer_embeddings"]
            if values.ndim != 4 or values.shape[0] != len(mappings) or mappings.shape[1:] != (n,):
                raise ValueError("Malformed mismatched-color arrays")
            for repeat, mapping in enumerate(mappings):
                if (not np.array_equal(np.sort(mapping), full)
                        or np.any(np.asarray(dataset.labels) == np.asarray(dataset.labels)[mapping])):
                    raise ValueError("Wrong-color mapping must be a different-category bijection")
                sources.append(Source(f"cee_mismatched_r{repeat}", values[repeat],
                    data["layer_indices"].astype(int).tolist(), full, mismatch_repeat=repeat))
    for source in sources:
        validate_source(source, n, layers)
    return sources, layers


def adjust_rows(rows, keys):
    from perceptual_alignment_analysis import fdr_bh
    families = {}
    for row in rows:
        family = "|".join(str(row[key]) for key in keys)
        row["fdr_family"] = family
        families.setdefault(family, []).append(row)
    for items in families.values():
        for row, value in zip(items, fdr_bh([r["p_value"] for r in items])):
            row["q_fdr"] = value


def write_rows(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(path, fields or ["status"], rows)


def select_values(source, layer, ids):
    positions = {int(idx): pos for pos, idx in enumerate(source.indices)}
    return source.at(layer)[[positions[int(i)] for i in ids]]


def run_extended_analysis(args):
    output = args.output_dir
    dataset = load_dataset(args.ground_truth, args.dataset_root)
    sources, layers = load_sources(output, dataset, not args.exclude_cee, not args.exclude_ceg,
                                  not getattr(args, "exclude_mismatched", False))
    with np.load(output / "embeddings/image_query_embeddings.npz", allow_pickle=False) as image:
        model_id = str(image["model_id"]) if "model_id" in image else "unknown_legacy_cache"
        stimulus_seed = int(image["seed"]) if "seed" in image else args.seed
        visual_pooling = str(image["pooling"]) if "pooling" in image else "unknown_legacy"
    dataset_metadata = output / "data/metadata.json"
    if dataset_metadata.exists():
        stimulus_seed = json.loads(dataset_metadata.read_text()).get("seed", stimulus_seed)
    design = dict(visual_pooling=visual_pooling, category_centroids="arithmetic_mean", perceptual_reference="sRGB_D65_CIEDE2000")
    with np.load(output / "embeddings/text_embeddings.npz", allow_pickle=False) as text:
        design["text_contexts"] = text["contexts"].tolist() if "contexts" in text else "unknown_legacy"
    if not args.exclude_cee and (output / "embeddings/cee_ground_truth_embeddings.npz").exists():
        with np.load(output / "embeddings/cee_ground_truth_embeddings.npz", allow_pickle=False) as cee:
            design["cee_template"] = str(cee["template"]) if "template" in cee else "unknown_legacy"
    distances = list(dict.fromkeys(x.strip() for x in args.embedding_distances.split(",")))
    coefficients = list(dict.fromkeys(x.strip() for x in args.rsa_coefficients.split(",")))
    if not set(distances) <= {"cosine", "pearson"} or not set(coefficients) <= {"spearman", "kendall_tau_b"}:
        raise ValueError("Unsupported distance or RSA coefficient")
    if min(args.sample_permutation_rounds, args.category_permutation_rounds) < 1:
        raise ValueError("Permutation counts must be positive")
    if args.sample_permutation_rounds < 999:
        warnings.warn("Sample permutation budget below 999: p-value resolution is limited; use a larger final-run budget.")
    sample_scope = getattr(args, "sample_permutation_scope", "core")
    core_perceptual_sources = {"image_tokens", "cee", "matched_text", "ceg_decision", "ceg_token"}
    labels = dataset.labels
    lab = rgb_to_lab(dataset.rgb)
    all_ids = np.arange(len(labels))
    category_sources = sources + [Source("perceptual", lab[None], [0], all_ids, fixed=True)]
    rsa_rows, cka_rows, probe_rows, delta_rows, regression_rows = [], [], [], [], []
    appendix = getattr(args, "appendix_metrics", False)
    run_id = str(output.resolve())
    generation = {"ceg_decision", "ceg_token"}
    written_rdms = set()
    support_records = {}
    human_cache = {}
    write_json(output / "analysis/summary.json", {"analysis_version": 2, "status": "running"})
    seed = args.seed

    # Per-layer cache bounds memory and avoids repeating distance construction.
    for layer in layers:
        print(f"Analysing layer {layer}: category and sample comparisons", flush=True)
        cache = {}

        def geometry(source, ids, level, distance):
            if source.name == "perceptual":
                distance = "ciede2000"
            key = (source.name, tuple(ids), level, distance)
            target_cache = human_cache if source.name == "perceptual" else cache
            if key in target_cache:
                return target_cache[key]
            active_labels = [labels[int(i)] for i in ids]
            colors = [c for c in BASIC_COLORS if c in active_labels]
            if source.category_only:
                values = source.at(layer)[[BASIC_COLORS.index(c) for c in colors]]
            else:
                values = select_values(source, layer, ids)
                if level == "category":
                    values = centroids(values, active_labels, colors)
            matrix = ciede2000_matrix(values) if source.name == "perceptual" else rdm(values, distance)
            target_cache[key] = (values, matrix, colors)
            return target_cache[key]

        for left, right in combinations(category_sources, 2):
            if (left.name in generation or right.name in generation) and not (
                    left.name in generation and right.name in {"image_tokens", "text_color_words", "perceptual"}
                    or right.name in generation and left.name in {"image_tokens", "text_color_words", "perceptual"}):
                continue
            if left.mismatch_repeat is not None and right.mismatch_repeat is not None:
                continue  # Pairing repeats are replicates, not distinct theoretical spaces.
            # Fixed-to-fixed comparisons have one physical value, not a layer trajectory.
            if left.fixed and right.fixed and layer != 0:
                continue
            ids = np.intersect1d(left.indices, right.indices)
            support_id = stable_seed(0, *ids)
            support_records[support_id] = dict(support_id=support_id, n_samples=len(ids),
                                               sample_indices="|".join(map(str, ids)))
            colors = [c for c in BASIC_COLORS if c in {labels[int(i)] for i in ids}]
            base = dict(model_id=model_id, model=model_id, run_id=run_id, layer_index=layer, source_a=left.name, source_b=right.name,
                        source_a_layer="fixed" if left.fixed else layer,
                        source_b_layer="fixed" if right.fixed else layer,
                        n_samples=len(ids), n_colors=len(colors), shared_colors="|".join(colors),
                        support="pairwise_intersection", support_id=support_id, seed=seed, stimulus_seed=stimulus_seed)
            base["mismatch_repeat"] = left.mismatch_repeat if left.mismatch_repeat is not None else right.mismatch_repeat
            if len(colors) < 3:
                continue
            # Persist full-support category RDMs separately from pair-specific ones.
            for distance in distances:
                a_values, a_cat, _ = geometry(left, ids, "category", distance)
                b_values, b_cat, _ = geometry(right, ids, "category", distance)
                rdm_paths = []
                for source, matrix in ((left, a_cat), (right, b_cat)):
                    stage = "fixed" if source.fixed else f"layer_{layer}"
                    metric = "ciede2000" if source.name == "perceptual" else distance
                    path = output / "rdm/category_v2" / f"{source.name}_{stage}_{metric}_support_{support_id}.csv"
                    if path not in written_rdms:
                        save_matrix_csv(path, colors, matrix, "color")
                        written_rdms.add(path)
                    rdm_paths.append(str(path.relative_to(output)))
                levels = [("category", a_cat, b_cat, args.category_permutation_rounds)]
                if not left.category_only and not right.category_only:
                    _, a_sample, _ = geometry(left, ids, "sample", distance)
                    _, b_sample, _ = geometry(right, ids, "sample", distance)
                    levels.append(("sample", a_sample, b_sample, args.sample_permutation_rounds))
                for level, a_rdm, b_rdm, rounds in levels:
                    if "perceptual" in (left.name, right.name):
                        model_rdm, human_rdm = (b_rdm, a_rdm) if left.name == "perceptual" else (a_rdm, b_rdm)
                        regression_rows.append(dict(base, level=level, embedding_distance=distance,
                            **distance_regression(model_rdm, human_rdm)))
                    for coefficient in coefficients:
                        test_seed = stable_seed(seed, layer, left.name, right.name, level)
                        pair = {left.name, right.name}
                        core_pair = "perceptual" in pair and bool(pair & core_perceptual_sources)
                        infer = level == "category" or sample_scope == "all" or (
                            sample_scope == "core" and core_pair and coefficient == "spearman" and distance == "cosine")
                        actual_rounds = rounds if infer else 0
                        if infer:
                            print(f"  {level}: {left.name} vs {right.name}, {distance}/{coefficient}, {rounds} permutations", flush=True)
                            value, p, n, status = compare_rdms(a_rdm, b_rdm, coefficient, rounds, test_seed)
                        else:
                            value = rdm_alignment(a_rdm, b_rdm, coefficient)
                            p, n = np.nan, len(a_rdm) * (len(a_rdm) - 1) // 2
                            status = "descriptive_only" if np.isfinite(value) else "undefined_observed"
                        rsa_rows.append(dict(base, level=level, embedding_distance=distance,
                            distance_a="ciede2000" if left.name == "perceptual" else distance,
                            distance_b="ciede2000" if right.name == "perceptual" else distance,
                            rsa_coefficient=coefficient, alignment=value, p_value=p, n_pairs=n,
                            permutation_rounds=actual_rounds, permutation_seed=test_seed if infer else "", status=status,
                            rdm_a_csv=rdm_paths[0] if level == "category" else "",
                            rdm_b_csv=rdm_paths[1] if level == "category" else "",
                            test="two_sided_RDM_label_permutation" if infer else "not_tested_descriptive_only"))
            # CKA operates on coordinates; human CKA uses LAB, not CIEDE2000.
            if appendix and not generation.intersection((left.name, right.name)):
                value, p, status = cka_permutation(a_values, b_values, args.category_permutation_rounds,
                                                 stable_seed(seed, layer, left.name, right.name, "cka"))
                cka_rows.append(dict(base, alignment=value, p_value=p, status=status,
                    level="category", metric="linear_cka", human_reference="CIELAB_coordinates" if "perceptual" in (left.name, right.name) else "none",
                    test="upper_tail_category_label_permutation", permutation_rounds=args.category_permutation_rounds))

        for source in sources:
            if not appendix or source.name in generation or source.category_only or source.name == "matched_text" or source.mismatch_repeat is not None or (source.fixed and layer != 0):
                continue
            ids = source.indices
            active_labels = [labels[int(i)] for i in ids]
            colors = [c for c in BASIC_COLORS if c in active_labels]
            probe = linear_probe(source.at(layer), active_labels, colors,
                                 args.linear_probe_test_size, args.ridge_alpha, seed)
            probe_rows.append(dict(model_id=model_id, source=source.name, layer_index="fixed" if source.fixed else layer,
                accuracy=probe["accuracy"], train_n=probe["train_n"], test_n=probe["test_n"], seed=seed))

        by_name = {source.name: source for source in sources}
        contrasts = []
        if "cee" in by_name and "matched_text" in by_name:
            contrasts.append(("cee", "matched_text", None))
            for source in sources:
                if source.mismatch_repeat is not None:
                    contrasts.extend([("cee", source.name, source.mismatch_repeat),
                                      (source.name, "matched_text", source.mismatch_repeat)])
        for name_a, name_b, repeat in contrasts:
            a, b = by_name[name_a], by_name[name_b]
            ids = np.intersect1d(a.indices, b.indices)
            active_labels = [labels[int(i)] for i in ids]
            colors = [c for c in BASIC_COLORS if c in active_labels]
            pa = centroids(select_values(a, layer, ids), active_labels, colors)
            pb = centroids(select_values(b, layer, ids), active_labels, colors)
            reference = ciede2000_matrix(centroids(lab[ids], active_labels, colors))
            for distance in distances:
                for coefficient in coefficients:
                    pseed = stable_seed(seed, "rq3_swap", layer, name_a, name_b, distance, coefficient)
                    row = paired_category_swap(pa, pb, reference, distance, coefficient, args.category_permutation_rounds, pseed)
                    row.update(model=model_id, model_id=model_id, run_id=run_id, seed=seed,
                        stimulus_seed=stimulus_seed,
                        layer_index=layer, level="category", n_samples=len(ids), n_colors=len(colors),
                        source_a=name_a, source_b=name_b, reference="perceptual", mismatch_repeat=repeat,
                        embedding_distance=distance, rsa_coefficient=coefficient,
                        n_pairs=len(colors)*(len(colors)-1)//2, shared_colors="|".join(colors),
                        pairing_scope="fixed_assignment" if repeat is not None else "no_mismatch",
                        alignment_a=row["rsa_a"], alignment_b=row["rsa_b"], delta_alignment=row["delta_rsa"])
                    delta_rows.append(row)

        # Checkpoint raw rows during expensive sample-level Kendall permutations.
        write_rows(output / "analysis/alignment_layerwise_v2.csv", rsa_rows)
        write_rows(output / "analysis/rq3_permutation_v2.csv", delta_rows)

    # One family per resolution, includes both metric choices, all pairs/layers in this run.
    adjust_rows(rsa_rows, ["level"])
    adjust_rows(cka_rows, ["metric"])
    if delta_rows:
        adjust_rows(delta_rows, ["source_a", "source_b", "embedding_distance", "rsa_coefficient"])
        for row in delta_rows:
            row.update(p_delta=row["p_value"], q_delta_fdr=row["q_fdr"])
    write_rows(output / "analysis/alignment_layerwise_v2.csv", rsa_rows)
    if appendix:
        write_rows(output / "analysis/appendix_cka_layerwise_v2.csv", cka_rows)
        write_rows(output / "analysis/appendix_linear_probe_v2.csv", probe_rows)
    write_rows(output / "analysis/rq3_permutation_v2.csv", delta_rows)
    mismatch_rows = mismatch_summary(delta_rows)
    write_rows(output / "analysis/rq3_mismatch_repeat_summary.csv", mismatch_rows)
    from analysis_plotting import repeat_alignment_summary
    write_rows(output / "analysis/mismatch_alignment_summary_v2.csv", repeat_alignment_summary(rsa_rows))
    write_rows(output / "analysis/mismatch_regression_summary_v2.csv", repeat_alignment_summary(regression_rows, "R2"))
    write_rows(output / "analysis/perceptual_distance_regression_v2.csv", regression_rows)
    write_rows(output / "analysis/sample_support_v2.csv", list(support_records.values()))
    template_rows, membership, template_status = template_robustness(output, lab, labels, layers, model_id,
        seed, args.category_permutation_rounds, coefficients)
    for row in template_rows:
        row["stimulus_seed"] = stimulus_seed
    adjust_rows(template_rows, ["comparison_type", "embedding_distance", "rsa_coefficient"])
    write_rows(output / "analysis/text_context_robustness_v2.csv", template_rows)
    write_rows(output / "analysis/text_template_subsets_v2.csv", membership)
    paired_rows = paired_contrasts(sources, lab, labels, layers, model_id, run_id, seed, distances,
        coefficients, args.category_permutation_rounds, getattr(args, "cee_reference_difference", False))
    adjust_rows(paired_rows, ["contrast", "reference", "embedding_distance", "rsa_coefficient"])
    for row in paired_rows:
        row["q_delta_fdr"] = row["q_fdr"]
        row["stimulus_seed"] = stimulus_seed
    write_rows(output / "analysis/paired_rsa_differences_v2.csv", paired_rows)
    write_rows(output / "analysis/cee_no_image_differences_v2.csv", [r for r in paired_rows if r["contrast"] == "cee_image_vs_no_image"])
    write_rows(output / "analysis/cee_visual_textual_differences_v2.csv", [r for r in paired_rows if r["contrast"] == "cee_visual_vs_textual"])
    write_rows(output / "analysis/generation_pre_post_differences_v2.csv", generation_wide(paired_rows))
    sanity_rows, missing_sanity = visual_equivalence(output, dataset, layers, model_id, seed, not args.exclude_cee, not args.exclude_ceg)
    for row in sanity_rows:
        row["stimulus_seed"] = stimulus_seed
    write_rows(output / "analysis/visual_state_equivalence_v2.csv", sanity_rows)
    plot_manifest = []
    if not args.no_plots:
        from analysis_plotting import plot_all_results
        plot_manifest = plot_all_results(output, rsa_rows, cka_rows, delta_rows, mismatch_rows, probe_rows,
            regression_rows=regression_rows, template_rows=template_rows, paired_rows=paired_rows, sanity_rows=sanity_rows,
            rdm_paths=written_rdms)
    write_json(output / "analysis/summary.json", dict(
        analysis_version=2, status="complete", model_id=model_id, sources=[s.name for s in sources], layers=layers,
        seed=seed, stimulus_seed=stimulus_seed, run_id=run_id, analysis_design=design,
        embedding_distances=distances, rsa_coefficients=coefficients,
        sample_permutation_rounds=args.sample_permutation_rounds,
        sample_permutation_scope=sample_scope,
        sample_core_sources=sorted(core_perceptual_sources),
        category_permutation_rounds=args.category_permutation_rounds,
        appendix_metrics=appendix, template_robustness_status=template_status,
        visual_sanity_missing=missing_sanity, canonical_visual="image_tokens (visual-only/neutral-prompt condition)",
        visual_sanity_note="Inspect raw differences before claiming equivalence; references are never silently substituted",
        rq3_status="computed" if delta_rows else "missing_matched_condition_rerun_CEe_extraction",
        mismatch_repeats=sum(s.mismatch_repeat is not None for s in sources),
        mismatch_reference="Original target RGB/category, not donor RGB; category swaps conditional on each fixed mapping",
        projector_comparisons="fixed pre/post against every dynamic layer; fixed/fixed once at layer 0",
        category_support="pairwise intersection of sample IDs, including CEg comparisons",
        multiple_testing="Main RSA: each level across all pairs/layers/metrics; delta: layers within contrast/reference/distance/coefficient; templates: layers/pairs within comparison_type/coefficient",
        inference_scope="Delta: category-wise condition-label exchangeability; not unrestricted equality of correlations",
        files=["alignment_layerwise_v2.csv", "rq3_permutation_v2.csv", "rq3_mismatch_repeat_summary.csv",
               "sample_support_v2.csv", "perceptual_distance_regression_v2.csv", "text_context_robustness_v2.csv",
               "text_template_subsets_v2.csv", "paired_rsa_differences_v2.csv", "cee_no_image_differences_v2.csv",
               "cee_visual_textual_differences_v2.csv", "generation_pre_post_differences_v2.csv", "visual_state_equivalence_v2.csv"]
              + (["appendix_cka_layerwise_v2.csv", "appendix_linear_probe_v2.csv"] if appendix else []),
        plots_disabled=bool(args.no_plots), plot_manifest="plots/v2/plot_manifest.csv" if plot_manifest else None,
        n_plots=len(plot_manifest), n_rsa_rows=len(rsa_rows), n_cka_rows=len(cka_rows)))


def mismatch_summary(rows):
    """Descriptive variation across assignments; do not treat repeats as subjects."""
    groups = {}
    for row in rows:
        if row.get("mismatch_repeat") is None:
            continue
        contrast = "correct_minus_mismatched" if row["source_a"] == "cee" else "mismatched_minus_text"
        key = (row["model_id"], row["layer_index"], row["embedding_distance"], row["rsa_coefficient"], contrast)
        groups.setdefault(key, []).append(row["delta_alignment"])
    output = []
    for (model, layer, distance, coefficient, contrast), values in groups.items():
        values = np.asarray(values, float)
        valid = values[np.isfinite(values)]
        output.append(dict(model_id=model, layer_index=layer, embedding_distance=distance,
            rsa_coefficient=coefficient, contrast=contrast, n_repeats=len(values), valid_repeats=len(valid),
            mean_delta=float(valid.mean()) if len(valid) else np.nan,
            sd_delta=float(valid.std(ddof=1)) if len(valid) > 1 else np.nan,
            min_delta=float(valid.min()) if len(valid) else np.nan,
            max_delta=float(valid.max()) if len(valid) else np.nan,
            fraction_positive=float((valid > 0).mean()) if len(valid) else np.nan,
            inference="descriptive_across_pairings_no_pvalue"))
    return output
