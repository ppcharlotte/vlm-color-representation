#!/usr/bin/env python3
"""Standalone shared-label, two-sided RSA max-statistics; never runs a model.

Reads v2 outputs, optionally reconstructing sample RDMs from saved embeddings.
See README_rsa_maxstat.md for inference scope and command examples.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from alignment_statistics import COEFFICIENTS, rdm_alignment, stable_seed
from plot_randomness_summary import read_category_rdm
from structured_color_utils import (
    BASIC_COLORS, ciede2000_matrix, load_dataset, rankdata_average, rdm,
    rgb_to_lab, upper_triangle_values,
)

# Logical labels are CLI/report labels only; source identifiers remain canonical.
ALLOWLIST = {
    "textual-perceptual": ("text_color_words", "perceptual"),
    "visual-perceptual": ("image_tokens", "perceptual"),
    "textual-visual": ("text_color_words", "image_tokens"),
    "pre_projector-textual": ("pre_projector", "text_color_words"),
    "pre_projector-perceptual": ("pre_projector", "perceptual"),
    "cee-perceptual": ("cee", "perceptual"),
    "cee-no_image": ("cee", "matched_text"),
    "cee-visual": ("cee", "image_tokens"),
    "cee-mismatched_image": ("cee", "cee_mismatched_mean"),
    "ceg_decision-perceptual": ("ceg_decision", "perceptual"),
    "ceg_decision-visual": ("ceg_decision", "image_tokens"),
    "ceg_decision-textual": ("ceg_decision", "text_color_words"),
    "ceg_token-perceptual": ("ceg_token", "perceptual"),
    "ceg_token-visual": ("ceg_token", "image_tokens"),
    "ceg_token-textual": ("ceg_token", "text_color_words"),
}
FAMILY_FIELDS = ("model_id", "comparison", "level", "embedding_distance", "rsa_coefficient")


def normalized_source(name):
    return "cee_mismatched_mean" if name.startswith("cee_mismatched_r") else name


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_close(actual, expected, context):
    if not np.isfinite(actual) or not np.isfinite(expected) or not np.isclose(
            actual, expected, rtol=1e-6, atol=1e-7):
        raise ValueError(f"{context}: recomputed {actual}, saved {expected}")


def shared_orders(n_items, rounds, seed):
    rng = np.random.default_rng(seed)
    return np.asarray([rng.permutation(n_items) for _ in range(rounds)], dtype=np.int32)


def permuted_rsa(a, b, coefficient, orders, batch_size=8):
    """The existing marginal null: permute BOTH axes of b, never a or edges.

    Streaming pairs uses the same pre-generated orders array for every layer,
    seed and mismatch repeat. It is algebraically identical to permutation-first
    evaluation, without retaining all sample RDMs in memory.
    """
    if a.shape != b.shape or a.shape != (orders.shape[1], orders.shape[1]):
        raise ValueError("RDM/permutation dimensions differ")
    tri = np.triu_indices(len(a), 1)
    av, bv = a[tri], b[tri]
    if not np.isfinite(av).all() or not np.isfinite(bv).all():
        raise ValueError("Nonfinite RDM entries")
    result = np.empty(len(orders))
    if coefficient == "spearman":
        av, bv = rankdata_average(av), rankdata_average(bv)
        av, bv = av - av.mean(), bv - bv.mean()
        an, bn = np.linalg.norm(av), np.linalg.norm(bv)
        if an == 0 or bn == 0:
            raise ValueError("Constant RDM ranks")
        av, bv = av / an, bv / bn
        rank_matrix = np.zeros_like(b, dtype=float)
        rank_matrix[tri] = bv
        rank_matrix[(tri[1], tri[0])] = bv
        for start in range(0, len(orders), batch_size):
            order = orders[start:start + batch_size]
            values = rank_matrix[order[:, tri[0]], order[:, tri[1]]]
            result[start:start + len(order)] = values @ av
    else:
        fn = COEFFICIENTS[coefficient]
        for i, order in enumerate(orders):
            result[i] = fn(av, b[order[tri[0]], order[tri[1]]])
    if not np.isfinite(result).all():
        raise ValueError("Undefined permuted RSA; no null draws were dropped")
    return result


def maxstat_pvalues(observed, null):
    """null is [permutation, layer], already averaged repeats then seeds."""
    observed, null = np.asarray(observed), np.asarray(null)
    if null.ndim != 2 or null.shape[1] != len(observed) or not np.isfinite(null).all():
        raise ValueError("Invalid joint null")
    maximum = np.max(np.abs(null), axis=1)
    # Same round-off allowance as the existing cross-seed mean RSA test.
    threshold = np.abs(observed) - 1e-14
    raw = (1 + np.sum(np.abs(null) >= threshold[None, :], axis=0)) / (len(null) + 1)
    corrected = (1 + np.sum(maximum[:, None] >= threshold[None, :], axis=0)) / (len(null) + 1)
    peak = int(np.argmax(np.abs(observed)))
    global_p = (1 + np.count_nonzero(maximum >= threshold[peak])) / (len(null) + 1)
    assert np.all(maximum[:, None] >= np.abs(null))
    assert np.all(corrected >= raw)
    assert corrected[peak] == global_p
    return raw, corrected, float(global_p), peak, maximum


class Inputs:
    """Read-only adapters; large NPZ members are memory-mapped in a temp dir."""

    def __init__(self, scratch):
        self.scratch = Path(scratch)
        self.arrays, self.datasets, self.supports, self.fixed_rdms = {}, {}, {}, {}

    def array(self, path, key):
        path = Path(path)
        ident = (str(path), key)
        if ident not in self.arrays:
            target = self.scratch / (hashlib.sha256(repr(ident).encode()).hexdigest() + ".npy")
            with zipfile.ZipFile(path) as archive, archive.open(key + ".npy") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            self.arrays[ident] = np.load(target, mmap_mode="r", allow_pickle=False)
        return self.arrays[ident]

    def dataset(self, run):
        if run not in self.datasets:
            data = load_dataset(run / "data/ground_truth.csv", run / "data")
            if [int(r["sample_index"]) for r in data.rows] != list(range(len(data.rows))):
                raise ValueError("Dataset sample_index must match original row order")
            self.datasets[run] = data
        return self.datasets[run]

    def support(self, run, row):
        if run not in self.supports:
            self.supports[run] = {r["support_id"]: r["sample_indices"] for r in
                                  read_csv(run / "analysis/sample_support_v2.csv")}
        ids = tuple(int(i) for i in self.supports[run][row["support_id"]].split("|"))
        if len(ids) != int(row["n_samples"]) or tuple(sorted(set(ids))) != ids:
            raise ValueError("Invalid original pairwise sample support")
        return ids

    def category(self, run, row, side):
        path = Path(row.get(f"rdm_{side}_csv", ""))
        if not row.get(f"rdm_{side}_csv"):
            stage = "fixed" if row[f"source_{side}_layer"] == "fixed" else f"layer_{row['layer_index']}"
            name = row[f"source_{side}"]
            metric = "ciede2000" if name == "perceptual" else row["embedding_distance"]
            path = Path("rdm/category_v2") / f"{name}_{stage}_{metric}_support_{row['support_id']}.csv"
        if path.is_absolute():
            # Relocate a copied run without depending on the old server path.
            parts = path.parts
            if "rdm" not in parts:
                raise ValueError(f"Cannot relocate RDM path {path}")
            path = Path(*parts[parts.index("rdm"):])
        path = (run / path).resolve()
        if not path.is_relative_to(run):
            raise ValueError("RDM path lies outside its input run")
        matrix = read_category_rdm(path)
        colors = row["shared_colors"].split("|")
        if len(colors) != 11 or set(colors) != set(BASIC_COLORS):
            raise ValueError("Existing cross-seed category aggregation requires all 11 colors")
        return matrix

    def sample(self, run, row, side):
        ids = np.asarray(self.support(run, row))
        name, layer = row[f"source_{side}"], int(row["layer_index"])
        fixed_key = (run, name, row["support_id"], row["embedding_distance"])
        if fixed_key in self.fixed_rdms:
            return self.fixed_rdms[fixed_key]
        data = self.dataset(run)
        if name == "perceptual":
            return self.cache_fixed(fixed_key, ciede2000_matrix(rgb_to_lab(data.rgb)[ids]))
        paths = {
            "image_tokens": ("embeddings/image_query_embeddings.npz", "image_layer_embeddings"),
            "pre_projector": ("embeddings/image_query_embeddings.npz", "pre_projector"),
            "cee": ("embeddings/cee_ground_truth_embeddings.npz", "cee_layer_embeddings"),
            "matched_text": ("embeddings/cee_ground_truth_embeddings.npz", "matched_text_layer_embeddings"),
            "ceg_decision": ("ceg_generation_side/embeddings/ceg_decision_embeddings.npz", "ceg_decision_layer_embeddings"),
            "ceg_token": ("ceg_generation_side/embeddings/ceg_token_from_generate_embeddings.npz", "ceg_token_from_generate_layer_embeddings"),
        }
        repeat = None
        if name.startswith("cee_mismatched_r"):
            repeat = int(name.removeprefix("cee_mismatched_r"))
            path, key = "embeddings/cee_mismatched_embeddings.npz", "mismatched_layer_embeddings"
        else:
            path, key = paths[name]
        path = run / path
        if str(self.array(path, "model_id")) != row["model_id"]:
            raise ValueError("Embedding model_id does not match RSA row")
        array = self.array(path, key)
        if name == "pre_projector":
            values = array
        else:
            layers = self.array(path, "layer_indices").tolist()
            values = array[repeat, layers.index(layer)] if repeat is not None else array[layers.index(layer)]
        if name.startswith("ceg_"):
            original_ids = self.array(path, "sample_indices").astype(int)
            if len(set(original_ids.tolist())) != len(original_ids):
                raise ValueError("Duplicate CEg cache sample IDs")
            if self.array(path, "human_categories").tolist() != [data.labels[i] for i in original_ids]:
                raise ValueError("CEg category metadata differs from dataset")
            positions = {int(v): i for i, v in enumerate(original_ids)}
            values = values[[positions[int(i)] for i in ids]]
        else:
            if len(values) != len(data.rows):
                raise ValueError("Cache/dataset sample count differs")
            values = values[ids]
        if repeat is not None:
            donor = self.array(path, "donor_indices")[repeat]
            labels = np.asarray(data.labels)
            if not np.array_equal(np.sort(donor), np.arange(len(labels))) or np.any(labels == labels[donor]):
                raise ValueError("Invalid saved mismatch assignment")
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite selected embeddings")
        matrix = rdm(values, row["embedding_distance"])
        return self.cache_fixed(fixed_key, matrix) if name == "pre_projector" else matrix

    def cache_fixed(self, key, matrix):
        if len(self.fixed_rdms) >= 8:
            self.fixed_rdms.pop(next(iter(self.fixed_rdms)))
        self.fixed_rdms[key] = matrix
        return matrix


def expected_layers(model, summaries, override):
    if override:
        layers = [int(i) for i in override.split(",")]
    else:
        declared = [s["layers"] for s in summaries if "layers" in s]
        if declared:
            if len(declared) != len(summaries) or any(x != declared[0] for x in declared):
                raise ValueError("Run metadata declares different/missing layer selections")
            layers = declared[0]
        elif "onevision" in model.lower() and "8b" in model.lower():
            layers = list(range(37))
        elif "llava-1.5-7b" in model.lower():
            layers = list(range(33))
        else:
            raise ValueError("Unknown expected layers: supply --layer-indices")
    if not layers or sorted(set(layers)) != layers or layers[0] < 0:
        raise ValueError("Expected layers must be sorted, unique, nonnegative")
    return layers


def aggregate_key(row):
    return tuple(str(row[k]) for k in ("model_id", "source_a", "source_b", "level",
                                      "embedding_distance", "rsa_coefficient", "layer_index"))


def analyze_family(key, entries, inputs, expected_seeds, summaries, args, aggregate):
    model, comparison, level, distance, coefficient = key
    base = dict(zip(FAMILY_FIELDS, key))
    family = "|".join(key)
    seed_rows = defaultdict(lambda: defaultdict(list))
    for run, row in entries:
        seed_rows[int(row.get("stimulus_seed") or row["seed"])][int(row["layer_index"])].append((run, row))
    if set(seed_rows) != set(expected_seeds):
        raise ValueError(f"Expected stimulus seeds {expected_seeds}; found {sorted(seed_rows)}")
    orientations = {(normalized_source(r["source_a"]), normalized_source(r["source_b"])) for _, r in entries}
    if len(orientations) != 1:
        raise ValueError("Source orientation differs; cannot preserve permutation side b")
    source_a, source_b = orientations.pop()
    fixed = all(r["source_a_layer"] == r["source_b_layer"] == "fixed" for _, r in entries)
    layers = [0] if fixed else expected_layers(model, summaries, args.layer_indices)
    support_by_seed, repeats_by_seed, n_by_seed = {}, {}, {}
    for seed in expected_seeds:
        if sorted(seed_rows[seed]) != layers:
            raise ValueError(f"Seed {seed}: expected layers {layers}, found {sorted(seed_rows[seed])}")
        supports, repeat_sets = [], []
        for layer in layers:
            group = seed_rows[seed][layer]
            repeats = [(r["source_a"], r["source_b"]) for _, r in group]
            if len(set(repeats)) != len(repeats):
                raise ValueError(f"Duplicate source/repeat rows at seed {seed}, layer {layer}")
            if comparison != "cee-mismatched_image" and len(group) != 1:
                raise ValueError("Unexpected repeated condition")
            repeat_sets.append(tuple(sorted(repeats)))
            for run, row in group:
                ids = inputs.support(run, row)
                supports.append((layer, ids, row["shared_colors"]))
                for side in ("a", "b"):
                    actual = row[f"source_{side}_layer"]
                    wanted = "fixed" if row[f"source_{side}"] in {"pre_projector", "perceptual"} else str(layer)
                    if str(actual) != wanted:
                        raise ValueError(f"Unexpected source layer {actual}, expected {wanted}")
        if any((ids, colors) != supports[0][1:] for _, ids, colors in supports):
            changed = sorted({layer for layer, ids, colors in supports if (ids, colors) != supports[0][1:]})
            raise ValueError(f"Seed {seed}: support/category masks differ across layers or mismatch repeats; baseline layer {supports[0][0]}, differing layers {changed}")
        if any(x != repeat_sets[0] for x in repeat_sets):
            raise ValueError(f"Seed {seed}: mismatch repeat set changes across layers")
        support_by_seed[seed] = supports[0][1]
        repeats_by_seed[seed] = len(repeat_sets[0])
        n_by_seed[seed] = 11 if level == "category" else len(supports[0][1])
        if comparison == "cee-mismatched_image":
            # Metadata declares the experimental assignments, not null draws.
            declared = [s.get("mismatch_repeats") for s in summaries
                        if int(s.get("stimulus_seed", s.get("seed", -1))) == seed]
            if declared and declared[0] is not None and repeats_by_seed[seed] != int(declared[0]):
                raise ValueError(f"Seed {seed}: missing original mismatch repeats")
    if len(set(n_by_seed.values())) != 1:
        raise ValueError(f"No identical sample permutation space across seeds: counts={n_by_seed}; no truncation/rematching performed")
    rounds = args.category_permutations if level == "category" else args.sample_permutations
    pseed = stable_seed(args.permutation_seed, "rsa_maxstat", *key)
    orders = shared_orders(next(iter(n_by_seed.values())), rounds, pseed)
    digest = hashlib.sha256(orders.tobytes()).hexdigest()
    observed_by_seed = np.empty((len(expected_seeds), len(layers)))
    null = np.zeros((rounds, len(layers)))
    old_p, old_q = defaultdict(dict), defaultdict(dict)
    # Streaming accumulation retains one shared permutation realization b across
    # ALL columns/seeds. No RNG call or fresh order appears inside these loops.
    for si, seed in enumerate(expected_seeds):
        for li, layer in enumerate(layers):
            group = seed_rows[seed][layer]
            observed_repeats = []
            for run, row in group:
                loader = inputs.category if level == "category" else inputs.sample
                a, b = loader(run, row, "a"), loader(run, row, "b")
                actual = rdm_alignment(a, b, coefficient)
                finite_close(actual, float(row["alignment"]), f"seed {seed}, layer {layer}, {row['source_b']}")
                observed_repeats.append(actual)
                null[:, li] += permuted_rsa(a, b, coefficient, orders, args.batch_size) / len(group) / len(expected_seeds)
                label = f"{seed}:{row['source_a']}:{row['source_b']}"
                old_p[layer][label] = row.get("p_value", "")
                old_q[layer][label] = row.get("q_fdr", "")
            observed_by_seed[si, li] = np.mean(observed_repeats)
            print(f"  seed {seed}, layer {layer}: verified RSA and shared null", flush=True)
    observed, sd = observed_by_seed.mean(axis=0), observed_by_seed.std(axis=0, ddof=1)
    raw, corrected, global_p, peak, maximum = maxstat_pvalues(observed, null)
    analysis_type = "maxstat_across_layers" if len(layers) > 1 else "single_point_permutation"
    results = []
    aggregate_status = "not_supplied; means/SD reconstructed from verified per-run rows"
    for li, layer in enumerate(layers):
        example = seed_rows[expected_seeds[0]][layer][0][1]
        row = dict(base, source_a=source_a, source_b=source_b, layer_index=layer,
                   source_a_layer=example["source_a_layer"], source_b_layer=example["source_b_layer"],
                   mean_alignment=float(observed[li]), sd_alignment=float(sd[li]), n_runs=len(expected_seeds),
                   stimulus_seeds="|".join(map(str, expected_seeds)), permutation_rounds=rounds,
                   maxstat_permutation_seed=pseed, raw_permutation_p=float(raw[li]),
                   maxstat_p=float(corrected[li]), **{"maxstat_significant_0.05": bool(corrected[li] < .05)},
                   observed_family_max_abs_rsa=float(abs(observed[peak])), global_maxstat_p=global_p,
                   maxstat_family=family, analysis_type=analysis_type,
                   borderline_sample_level=bool(level == "sample" and .03 <= corrected[li] <= .07),
                   existing_run_raw_p=json.dumps(old_p[layer], sort_keys=True),
                   existing_run_q_fdr=json.dumps(old_q[layer], sort_keys=True),
                   existing_mean_perm_p="", existing_mean_perm_q_fdr="",
                   permutation_side="source_b", status="ok")
        if aggregate:
            saved = aggregate.get(aggregate_key(row))
            if saved is None:
                raise ValueError(f"Supplied aggregate table missing {aggregate_key(row)}")
            finite_close(row["mean_alignment"], float(saved["mean_alignment"]), "Aggregate mean")
            finite_close(row["sd_alignment"], float(saved["sd_alignment"]), "Aggregate SD")
            if int(saved["n_runs"]) != len(expected_seeds):
                raise ValueError("Existing aggregate has different seed count")
            if saved.get("stimulus_seeds") and set(map(int, saved["stimulus_seeds"].split("|"))) != set(expected_seeds):
                raise ValueError("Existing aggregate uses different stimulus seeds")
            for name in ("mean_perm_p", "mean_perm_q_fdr"):
                row["existing_" + name] = saved.get(name, "")
            aggregate_status = "verified mean and SD against supplied aggregate"
        results.append(row)
    significant = [layers[i] for i, p in enumerate(corrected) if p < .05]
    summary = dict(base, model=model, source_a=source_a, source_b=source_b, rsa_metric=coefficient,
                   n_layers=len(layers), n_permutations=rounds, peak_layer="" if fixed else layers[peak],
                   peak_rsa=float(observed[peak]), observed_family_max_abs_rsa=float(abs(observed[peak])),
                   global_maxstat_p=global_p, significant_layers="|".join(map(str, significant)),
                   n_significant_layers=len(significant), minimum_layerwise_maxstat_p=float(corrected.min()),
                   borderline_sample_level=any(r["borderline_sample_level"] for r in results),
                   maxstat_family=family, analysis_type=analysis_type, status="ok", reason="",
                   stimulus_seeds="|".join(map(str, expected_seeds)), n_runs=len(expected_seeds),
                   layers="|".join(map(str, layers)), permutation_side="source_b", permutation_sha256=digest,
                   maxstat_permutation_seed=pseed, aggregate_check=aggregate_status,
                   repeats_per_seed=json.dumps(repeats_by_seed, sort_keys=True),
                   support_sha256_by_seed=json.dumps({s: hashlib.sha256(repr(v).encode()).hexdigest()
                                                     for s, v in support_by_seed.items()}, sort_keys=True),
                   permutation_unit="category label" if level == "category" else "sample position within original pairwise support",
                   averaging="mean RSA across fixed mismatch repeats within seed, then equal-weight seed mean",
                   layer_selection="fixed" if fixed else ("explicit CLI subset" if args.layer_indices else "run metadata or model default"))
    return results, summary, dict(maximum_null=maximum, mean_null_by_layer=null,
                                 observed_by_seed=observed_by_seed, layers=layers, permutation_orders=orders)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing directories are refused")
    parser.add_argument("--expected-seeds", default="42,43,44")
    parser.add_argument("--comparison", action="append", choices=sorted(ALLOWLIST))
    parser.add_argument("--level", action="append", choices=["sample", "category"])
    parser.add_argument("--embedding-distance", action="append", choices=["cosine", "pearson"])
    parser.add_argument("--rsa-coefficient", action="append", choices=["spearman", "kendall_tau_b"])
    parser.add_argument("--layer-indices", help="Only for an intentionally defined subset; comma-separated")
    parser.add_argument("--sample-permutations", type=int, default=1000)
    parser.add_argument("--category-permutations", type=int, default=5000)
    parser.add_argument("--permutation-seed", type=int, default=20261013)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--aggregate-csv", type=Path, action="append", default=[],
                        help="Optional existing mean/SD tables; supply main AND mismatch tables for all comparisons")
    parser.add_argument("--save-null", action="store_true", help="Save shared orders and joint null for auditing")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if min(args.sample_permutations, args.category_permutations, args.batch_size) < 1:
        raise ValueError("Permutation budgets and batch size must be positive")
    seeds = [int(s) for s in args.expected_seeds.split(",")]
    if len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError("Expected seeds must be unique, with at least two runs for SD")
    runs = [p.resolve() for p in args.run_dirs]
    if len(set(runs)) != len(runs):
        raise ValueError("Duplicate input run directories")
    output = args.output_dir.resolve()
    if output.exists() or any(output.is_relative_to(run) or run.is_relative_to(output) for run in runs):
        raise ValueError("Choose a NEW output directory outside all existing input runs")
    requested = set(args.comparison or ALLOWLIST)
    lookup = {frozenset(v): k for k, v in ALLOWLIST.items() if k in requested}
    groups, summaries, models = defaultdict(list), defaultdict(list), set()
    seen_runs = set()
    for run in runs:
        summary_path = run / "analysis/summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if summary and summary.get("status") != "complete":
            raise ValueError(f"Incomplete input analysis: {run}")
        rows = read_csv(run / "analysis/alignment_layerwise_v2.csv")
        identities = {(r.get("model_id") or r.get("model"), int(r.get("stimulus_seed") or r["seed"])) for r in rows}
        if len(identities) != 1:
            raise ValueError(f"Mixed or empty model/seed data in {run}")
        model, seed = identities.pop()
        if summary.get("model_id", model) != model or int(summary.get("stimulus_seed", seed)) != seed:
            raise ValueError(f"Run metadata/model/seed differs from RSA rows in {run}")
        if (model, seed) in seen_runs:
            raise ValueError(f"Duplicate model/stimulus seed: {model}, {seed}")
        seen_runs.add((model, seed))
        models.add(model)
        summaries[model].append(summary)
        for row in rows:
            row["model_id"] = model
            pair = frozenset(normalized_source(row[k]) for k in ("source_a", "source_b"))
            if pair not in lookup:
                continue
            if any(selected and row[column] not in selected for selected, column in (
                    (args.level, "level"), (args.embedding_distance, "embedding_distance"),
                    (args.rsa_coefficient, "rsa_coefficient"))):
                continue
            if args.layer_indices and row["layer_index"] not in args.layer_indices.split(",") and not (
                    row["source_a_layer"] == row["source_b_layer"] == "fixed"):
                continue
            groups[(model, lookup[pair], row["level"], row["embedding_distance"], row["rsa_coefficient"])].append((run, row))
    aggregate = {}
    for path in args.aggregate_csv:
        for row in read_csv(path):
            key = aggregate_key(row)
            if key in aggregate:
                raise ValueError(f"Duplicate aggregate reference row {key}")
            aggregate[key] = row
    if args.aggregate_csv and not aggregate:
        raise ValueError("Supplied aggregate tables are empty")
    output.mkdir(parents=True)
    layer_rows, summary_rows = [], []
    metadata = dict(input_runs=list(map(str, runs)), arguments={k: str(v) for k, v in vars(args).items()},
                    allowlist=ALLOWLIST, status="running", inference="two-sided shared source_b row/column relabeling",
                    raw_p_note="raw_permutation_p is recomputed using the SAME shared draws as maxstat; old p/q are copied separately",
                    sample_note="Seeds contain different RGB samples. Shared indices mean positions, not identical physical stimuli. Unequal support sizes are skipped.",
                    hypothesis_note="Tests RSA alignment, not a difference of two RSA correlations or superiority of a peak layer")
    (output / "rsa_maxstat_metadata.json").write_text(json.dumps(metadata, indent=2))
    # Use the new output filesystem, not a possibly RAM-backed /tmp on servers.
    with tempfile.TemporaryDirectory(prefix="cache_", dir=output) as scratch:
        inputs = Inputs(scratch)
        for key, entries in sorted(groups.items()):
            print("Family: " + " | ".join(key), flush=True)
            try:
                designs = [s.get("analysis_design") for s in summaries[key[0]]]
                if any(d != designs[0] for d in designs):
                    raise ValueError("Different run analysis designs")
                rows, summary, audit = analyze_family(key, entries, inputs, seeds, summaries[key[0]], args, aggregate)
            except (ValueError, KeyError, FileNotFoundError, IndexError) as error:
                summary = dict(zip(FAMILY_FIELDS, key), status="skipped", reason=str(error))
                print(f"  SKIPPED: {error}", flush=True)
            else:
                layer_rows.extend(rows)
                if args.save_null:
                    name = hashlib.sha256("|".join(key).encode()).hexdigest()[:20] + ".npz"
                    np.savez_compressed(output / name, **audit)
                    summary["null_file"] = name
                label = "single-point permutation" if summary["n_layers"] == 1 else f"peak layer {summary['peak_layer']}"
                print(f"  {label}: RSA={summary['peak_rsa']:.6f}, p={summary['global_maxstat_p']:.6g}; significant layers=[{summary['significant_layers']}]", flush=True)
            summary_rows.append(summary)
            write_csv(output / "rsa_maxstat_layerwise.csv", layer_rows)
            write_csv(output / "rsa_maxstat_summary.csv", summary_rows)
    for model in sorted(models):
        for comparison in sorted(requested):
            if not any(k[:2] == (model, comparison) for k in groups):
                summary_rows.append(dict(model_id=model, comparison=comparison, status="skipped",
                                         reason="No existing rows for requested comparison/filters; no new level constructed"))
                print(f"  SKIPPED {model} | {comparison}: no existing rows for requested comparison/filters", flush=True)
    write_csv(output / "rsa_maxstat_layerwise.csv", layer_rows)
    write_csv(output / "rsa_maxstat_summary.csv", summary_rows)
    skipped = [s for s in summary_rows if s["status"] != "ok"]
    metadata.update(status="complete_with_skips" if skipped else "complete", n_families=len(summary_rows),
                    n_tested=len(summary_rows) - len(skipped), skipped=skipped)
    (output / "rsa_maxstat_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Finished: {metadata['n_tested']} tested, {len(skipped)} skipped. Outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
