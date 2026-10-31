#!/usr/bin/env python3
"""Aggregate v2 RSA across sampled RGB sets; common-category mean permutation test."""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np

from alignment_statistics import mean_rsa_permutation, rdm_alignment, stable_seed
from structured_color_utils import BASIC_COLORS, try_import_matplotlib, write_json
from extended_alignment_analysis import adjust_rows, write_rows

GROUP = ("model_id", "source_a", "source_b", "level", "embedding_distance", "rsa_coefficient",
         "layer_index", "source_a_layer", "source_b_layer")
FAMILY = ("model_id", "source_a", "source_b", "level", "embedding_distance", "rsa_coefficient")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--run-labels", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/randomness_plots"))
    parser.add_argument("--source", action="append", help="Optional source filter (either side)")
    parser.add_argument("--permutation-rounds", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42, help="Permutation seed, not stimulus-sampling seed")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


@lru_cache(maxsize=512)
def read_category_rdm(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header, rows = next(reader), list(reader)
    columns, labels = header[1:], [row[0] for row in rows]
    if (len(labels) != 11 or len(columns) != 11 or set(labels) != set(BASIC_COLORS)
            or set(columns) != set(BASIC_COLORS)):
        raise ValueError("Aggregated category inference requires exactly the same 11 labelled colors")
    matrix = np.asarray([[float(v) for v in row[1:]] for row in rows])
    return matrix[np.ix_([labels.index(c) for c in BASIC_COLORS], [columns.index(c) for c in BASIC_COLORS])]


def category_pair(run, row):
    arrays = []
    for side in ("a", "b"):
        path = row.get(f"rdm_{side}_csv", "")
        if path:
            path = Path(path)
            path = path if path.is_absolute() else run / path
        else:
            name = row[f"source_{side}"]
            stage = "fixed" if row[f"source_{side}_layer"] == "fixed" else f"layer_{row['layer_index']}"
            distance = "ciede2000" if name == "perceptual" else row["embedding_distance"]
            path = run / "rdm/category_v2" / f"{name}_{stage}_{distance}_support_{row['support_id']}.csv"
        arrays.append(read_category_rdm(path))
    coefficient = rdm_alignment(*arrays, row["rsa_coefficient"])
    if not np.isclose(coefficient, float(row["alignment"]), rtol=1e-6, atol=1e-7):
        raise ValueError(f"Saved RDMs do not reproduce RSA in {run}")
    return tuple(arrays)


def aggregate_runs(run_dirs, rounds=10000, seed=42, sources=None):
    runs = [Path(path).resolve() for path in run_dirs]
    if len(set(runs)) != len(runs):
        raise ValueError("Do not include the same run directory twice")
    if rounds < 1:
        raise ValueError("Permutation rounds must be positive")
    read_category_rdm.cache_clear()
    groups, model_runs, designs = defaultdict(list), defaultdict(set), {}
    for run in runs:
        summary_path = run / "analysis/summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary["status"] != "complete":
                raise ValueError(f"Incomplete analysis run: {run}")
            if "analysis_design" in summary:
                model = summary["model_id"]
                if model in designs and designs[model] != summary["analysis_design"]:
                    raise ValueError(f"Different prompts/pooling/designs cannot be aggregated for {model}")
                designs[model] = summary["analysis_design"]
        with (run / "analysis/alignment_layerwise_v2.csv").open(newline="", encoding="utf-8-sig") as handle:
            seen = set()
            for row in csv.DictReader(handle):
                row["model_id"] = row.get("model_id") or row.get("model")
                if not row["model_id"]:
                    raise ValueError("Model metadata is required")
                model_runs[row["model_id"]].add(run)
                if "query_tokens" in (row["source_a"], row["source_b"]):
                    continue
                if sources and not set(sources).intersection((row["source_a"], row["source_b"])):
                    continue
                key = tuple(row[field] for field in GROUP)
                if key in seen:
                    raise ValueError(f"Duplicate analysis condition in {run}: {key}")
                seen.add(key)
                groups[key].append((run, row))
    output = []
    for key, items in groups.items():
        record = dict(zip(GROUP, key))
        values = np.asarray([float(row["alignment"]) for _, row in items])
        finite = np.isfinite(values)
        valid = values[finite]
        pseed = stable_seed(seed, "mean_rsa", *key)
        record.update(model=record["model_id"], mean_alignment=float(valid.mean()) if len(valid) else np.nan,
            sd_alignment=float(valid.std(ddof=1)) if len(valid) > 1 else np.nan,
            n_runs=len(valid), n_runs_present=len(items), expected_runs=len(model_runs[record["model_id"]]),
            run_ids="|".join(str(run) for run, _ in items),
            stimulus_seeds="|".join(str(row.get("stimulus_seed", row.get("seed", "unknown"))) for _, row in items),
            permutation_seed=pseed, permutation_rounds=0, p_value=np.nan,
            status="sample_level_descriptive_only", test="none",
            sampling_note="Different sampled RGB stimulus sets; not stochastic model inference")
        if record["level"] == "category":
            record["status"] = "insufficient_or_incomplete_runs"
            if len(items) >= 2 and len(items) == record["expected_runs"] and finite.all():
                try:
                    pairs = [category_pair(run, row) for run, row in items]
                except (FileNotFoundError, ValueError, KeyError) as error:
                    record.update(status="category_inference_unavailable", reason=str(error))
                else:
                    value, p, status = mean_rsa_permutation(pairs, record["rsa_coefficient"], rounds, pseed)
                    record.update(mean_alignment=value, p_value=p, status=status, permutation_rounds=rounds,
                                  test="two_sided_shared_category_label_permutation_of_mean_RSA")
        output.append(record)
    adjust_rows(output, list(FAMILY))
    for row in output:
        row["mean_perm_p"], row["mean_perm_q_fdr"] = row["p_value"], row["q_fdr"]
    return output


def aggregate_mismatch_runs(run_dirs, sources=None):
    """Average assignments within each run first, then describe variation across runs."""
    from analysis_plotting import repeat_alignment_summary
    groups = defaultdict(list)
    runs = [Path(p).resolve() for p in run_dirs]
    if len(set(runs)) != len(runs):
        raise ValueError("Do not include the same run directory twice")
    for run in runs:
        with (run / "analysis/alignment_layerwise_v2.csv").open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        for row in repeat_alignment_summary(rows):
            if sources and not set(sources).intersection((row["source_a"], row["source_b"])):
                continue
            groups[tuple(row[k] for k in GROUP)].append((run, row))
    output = []
    for key, items in groups.items():
        values = np.asarray([r["mean_alignment"] for _, r in items])
        valid = values[np.isfinite(values)]
        output.append(dict(zip(GROUP, key),
            mean_alignment=float(valid.mean()) if len(valid) else np.nan,
            sd_alignment=float(valid.std(ddof=1)) if len(valid) > 1 else np.nan,
            n_runs=len(valid), n_runs_present=len(items),
            repeats_per_run="|".join(str(r["n_repeats"]) for _, r in items),
            within_run_sd="|".join(str(r["sd_alignment"]) for _, r in items),
            run_ids="|".join(str(run) for run, _ in items),
            p_value=np.nan, q_fdr=np.nan, mean_perm_p=np.nan, mean_perm_q_fdr=np.nan,
            permutation_rounds=0, status="descriptive_nested_pairing_seed_summary",
            test="none", averaging="equal_weight_seed_means_after_within_seed_pairing_mean"))
    return output


def plot_aggregates(output, rows):
    from analysis_plotting import _slug, _fixed_axis
    from datetime import datetime
    import shutil
    archive = output.parent / "archived_plots" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for old in output.glob("*.png"):
        if "cee_mismatched_r" in old.name or "query_tokens" in old.name:
            archive.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(archive / old.name))
    plt = try_import_matplotlib()
    if plt is None:
        raise RuntimeError("Matplotlib is required unless --no-plots is set")
    groups = defaultdict(list)
    for row in rows:
        if any(str(row[k]).startswith("cee_mismatched_r") or row[k] == "query_tokens"
               for k in ("source_a", "source_b")):
            continue
        groups[tuple(row[k] for k in FAMILY)].append(row)
    for key, group in groups.items():
        group.sort(key=lambda row: int(row["layer_index"]))
        x = np.asarray([int(row["layer_index"]) for row in group])
        mean, sd = (np.asarray([row[k] for row in group]) for k in ("mean_alignment", "sd_alignment"))
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        ax.plot(x, mean, "o-", label="Mean RSA")
        if np.isfinite(sd).any():
            ax.fill_between(x, mean-sd, mean+sd, alpha=.2, label="±1 SD across stimulus-sampling runs")
        significant = np.asarray([row["mean_perm_q_fdr"] <= .05 for row in group])
        if significant.any():
            ax.scatter(x[significant], mean[significant], marker="*", s=90, label="Mean-test BH q ≤ .05")
        fixed = _fixed_axis(group)
        ax.set_xlabel("Fixed-stage comparison" if fixed else "LM hidden-state index")
        if fixed:
            ax.set_xticks([0], ["fixed"])
        ax.set_ylabel(key[-1])
        ax.set_title(f"{key[1]} vs {key[2]} ({key[3]}; {key[4]})")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.tight_layout()
        output.mkdir(parents=True, exist_ok=True)
        fig.savefig(output / f"{_slug(*key)}.png")
        plt.close(fig)


def main():
    args = parse_args()
    labels = [value.strip() for value in args.run_labels.split(",") if value.strip()]
    if labels and len(labels) != len(args.run_dirs):
        raise ValueError("--run-labels must match --run-dirs")
    rows = aggregate_runs(args.run_dirs, args.permutation_rounds, args.seed, args.source)
    mismatch_rows = aggregate_mismatch_runs(args.run_dirs, args.source)
    write_rows(args.output_dir / "seed_averaged_alignment_v2.csv", rows)
    write_rows(args.output_dir / "seed_averaged_mismatch_alignment_v2.csv", mismatch_rows)
    if not args.no_plots:
        plot_aggregates(args.output_dir / "plots", rows + mismatch_rows)
    write_json(args.output_dir / "summary_v2.json", dict(run_dirs=list(map(str, args.run_dirs)),
        run_labels=labels, n_conditions=len(rows), permutation_seed=args.seed,
        requested_permutation_rounds=args.permutation_rounds, sd="sample std ddof=1",
        category_inference="same labelled category permutation across runs; compare mean RSA, never average p-values",
        sample_inference="descriptive only; sample IDs do not correspond across runs",
        mismatch_summary="average pairings within each seed, then mean and sample SD across seeds; no combined p-value",
        fdr="layers within model/source pair/level/distance/RSA coefficient"))


if __name__ == "__main__":
    main()
