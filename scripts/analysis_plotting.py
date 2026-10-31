"""Plots for every numeric result family emitted by alignment v2."""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from structured_color_utils import try_import_matplotlib, write_csv


def _slug(*parts):
    return "__".join(re.sub(r"[^A-Za-z0-9_.-]+", "-", str(p)).strip("-") for p in parts)


def _groups(rows, keys):
    output = defaultdict(list)
    for row in rows:
        output[tuple(row.get(key) for key in keys)].append(row)
    return output


def _float(row, key):
    try:
        value = float(row[key])
        return value if np.isfinite(value) else np.nan
    except (KeyError, TypeError, ValueError):
        return np.nan


def _excluded(row):
    return any("cee_mismatched_r" in str(row.get(k, "")) or "query_tokens" == row.get(k)
               for k in ("source", "source_a", "source_b", "reference"))


def repeat_alignment_summary(rows, metric="alignment"):
    """Average effect sizes across pairing assignments, never average p-values."""
    groups = defaultdict(list)
    keys = ("model_id", "run_id", "layer_index", "source_a", "source_b", "source_a_layer",
            "source_b_layer", "level", "embedding_distance", "rsa_coefficient", "support_id")
    for row in rows:
        if not any(str(row.get(k, "")).startswith("cee_mismatched_r") for k in ("source_a", "source_b")):
            continue
        if "query_tokens" in (row.get("source_a"), row.get("source_b")):
            continue
        normalized = dict(row)
        for k in ("source_a", "source_b"):
            if str(normalized.get(k, "")).startswith("cee_mismatched_r"):
                normalized[k] = "cee_mismatched_mean"
        groups[tuple(normalized.get(k) for k in keys)].append(_float(row, metric))
    result = []
    for key, values in groups.items():
        valid = np.asarray(values)[np.isfinite(values)]
        mean = float(valid.mean()) if len(valid) else np.nan
        sd = float(valid.std(ddof=1)) if len(valid) > 1 else np.nan
        result.append(dict(zip(keys, key), mean_alignment=mean, sd_alignment=sd,
                           lower_sd=mean-sd, upper_sd=mean+sd, n_repeats=len(values),
                           valid_repeats=len(valid), metric=metric, inference="descriptive_across_pairings_no_pvalue"))
    return result


def _fixed_axis(rows):
    return bool(rows) and all(str(row.get("layer_index")) == "fixed" or
                              (row.get("source_a_layer") == "fixed" and row.get("source_b_layer") == "fixed")
                              for row in rows)


def _line_plot(path, rows, y_key, title, ylabel, q_key=None, interval=None, zero=False, secondary=(), interval_label="95% CI"):
    plt = try_import_matplotlib()
    if plt is None or not rows:
        return False
    fixed_axis = _fixed_axis(rows)
    layer_value = lambda row: 0 if fixed_axis else int(row["layer_index"])
    rows = sorted(rows, key=layer_value)
    x = np.asarray([layer_value(row) for row in rows])
    y = np.asarray([_float(row, y_key) for row in rows])
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=160)
    ax.plot(x, y, marker="o", linewidth=1.8, label=ylabel)
    for key, label in secondary:
        values = np.asarray([_float(row, key) for row in rows])
        ax.plot(x, values, marker="o", linewidth=1.4, label=label)
    if interval:
        low = np.asarray([_float(row, interval[0]) for row in rows])
        high = np.asarray([_float(row, interval[1]) for row in rows])
        ax.fill_between(x, low, high, alpha=.2, label=interval_label)
    if q_key:
        significant = np.asarray([_float(row, q_key) <= .05 for row in rows])
        if significant.any():
            ax.scatter(x[significant], y[significant], marker="*", s=85, zorder=5,
                       label="BH-FDR q ≤ .05")
    if zero:
        ax.axhline(0, color="black", linewidth=.8, alpha=.55)
    ax.set_xlabel("Fixed-stage comparison" if fixed_axis else "LM hidden-state index (0 = input state)")
    if fixed_axis:
        ax.set_xticks([0], ["fixed"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=.25)
    if secondary or interval or (q_key and any(_float(row, q_key) <= .05 for row in rows)):
        ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return True


def _mismatch_plot(path, rows, title):
    plt = try_import_matplotlib()
    if plt is None or not rows:
        return False
    rows = sorted(rows, key=lambda row: int(row["layer_index"]))
    x = np.asarray([int(row["layer_index"]) for row in rows])
    mean = np.asarray([_float(row, "mean_delta") for row in rows])
    sd = np.asarray([_float(row, "sd_delta") for row in rows])
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=160)
    ax.plot(x, mean, marker="o", linewidth=1.8)
    if np.isfinite(sd).any():
        ax.fill_between(x, mean - sd, mean + sd, alpha=.2, label="mean ± SD across pairings")
        ax.legend(fontsize=8)
    ax.axhline(0, color="black", linewidth=.8, alpha=.55)
    ax.set_xlabel("LM hidden-state index (0 = input state)")
    ax.set_ylabel("Mean alignment difference")
    ax.set_title(title)
    ax.grid(True, alpha=.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return True


def _rdm_plot(csv_path, output_root):
    plt = try_import_matplotlib()
    if plt is None:
        return None
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return None
    labels = [next(iter(row.values())) for row in rows]
    columns = list(rows[0])[1:]
    if len(rows) != len(columns):
        return None
    try:
        matrix = np.asarray([[float(row[column]) for column in columns] for row in rows])
    except (TypeError, ValueError):
        return None
    relative = csv_path.relative_to(output_root / "rdm")
    path = output_root / "plots/v2/rdm" / relative.with_suffix(".png")
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    image = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(csv_path.stem)
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04, label="Dissimilarity")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path, len(rows)


def plot_all_results(output, rsa_rows, cka_rows, delta_rows, mismatch_rows, probe_rows,
                     regression_rows=(), template_rows=(), paired_rows=(), sanity_rows=(), rdm_paths=None):
    """Write one plot per meaningful CSV subgroup plus all square RDM heatmaps."""
    if try_import_matplotlib() is None:
        raise RuntimeError("Plotting was requested but Matplotlib is unavailable; install project requirements or use --no-plots.")
    manifest = []
    # Preserve obsolete plots outside the active plot directory on reruns.
    from datetime import datetime
    import shutil
    archive = output / "plots/archived" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for old in (output / "plots/v2").rglob("*.png"):
        if "cee_mismatched_r" in old.name or "query_tokens" in old.name:
            target = archive / old.relative_to(output / "plots/v2")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(target))

    summaries = [(repeat_alignment_summary(rsa_rows), "mismatch_alignment_summary_v2.csv"),
                 (repeat_alignment_summary(regression_rows, "R2"), "mismatch_regression_summary_v2.csv")]
    rsa_rows, cka_rows, delta_rows, probe_rows, regression_rows, paired_rows = (
        [r for r in data if not _excluded(r)]
        for data in (rsa_rows, cka_rows, delta_rows, probe_rows, regression_rows, paired_rows))

    def record(path, source_csv, plot_type, filters, rows):
        manifest.append(dict(plot_path=str(path), source_csv=source_csv, plot_type=plot_type,
                             filters=filters, n_rows=len(rows)))

    rsa_keys = ("source_a", "source_b", "level", "embedding_distance", "rsa_coefficient")
    for summary, filename in summaries:
        for key, rows in _groups(summary, rsa_keys).items():
            path = output / "plots/v2/mismatch_summary" / f"{_slug(filename, *key)}.png"
            if _line_plot(path, rows, "mean_alignment", " / ".join(map(str, key)),
                          "Mean " + rows[0]["metric"], interval=("lower_sd", "upper_sd"),
                          interval_label="mean ± SD across pairings"):
                record(path, "analysis/" + filename, "mismatch_alignment_summary", repr(key), rows)
    for key, rows in _groups(rsa_rows, rsa_keys).items():
        a, b, level, distance, coefficient = key
        path = output / "plots/v2/rsa" / f"{_slug(*key)}.png"
        if _line_plot(path, rows, "alignment", f"{a} vs {b} ({level}; {distance}; {coefficient})",
                      "RSA alignment", "q_fdr"):
            record(path, "analysis/alignment_layerwise_v2.csv", "rsa_layerwise", repr(dict(zip(rsa_keys, key))), rows)

    cka_keys = ("source_a", "source_b")
    for key, rows in _groups(cka_rows, cka_keys).items():
        path = output / "plots/v2/appendix/cka" / f"{_slug(*key)}.png"
        if _line_plot(path, rows, "alignment", f"{key[0]} vs {key[1]} (linear CKA)", "Linear CKA", "q_fdr"):
            record(path, "analysis/appendix_cka_layerwise_v2.csv", "cka_layerwise", repr(dict(zip(cka_keys, key))), rows)

    delta_keys = ("source_a", "source_b", "mismatch_repeat", "embedding_distance", "rsa_coefficient")
    for key, rows in _groups(delta_rows, delta_keys).items():
        path = output / "plots/v2/rq3" / f"{_slug(*key)}.png"
        title = f"{key[0]} minus {key[1]} ({key[3]}; {key[4]})"
        if _line_plot(path, rows, "delta_alignment", title, "Alignment difference", "q_fdr",
                      ("ci_low", "ci_high") if "ci_low" in rows[0] else None, True,
                      (("alignment_a", f"{key[0]}–perceptual"), ("alignment_b", f"{key[1]}–perceptual"))):
            record(path, "analysis/rq3_permutation_v2.csv", "rq3_alignment_and_delta",
                   repr(dict(zip(delta_keys, key))), rows)

    mismatch_keys = ("contrast", "embedding_distance", "rsa_coefficient")
    for key, rows in _groups(mismatch_rows, mismatch_keys).items():
        path = output / "plots/v2/mismatch_summary" / f"{_slug(*key)}.png"
        if _mismatch_plot(path, rows, f"{key[0]} ({key[1]}; {key[2]})"):
            record(path, "analysis/rq3_mismatch_repeat_summary.csv", "mismatch_repeat_summary",
                   repr(dict(zip(mismatch_keys, key))), rows)

    for (source,), rows in _groups(probe_rows, ("source",)).items():
        path = output / "plots/v2/appendix/probe" / f"{_slug(source)}.png"
        if _line_plot(path, rows, "accuracy", f"Color decoding: {source}", "Probe accuracy"):
            record(path, "analysis/appendix_linear_probe_v2.csv", "linear_probe_layerwise", repr({"source": source}), rows)

    families = [
        (regression_rows, ("source_a", "source_b", "level", "embedding_distance"),
         "perceptual_distance_regression_v2.csv", "perceptual_R2", "R2", None, "Descriptive distance-regression R²"),
        (template_rows, ("subset_a", "subset_b", "comparison_type", "rsa_coefficient"),
         "text_context_robustness_v2.csv", "template_robustness", "alignment", "q_fdr", "Template RSA"),
        (paired_rows, ("contrast", "source_a", "source_b", "reference", "embedding_distance", "rsa_coefficient"),
         "paired_rsa_differences_v2.csv", "paired_differences", "delta_rsa", "q_delta_fdr", "RSA difference (a − b)"),
    ]
    for data, keys, filename, kind, ykey, qkey, label in families:
        for key, rows in _groups(data, keys).items():
            path = output / "plots/v2" / kind / f"{_slug(*key)}.png"
            if _line_plot(path, rows, ykey, " / ".join(map(str, key)), label, qkey, zero=kind == "paired_differences"):
                record(path, "analysis/" + filename, kind, repr(dict(zip(keys, key))), rows)

    for key, rows in _groups(sanity_rows, ("condition_a", "condition_b")).items():
        for metric in ("mean_cosine_similarity", "min_cosine_similarity", "max_abs_difference", "mean_abs_difference"):
            path = output / "plots/v2/visual_sanity" / f"{_slug(*key, metric)}.png"
            if _line_plot(path, rows, metric, f"{key[0]} vs {key[1]}", metric):
                record(path, "analysis/visual_state_equivalence_v2.csv", "visual_sanity", repr(key + (metric,)), rows)

    active_rdms = ((output / "rdm").rglob("*.csv") if rdm_paths is None else rdm_paths)
    for csv_path in sorted(active_rdms):
        if "cee_mismatched_r" in csv_path.name or "query_tokens" in csv_path.name:
            continue
        result = _rdm_plot(csv_path, output)
        if result:
            path, n_rows = result
            manifest.append(dict(plot_path=str(path), source_csv=str(csv_path.relative_to(output)),
                                 plot_type="rdm_heatmap", filters="{}", n_rows=n_rows))

    manifest_path = output / "plots/v2/plot_manifest.csv"
    write_csv(manifest_path, ["plot_path", "source_csv", "plot_type", "filters", "n_rows"], manifest)
    return manifest
