"""RSA and paired, stratified stimulus bootstrap; no independent-distance tests."""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.stats import kendalltau

from structured_color_utils import (
    centroids, ciede2000_matrix, rdm, rdm_permutation_pvalue,
    spearman_corr, upper_triangle_values,
)


def stable_seed(seed, *parts):
    digest = hashlib.sha256("|".join(map(str, (seed, *parts))).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def kendall_tau_b(a, b):
    # Only the coefficient is used. SciPy's independent-pair p-value is discarded.
    return float(kendalltau(a, b, variant="b", method="asymptotic").statistic)


COEFFICIENTS = {"spearman": spearman_corr, "kendall_tau_b": kendall_tau_b}


def rdm_alignment(a, b, coefficient):
    av, bv = upper_triangle_values(a), upper_triangle_values(b)
    if (a.shape != b.shape or len(av) < 3 or
            not all(np.isfinite(v).all() and np.ptp(v) > 0 for v in (av, bv))):
        return np.nan
    return float(COEFFICIENTS[coefficient](av, bv))


def distance_regression(model_rdm, perceptual_rdm):
    """Descriptive OLS with intercept; no independent-pair standard errors/p-values."""
    x, y = upper_triangle_values(model_rdm), upper_triangle_values(perceptual_rdm)
    result = dict(R2=np.nan, slope=np.nan, intercept=np.nan, n_pairs=len(x),
                  status="undefined_or_constant_distance", inference="descriptive_only")
    if model_rdm.shape != perceptual_rdm.shape:
        raise ValueError("RDM shapes differ")
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all() or np.ptp(x) == 0 or np.ptp(y) == 0:
        return result
    xc, yc = x - x.mean(), y - y.mean()
    slope = np.dot(xc, yc) / np.dot(xc, xc)
    intercept = y.mean() - slope * x.mean()
    return dict(R2=float(1 - np.sum((y - (intercept + slope * x)) ** 2) / np.dot(yc, yc)),
                slope=float(slope), intercept=float(intercept), n_pairs=len(x),
                status="ok", inference="descriptive_only")


def paired_category_swap(a, b, reference_rdm, distance, coefficient, rounds, seed):
    """Swap whole category prototypes between paired conditions; reference stays fixed.

    Tests the stronger null that the two condition labels are exchangeable within
    category, not arbitrary equality of population correlations. Categories are
    permutation units, never individual RDM edges. Pooling precedes this function.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 2 or len(a) != len(reference_rdm):
        raise ValueError("Paired category swap needs identical category support and feature dimensions")
    if rounds < 1:
        raise ValueError("Permutation rounds must be positive")
    ra = rdm_alignment(rdm(a, distance), reference_rdm, coefficient)
    rb = rdm_alignment(rdm(b, distance), reference_rdm, coefficient)
    observed = ra - rb
    exact = len(a) <= 16 and rounds >= 2 ** len(a)
    n = 2 ** len(a) if exact else rounds
    result = dict(rsa_a=ra, rsa_b=rb, delta_rsa=observed, p_value=np.nan,
                  permutation_rounds=n, requested_permutation_rounds=rounds, permutation_seed=seed,
                  test="paired_category_prototype_swap_two_sided", exact_permutation=exact,
                  null_hypothesis="condition_labels_exchangeable_within_category",
                  status="undefined_observed")
    if not np.isfinite(observed):
        return result
    rng = np.random.default_rng(seed)
    extreme = 0
    for index in range(n):
        mask = (((index >> np.arange(len(a))) & 1).astype(bool) if exact
                else rng.integers(0, 2, len(a)).astype(bool))[:, None]
        first, second = np.where(mask, b, a), np.where(mask, a, b)
        delta = (rdm_alignment(rdm(first, distance), reference_rdm, coefficient)
                 - rdm_alignment(rdm(second, distance), reference_rdm, coefficient))
        if not np.isfinite(delta):
            result["status"] = "undefined_permutation_no_pvalue"
            return result
        extreme += abs(delta) >= abs(observed) - 1e-14
    result.update(p_value=extreme / n if exact else (extreme + 1) / (n + 1), status="ok")
    return result


def mean_rsa_permutation(rdm_pairs, coefficient, rounds, seed):
    """One common category-label permutation in every run, two-sided mean RSA test."""
    if not rdm_pairs or rounds < 1:
        raise ValueError("Runs and positive permutation budget required")
    shape = rdm_pairs[0][0].shape
    if any(a.shape != shape or b.shape != shape for a, b in rdm_pairs):
        raise ValueError("All category RDMs must have the same labelled support")
    observed = np.asarray([rdm_alignment(a, b, coefficient) for a, b in rdm_pairs])
    if not np.isfinite(observed).all():
        return np.nan, np.nan, "undefined_run_alignment"
    value = float(observed.mean())
    extreme = 0
    rng = np.random.default_rng(seed)
    ranked = None
    if coefficient == "spearman":
        # Category relabelling preserves the complete set of pair ranks.
        from structured_color_utils import rankdata_average
        tri = np.triu_indices(shape[0], k=1)
        ranked = []
        for a, b in rdm_pairs:
            av, bv = (rankdata_average(upper_triangle_values(m)) for m in (a, b))
            av, bv = av - av.mean(), bv - bv.mean()
            av, bv = av / np.linalg.norm(av), bv / np.linalg.norm(bv)
            matrix = np.zeros_like(b, dtype=float)
            matrix[tri], matrix[(tri[1], tri[0])] = bv, bv
            ranked.append((av, matrix))
    for _ in range(rounds):
        order = rng.permutation(shape[0])
        if ranked is not None:
            null = np.mean([np.dot(a, b[order[tri[0]], order[tri[1]]]) for a, b in ranked])
        else:
            null = np.mean([rdm_alignment(a, b[np.ix_(order, order)], coefficient) for a, b in rdm_pairs])
        if not np.isfinite(null):
            return value, np.nan, "undefined_permutation"
        extreme += abs(null) >= abs(value) - 1e-14
    return value, (extreme + 1) / (rounds + 1), "ok"


def compare_rdms(a, b, coefficient, rounds, seed):
    if rounds < 1:
        raise ValueError("Permutation rounds must be positive")
    if a.shape != b.shape:
        raise ValueError("RDM shapes differ")
    av, bv = upper_triangle_values(a), upper_triangle_values(b)
    # Never silently drop undefined distances: that changes the test's support.
    if not np.isfinite(av).all() or not np.isfinite(bv).all():
        return np.nan, np.nan, len(av), "undefined_distance"
    if len(av) < 3 or np.ptp(av) == 0 or np.ptp(bv) == 0:
        return np.nan, np.nan, len(av), "constant_or_insufficient_rdm"
    value, p, n = rdm_permutation_pvalue(a, b, COEFFICIENTS[coefficient], rounds, seed)
    return value, p, n, "ok" if np.isfinite(value) else "undefined_coefficient"


def cka_permutation(a, b, rounds, seed):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or len(a) < 3:
        return np.nan, np.nan, "undefined_coordinates"
    a, b = a - a.mean(0), b - b.mean(0)
    ga, gb = a @ a.T, b @ b.T
    denominator = np.linalg.norm(ga) * np.linalg.norm(gb)
    if denominator <= 1e-15:
        return np.nan, np.nan, "constant_coordinates"
    value = float(np.sum(ga * gb) / denominator)
    rng, extreme = np.random.default_rng(seed), 0
    for _ in range(rounds):
        order = rng.permutation(len(a))
        null = np.sum(ga * gb[np.ix_(order, order)]) / denominator
        extreme += null >= value - 1e-14
    return value, (extreme + 1) / (rounds + 1), "ok"


def paired_category_bootstrap(with_image, without_image, lab, labels, colors,
                              distances, coefficients, rounds, seed):
    """Conditional inference over RGB sampling within FIXED color categories.

    Rebuild model and perceptual centroids from the same stratified draw. The
    zero-centred bootstrap test is approximate; it is NOT a randomized-treatment
    test or an exact test for equality of correlations. Percentile CIs are
    pointwise. Labels, templates, models and the set of categories are fixed.
    """
    if rounds < 2:
        raise ValueError("At least two bootstrap replicates required")
    if with_image.shape != without_image.shape or len(lab) != len(labels) or len(labels) != len(with_image):
        raise ValueError("Paired bootstrap requires identical sample support and feature shape")
    groups = [np.flatnonzero(np.asarray(labels) == color) for color in colors]
    if any(len(group) < 2 for group in groups):
        raise ValueError("Need at least two RGB samples per category for within-category bootstrap")
    keys = [(distance, coefficient) for distance in distances for coefficient in coefficients]

    def evaluate(indices):
        sampled_labels = [labels[int(i)] for i in indices]
        visual = centroids(with_image[indices], sampled_labels, colors)
        textual = centroids(without_image[indices], sampled_labels, colors)
        human = upper_triangle_values(ciede2000_matrix(centroids(lab[indices], sampled_labels, colors)))
        values = {}
        for distance in distances:
            va, vb = upper_triangle_values(rdm(visual, distance)), upper_triangle_values(rdm(textual, distance))
            for coefficient in coefficients:
                fn = COEFFICIENTS[coefficient]
                if not all(np.isfinite(v).all() and np.ptp(v) > 0 for v in (va, vb, human)):
                    values[distance, coefficient] = (np.nan, np.nan)
                else:
                    values[distance, coefficient] = (fn(va, human), fn(vb, human))
        return values

    observed = evaluate(np.arange(len(labels)))
    draws = {key: [] for key in keys}
    rng = np.random.default_rng(seed)
    for _ in range(rounds):
        indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        for key, (a, b) in evaluate(indices).items():
            draws[key].append(a - b)
    rows = []
    for key in keys:
        a, b = observed[key]
        delta = a - b
        samples = np.asarray(draws[key])
        valid = samples[np.isfinite(samples)]
        status = "ok"
        lo = hi = p = np.nan
        if not np.isfinite(delta) or len(valid) != rounds:
            status = "undefined_bootstrap_replicates"
        else:
            lo, hi = np.quantile(valid, [0.025, 0.975])
            if np.std(valid) <= 1e-14:
                if abs(delta) <= 1e-14:
                    p, status = 1.0, "identical_alignment"
                else:
                    status = "degenerate_bootstrap_no_pvalue"
            else:
                p = (1 + np.count_nonzero(np.abs(valid - delta) >= abs(delta))) / (rounds + 1)
        rows.append(dict(embedding_distance=key[0], rsa_coefficient=key[1],
                         with_image_alignment=a, without_image_alignment=b, delta_alignment=delta,
                         ci_low=lo, ci_high=hi, p_value=p, status=status,
                         bootstrap_rounds=rounds, valid_bootstrap_rounds=len(valid),
                         test="paired_stratified_centered_bootstrap_approximate_two_sided",
                         ci_method="percentile_95_pointwise", inference_scope="RGB_sampling_within_fixed_color_categories"))
    return rows
