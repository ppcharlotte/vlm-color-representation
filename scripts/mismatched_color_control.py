"""Random one-to-one image assignments that always cross human color categories."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def mismatched_assignments(labels, repeats=5, seed=42):
    labels = np.asarray(labels)
    n = len(labels)
    if repeats < 0:
        raise ValueError("Mismatch repeats cannot be negative")
    if repeats == 0:
        return np.empty((0, n), dtype=int)
    _, counts = np.unique(labels, return_counts=True)
    if not len(counts) or counts.max() * 2 > n:
        raise ValueError("A different-category bijection requires no category to exceed half the samples")
    rng = np.random.default_rng(seed)
    assignments = []
    for _ in range(repeats):
        # Random-cost bipartite matching: not a uniform draw from all bijections.
        cost = rng.random((n, n))
        cost[labels[:, None] == labels[None, :]] = np.inf
        recipient, donor = linear_sum_assignment(cost)
        mapping = np.empty(n, dtype=int)
        mapping[recipient] = donor
        if len(np.unique(mapping)) != n or np.any(labels == labels[mapping]):
            raise RuntimeError("Invalid mismatched color assignment")
        assignments.append(mapping)
    return np.stack(assignments)
