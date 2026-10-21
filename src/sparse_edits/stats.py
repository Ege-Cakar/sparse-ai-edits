"""p-values, global tests, support recovery, and the rare/weak phase boundaries.

Matrices are (documents, n sentences). Global test statistics are oriented so larger = more evidence of edits.
"""
from functools import cache

import numpy as np
from scipy.stats import norm


def conformal_pvalues(null_scores, null_lengths, scores, lengths, n_bins, rng):
    """Randomized conformal p-values within length bins. Scores: higher = more AI-like.

    p = (#{null > s} + U (1 + #{null = s})) / (1 + #null), U ~ Uniform(0, 1), over null sentences in the same
    length bin. Exactly Uniform(0, 1) when a human test sentence is exchangeable with the null sentences of its
    bin, and never exactly 0 or 1. Bins are null-length quantiles; test lengths are clipped to the null range
    so every bin is non-empty. NaN scores get NaN p-values.
    """
    edges = np.unique(np.quantile(null_lengths, np.linspace(0, 1, n_bins + 1)[1:-1], method="lower"))
    null_bin = np.digitize(null_lengths, edges)
    test_bin = np.digitize(np.clip(lengths, null_lengths.min(), null_lengths.max()), edges)
    p = np.full(len(scores), np.nan)
    for b in np.unique(test_bin):
        ref, idx = np.sort(null_scores[null_bin == b]), (test_bin == b) & ~np.isnan(scores)
        lo, hi = np.searchsorted(ref, scores[idx], "left"), np.searchsorted(ref, scores[idx], "right")
        p[idx] = (len(ref) - hi + rng.random(idx.sum()) * (1 + hi - lo)) / (1 + len(ref))
    return p


def empirical_null(z, central=0.5, bins=30):
    """Efron (2004) central matching: fit a quadratic to log histogram counts over the central fraction of z.

    log f0(z) = c - (z - delta0)^2 / (2 sigma0^2). Returns (delta0, sigma0), or (nan, nan) if the fitted
    quadratic is not concave (too few z values in the central region).
    """
    lo, hi = np.quantile(z, [(1 - central) / 2, (1 + central) / 2])
    counts, edges = np.histogram(z[(z >= lo) & (z <= hi)], bins=bins)
    mids, keep = (edges[:-1] + edges[1:]) / 2, counts > 0
    b2, b1, _ = np.polyfit(mids[keep], np.log(counts[keep]), 2, w=np.sqrt(counts[keep]))
    return (-b1 / (2 * b2), np.sqrt(-1 / (2 * b2))) if b2 < 0 else (np.nan, np.nan)


def hc(p, alpha0=0.5, plus=False):
    """Higher Criticism (Donoho & Jin 2004): max over i <= alpha0 n of sqrt(n)(i/n - p_(i)) / sqrt(p_(i)(1 - p_(i))).
    plus=True (HC+) skips p_(i) < 1/n."""
    n = p.shape[1]
    ps, i = np.sort(p, axis=1), np.arange(1, n + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.sqrt(n) * (i / n - ps) / np.sqrt(ps * (1 - ps))
    keep = (i <= alpha0 * n) & ((ps >= 1 / n) if plus else True)
    return np.where(keep & np.isfinite(h), h, -np.inf).max(axis=1)


def global_tests(alpha0):
    return {"hc": lambda p: hc(p, alpha0),
            "hc_plus": lambda p: hc(p, alpha0, plus=True),
            "min_p": lambda p: -np.log(p.min(axis=1)),
            "sum": lambda p: norm.isf(p).sum(axis=1) / np.sqrt(p.shape[1])}   # Stouffer


@cache
def null_thresholds(n, alpha, alpha0, reps, seed):
    """Level-alpha critical values of each global test under iid Uniform(0, 1) p-values, by Monte Carlo."""
    p = np.random.default_rng(seed).random((reps, n))
    return {name: np.quantile(f(p), 1 - alpha) for name, f in global_tests(alpha0).items()}


def bh(p, q):
    """Benjamini-Hochberg rejections, row-wise."""
    n = p.shape[1]
    order = np.argsort(p, axis=1)
    below = np.take_along_axis(p, order, axis=1) <= q * np.arange(1, n + 1) / n
    k = np.where(below.any(axis=1), n - np.argmax(below[:, ::-1], axis=1), 0)   # largest i with p_(i) <= qi/n
    return np.argsort(order, axis=1) < k[:, None]


def sample_truth(docs, n, m, rng):
    """Boolean (docs, n) with exactly m True per row at uniformly random positions."""
    return rng.random((docs, n)).argsort(axis=1) < m


def n_edited(n, theta):
    return np.round(np.asarray(n) ** (1 - theta)).astype(int)


def evaluate(p, truth, thresholds, alpha, alpha0, q):
    """Per-document outcomes: rejection by each global test, and Hamming error / m and exact recovery for
    BH(q) and Bonferroni(alpha) support estimates."""
    out = {f"reject_{name}": f(p) > thresholds[name] for name, f in global_tests(alpha0).items()}
    m = truth.sum(axis=1)
    for name, est in {"bh": bh(p, q), "bonferroni": p <= alpha / p.shape[1]}.items():
        errors = (est != truth).sum(axis=1)
        out[f"hamming_{name}"] = errors / np.maximum(m, 1)
        out[f"exact_{name}"] = errors == 0
    return out


# Rare/weak phase boundaries with eps = n^-theta, mu = sqrt(2 r log n) (Donoho & Jin 2004; Jin & Ke 2016).
def detection_boundary(theta):
    theta = np.asarray(theta, float)
    return np.where(theta <= 0.5, 0.0, np.where(theta <= 0.75, theta - 0.5, (1 - np.sqrt(1 - theta)) ** 2))


def almost_full_recovery_boundary(theta):
    return np.asarray(theta, float)


def exact_recovery_boundary(theta):
    return (1 + np.sqrt(1 - np.asarray(theta, float))) ** 2
