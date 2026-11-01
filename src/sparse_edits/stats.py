"""p-values, global tests, support recovery, edited-fraction estimates, and the rare/weak phase boundaries.

Matrices are (documents, n sentences). Global test statistics are oriented so larger = more evidence of edits.
"""
from functools import cache

import numpy as np
import pandas as pd
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
def null_thresholds(n, alpha, alpha0, reps, seed, rho=0.0):
    """Level-alpha critical values of each global test by Monte Carlo, plus "hc_all": HC over every i
    (alpha0 = 1), the bounding constant of eps_lower.

    rho = 0: iid Uniform(0, 1) p-values. rho > 0: equicorrelated null documents, z_j = sqrt(rho) a +
    sqrt(1 - rho) e_j with a, e_j iid N(0, 1) and p_j = 1 - Phi(z_j): each p_j is still Uniform(0, 1), and
    every pair of sentences in a document has correlation rho in z (a shared author / topic effect).
    """
    rng = np.random.default_rng(seed)
    if rho == 0:
        p = rng.random((reps, n))
    else:
        p = norm.sf(np.sqrt(rho) * rng.standard_normal((reps, 1)) + np.sqrt(1 - rho) * rng.standard_normal((reps, n)))
    out = {name: np.quantile(f(p), 1 - alpha) for name, f in global_tests(alpha0).items()}
    return out | {"hc_all": np.quantile(hc(p, 1.0), 1 - alpha)}


def intraclass_correlation(z, groups):
    """One-way ANOVA estimate of the correlation of z within groups (documents), floored at 0; groups with a
    single member are ignored. Unbiased for the equicorrelated model of null_thresholds at any group sizes."""
    df = pd.DataFrame({"z": z, "g": groups})
    df = df[df.groupby("g").z.transform("size") >= 2]
    k = df.groupby("g").z.agg(["mean", "var", "size"])
    if len(k) < 2:
        return np.nan
    msw = (k["var"] * (k["size"] - 1)).sum() / (k["size"] - 1).sum()
    msb = (k["size"] * (k["mean"] - df.z.mean()) ** 2).sum() / (len(k) - 1)
    n0 = (k["size"].sum() - (k["size"] ** 2).sum() / k["size"].sum()) / (len(k) - 1)
    return max(0.0, (msb - msw) / (msb + (n0 - 1) * msw))


def bh(p, q):
    """Benjamini-Hochberg rejections, row-wise."""
    n = p.shape[1]
    order = np.argsort(p, axis=1)
    below = np.take_along_axis(p, order, axis=1) <= q * np.arange(1, n + 1) / n
    k = np.where(below.any(axis=1), n - np.argmax(below[:, ::-1], axis=1), 0)   # largest i with p_(i) <= qi/n
    return np.argsort(order, axis=1) < k[:, None]


def eps_storey(p, lam):
    """Storey (2002) point estimate of the edited fraction, row-wise: 1 - #{p > lam} / ((1 - lam) n), clipped to
    [0, 1]. Biased down by eps P(edited p > lam) / (1 - lam) when edits are weak, and up by the clip at 0 when n
    is small (about +0.01 at n = 16, eps = 1/4)."""
    return np.clip(1 - (p > lam).mean(axis=1) / (1 - lam), 0, 1)


def eps_lower(p, c):
    """Meinshausen & Rice (2006) lower confidence bound for the edited fraction with the HC bounding function,
    row-wise: max over i of (i/n - p_(i) - c sqrt(p_(i)(1 - p_(i)) / n)) / (1 - p_(i)), floored at 0.

    With c the level-alpha null quantile of HC over all i (null_thresholds "hc_all"), P(bound <= true fraction)
    >= 1 - alpha, exactly 1 - alpha with no edits; the bound is positive iff that HC rejects. Edited p-values
    only lower F_n(t) - t relative to (1 - t), and c is taken at n rather than at the number of human
    sentences, which assumes the null quantile of HC does not shrink with n.
    """
    n = p.shape[1]
    ps, i = np.sort(p, axis=1), np.arange(1, n + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (i / n - ps - c * np.sqrt(ps * (1 - ps) / n)) / (1 - ps)
    return np.clip(np.where(np.isfinite(b), b, 0).max(axis=1), 0, 1)


def wilson(k, n, level):
    """Wilson score interval for a binomial proportion k / n."""
    z, phat = norm.isf((1 - level) / 2), k / n
    center = (phat + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return center - half, center + half


def sample_truth(docs, n, m, rng):
    """Boolean (docs, n) with exactly m True per row at uniformly random positions."""
    return rng.random((docs, n)).argsort(axis=1) < m


def n_edited(n, theta):
    return np.round(np.asarray(n) ** (1 - theta)).astype(int)


def evaluate(p, truth, thresholds, alpha, alpha0, q, lam):
    """Per-document outcomes: rejection by each global test; Hamming error / m and exact recovery for BH(q) and
    Bonferroni(alpha) support estimates; the true edited fraction m / n, its Storey estimate and squared error,
    and the Meinshausen-Rice lower bound and whether it covers."""
    out = {f"reject_{name}": f(p) > thresholds[name] for name, f in global_tests(alpha0).items()}
    m = truth.sum(axis=1)
    eps, storey, lower = m / p.shape[1], eps_storey(p, lam), eps_lower(p, thresholds["hc_all"])
    out |= dict(eps_true=eps, eps_storey=storey, sqerr_eps_storey=(storey - eps) ** 2, eps_lower=lower,
                cover_eps_lower=lower <= eps)
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
