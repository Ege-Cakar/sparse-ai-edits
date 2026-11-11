import numpy as np

from sparse_edits.doc_conformal import band_edges, conformal_p

rng = np.random.default_rng(0)


def test_conformal_p_is_uniform_for_exchangeable_statistics_with_ties():
    cal = rng.integers(0, 5, (4000, 59)).astype(float)          # discrete statistics: many ties
    test = rng.integers(0, 5, (4000, 1)).astype(float)
    p = np.concatenate([conformal_p(c, t, rng) for c, t in zip(cal, test)])
    assert abs((p <= 0.05).mean() - 0.05) < 0.01 and abs(p.mean() - 0.5) < 0.02


def test_band_edges_keep_min_docs_per_band():
    n = rng.integers(16, 200, 130)
    edges = band_edges(n, 60, 3)
    assert len(edges) == 1 and np.bincount(np.digitize(n, edges)).min() >= 60
    assert len(band_edges(n[:59], 60, 3)) == 0                   # too few documents: one band
