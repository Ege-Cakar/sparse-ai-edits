import numpy as np
from scipy.stats import binom, false_discovery_control, kstest, norm

from sparse_edits.stats import (bh, conformal_pvalues, detection_boundary, empirical_null, eps_lower, eps_storey,
                                exact_recovery_boundary, global_tests, hc, intraclass_correlation, n_edited,
                                null_thresholds, sample_truth, wilson)

rng = np.random.default_rng(0)


def test_conformal_pvalues_uniform_under_null_with_length_dependence():
    lengths = rng.integers(8, 60, 40000)
    scores = rng.normal(lengths / 10, 1)                  # score depends on length; bins must absorb it
    null, test = slice(0, 20000), slice(20000, None)
    p = conformal_pvalues(scores[null], lengths[null], scores[test], lengths[test], 10, rng)
    assert kstest(p, "uniform").pvalue > 0.01
    assert 0 < p.min() and p.max() < 1


def test_conformal_pvalues_nan_scores():
    p = conformal_pvalues(rng.normal(size=100), np.full(100, 10), np.array([0.0, np.nan]), np.array([10, 10]), 5, rng)
    assert np.isfinite(p[0]) and np.isnan(p[1])


def test_empirical_null_recovers_shifted_null():
    z = np.concatenate([rng.normal(0.3, 1.2, 95000), rng.normal(4, 1, 5000)])
    delta0, sigma0 = empirical_null(z)
    assert abs(delta0 - 0.3) < 0.05 and abs(sigma0 - 1.2) < 0.05


def test_hc_matches_definition():
    p = np.array([[0.001, 0.02, 0.3, 0.5, 0.7, 0.9]])
    n, i = 6, np.arange(1, 4)
    expected = max(np.sqrt(n) * (i / n - p[0, :3]) / np.sqrt(p[0, :3] * (1 - p[0, :3])))
    assert np.isclose(hc(p, 0.5)[0], expected)
    assert np.isclose(hc(p, 0.5, plus=True)[0], np.sqrt(n) * (3 / n - 0.3) / np.sqrt(0.3 * 0.7))   # only p >= 1/n


def test_bh_matches_scipy():
    p = rng.random((200, 20)) ** 3
    assert np.array_equal(bh(p, 0.1), false_discovery_control(p, axis=1) <= 0.1)


def test_thresholds_give_nominal_size():
    thresholds = null_thresholds(16, 0.05, 0.5, 100000, 0)
    p = rng.random((50000, 16))
    for name, f in global_tests(0.5).items():
        assert abs((f(p) > thresholds[name]).mean() - 0.05) < 0.005, name


def test_sample_truth_and_n_edited():
    truth = sample_truth(1000, 16, 3, rng)
    assert (truth.sum(axis=1) == 3).all() and abs(truth.mean(axis=0) - 3 / 16).max() < 0.05
    assert n_edited(16, 0.5) == 4 and n_edited(16, np.inf) == 0


def test_phase_boundaries_match_known_points():
    assert np.allclose(detection_boundary([0.3, 0.5, 0.75, 1.0]), [0, 0, 0.25, 1])
    assert np.allclose(exact_recovery_boundary([0.0, 1.0]), [4, 1])


def test_hc_detects_strong_sparse_signal():
    n = 32
    truth = sample_truth(2000, n, 2, rng)
    p = norm.sf(rng.standard_normal(truth.shape) + np.sqrt(2 * 2.0 * np.log(n)) * truth)
    assert (hc(p) > null_thresholds(n, 0.05, 0.5, 100000, 0)["hc"]).mean() > 0.8


def test_eps_lower_covers_and_is_positive_iff_full_hc_rejects():
    for n in [16, 32]:
        c = null_thresholds(n, 0.05, 0.5, 100000, 0)["hc_all"]
        for m, r in [(0, 0.0), (1, 1.0), (4, 0.5), (4, 2.0), (12, 1.0)]:
            truth = sample_truth(20000, n, m, rng)
            p = norm.sf(rng.standard_normal(truth.shape) + np.sqrt(2 * r * np.log(n)) * truth)
            lower = eps_lower(p, c)
            cover = (lower <= m / n).mean()
            assert cover > 0.94 if m else abs(cover - 0.95) < 0.006, (n, m, r, cover)
            assert np.array_equal(lower > 0, hc(p, 1.0) > c)


def test_eps_storey_mean_when_edits_are_strong():
    truth = sample_truth(20000, 16, 4, rng)
    p = np.where(truth, 1e-9, rng.random(truth.shape))
    x = np.arange(13)                                      # #{p > 1/2} among the 12 human sentences ~ Bin(12, 1/2)
    exact = (binom.pmf(x, 12, 0.5) * np.clip(1 - x / 8, 0, 1)).sum()    # unbiased (1/4) but for the clip at 0
    assert abs(eps_storey(p, 0.5).mean() - exact) < 0.003 and exact > 4 / 16


def test_wilson_known_values():
    assert np.allclose(wilson(0, 10, 0.95), (0, 0.2775), atol=1e-4)
    lo, hi = wilson(50, 100, 0.95)
    assert np.isclose(lo + hi, 1) and np.isclose(hi, 0.5962, atol=1e-4)


def equicorrelated_p(docs, n, rho, rng):
    return norm.sf(np.sqrt(rho) * rng.standard_normal((docs, 1)) + np.sqrt(1 - rho) * rng.standard_normal((docs, n)))


def test_dependent_thresholds_restore_level_that_iid_thresholds_lose():
    n, rho = 16, 0.2
    p = equicorrelated_p(50000, n, rho, rng)
    iid, dep = null_thresholds(n, 0.05, 0.5, 100000, 0), null_thresholds(n, 0.05, 0.5, 100000, 0, rho)
    assert null_thresholds(n, 0.05, 0.5, 100000, 0, 0.0) == iid                 # rho = 0 is the old iid null
    for name, f in global_tests(0.5).items():
        assert abs((f(p) > dep[name]).mean() - 0.05) < 0.006, name
    assert (global_tests(0.5)["sum"](p) > iid["sum"]).mean() > 0.15             # iid Stouffer threshold: ~0.20


def test_intraclass_correlation_recovers_rho_with_unequal_groups():
    sizes = rng.integers(3, 60, 3000)
    groups = np.repeat(np.arange(len(sizes)), sizes)
    z = np.sqrt(0.15) * rng.standard_normal(len(sizes))[groups] + np.sqrt(0.85) * rng.standard_normal(len(groups))
    assert abs(intraclass_correlation(z, groups) - 0.15) < 0.015
    assert intraclass_correlation(rng.standard_normal(len(groups)), groups) < 0.01
