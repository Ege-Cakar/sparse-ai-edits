import numpy as np
from scipy.stats import false_discovery_control, kstest, norm

from sparse_edits.stats import (bh, conformal_pvalues, detection_boundary, empirical_null, exact_recovery_boundary,
                                global_tests, hc, n_edited, null_thresholds, sample_truth)

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
