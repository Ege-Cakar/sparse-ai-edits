"""Lecture 4 claims, checked by simulation at the project's window size p = 32."""
import numpy as np
import pytest

from sparse_edits import hc_variants as hv
from sparse_edits.stats import hc, null_thresholds

p = 32


def power(Sigma, pvalue_fn, theta, r, rng, reps=3000):
    X, _ = hv.simulate_windows(reps, Sigma, theta, r, rng)
    t = hv.null_thresholds_dependent(pvalue_fn, Sigma, 0.05, 0.5, 10000, 1)
    return np.mean(hc(pvalue_fn(X)) > t)


def test_pc_adjust_closed_form_for_style_factor():
    Sigma = hv.factor_covariance(p, 1.0)
    X = np.random.default_rng(0).standard_normal((5, p))
    X_tilde, Sigma0 = hv.pc_adjust(X, Sigma, K=1)
    assert np.allclose(X_tilde, X - X.mean(axis=1, keepdims=True))        # removing xi_1 = 1/sqrt(p)
    assert np.allclose(Sigma0, np.eye(p) - np.ones((p, p)) / p)


def test_style_factor_inflates_plain_hc_and_pc_adjustment_restores_size():
    rng = np.random.default_rng(1)
    Sigma = hv.factor_covariance(p, 1.0)
    X0 = rng.multivariate_normal(np.zeros(p), Sigma, size=4000)
    t_iid = null_thresholds(p, 0.05, 0.5, 100000, 0)["hc"]                  # what phase.py uses today
    assert np.mean(hc(hv.brute_force_pvalues(X0, Sigma)) > t_iid) > 0.09
    assert 0.035 < np.mean(hc(hv.pc_adjusted_pvalues(X0, Sigma, 1)) > t_iid) < 0.07


def test_pc_adjusted_hc_more_powerful_under_style_factor():
    rng = np.random.default_rng(2)
    Sigma = hv.factor_covariance(p, 1.0)
    brute = power(Sigma, lambda X: hv.brute_force_pvalues(X, Sigma), 0.5, 0.5, rng)
    pc = power(Sigma, lambda X: hv.pc_adjusted_pvalues(X, Sigma, 1), 0.5, 0.5, rng)
    assert pc > brute + 0.15


def test_innovated_hc_more_powerful_under_neighbor_dependence():
    rng = np.random.default_rng(3)
    Sigma = hv.tridiagonal_covariance(p, 0.4)
    Omega = hv.precision(Sigma)
    brute = power(Sigma, lambda X: hv.brute_force_pvalues(X, Sigma), 0.5, 0.3, rng)
    innovated = power(Sigma, lambda X: hv.innovated_pvalues(X, Omega), 0.5, 0.3, rng)
    assert innovated > brute + 0.1


def test_null_pvalues_are_uniform_for_every_transform():
    rng = np.random.default_rng(4)
    Sigma = hv.tridiagonal_covariance(p, 0.4)
    X0 = rng.multivariate_normal(np.zeros(p), Sigma, size=20000)
    for pv in (hv.brute_force_pvalues(X0, Sigma), hv.whitening_pvalues(X0, Sigma),
               hv.innovated_pvalues(X0, hv.precision(Sigma)), hv.pc_adjusted_pvalues(X0, Sigma, 1)):
        assert abs(np.mean(pv < 0.05) - 0.05) < 0.006


def test_toeplitz_covariance_recovers_tridiagonal():
    rng = np.random.default_rng(5)
    Sigma = hv.tridiagonal_covariance(p, 0.4)
    X0 = rng.multivariate_normal(np.zeros(p), Sigma, size=20000)
    assert np.abs(hv.toeplitz_covariance(X0, max_lag=3) - Sigma).max() < 0.03


def test_q_star_matches_theorem_2_and_threshold_recovers_strong_signal():
    assert np.isclose(hv.q_star(0.5, 2.0), (2.5 ** 2) / 8)                 # r > theta
    assert np.isclose(hv.q_star(0.5, 0.3), 0.5)                            # r < theta
    rng = np.random.default_rng(6)
    X, truth = hv.simulate_windows(2000, np.eye(1000), 0.5, 4.0, rng)     # above exact recovery (1 + sqrt(.5))^2
    sel = hv.threshold_select(X, 0.5, 4.0)
    assert sel[truth].mean() > 0.95 and (sel & ~truth).sum() / truth.sum() < 0.05


def test_stubs_are_declared():
    for call in (lambda: hv.calib_window_z({}, "x", 16), lambda: hv.choose_K(np.ones(3), 10),
                 lambda: hv.precision_sparse(np.zeros((3, 3)))):
        with pytest.raises(NotImplementedError):
            call()
