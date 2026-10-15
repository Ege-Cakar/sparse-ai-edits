"""Higher Criticism under dependent sentences: innovated HC and PC-adjusted HC (Lecture 4), plus the
thresholding rule for signal recovery (Lecture 3).

Notation follows the lecture slides. Lecture <-> project:
  p           number of tests             = n sentences in a window (phase.window_n)
  X = mu + Z  z-scores of one window,     Z ~ N_p(0, Sigma); rows of a (docs, p) matrix here
  mu_j        0 for human sentences, tau_p = sqrt(2 r log p) for edited ones (eps_p = p^-theta)
  Sigma, Omega = Sigma^{-1}               null covariance / precision of a window's z-scores
  "n iid copies" for estimating Sigma   = human-only windows (calib split), all mu = 0
  pi_j        p-values that go into HC (stats.hc)

Why dependence matters here: sentences of one document share an author, topic and site, so their z-scores
move together. Two structures from Lecture 4, slide 6:
  * Sigma low-rank plus sparse (a confounding factor, e.g. per-document style shifting every sentence)
      -> PC adjustment: remove the top K principal components (Ke, Xue and Yang 2020).
  * Omega sparse (conditional dependence only between nearby sentences)
      -> innovated transform Y = Omega X (Hall and Jin 2010), which maximizes every site's SNR at once.
Plain HC with i.i.d. Monte Carlo thresholds (phase.py) assumes Sigma = I; the factor case inflates its type I
error, and both cases cost power.

Status: the transforms, p-values and HC wrappers are implemented and tested on simulated Gaussian data.
Stubs (NotImplementedError) mark what needs project data or a team decision: the calib-window z-scores,
choosing K, and a sparse precision estimator.
"""
import numpy as np
from scipy.stats import norm

from .stats import hc


# ----- Estimating Sigma and Omega from null (human-only) windows --------------------------------------------

def sample_covariance(X0):
    """Sigma-hat from null windows X0 (copies, p). Mean is 0 under the null, so no centering."""
    X0 = np.asarray(X0, float)
    return X0.T @ X0 / len(X0)


def toeplitz_covariance(X0, max_lag=None):
    """Sigma-hat assuming stationarity in sentence position: Sigma_jk depends only on |j - k|.

    Averages each diagonal of the sample covariance (p^2 parameters -> p), and bands it at max_lag
    (entries with |j - k| > max_lag set to 0). The natural estimator if correlation decays with distance
    between sentences."""
    S = sample_covariance(X0)
    p = S.shape[0]
    lags = np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    acov = np.array([S[lags == h].mean() for h in range(p)])
    if max_lag is not None:
        acov[max_lag + 1:] = 0.0
    return acov[lags]


def precision(Sigma):
    """Omega = Sigma^{-1} by direct inversion. Fine when the number of null windows is much larger than p
    (p = 16 or 32 here); use precision_sparse when it is not."""
    return np.linalg.inv(Sigma)


def precision_sparse(X0, **kwargs):
    """Sparse precision estimate (graphical lasso, CLIME, ...): 'another chapter' in the lecture.
    TODO: needed only if the number of calib windows is not >> p; choose method and tuning."""
    raise NotImplementedError("sparse precision estimation not chosen yet; see docs/review.md item 54")


def calib_window_z(cfg, detector, n):
    """Null windows X0 (copies, n) of z-scores for human calib sentences, to estimate Sigma.

    TODO: calibrate.py computes p-values only outside the calib split, because calib is the conformal null.
    Using the same sentences both as the null and as X0 would understate dependence. Options: split calib into
    two halves (null for p-values / copies for Sigma), or cross-fit. Then take windows of n consecutive
    sentences, as phase.test_documents does."""
    raise NotImplementedError("needs out-of-fold z-scores for calib sentences; see docs/review.md item 54")


# ----- Transforms and marginal p-values (Lecture 4, slide 9) ------------------------------------------------

def brute_force_pvalues(X, Sigma):
    """Y = X:  pi_j = P(N(0, Sigma_jj) >= Y_j). Right marginals, but the dependence is left in."""
    return norm.sf(np.asarray(X, float) / np.sqrt(np.diag(Sigma)))


def whitening_pvalues(X, Sigma):
    """Y = Sigma^{-1/2} X:  Y ~ N(Sigma^{-1/2} mu, I),  pi_j = P(N(0, 1) >= Y_j)."""
    lam, U = np.linalg.eigh(Sigma)
    W = U @ np.diag(lam ** -0.5) @ U.T
    return norm.sf(np.asarray(X, float) @ W.T)


def innovated_pvalues(X, Omega):
    """Innovated transform Y = Omega X:  Y ~ N(Omega mu, Omega),  pi_j = P(N(0, Omega_jj) >= Y_j).

    A = Sigma^{-1} maximizes SNR_j = (A mu)_j / sqrt((A Sigma A')_jj) for every j at once when mu is sparse
    (Lecture 4, slides 11-12)."""
    Y = np.asarray(X, float) @ np.asarray(Omega).T
    return norm.sf(Y / np.sqrt(np.diag(Omega)))


def pc_adjust(X, Sigma, K):
    """Remove the top K principal components (Lecture 4, slide 20).

    X-tilde = X - sum_k (xi_k' X) xi_k,   Sigma0-hat = Sigma-hat - sum_k lambda_k xi_k xi_k'.
    With a per-document style factor, Sigma = sigma_a^2 11' + I, xi_1 = 1/sqrt(p), and X-tilde is X minus its
    window mean. Signal loss (xi_k' mu) xi_k is small when mu is sparse (slide 23)."""
    lam, xi = np.linalg.eigh(Sigma)
    lam, xi = lam[::-1][:K], xi[:, ::-1][:, :K]                     # K largest
    X = np.asarray(X, float)
    X_tilde = X - (X @ xi) @ xi.T
    Sigma0 = Sigma - (xi * lam) @ xi.T
    return X_tilde, Sigma0


def pc_adjusted_pvalues(X, Sigma, K):
    """pi_j = P(N(0, Sigma0-hat(j, j)) >= X-tilde_j)."""
    X_tilde, Sigma0 = pc_adjust(X, Sigma, K)
    return norm.sf(X_tilde / np.sqrt(np.diag(Sigma0)))


def choose_K(lam, n_copies):
    """Number of principal components to remove.

    TODO: options are a scree / eigen-gap rule, eigenvalues above the Marchenko-Pastur edge
    (1 + sqrt(p / n_copies))^2 times the noise level, or the K that brings the null type I error of HC back to
    alpha on held-out calib windows. Until decided, pass K explicitly (K = 1 for a single style factor)."""
    raise NotImplementedError("rule for K not chosen yet; see docs/review.md item 55")


# ----- HC on adjusted p-values (Lecture 4: "plug pi_1, ..., pi_p into HC") ---------------------------------

def innovated_hc(X, Omega, alpha0=0.5, plus=False):
    return hc(innovated_pvalues(X, Omega), alpha0, plus)


def pc_adjusted_hc(X, Sigma, K, alpha0=0.5, plus=False):
    return hc(pc_adjusted_pvalues(X, Sigma, K), alpha0, plus)


def null_thresholds_dependent(pvalue_fn, Sigma, alpha, alpha0, reps, seed, plus=False):
    """Level-alpha critical value of HC on pvalue_fn(X) when X ~ N_p(0, Sigma), by Monte Carlo.

    phase.null_thresholds assumes i.i.d. uniform p-values. After an adjustment the p-values are only
    approximately independent (e.g. X-tilde has correlation -1/p after removing the mean), so this gives the
    exact null for a known Sigma; with Sigma-hat it is a parametric bootstrap."""
    rng = np.random.default_rng(seed)
    X0 = rng.multivariate_normal(np.zeros(len(Sigma)), Sigma, size=reps)
    return np.quantile(hc(pvalue_fn(X0), alpha0, plus), 1 - alpha)


# ----- Signal recovery by thresholding (Lecture 3, slides 11-17) -------------------------------------------

def q_star(theta, r):
    """Optimal threshold exponent (Theorem 2): (theta + r)^2 / (4r) if r > theta, else theta."""
    theta, r = np.asarray(theta, float), np.asarray(r, float)
    return np.where(r > theta, (theta + r) ** 2 / (4 * r), theta)


def threshold_select(X, theta, r):
    """S-hat(q*) = {j : X_j > sqrt(2 q* log p)}. An oracle rule: it needs (theta, r). In the empirical phase
    diagrams theta is known by design and r can be r_eff, which gives the ideal recovery benchmark
    (docs/review.md item 39)."""
    X = np.asarray(X, float)
    p = X.shape[-1]
    return X > np.sqrt(2 * q_star(theta, r) * np.log(p))


def hc_critical_value(p, delta=0.1):
    """Asymptotic HC rejection level sqrt(2 (1 + delta) log log p) (Lecture 3, slide 23). Loose at p = 16-32,
    where phase.null_thresholds (Monte Carlo) should be used instead; kept for comparison."""
    return np.sqrt(2 * (1 + delta) * np.log(np.log(p)))


# ----- Simulated windows with dependence (for tests and for the report's sanity figures) -------------------

def factor_covariance(p, sigma_a2):
    """Sigma = sigma_a^2 11' + I: every sentence of a window shares one style shift a ~ N(0, sigma_a^2)."""
    return sigma_a2 * np.ones((p, p)) + np.eye(p)


def tridiagonal_covariance(p, rho):
    """Sigma = I with rho on the sub-diagonals (Lecture 4, slide 10 uses rho = 0.4): neighbor dependence."""
    return np.eye(p) + rho * (np.eye(p, k=1) + np.eye(p, k=-1))


def simulate_windows(docs, Sigma, theta, r, rng):
    """X = mu + Z with exactly m = round(p^(1 - theta)) entries of mu equal to tau_p = sqrt(2 r log p)."""
    p = len(Sigma)
    m = int(round(p ** (1 - theta)))
    truth = rng.random((docs, p)).argsort(axis=1) < m
    Z = rng.multivariate_normal(np.zeros(p), Sigma, size=docs)
    return Z + np.sqrt(2 * r * np.log(p)) * truth, truth
