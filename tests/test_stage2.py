import numpy as np
import pandas as pd
from scipy.stats import kstest, norm

from sparse_edits.stage2 import generated_correlation, lower_pvalues, outcomes

rng = np.random.default_rng(0)
CFG = {"seed": 0, "calibrate": {"length_bins": 5}, "stage2": {"eps0": [0.9], "null_mc_reps": 20000},
       "phase": {"alpha": 0.05, "hc_alpha0": 0.5, "null_mc_reps": 20000, "storey_lambda": 0.5}}


def test_lower_pvalues_uniform_for_generated_like_text_and_small_for_human_like():
    def sents(n, shift):
        lengths = rng.integers(5, 40, n)
        return pd.DataFrame({"source": "a", "n_words": lengths, "score": rng.normal(lengths / 10 + shift, 1)})
    null = sents(20000, 0)
    assert kstest(lower_pvalues(null, sents(5000, 0), 5, rng), "uniform").pvalue > 0.01
    assert lower_pvalues(null, sents(5000, -2), 5, rng).median() < 0.05   # lower score = more human-like


def test_composite_null_keeps_its_level_at_the_boundary_and_has_power_inside_the_alternative():
    """Stage-2 p-values: AI sentences Uniform(0, 1), equicorrelated in z at rho; human sentences far in the lower
    tail. At a human fraction of exactly 1 - eps0 the composite null holds, so it is rejected at most ~alpha of the
    time; at one half it is rejected nearly always."""
    n, docs, rho = 40, 4000, 0.2
    z = np.sqrt(rho) * rng.standard_normal((docs, 1)) + np.sqrt(1 - rho) * rng.standard_normal((docs, n))
    for human, check in [(4, lambda r: r <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / docs)), (20, lambda r: r > 0.95)]:
        p2 = norm.sf(np.where(np.arange(n) < human, z + 8, z))
        out = outcomes(rng.random((docs, n)), p2, 0.0, rho, CFG)
        assert check(out["reject2_eps0.9"].mean())
        assert (out["eta_lower"] <= human / n).mean() >= 0.95 - 3 * np.sqrt(0.05 * 0.95 / docs)


def test_generated_correlation_recovers_within_document_correlation():
    rho, docs, n = 0.3, 600, 20
    z = np.sqrt(rho) * rng.standard_normal((docs, 1)) + np.sqrt(1 - rho) * rng.standard_normal((docs, n))
    null = pd.DataFrame({"source": "a", "editor": "generated-x", "doc_id": np.repeat(np.arange(docs), n),
                         "n_words": 10, "score": z.ravel()})
    assert abs(generated_correlation(CFG, null, rng)["a"] - rho) < 0.05
