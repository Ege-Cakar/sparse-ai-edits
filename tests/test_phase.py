import numpy as np
import pandas as pd
import pytest

from sparse_edits import phase
from sparse_edits.stats import sample_truth_available


def test_sample_truth_available_uses_only_available_positions():
    rng = np.random.default_rng(0)
    available = rng.random((2000, 16)) > 0.1
    available[0, :14] = False                                  # only 2 available: too few for m = 3
    truth, ok = sample_truth_available(available, 3, rng)
    assert not ok[0] and ok[1:].all()
    assert (truth[ok].sum(axis=1) == 3).all()
    assert not (truth & ~available).any()
    full = np.ones((20000, 8), bool)                           # uniform over positions when all are available
    assert abs(sample_truth_available(full, 2, rng)[0].mean(axis=0) - 2 / 8).max() < 0.02


def make_data(tmp_path, n_docs=60, n_sent=12, missing=0.1, seed=0):
    rng = np.random.default_rng(seed)
    rows, vid = [], 0
    for d in range(n_docs):
        for j in range(n_sent):
            sid = d * n_sent + j
            rows.append(dict(version_id=vid, sentence_id=sid, doc_id=d, sent_idx=j, split="test", editor="human",
                             target_r=0.0, rep=0, actual_r=0.0)); vid += 1
            for rep in range(2):
                if rng.random() > missing:                     # some edits rejected, as in the real pipeline
                    rows.append(dict(version_id=vid, sentence_id=sid, doc_id=d, sent_idx=j, split="test",
                                     editor="ed", target_r=0.3, rep=rep, actual_r=0.3)); vid += 1
    versions = pd.DataFrame(rows)
    (tmp_path / "pvalues").mkdir()
    for det, shift in [("a", 2.0), ("b", 1.0)]:
        z = rng.standard_normal(len(versions)) + shift * (versions.editor != "human")
        from scipy.stats import norm
        pd.DataFrame({"version_id": versions.version_id, "p": norm.sf(z)}).to_parquet(
            tmp_path / "pvalues" / f"{det}.parquet")
    cfg = dict(data_dir=str(tmp_path), seed=0, phase=dict(
        modes=["window"], window_n=[8], thetas=[0.5], alpha=0.05, bh_q=0.1, hc_alpha0=0.5, null_mc_reps=2000,
        empirical_null_central=0.5))
    return cfg, versions


def test_detectors_get_identical_draws_regardless_of_order(tmp_path):
    cfg, versions = make_data(tmp_path)
    a_alone = phase.empirical(cfg, "a", versions)
    phase.empirical(cfg, "b", versions)                         # running another detector first changes nothing
    a_after = phase.empirical(cfg, "a", versions)
    pd.testing.assert_frame_equal(a_alone, a_after)
    b = phase.empirical(cfg, "b", versions)
    cols = ["mode", "n", "editor", "target_r", "theta", "null", "n_docs", "n_dropped", "m_mean"]
    conformal = lambda df: df[df["null"] == "conformal"][cols].reset_index(drop=True)   # empirical-null fit may skip
    pd.testing.assert_frame_equal(conformal(a_alone), conformal(b))                       # same docs, same edits


def test_rejected_edits_cost_sentences_not_documents(tmp_path):
    cfg, versions = make_data(tmp_path, missing=0.3)
    out = phase.empirical(cfg, "a", versions)
    edited = out[(out.editor == "ed") & (out["null"] == "conformal")].iloc[0]
    # With both reps rejected for ~9% of sentences, requiring every sentence to have an edit (the old rule)
    # would keep about 0.91^8 = 47% of windows; drawing among available sentences keeps nearly all.
    assert edited.n_docs >= 0.95 * (edited.n_docs + edited.n_dropped)
    assert edited.m_mean == pytest.approx(round(8 ** 0.5))
    assert np.isfinite(out.mu_eff.dropna()).all()
