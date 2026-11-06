import numpy as np
import pandas as pd
import pytest

from sparse_edits import phase
from scipy.stats import norm


def test_place_edits_keeps_human_sentence_where_edit_rejected():
    rng = np.random.default_rng(0)
    h, a = rng.random((5000, 8)), rng.random((5000, 8))
    a[rng.random(a.shape) < 0.2] = np.nan                       # 20% of edits rejected
    z, truth, kept = phase.place_edits(h, a, 3, rng)
    assert not (truth & np.isnan(a)).any()                      # never marked edited where no edit exists
    assert truth.sum() + kept == 3 * len(h)                     # each chosen sentence is edited or stays human
    assert np.allclose(z, norm.isf(np.where(truth, a, h)))
    assert abs(kept / (3 * len(h)) - 0.2) < 0.02                # positions ignore availability, so ~20% stay human


def test_cell_seed_does_not_depend_on_number_types():
    cfg = dict(seed=0)
    python = phase.cell_rng(cfg, "truth", "c4", "window", 16, "ed", 0.1, 0.5).random()
    numpy = phase.cell_rng(cfg, "truth", "c4", "window", np.int64(16), "ed", np.float64(0.1), np.float64(0.5)).random()
    assert python == numpy


def test_summarize_intervals():
    row = phase.summarize(dict(reject=np.array([True] * 30 + [False] * 70), err=np.arange(100.0)), 0.95)
    assert row["reject"] == 0.3 and row["reject_lo"] < 0.3 < row["reject_hi"] < 0.4
    assert row["err"] == 49.5 and np.isclose(row["err_hi"] - 49.5, 1.96 * np.arange(100.0).std(ddof=1) / 10, rtol=1e-3)


def make_data(tmp_path, n_docs=60, n_sent=12, missing=0.1, seed=0):
    rng = np.random.default_rng(seed)
    rows, vid = [], 0
    for d in range(n_docs):
        for j in range(n_sent):
            sid = d * n_sent + j
            rows.append(dict(version_id=vid, sentence_id=sid, source="s", doc_id=d, sent_idx=j, split="test", editor="human",
                             target_r=0.0, rep=0, actual_r=0.0)); vid += 1
            for rep in range(2):
                if rng.random() > missing:                     # some edits rejected, as in the real pipeline
                    rows.append(dict(version_id=vid, sentence_id=sid, source="s", doc_id=d, sent_idx=j, split="test",
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
        empirical_null_central=0.5, storey_lambda=0.5, ci_level=0.95, whole_min_sentences=1))
    return cfg, versions


def test_detectors_get_identical_draws_regardless_of_order(tmp_path):
    cfg, versions = make_data(tmp_path)
    a_alone, _ = phase.empirical(cfg, "a", versions)
    phase.empirical(cfg, "b", versions)                         # running another detector first changes nothing
    a_after, _ = phase.empirical(cfg, "a", versions)
    pd.testing.assert_frame_equal(a_alone, a_after)
    b, _ = phase.empirical(cfg, "b", versions)
    cols = ["mode", "n", "editor", "target_r", "theta", "null", "n_docs", "n_kept_human", "m_mean"]
    conformal = lambda df: df[df["null"] == "conformal"][cols].reset_index(drop=True)   # empirical-null fit may skip
    pd.testing.assert_frame_equal(conformal(a_alone), conformal(b))                       # same docs, same edits


def test_rejected_edits_keep_the_human_sentence_and_every_document(tmp_path):
    cfg, versions = make_data(tmp_path, missing=0.3)
    out, _ = phase.empirical(cfg, "a", versions)
    edited = out[(out.editor == "ed") & (out["null"] == "conformal")].iloc[0]
    m = round(8 ** 0.5)
    assert edited.n_docs == 60                                  # one window per document, none dropped
    assert edited.m_mean + edited.n_kept_human / edited.n_docs == pytest.approx(m)
    assert 0 < edited.n_kept_human < 0.2 * m * edited.n_docs    # both reps rejected for ~9% of sentences
    assert np.isfinite(out.mu_eff.dropna()).all()


def test_per_document_rows_reproduce_cell_means(tmp_path):
    cfg, versions = make_data(tmp_path)
    cells, docs = phase.empirical(cfg, "a", versions)
    assert len(docs) == cells.n_docs.sum() and (docs.doc_n == 8).all()
    regrouped = docs.groupby(phase.CELL_KEYS, dropna=False)[["reject_hc", "eps_storey"]].mean().reset_index()
    merged = cells.merge(regrouped, on=phase.CELL_KEYS, suffixes=("", "_docs"))
    assert len(merged) == len(cells)
    assert np.allclose(merged.reject_hc, merged.reject_hc_docs) and np.allclose(merged.eps_storey, merged.eps_storey_docs)
    edited = docs.eps_true > 0
    assert docs.mu_doc[edited].notna().all() and docs.mu_doc[~edited].isna().all()
    conformal = docs[edited & (docs["null"] == "conformal")]
    assert 1.0 < conformal.mu_doc.mean() < 3.0                  # detector "a" shifts edited z by 2


def test_whole_mode_skips_short_documents():
    lengths = [3, 5, 16, 16, 40]
    versions = pd.DataFrame([dict(sentence_id=100 * d + j, doc_id=d, sent_idx=j, split="test", editor="human")
                             for d, n in enumerate(lengths) for j in range(n)])
    rng = np.random.default_rng(0)
    groups = phase.test_documents(versions, "whole", np.nan, rng, min_whole=16)
    assert sorted(g.shape for g in groups) == [(1, 40), (2, 16)]
    assert len(phase.test_documents(versions, "whole", np.nan, rng)) == 4      # default keeps every length


def test_cross_fit_calib_gives_each_calib_sentence_one_p_value_from_the_other_fold():
    from sparse_edits.calibrate import calib_fold, cross_fit_calib
    rng = np.random.default_rng(0)
    df = pd.DataFrame(dict(version_id=np.arange(4000), source="s", doc_id=np.repeat(np.arange(200), 20),
                           split="calib", editor="human", n_words=rng.integers(5, 30, 4000),
                           score=rng.standard_normal(4000)))
    cfg = dict(seed=0, calibrate=dict(length_bins=5))
    out = cross_fit_calib(cfg, df)
    assert sorted(out.version_id) == list(range(4000)) and out.p.between(0, 1).all()
    folds = df.assign(fold=calib_fold(cfg, df.doc_id)).groupby("doc_id").fold.nunique()
    assert (folds == 1).all() and 0.4 < calib_fold(cfg, df.doc_id).mean() < 0.6   # whole documents per fold


def test_dependent_null_rows_use_calib_correlation(tmp_path):
    cfg, versions = make_data(tmp_path)
    cells, _ = phase.empirical(cfg, "a", versions)
    assert "conformal_dependent" not in set(cells["null"])                    # no calib p-values yet
    rng = np.random.default_rng(1)
    human = versions[versions.editor == "human"]
    z = np.sqrt(0.3) * rng.standard_normal(human.doc_id.max() + 1)[human.doc_id] + np.sqrt(0.7) * rng.standard_normal(len(human))
    (tmp_path / "pvalues_calib").mkdir()
    pd.DataFrame({"version_id": human.version_id, "p": norm.sf(z), "fold": 0}).to_parquet(
        tmp_path / "pvalues_calib" / "a.parquet")
    cells, docs = phase.empirical(cfg, "a", versions)
    dep = cells[cells["null"] == "conformal_dependent"]
    assert len(dep) == len(cells[cells["null"] == "conformal"]) and abs(dep.rho.iloc[0] - 0.3) < 0.1
    assert (cells[cells["null"] != "conformal_dependent"].rho == 0).all()
    assert set(docs["null"]) >= {"conformal", "conformal_dependent"}
