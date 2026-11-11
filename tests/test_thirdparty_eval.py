import numpy as np
import pandas as pd

from sparse_edits import thirdparty_eval as te

CFG = dict(seed=0, calibrate=dict(length_bins=10), thirdparty=dict(null_sentences_per_bin=200),
           phase=dict(alpha=0.05, hc_alpha0=0.5, bh_q=0.1, storey_lambda=0.5, null_mc_reps=4000))


def docs_frame():
    # dataset a: designated null (fold -1) and test docs; dataset b: cross-fit humans; corpus: the corpus null
    return pd.DataFrame(dict(
        doc_id=range(8), dataset=["a", "a", "a", "b", "b", "b", "corpus", "corpus"],
        scope_domain=["a/x"] * 3 + ["b/y"] * 3 + ["corpus/z"] * 2,
        null_domain=[True, False, False, True, True, False, True, True],
        null_dataset=[True, False, False, True, True, False, True, True],
        xgroup=["", "", "", "b", "b", "b", "", ""], fold=[-1, -1, -1, 0, 1, 0, -1, -1]))


def test_null_sets_cross_fit_and_never_test_a_designated_null():
    sets = {key: (sorted(t), sorted(n)) for t, n, key in te.null_sets(docs_frame(), "dataset")}
    assert sets[("a", "", -1)] == ([1, 2], [0])                  # doc 0 is the null, never tested
    assert sets[("b", "b", 0)] == ([3, 5], [4]) and sets[("b", "b", 1)] == ([4], [3])   # other fold only
    corpus = {key: sorted(n) for _, n, key in te.null_sets(docs_frame(), "corpus")}
    assert all(n == [6, 7] for n in corpus.values())


def test_human_documents_reject_at_about_alpha():
    rng = np.random.default_rng(1)
    n_docs, n = 3000, 12
    docs = pd.DataFrame(dict(doc_id=range(n_docs), dataset="a", scope_domain="a/x", null_domain=np.arange(n_docs) < 1500,
                             xgroup="", fold=-1))
    docs["null_dataset"] = docs.null_domain
    sents = pd.DataFrame(dict(doc_id=np.repeat(range(n_docs), n), sent_idx=np.tile(range(n), n_docs),
                              n_words=rng.integers(5, 30, n_docs * n), label=0))
    score = rng.standard_normal(len(sents))                      # every document human: exchangeable scores
    p, rho = te.sentence_pvalues(CFG, docs, sents, score, "dataset", "test")
    assert np.isnan(p[: 1500 * n]).all() and not np.isnan(p[1500 * n:]).any()
    out = te.outcomes(CFG, sents, p, rho)
    assert len(out) == 1500 and (out.n == n).all()
    assert abs(out.reject_hc_iid.mean() - 0.05) < 0.02 and abs(out.reject_min_p_iid.mean() - 0.05) < 0.02
    assert ((out.hc_p_iid > 0) & (out.hc_p_iid <= 1)).all() and abs((out.hc_p_iid <= 0.05).mean() - 0.05) < 0.02
