"""Our tests on the third-party datasets (inputs from thirdparty.py): conformal p-values and document tests per detector
and null variant, written to thirdparty/results/.

Sentence detectors: "ensemble" (thirdparty.ensemble's runs, each standardized by its scores on our corpus's val human
sentences, then averaged, as ensemble.py) and every thirdparty.detectors entry that is not one of our runs. Null
variants, each a set of human documents:
- domain: the documents of the tested document's scope_domain marked null_domain;
- dataset: the dataset's designated null (null_dataset);
- corpus: our corpus's calib documents, all sources pooled.
A document of a cross-fit group (fold f) is tested only against null documents of that group's other fold, and a null
document outside a cross-fit group is not tested in its own variant. Sentence p-values are conformal within length bins,
min(calibrate.length_bins, null sentences / thirdparty.null_sentences_per_bin) of them; each null set's
within-document correlation rho is the intraclass correlation of its sentences' z, cross-fitted between two halves of
its documents, rounded to 0.01. Sentences without a score (MELD: nothing left after its canonicalization) are left
out.

results/<detector>_<variant>.parquet, one row per tested document: n, rho, every stats.evaluate outcome under iid
thresholds (suffix _iid) and rho-correlated ones (_rho), the HC statistic and its p-value at the document's n (hc_p_*),
and the share of its sentences flagged at p < 0.01 and by Bonferroni at phase.alpha.
results/<detector>_<variant>_sentences.parquet: (doc_id, sent_idx, p).
results/<detector>_<variant>_documents.parquet for whole-document detectors: each document's conformal p-value against
the same null documents (one length bin) and rejection at phase.alpha.
"""
import zlib
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import cli, path
from .stats import conformal_pvalues, evaluate, global_tests, hc, intraclass_correlation, null_pvalues

VARIANTS = ("domain", "dataset", "corpus")


def ensemble_scores(cfg, sents):
    v = pd.read_parquet(path(cfg, "versions.parquet"), columns=["version_id", "split", "editor"])
    ref = v.version_id[(v.split == "val") & (v.editor == "human")]
    z = []
    for run in cfg["thirdparty"]["ensemble"]:
        r = pd.read_parquet(path(cfg, "scores", f"{run}.parquet")).set_index("version_id").score.reindex(ref)
        assert r.notna().all(), f"{run}: val human sentences without a score"
        s = sents[["doc_id", "sent_idx"]].merge(pd.read_parquet(path(cfg, "thirdparty", "scores", f"{run}.parquet")),
                                                how="left").score.to_numpy()
        z.append((s - r.mean()) / r.std())
    return np.mean(z, axis=0)


def membership(docs, variant):
    """(null pool of each document, whether it is a member of its pool's null) under a variant."""
    if variant == "domain":
        return docs.scope_domain, docs.null_domain
    if variant == "dataset":
        return docs.dataset, docs.null_dataset
    return pd.Series("corpus", index=docs.index), docs.dataset.eq("corpus")


def null_sets(docs, variant):
    """[(doc_ids tested, doc_ids of their null, key)] under a variant (see the module docstring)."""
    pool, member = membership(docs, variant)
    d = docs.assign(pool=pool.to_numpy(), member=member.to_numpy())
    tested = d[d.dataset.ne("corpus") & ~(d.member & (d.fold < 0))]
    out = []
    for (pl, xg, f), g in tested.groupby(["pool", "xgroup", "fold"]):
        null = d[(d.pool == pl) & d.member & ~((d.xgroup == xg) & (d.fold == f) & (f >= 0))]
        if len(null):
            out.append((g.doc_id.to_numpy(), null.doc_id.to_numpy(), (pl, xg, int(f))))
    return out


def n_bins(cfg, n_sentences):
    return int(max(1, min(cfg["calibrate"]["length_bins"], n_sentences // cfg["thirdparty"]["null_sentences_per_bin"])))


def null_rho(cfg, s, rng):
    """Intraclass correlation of z over the null sentences s (doc_id, n_words, score), each half of its documents
    tested against the other half."""
    half = np.array([zlib.crc32(repr((cfg["seed"], int(d))).encode()) % 2 for d in s.doc_id])
    p = np.empty(len(s))
    for k in (0, 1):
        a, b = half != k, half == k
        if not a.any() or not b.any():
            return np.nan
        p[b] = conformal_pvalues(s.score.to_numpy()[a], s.n_words.to_numpy()[a], s.score.to_numpy()[b],
                                 s.n_words.to_numpy()[b], n_bins(cfg, a.sum()), rng)
    return intraclass_correlation(norm.isf(p), s.doc_id.to_numpy())


def sentence_pvalues(cfg, docs, sents, score, variant, name):
    """(p per sentence, NaN where its document is not tested or it has no score; rho per tested doc_id)."""
    s = sents.assign(score=score)[lambda x: np.isfinite(x.score)]
    p, rho = np.full(len(sents), np.nan), {}
    for tested, null, key in null_sets(docs, variant):
        rng = np.random.default_rng([cfg["seed"], zlib.crc32(repr((name, variant) + key).encode())])
        ns, ts = s[s.doc_id.isin(null)], s[s.doc_id.isin(tested)]
        if ns.empty or ts.empty:
            continue
        p[ts.index] = conformal_pvalues(ns.score.to_numpy(), ns.n_words.to_numpy(), ts.score.to_numpy(),
                                        ts.n_words.to_numpy(), n_bins(cfg, len(ns)), rng)
        r = null_rho(cfg, ns, rng)
        rho |= dict.fromkeys(tested.tolist(), round(r, 2) if np.isfinite(r) else 0.0)
    return p, rho


@lru_cache(maxsize=4096)
def null_distribution(n, rho, alpha, alpha0, reps, seed):
    """Level-alpha thresholds of every global test (as stats.null_thresholds, from the same draws) and the sorted null
    HC statistics, for documents of n sentences."""
    p = null_pvalues(n, reps, seed, rho)
    stats = {name: f(p) for name, f in global_tests(alpha0, rho).items()}
    th = {k: np.quantile(v, 1 - alpha) for k, v in stats.items()} | {"hc_all": np.quantile(hc(p, 1.0), 1 - alpha)}
    return th, np.sort(stats["hc"])


def outcomes(cfg, sents, p, rho):
    """One row per tested document with at least one p-value (see the module docstring)."""
    c = cfg["phase"]
    s = sents.assign(p=p)[lambda x: x.p.notna()].sort_values(["doc_id", "sent_idx"])
    ids, start = np.unique(s.doc_id.to_numpy(), return_index=True)
    n = np.diff(np.append(start, len(s)))
    r = np.array([rho[d] for d in ids])
    pv, truth = s.p.to_numpy(), s.label.to_numpy() == 1
    rows = []
    for (k, rr), g in pd.DataFrame(dict(n=n, r=r)).groupby(["n", "r"]):
        idx = start[g.index.to_numpy()][:, None] + np.arange(k)
        P, T = pv[idx], truth[idx]
        row = dict(doc_id=ids[g.index], n=k, rho=rr, hc=hc(P, c["hc_alpha0"]), flag_p01=(P < 0.01).mean(1),
                   flag_bonferroni=(P <= c["alpha"] / k).mean(1))
        for sfx, rv in (("iid", 0.0), ("rho", rr)):
            th, null_hc = null_distribution(int(k), float(rv), c["alpha"], c["hc_alpha0"], c["null_mc_reps"], cfg["seed"])
            row |= {f"{key}_{sfx}": v for key, v in evaluate(P, T, th, c["alpha"], c["hc_alpha0"], c["bh_q"],
                                                             c["storey_lambda"], rv).items()}
            row[f"hc_p_{sfx}"] = (1 + len(null_hc) - np.searchsorted(null_hc, row["hc"], "left")) / (1 + len(null_hc))
        rows.append(pd.DataFrame(row))
    return pd.concat(rows, ignore_index=True)


def document_pvalues(cfg, docs, scores, variant, name):
    """Conformal p-values of whole-document scores against each tested document's null documents."""
    sc = docs[["doc_id"]].merge(scores, how="left").set_index("doc_id").score
    out = []
    for tested, null, key in null_sets(docs, variant):
        rng = np.random.default_rng([cfg["seed"], zlib.crc32(repr((name, variant, "documents") + key).encode())])
        ns, ts = sc.reindex(null).dropna(), sc.reindex(tested)
        p = conformal_pvalues(ns.to_numpy(), np.zeros(len(ns)), ts.to_numpy(), np.zeros(len(ts)), 1, rng)
        out.append(pd.DataFrame(dict(doc_id=tested, p=p, reject=p <= cfg["phase"]["alpha"])))
    return pd.concat(out, ignore_index=True)


def analyze(cfg, names=None, variants=VARIANTS):
    c = cfg["thirdparty"]
    docs = pd.read_parquet(path(cfg, "thirdparty", "docs.parquet"),
                           columns=["doc_id", "dataset", "scope_domain", "null_domain", "null_dataset", "xgroup", "fold"])
    sents = pd.read_parquet(path(cfg, "thirdparty", "sentences.parquet"),
                            columns=["doc_id", "sent_idx", "n_words", "label"])
    names = names or ["ensemble"] + [d["name"] for d in c["detectors"] if d["kind"] != "run"]
    for name in names:
        score = ensemble_scores(cfg, sents) if name == "ensemble" else sents[["doc_id", "sent_idx"]].merge(
            pd.read_parquet(path(cfg, "thirdparty", "scores", f"{name}.parquet")), how="left").score.to_numpy()
        for variant in variants:
            p, rho = sentence_pvalues(cfg, docs, sents, score, variant, name)
            sents[["doc_id", "sent_idx"]].assign(p=p).to_parquet(
                path(cfg, "thirdparty", "results", f"{name}_{variant}_sentences.parquet"), index=False)
            out = outcomes(cfg, sents, p, rho)
            out.to_parquet(path(cfg, "thirdparty", "results", f"{name}_{variant}.parquet"), index=False)
            print(f"{name} / {variant}: {len(out)} documents, {np.isnan(p).sum()} sentences without a p-value", flush=True)
    for d in c["detectors"]:
        if d.get("documents") and d["name"] in names:
            scores = pd.read_parquet(path(cfg, "thirdparty", "docscores", f"{d['name']}.parquet"))
            for variant in variants:
                document_pvalues(cfg, docs, scores, variant, d["name"]).to_parquet(
                    path(cfg, "thirdparty", "results", f"{d['name']}_{variant}_documents.parquet"), index=False)


def main():
    def args(p):
        p.add_argument("--detectors", nargs="+", help="default: ensemble and every non-run thirdparty.detectors entry; "
                       "a run's name analyzes that run alone")
        p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    cfg, args = cli(__doc__, args)
    analyze(cfg, args.detectors, args.variants)


if __name__ == "__main__":
    main()
