"""Document-level conformal thresholds for the global tests (HC, HC+, min-p, Stouffer, PC-adjusted and innovated HC).

Each whole test document's statistic (stat_* in phase_docs/<detector>.parquet) is ranked among the same statistic on
the calib human documents of its source and length band, computed from the cross-fitted calib p-values in
pvalues_calib/. The randomized rank p-value is exactly Uniform(0, 1) for a human test document exchangeable with those
calib documents, whatever the dependence between its sentences, so rejecting at p <= alpha holds level alpha without
the equicorrelated model behind phase.py's Monte Carlo thresholds. Length bands are quantiles of the calib documents'
lengths, as many as allow phase.doc_conformal.min_docs per band (at most max_bands). Writes
doc_conformal/<detector>.parquet: rejection rates per (source, editor, theta) for each test, with the document counts.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import cli, path
from .stats import global_tests, intraclass_correlation


def calib_stats(cfg, detector, versions):
    """stat_* of every complete calib human document with at least whole_min_sentences sentences, per source at its
    own rho-hat (the same rho phase.py's conformal_dependent rows use)."""
    c = cfg["phase"]
    cal = versions.merge(pd.read_parquet(path(cfg, "pvalues_calib", f"{detector}.parquet")), on="version_id")
    out = []
    for source, g in cal.groupby("source"):
        rho = intraclass_correlation(norm.isf(g.p), g.doc_id)
        docs = g.groupby("doc_id").filter(lambda d: len(d) >= c["whole_min_sentences"] and d.p.notna().all())
        for n, same_n in docs.groupby(docs.groupby("doc_id").p.transform("size")):
            ids = same_n.doc_id.unique()
            p = np.stack([same_n.p[same_n.doc_id == i].to_numpy() for i in ids])
            out.append(pd.DataFrame({"source": source, "doc_id": ids, "doc_n": n,
                                     **{f"stat_{k}": f(p) for k, f in global_tests(c["hc_alpha0"], rho).items()}}))
    return pd.concat(out, ignore_index=True)


def conformal_p(cal, test, rng):
    """Randomized rank p-values of test statistics among calib statistics (larger = more evidence)."""
    cal = np.sort(cal)
    above = len(cal) - np.searchsorted(cal, test, side="right")
    ties = np.searchsorted(cal, test, side="right") - np.searchsorted(cal, test, side="left")
    return (above + rng.random(len(test)) * (1 + ties)) / (1 + len(cal))


def band_edges(n, min_docs, max_bands):
    k = int(np.clip(len(n) // min_docs, 1, max_bands))
    return np.unique(np.quantile(n, np.linspace(0, 1, k + 1)[1:-1], method="lower"))


def doc_conformal(cfg, detector, versions):
    c, rng = cfg["phase"], np.random.default_rng(cfg["seed"])
    cal = calib_stats(cfg, detector, versions)
    docs = pd.read_parquet(path(cfg, "phase_docs", f"{detector}.parquet"))
    docs = docs[(docs["mode"] == "whole") & (docs["null"] == "conformal_dependent")].copy()
    tests = [k[5:] for k in cal if k.startswith("stat_")]
    for source, cs in cal.groupby("source"):
        edges = band_edges(cs.doc_n.to_numpy(), c["doc_conformal"]["min_docs"], c["doc_conformal"]["max_bands"])
        cb, sel = np.digitize(cs.doc_n, edges), docs.source == source
        tb = np.digitize(docs.loc[sel, "doc_n"], edges)
        for b in np.unique(tb):
            rows = docs.index[sel][tb == b]
            for t in tests:
                p = conformal_p(cs[f"stat_{t}"].to_numpy()[cb == b], docs.loc[rows, f"stat_{t}"].to_numpy(), rng)
                docs.loc[rows, f"reject_{t}"] = p <= c["alpha"]
        docs.loc[sel, "n_calib_docs"] = len(cs)
    agg = {f"reject_{t}": "mean" for t in tests} | {"n_calib_docs": "first", "doc_n": "size"}
    return docs.groupby(["source", "editor", "theta"]).agg(agg).rename(columns={"doc_n": "n_docs"}).reset_index()


def main():
    cfg, _ = cli(__doc__)
    versions = pd.read_parquet(path(cfg, "versions.parquet"), columns=["version_id", "source", "doc_id", "split"])
    versions = versions[versions.split == "calib"]
    for f in sorted(Path(cfg["data_dir"], "phase_docs").glob("*.parquet")):
        doc_conformal(cfg, f.stem, versions).to_parquet(path(cfg, "doc_conformal", f.name), index=False)
        print(f"{f.stem}: done")


if __name__ == "__main__":
    main()
