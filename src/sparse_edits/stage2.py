"""Second stage: among documents that stage 1 flags (H0: every sentence human), tell human-drafted, AI-edited ones from
fully AI-generated ones, with the null H0': every sentence AI.

Stage-2 p-values are lower-tail conformal p-values ("more human-like than generated text"): each test sentence's score
against the sentences of the generated calib documents of generate.calibration (pooled; generate.py), within the
test sentence's source and length bin, as calibrate.py does for stage 1. Calibrating on generated documents, not on
AI-edited sentences, keeps H0' "every sentence AI"; it holds only for text like the calibration generators', which the
held-out generators check. The within-document correlation rho' of their z is estimated on generated calib documents,
cross-fitted (two folds by calibrate.calib_fold of doc_id, each fold against the other; documents are grouped by
generator and doc_id), as stage 1's rho is on human calib documents (phase.calib_correlation).
Per document: the global tests (stats.global_tests) at Monte Carlo thresholds for equicorrelated z (rho for stage 1,
rho' for stage 2; stats.null_thresholds); the edited fraction's Storey estimate and Meinshausen-Rice lower bound from
the stage-1 p-values; and, from the stage-2 p-values, eta_lower, the Meinshausen-Rice lower bound on the human
(non-AI) fraction. The composite null "at least a fraction eps0 of sentences AI" (stage2.eps0) is rejected when
eta_lower > 1 - eps0: a level-alpha test, since eta_lower stays at or below the true human fraction with probability
at least 1 - alpha. The level of the dependent-null bound rests on the same equicorrelation model as stage 1's.
Documents: whole test documents with at least stage2.min_sentences sentences: human ones; sparse-edit ones, the cells
of phase.py's whole mode with the same seeds, so the same edited positions; and every generator's generated ones.
Needs pvalues/ and pvalues_calib/ from calibrate.py, generated/versions.parquet and generated/scores/ (generate.py,
detect.py --versions). Writes stage2/<detector>.parquet, one row per document and cell; --summary prints rates with
Wilson intervals.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import cli, path
from .calibrate import HIGHER_IS_HUMAN, calib_fold
from .phase import calib_correlation, cell_rng, lookup, place_edits, rep_key, test_documents
from .stats import (conformal_pvalues, eps_lower, eps_storey, global_tests, intraclass_correlation, n_edited,
                    null_thresholds, wilson)

V_COLS = ["version_id", "sentence_id", "source", "doc_id", "sent_idx", "split", "editor", "target_r", "rep",
          "actual_r", "n_words"]


def lower_pvalues(null, test, bins, rng):
    """Conformal p-values of test sentences' scores in the lower tail of the null's, within source and length bin
    (frames with source, score, n_words); NaN for a source without null sentences."""
    p = pd.Series(np.nan, index=test.index)
    for source, t in test.groupby("source"):
        if not len(n := null[null.source == source]):
            continue
        p[t.index] = conformal_pvalues(-n.score.to_numpy(), n.n_words.to_numpy(), -t.score.to_numpy(),
                                       t.n_words.to_numpy(), bins, rng)
    return p


def generated_correlation(cfg, null, rng):
    """rho' per source: intraclass correlation of z = Phi^-1(1 - p') within generated calib documents, each fold's
    p-values against the other fold."""
    fold = calib_fold(cfg, null.doc_id)
    p = pd.concat([lower_pvalues(null[fold != k], null[fold == k], cfg["calibrate"]["length_bins"], rng)
                   for k in (0, 1)])
    z, doc = norm.isf(p[null.index]), null.editor + "/" + null.doc_id.astype(str)
    return {s: intraclass_correlation(z[m], doc[m].to_numpy()) for s in sorted(null.source.unique())
            for m in [(null.source == s).to_numpy()]}


def outcomes(p1, p2, rho1, rho2, cfg):
    """Per-document outcomes of both stages for (docs, n) p-value matrices (stage 1, stage 2)."""
    c, n = cfg["phase"], p1.shape[1]
    assert np.isfinite(rho1) and np.isfinite(rho2), "no correlation estimate for this source"
    reps = cfg["stage2"].get("null_mc_reps", c["null_mc_reps"])
    t1, t2 = (null_thresholds(n, c["alpha"], c["hc_alpha0"], reps, cfg["seed"], float(r)) for r in (rho1, rho2))
    out = {}
    for name, f in global_tests(c["hc_alpha0"]).items():
        out[f"reject_{name}"], out[f"reject2_{name}"] = f(p1) > t1[name], f(p2) > t2[name]
    out |= dict(eps_storey=eps_storey(p1, c["storey_lambda"]), eps_lower=eps_lower(p1, t1["hc_all"]),
                eta_lower=eps_lower(p2, t2["hc_all"]))
    return out | {f"reject2_eps{e:g}": out["eta_lower"] > 1 - e for e in cfg["stage2"]["eps0"]}


def frame(out, **cols):
    return pd.DataFrame(cols | out)


def generated_docs(cfg, g, rho1, rho2):
    """Rows for each generator's generated test documents (g: their sentences with p1, p2)."""
    rows = []
    g = g.sort_values(["editor", "doc_id", "sent_idx"])
    n = g.groupby(["editor", "doc_id"]).sent_idx.transform("size")
    for (editor, k), d in g[n >= cfg["stage2"]["min_sentences"]].groupby(["editor", n]):
        p1, p2 = (d[c].to_numpy().reshape(-1, k) for c in ("p1", "p2"))
        ok = ~np.isnan(p1).any(axis=1) & ~np.isnan(p2).any(axis=1)
        rows.append(frame(outcomes(p1[ok], p2[ok], rho1, rho2, cfg), kind="generated", label=editor.split("-", 1)[1],
                          theta=np.nan, doc_id=d.doc_id.to_numpy()[::k][ok], n=k, m=k))
    return rows


def human_and_edited_docs(cfg, source, versions, test, rho1, rho2):
    """Rows for human test documents and for phase.py's whole-mode sparse-edit cells on them (same seeds, same
    rejected-edit rule: a chosen sentence without an accepted edit stays human)."""
    c, rows = cfg["phase"], []
    groups = test_documents(versions, "whole", np.nan, cell_rng(cfg, "window", source, "whole", np.nan),
                            cfg["stage2"]["min_sentences"])
    human = test[test.editor == "human"].set_index("sentence_id")
    groups = [ids[~np.isnan(lookup(human.p1, ids)).any(axis=1)] for ids in groups]   # as phase.complete_human
    h = [(lookup(human.p1, ids), lookup(human.p2, ids), lookup(human.doc_id, ids[:, :1])[:, 0]) for ids in groups]
    for ids, (h1, h2, doc) in zip(groups, h):
        if len(ids):
            rows.append(frame(outcomes(h1, h2, rho1, rho2, cfg), kind="human", label="human", theta=np.inf,
                              doc_id=doc, n=ids.shape[1], m=0))
    edits = test[test.editor != "human"]
    edits = edits.assign(key=rep_key(cfg, edits)).sort_values("key").drop_duplicates(["editor", "target_r",
                                                                                     "sentence_id"])
    for (editor, target_r), e in edits.groupby(["editor", "target_r"], dropna=False):
        e = e.set_index("sentence_id")
        a = [(lookup(e.p1, ids), lookup(e.p2, ids)) for ids in groups]
        for theta in c["thetas"]:
            rng = cell_rng(cfg, "truth", source, "whole", np.nan, editor, target_r, theta)
            for ids, (h1, h2, doc), (a1, a2) in zip(groups, h, a):
                z1, truth, _ = place_edits(h1, a1, n_edited(ids.shape[1], theta), rng)   # as phase.run_cell
                if len(ids):
                    rows.append(frame(outcomes(norm.sf(z1), np.where(truth, a2, h2), rho1, rho2, cfg), kind="edited",
                                      label=editor, theta=theta, doc_id=doc, n=ids.shape[1], m=truth.sum(axis=1)))
    return rows


def run(cfg, detector, versions, gen_versions):
    """Per-document rows of one detector, from its stage-1 p-values (pvalues/) and both score files."""
    g = cfg["generate"]
    sign = -1 if detector.startswith(tuple(HIGHER_IS_HUMAN)) else 1
    scores = pd.concat([pd.read_parquet(path(cfg, d, f"{detector}.parquet")) for d in ("scores", "generated/scores")])
    gen = gen_versions.merge(scores, on="version_id").assign(score=lambda d: sign * d.score)
    test = versions[versions.split == "test"].merge(scores, on="version_id").assign(score=lambda d: sign * d.score)
    test = test.merge(pd.read_parquet(path(cfg, "pvalues", f"{detector}.parquet")).rename(columns={"p": "p1"}),
                      on="version_id")
    human_calib = versions[(versions.split == "calib") & (versions.editor == "human")].merge(scores, on="version_id")
    human_calib["score"] *= sign
    null = gen[(gen.split == "calib") & gen.editor.isin([f"generated-{x}" for x in g["calibration"]])]
    gen_test = gen[gen.split == "test"].copy()
    rng, bins = np.random.default_rng([cfg["seed"], 2]), cfg["calibrate"]["length_bins"]   # a stream of its own
    gen_test["p1"] = pd.concat([pd.Series(conformal_pvalues(
        h.score.to_numpy(), h.n_words.to_numpy(), t.score.to_numpy(), t.n_words.to_numpy(), bins, rng), t.index)
        for s, t in gen_test.groupby("source") for h in [human_calib[human_calib.source == s]]])
    gen_test["p2"], test["p2"] = lower_pvalues(null, gen_test, bins, rng), lower_pvalues(null, test, bins, rng)
    rho1, rho2 = calib_correlation(cfg, detector, versions), generated_correlation(cfg, null, rng)
    print(f"{detector}: rho (human calib) {rho1}\n{detector}: rho' (generated calib) {rho2}")
    rows = []
    for s in sorted(test.source.unique()):
        rows += [r.assign(source=s) for r in
                 human_and_edited_docs(cfg, s, versions[versions.source == s], test[test.source == s], rho1[s], rho2[s])
                 + generated_docs(cfg, gen_test[gen_test.source == s], rho1[s], rho2[s])]
    return pd.concat(rows, ignore_index=True).assign(detector=detector, rho=lambda d: d.source.map(rho1),
                                                      rho2=lambda d: d.source.map(rho2))


def group_of(cfg, d):
    """Report group of each row: generators as calibration / held out, editors as training / held out."""
    held = {e["name"] for e in cfg["edit"]["editors"] if e["heldout"]}
    cal = set(cfg["generate"]["calibration"])
    return np.select([d.kind == "human", d.kind == "generated"],
                     ["human", np.where(d.label.isin(cal), "generated, calibration", "generated, held out")],
                     np.where(d.label.isin(held), "edited, held-out editor", "edited, training editors"))


def summary(cfg, d, test="hc"):
    """Rates per detector and group (and theta for edited documents) with Wilson intervals: stage 1, stage 2, both."""
    d = d.assign(group=group_of(cfg, d), both=d[f"reject_{test}"] & d[f"reject2_{test}"])
    cols = {f"reject_{test}": "stage 1", f"reject2_{test}": "H0'", "both": "both stages"} | {
        f"reject2_eps{e:g}": f"eps >= {e:g}" for e in cfg["stage2"]["eps0"]}
    level, out = cfg["phase"]["ci_level"], []
    keys = ["detector", "group", "label", "theta"]
    for k, g in d.groupby(keys, dropna=False):
        row = dict(zip(keys, k), docs=len(g), eps_storey=g.eps_storey.mean(), eps_lower=g.eps_lower.mean())
        for c, name in cols.items():
            lo, hi = wilson(g[c].sum(), len(g), level)
            row[name] = f"{g[c].mean():.3f} [{lo:.3f}, {hi:.3f}]"
        out.append(row)
    return pd.DataFrame(out)


def main():
    def add_args(p):
        p.add_argument("--summary", action="store_true", help="print rates from stage2/*.parquet and exit")
        p.add_argument("--test", default="hc", help="global test for the summary (hc, hc_plus, min_p, sum)")
        p.add_argument("--detectors", nargs="*", help="only these (default: every file in generated/scores)")
    cfg, args = cli(__doc__, add_args)
    files = sorted(Path(cfg["data_dir"], "stage2").glob("*.parquet"))
    if args.summary:
        with pd.option_context("display.width", 250, "display.max_rows", 2000, "display.max_columns", 30):
            print(summary(cfg, pd.concat(map(pd.read_parquet, files)), args.test).to_string(index=False))
        return
    versions = pd.read_parquet(path(cfg, "versions.parquet"), columns=V_COLS)
    gen_versions = pd.read_parquet(path(cfg, "generated", "versions.parquet"), columns=V_COLS)
    for f in sorted(Path(cfg["data_dir"], "generated", "scores").glob("*.parquet")):
        if args.detectors and f.stem not in args.detectors:
            continue
        if not path(cfg, "pvalues", f.name).exists():
            print(f"{f.stem}: no pvalues/{f.name}; run calibrate.py first")
            continue
        run(cfg, f.stem, versions, gen_versions).to_parquet(path(cfg, "stage2", f.name), index=False)


if __name__ == "__main__":
    main()
