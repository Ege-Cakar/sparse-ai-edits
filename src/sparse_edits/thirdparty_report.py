"""Tables of the third-party results (thirdparty_eval.py) and the metrics each dataset's paper reports, as CSVs in
thirdparty/report/ and thirdparty/report.md.

Document scores for the papers' threshold-free metrics: our pipeline's -log10 of the HC p-value at the document's n
under rho-correlated thresholds ("hc_p") and the raw HC statistic ("hc"), per sentence detector and null variant;
whole-document detectors (EditLens, MELD, GPTTrace) by their own scores. Sentence scores: -log10 of the conformal p-value.

- tests.csv: rejection rates of every global test (iid and rho thresholds) per dataset, part, label and group, with
  Wilson intervals; false alarms are the rates on held-out human documents (HELD_OUT parts).
- editlens.csv: EditLens's binary (human vs rest, AI vs rest) and ternary metrics with its own procedure
  (scripts/eval/{binary,ternary}_eval.py): thresholds maximizing F1 on min-max scaled val scores, applied to test,
  test_enron and test_llama (binary: scaled by val's range; ternary: by the evaluated split's own range, as its script).
- apt_eval.csv: detection rate per polish level and polisher, at APT-Eval's threshold (maximum accuracy on its 300
  originals vs 300 MixSet AI texts) and at our alpha; Pearson r of the score with sem_similarity, Levenshtein and Jaccard
  (EditLens Table 4).
- opai.csv: sentence accuracy and F1 on the AI class per domain and version (OpAI-Bench Table 4), for sentences flagged
  by Bonferroni at alpha within their document and by p < 0.01.
- pasted.csv: sentence AUROC and TPR at FPR < 1% (PASTED Table 1, in-distribution test and out-of-distribution).
- raid.csv: TPR at 5% FPR with per-domain thresholds on human documents (RAID's metric, reimplemented), and AUROC.
- aitdna.csv: document AUROC with AI = at least half the tokens AI-written (AITDNA Table 3, tau = 0.5), sentence
  accuracy / F1.
- peer_review.csv: share of polished reviews (H-AI) and of human reviews flagged at alpha (Saha et al. Table 2 counts
  flagged H-AI reviews as false positives; here they are the edited documents).
- own_rules.csv: each whole-document detector used as its authors intend: flagged when its score exceeds its shipped
  thresholds (thirdparty.detectors own_thresholds: MELD's meld_config.json offsets, GPTTrace's model-card logits) or,
  for EditLens, which ships none, its paper's threshold (maximum F1, human vs rest, on EditLens val), on every dataset.
- fraction.csv: Pearson r of the edited-fraction estimates with the true AI share or edit distance.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

from . import cli, path
from .thirdparty_eval import VARIANTS

HELD_OUT = {"editlens": ["test", "test_enron"], "apt_eval": ["original"], "opai": ["test"], "pasted": ["test", "ood"],
            "aitdna": ["test"], "peer_review": ["human_test"], "raid": ["train"]}
TESTS = ["reject_hc", "reject_hc_plus", "reject_min_p", "reject_sum"]


def load(cfg):
    c = cfg["thirdparty"]
    docs = pd.read_parquet(path(cfg, "thirdparty", "docs.parquet"),
                           columns=["doc_id", "dataset", "part", "domain", "label", "group", "meta"])
    docs = docs[docs.dataset != "corpus"].reset_index(drop=True)
    sents = pd.read_parquet(path(cfg, "thirdparty", "sentences.parquet"), columns=["doc_id", "sent_idx", "label"])
    names = ["ensemble"] + [d["name"] for d in c["detectors"] if d["kind"] != "run"]
    res = {(n, v): pd.read_parquet(path(cfg, "thirdparty", "results", f"{n}_{v}.parquet")) for n in names for v in VARIANTS}
    sp = {(n, v): sents.merge(pd.read_parquet(path(cfg, "thirdparty", "results", f"{n}_{v}_sentences.parquet")))
          for n in names for v in VARIANTS}
    whole = {d["name"]: pd.read_parquet(path(cfg, "thirdparty", "docscores", f"{d['name']}.parquet"))
             for d in c["detectors"] if d.get("documents")}
    whole_p = {(n, v): pd.read_parquet(path(cfg, "thirdparty", "results", f"{n}_{v}_documents.parquet"))
               for n in whole for v in VARIANTS}
    return docs, res, sp, whole, whole_p


def doc_scores(docs, res, whole):
    """{scorer: Series of document scores indexed by doc_id}: our -log10 p and raw HC per detector and variant, and the
    whole-document detectors' own scores."""
    out = {}
    for (n, v), r in res.items():
        r = r.set_index("doc_id")
        out[f"{n}/{v}/hc_p"], out[f"{n}/{v}/hc"] = -np.log10(r.hc_p_rho), r.hc
    out |= {n: w.set_index("doc_id").score for n, w in whole.items()}
    return out


def tests_table(docs, res, whole_p):
    rows = []
    for (n, v), r in list(res.items()) + [(k, w.rename(columns={"reject": "reject_whole"})) for k, w in whole_p.items()]:
        d = docs.merge(r, on="doc_id")
        cols = [f"{t}_{s}" for t in TESTS for s in ("iid", "rho")] if "reject_whole" not in r else ["reject_whole"]
        for keys, g in d.groupby(["dataset", "part", "label", "group"]):
            rows.append(dict(detector=n, variant=v, **dict(zip(["dataset", "part", "label", "group"], keys)),
                             docs=len(g), **{c: g[c].mean() for c in cols}))
    return pd.DataFrame(rows)


GRID = np.linspace(0, 1, 1000)


def val_threshold(s, y):
    """EditLens's threshold.py on min-max scaled scores: the first grid point of maximum F1, mapped back to raw units."""
    lo, hi = s.min(), s.max()
    f = [f1_score(y, (s - lo) / (hi - lo) >= t, zero_division=0) for t in GRID]
    return lo + GRID[int(np.argmax(f))] * (hi - lo)


def editlens_table(docs, scores):
    """EditLens's own procedure; labels 0 human, 1 generated, 2 edited as its ternary script."""
    d = docs[docs.dataset == "editlens"]
    lab = d.label.map({"human": 0, "generated": 1, "edited": 2}).to_numpy()
    val, rows = d.part.eq("val").to_numpy(), []
    splits = {"test": d.part.eq("test"), "test_enron": d.part.eq("test_enron"),
              "test_llama": d.part.eq("test_llama") | (d.part.eq("test") & d.label.eq("human"))}
    for name, sc in scores.items():
        s = sc.reindex(d.doc_id).to_numpy()
        ok = np.isfinite(s)
        lo, hi = s[val & ok].min(), s[val & ok].max()
        th = {m: val_threshold(s[val & ok], y[val & ok]) for m, y in (("human_vs_rest", lab > 0), ("ai_vs_rest", lab == 1))}
        for split, mask in splits.items():
            m = mask.to_numpy() & ok
            row = dict(scorer=name, split=split, docs=int(m.sum()))
            for mode, y in (("human_vs_rest", lab > 0), ("ai_vs_rest", lab == 1)):
                pred = s[m] >= th[mode]
                row |= {f"{mode}_acc": accuracy_score(y[m], pred), f"{mode}_f1": f1_score(y[m], pred)}
            # the ternary script rescales each split by its own range, then applies val's scaled thresholds
            t = (s[m] - s[m].min()) / (s[m].max() - s[m].min())
            th_scaled = {k: (v - lo) / (hi - lo) for k, v in th.items()}
            pred = np.where(t > th_scaled["ai_vs_rest"], 1, np.where(t < th_scaled["human_vs_rest"], 0, 2))
            row |= dict(ternary_acc=accuracy_score(lab[m], pred), ternary_macro_f1=f1_score(lab[m], pred, average="macro"),
                        **{f"ternary_f1_{k}": v for k, v in zip(["human", "generated", "edited"],
                                                                f1_score(lab[m], pred, average=None, labels=[0, 1, 2]))})
            rows.append(row)
    return pd.DataFrame(rows)


def apt_table(docs, scores, rejections):
    d = docs[docs.dataset == "apt_eval"].assign(m=lambda x: x.meta.map(json.loads))
    pol = d[d.part == "polished"]
    level = pol.m.map(lambda m: m["polishing_degree"] if isinstance(m["polishing_degree"], str)
                      else f"{m['polishing_percent']:g}%")
    calib = d[d.part.isin(["original", "threshold_ai"])]
    rows = []
    for name, sc in scores.items():
        c, y = sc.reindex(calib.doc_id).to_numpy(), calib.part.eq("threshold_ai").to_numpy()
        cand = np.unique(c[np.isfinite(c)])
        acc = [((c >= t) == y).mean() for t in cand]
        t = cand[int(np.argmax(acc))]                    # APT-Eval: the threshold of maximum accuracy
        s = sc.reindex(pol.doc_id).to_numpy()
        row = dict(scorer=name, threshold_accuracy=max(acc), fpr_originals=(c[~y] >= t).mean())
        row |= {f"rate_{lv}": (s[level.to_numpy() == lv] >= t).mean() for lv in sorted(level.unique())}
        row |= {f"rate_{p}": (s[pol.m.map(lambda m: m['polisher']).to_numpy() == p] >= t).mean()
                for p in sorted(pol.m.map(lambda m: m["polisher"]).unique())}
        ok = np.isfinite(s)
        for k in ("sem_similarity", "levenshtein_distance", "jaccard_distance"):
            row[f"r_{k}"] = np.corrcoef(s[ok], pol.m.map(lambda m: m[k]).to_numpy()[ok])[0, 1]
        rows.append(row)
    for name, rej in rejections.items():                # our alpha rule
        r = rej.reindex(pol.doc_id).to_numpy()
        rows.append(dict(scorer=name + " (alpha)", fpr_originals=rej.reindex(d[d.part == "original"].doc_id).mean(),
                         **{f"rate_{lv}": np.nanmean(r[level.to_numpy() == lv]) for lv in sorted(level.unique())}))
    return pd.DataFrame(rows)


def sentence_rules(sp, alpha):
    """Sentences with p, flagged by Bonferroni within their document and at p < 0.01."""
    s = sp.dropna(subset=["p"])
    n = s.groupby("doc_id").p.transform("size")
    return s.assign(bonferroni=s.p <= alpha / n, p01=s.p < 0.01)


def opai_table(docs, sp, alpha):
    d = docs[docs.dataset == "opai"].assign(version=lambda x: x.meta.map(lambda m: json.loads(m)["version"]))
    rows = []
    for (n, v), s in sp.items():
        s = sentence_rules(s, alpha).merge(d[["doc_id", "domain", "version"]])
        for (dom, ver), g in s.groupby(["domain", "version"]):
            for rule in ("bonferroni", "p01"):
                rows.append(dict(detector=n, variant=v, rule=rule, domain=dom, version=ver, sentences=len(g),
                                 acc=accuracy_score(g.label == 1, g[rule]), f1_ai=f1_score(g.label == 1, g[rule],
                                                                                           zero_division=0)))
    return pd.DataFrame(rows)


def own_rule_table(docs, whole, cfg):
    """Flag rates per dataset, part and label of each whole-document detector under its authors' decision rule."""
    rules = {d["name"]: d.get("own_thresholds", {}) for d in cfg["thirdparty"]["detectors"] if d.get("documents")}
    e = docs[(docs.dataset == "editlens") & docs.part.eq("val")]
    s = whole["editlens"].set_index("doc_id").score.reindex(e.doc_id).to_numpy()
    rules["editlens"] = {"paper_val_max_f1": val_threshold(s, e.label.ne("human").to_numpy())}
    rows = []
    for name, th in rules.items():
        d = docs.merge(whole[name], on="doc_id")
        for rule, t in th.items():
            g = d.assign(flag=d.score > t).groupby(["dataset", "part", "label"]).flag.agg(["size", "mean"])
            rows.append(g.reset_index().assign(detector=name, rule=rule, threshold=t))
    return pd.concat(rows, ignore_index=True).rename(columns={"size": "docs", "mean": "flag_rate"})


def tpr_at(y, s, fpr):
    f, t, _ = roc_curve(y, s)
    return t[f < fpr][-1]


def pasted_table(docs, sp):
    d = docs[docs.dataset == "pasted"]
    rows = []
    for part in ("test", "ood"):
        ids = d.doc_id[d.part == part]
        for (n, v), s in sp.items():
            g = s[s.doc_id.isin(ids)].dropna(subset=["p"])
            rows.append(dict(part=part, scorer=f"{n}/{v}", sentences=len(g), auroc=roc_auc_score(g.label, -np.log10(g.p)),
                             tpr_at_1pct_fpr=tpr_at(g.label, -np.log10(g.p), 0.01)))
    return pd.DataFrame(rows)


def raid_table(docs, scores):
    d = docs[(docs.dataset == "raid") & (docs.part == "train")]
    human = d.label.eq("human").to_numpy()
    rows = []
    for name, sc in scores.items():
        s, hit = sc.reindex(d.doc_id).to_numpy(), np.zeros(len(d), bool)
        for dom in d.domain.unique():
            m = d.domain.eq(dom).to_numpy() & np.isfinite(s)
            h = np.unique(s[m & human])
            t = h[np.argmax([(s[m & human] >= x).mean() <= 0.05 for x in h])]   # smallest threshold with FPR <= 5%
            hit[m] = s[m] >= t
        ok = np.isfinite(s)
        rows.append(dict(scorer=name, ai_docs=int((~human & ok).sum()), tpr_at_5pct_fpr=hit[~human & ok].mean(),
                         fpr=hit[human & ok].mean(), auroc=roc_auc_score(~human[ok], s[ok])))
    return pd.DataFrame(rows)


def aitdna_table(docs, scores, sp, alpha):
    d = docs[docs.dataset == "aitdna"].assign(share=lambda x: x.meta.map(lambda m: json.loads(m)["ai_token_share"]))
    rows = []
    for name, sc in scores.items():
        s = sc.reindex(d.doc_id).to_numpy()
        ok = np.isfinite(s)
        rows.append(dict(scorer=name, auroc_tau_0_5=roc_auc_score(d.share.to_numpy()[ok] >= 0.5, s[ok]),
                         auroc_any_ai=roc_auc_score(d.label.ne("human").to_numpy()[ok], s[ok])))
    for (n, v), s in sp.items():
        g = sentence_rules(s[s.doc_id.isin(d.doc_id)], alpha)
        for rule in ("bonferroni", "p01"):
            rows.append(dict(scorer=f"{n}/{v} sentences, {rule}", sentence_acc=accuracy_score(g.label == 1, g[rule]),
                             sentence_f1_ai=f1_score(g.label == 1, g[rule], zero_division=0)))
    return pd.DataFrame(rows)


def peer_table(docs, rejections):
    d = docs[docs.dataset == "peer_review"]
    rows = []
    for name, rej in rejections.items():
        r = d.assign(reject=rej.reindex(d.doc_id).to_numpy()).dropna(subset=["reject"])
        rows.append(dict(scorer=name, human_test=r[r.part == "human_test"].reject.mean(),
                         **{p: r[r.part == p].reject.mean() for p in ("polished_easy", "polished_hard")}))
    return pd.DataFrame(rows)


def fraction_table(docs, res, sents_label):
    """Edited-fraction estimates against the truth: AI sentence share (labelled datasets) or the papers' distances."""
    m = docs.assign(m=docs.meta.map(json.loads))
    truth = {"apt_eval": lambda x: x.get("levenshtein_distance"), "editlens": lambda x: x.get("cosine_score"),
             "opai": lambda x: x.get("ai_sentence_ratio"), "aitdna": lambda x: x.get("ai_token_share")}
    share = sents_label[sents_label.label >= 0].groupby("doc_id").label.mean()
    rows = []
    for (n, v), r in res.items():
        d = m.merge(r, on="doc_id")
        for ds, g in d[d.label.ne("human")].groupby("dataset"):
            t = g.m.map(truth[ds]) if ds in truth else g.doc_id.map(share)
            ok = t.notna().to_numpy()
            if ok.sum() > 2:
                rows.append(dict(detector=n, variant=v, dataset=ds, docs=int(ok.sum()),
                                 **{f"r_{e}": np.corrcoef(g[e].to_numpy()[ok], t.to_numpy(float)[ok])[0, 1]
                                    for e in ("eps_storey_rho", "eps_lower_rho")}))
    return pd.DataFrame(rows)


def markdown(t):
    cells = t.map(lambda x: f"{x:.3f}" if isinstance(x, float) else str(x))
    return "\n".join(["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
                     + ["| " + " | ".join(r) + " |" for r in cells.to_numpy()])


def main():
    cfg, _ = cli(__doc__)
    alpha = cfg["phase"]["alpha"]
    docs, res, sp, whole, whole_p = load(cfg)
    scores = doc_scores(docs, res, whole)
    rejections = {f"{n}/{v}": r.set_index("doc_id").reject_hc_rho.astype(float) for (n, v), r in res.items()}
    rejections |= {f"{n}/{v} (whole document)": w.set_index("doc_id").reject.astype(float)
                   for (n, v), w in whole_p.items()}
    sents = pd.read_parquet(path(cfg, "thirdparty", "sentences.parquet"), columns=["doc_id", "label"])
    tables = dict(tests=tests_table(docs, res, whole_p), editlens=editlens_table(docs, scores),
                  apt_eval=apt_table(docs, scores, rejections), opai=opai_table(docs, sp, alpha),
                  pasted=pasted_table(docs, sp), raid=raid_table(docs, scores),
                  aitdna=aitdna_table(docs, scores, sp, alpha), peer_review=peer_table(docs, rejections),
                  own_rules=own_rule_table(docs, whole, cfg), fraction=fraction_table(docs, res, sents))
    with open(path(cfg, "thirdparty", "report.md"), "w") as f:
        for k, t in tables.items():
            t.to_csv(path(cfg, "thirdparty", "report", f"{k}.csv"), index=False)
            f.write(f"## {k}\n\n{markdown(t)}\n\n")


if __name__ == "__main__":
    main()
