"""Style comparison study: how similar are different editors' edits of the same sentence at the same strength?

  sample   after `python -m sparse_edits.corpus --config studies/style_comparison.yaml`, mark study.per_source
           sentences per source (study.min_words-study.max_words words) as split "study", the rest "pool"
  agent-tasks    write agent_tasks/rep<rep>_r<strength>.jsonl: {"id", "prompt"} per study sentence, with the exact
                 editor prompt from sparse_edits.edit, for editors run as Claude Code subagents
  agent-collect  convert agent_outputs/<editor>__rep<rep>_r<strength>.jsonl ({"id", "text"} lines) into
                 edits/<editor>.jsonl, applying the pipeline's cleaning and rejection rules (one attempt, no resampling)
  analyze  after `python -m sparse_edits.edit ... --consolidate`, compare accepted edits pairwise:
             site overlap   Jaccard overlap of edit sites on the original (substituted/deleted word i, insertion before i)
             distance       word-level Levenshtein distance between the two edited sentences / longer length
             vocabulary     Jensen-Shannon divergence (bits) of introduced-word counts, rep 0 vs rep 0 across
                            editors and rep 0 vs rep 1 within an editor
           Within an editor, pairs are its two samples (rep 0 vs rep 1): its self-similarity. Across editors, all
           rep pairs. Only (sentence, strength) units where every editor has 2 accepted edits are used, so all
           comparisons share one set of units and sample sizes. Means are per sentence, then over sentences; 95%
           intervals bootstrap over sentences.
"""
import json
from collections import Counter
from itertools import combinations_with_replacement, product
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from scipy.spatial.distance import jensenshannon

from sparse_edits import cli, path
from sparse_edits.edit import clean, record, rejection, request
from sparse_edits.text import words


def sample(cfg):
    s, p = cfg["study"], path(cfg, "sentences.parquet")
    df = pd.read_parquet(p)
    pool = df[df.n_words.between(s["min_words"], s["max_words"])]
    chosen = pool.groupby("source").sample(n=s["per_source"], random_state=cfg["seed"]).sentence_id
    df["split"] = np.where(df.sentence_id.isin(chosen), "study", "pool")
    df.to_parquet(p, index=False)
    print(df[df.split == "study"].groupby("source").n_words.describe()[["count", "mean", "min", "max"]])


def agent_tasks(cfg):
    study = pd.read_parquet(path(cfg, "sentences.parquet")).query("split == 'study'")
    for rep, r in product(range(cfg["edit"]["reps"]), cfg["edit"]["target_strengths"]):
        with open(path(cfg, "agent_tasks", f"rep{rep}_r{r}.jsonl"), "w") as f:
            for row in study.itertuples():
                prompt = request({"model": None}, row, r)["messages"][1]["content"]
                f.write(json.dumps({"id": f"{row.sentence_id}:{r}:{rep}", "prompt": prompt}) + "\n")


def agent_collect(cfg):
    sents = pd.read_parquet(path(cfg, "sentences.parquet")).set_index("sentence_id", drop=False)
    files = sorted(Path(cfg["data_dir"], "agent_outputs").glob("*.jsonl"))
    for editor in sorted({f.stem.split("__")[0] for f in files}):
        rows, bad = [], 0
        for f in [f for f in files if f.stem.split("__")[0] == editor]:
            for line in f.read_text().splitlines():
                try:
                    d = json.loads(line)
                    sid, r, rep = d["id"].split(":")
                except (json.JSONDecodeError, KeyError, ValueError):
                    bad += 1
                    continue
                row = sents.loc[int(sid)]
                out = clean(d.get("text"), row.text)
                rows.append(record({"name": editor}, row, float(r), int(rep), out, 1, rejection(row.text, out)))
        path(cfg, "edits", f"{editor}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
        print(f"{editor}: {len(rows)} edits, {bad} unparseable lines, "
              f"{sum(x['rejected'] is not None for x in rows)} rejected")


def edit_sites(original, edited):
    return {("ins" if op.tag == "insert" else "sub", op.src_pos)
            for op in Levenshtein.editops(words(original), words(edited))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def distance(a, b):
    wa, wb = words(a), words(b)
    return Levenshtein.distance(wa, wb) / max(len(wa), len(wb), 1)


def pair_table(edits, originals):
    """One row per (sentence, strength, editor pair) with mean site overlap and distance over the rep pairs."""
    rows, arms = [], sorted(edits.editor.unique())
    for (sid, r), unit in edits.groupby(["sentence_id", "target_r"]):
        by_arm = {a: g.text.tolist() for a, g in unit.groupby("editor")}
        sites = {a: [edit_sites(originals[sid], t) for t in ts] for a, ts in by_arm.items()}
        for a, b in combinations_with_replacement(arms, 2):
            if a not in by_arm or b not in by_arm:
                continue
            idx = [(0, 1)] if a == b else [(i, j) for i in range(len(by_arm[a])) for j in range(len(by_arm[b]))]
            idx = [(i, j) for i, j in idx if i < len(by_arm[a]) and j < len(by_arm[b])]
            if idx:
                rows.append(dict(sentence_id=sid, target_r=r, a=a, b=b,
                                 site_overlap=np.mean([jaccard(sites[a][i], sites[b][j]) for i, j in idx]),
                                 distance=np.mean([distance(by_arm[a][i], by_arm[b][j]) for i, j in idx])))
    return pd.DataFrame(rows)


def bootstrap(per_sentence, n, rng):
    """Mean and 95% interval of per-sentence values, resampling sentences."""
    v = per_sentence.to_numpy()
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return f"{v.mean():.3f} [{np.quantile(means, 0.025):.3f}, {np.quantile(means, 0.975):.3f}]"


def vocabulary(edits):
    """Counter of introduced words per (editor, rep), from the diff mask."""
    return {(a, rep): Counter(w for t, m in zip(g.text, g.ai_word_mask) for w, x in zip(words(t), m) if x)
            for (a, rep), g in edits.groupby(["editor", "rep"])}


def jsd(c1, c2):
    keys = sorted(c1.keys() | c2.keys())
    return jensenshannon([c1[k] for k in keys], [c2[k] for k in keys], base=2)


def analyze(cfg, editors=None, reference=None):
    s, rng = {**cfg["study"], **({"reference": reference} if reference else {})}, np.random.default_rng(cfg["seed"])
    sents = pd.read_parquet(path(cfg, "sentences.parquet")).set_index("sentence_id")
    edits = pd.read_parquet(path(cfg, "edits.parquet"))
    edits = edits[edits.editor.isin(editors)] if editors else edits
    accepted = edits[edits.rejected.isna()]
    print("Accepted edits and median measured strength by editor (columns: target strength):")
    print(accepted.pivot_table(index="editor", columns="target_r", values="actual_r", aggfunc=["count", "median"]).round(2))
    print("\nRejected share:", edits.groupby("editor").rejected.apply(lambda x: x.notna().mean()).round(3).to_dict())

    counts = accepted.groupby(["sentence_id", "target_r", "editor"]).size().unstack(fill_value=0)
    units = counts[(counts >= 2).all(axis=1)].index
    accepted = accepted.set_index(["sentence_id", "target_r"]).loc[units].reset_index()
    print(f"\nUsing {len(units)} units from {units.get_level_values(0).nunique()} sentences with 2 accepted edits "
          f"from every editor.")
    pairs = pair_table(accepted, sents.text)
    pairs.to_csv(path(cfg, "pairs.csv"), index=False)
    for metric in ["site_overlap", "distance"]:
        per_sentence = pairs.groupby(["a", "b", "sentence_id"])[metric].mean()
        table = per_sentence.groupby(["a", "b"]).apply(lambda v: bootstrap(v, s["bootstrap"], rng)).unstack()
        print(f"\n{metric} (mean over sentences [95% CI]; diagonal = self-similarity):\n{table.to_string()}")

    ref = s["reference"]
    print(f"\nDifference from {ref}'s self-similarity (editor-vs-{ref} minus {ref}-vs-{ref}; paired over sentences):")
    for metric in ["site_overlap", "distance"]:
        per = pairs.groupby(["a", "b", "sentence_id"])[metric].mean()
        self_ref = per.loc[(ref, ref)]
        for arm in sorted((set(pairs.a) | set(pairs.b)) - {ref}):
            cross = per.loc[(min(arm, ref), max(arm, ref))]
            diff = (cross - self_ref).dropna()
            print(f"  {metric:12s} {arm:18s} {bootstrap(diff, s['bootstrap'], rng)}  (n={len(diff)} sentences)")

    vocab = vocabulary(accepted)
    arms = sorted(accepted.editor.unique())
    print("\nVocabulary JSD in bits (off-diagonal: rep 0 vs rep 0; diagonal: rep 0 vs rep 1 of the same editor):")
    print(pd.DataFrame([[jsd(vocab[(a, 0)], vocab[(b, 1 if a == b else 0)]) for b in arms] for a in arms],
                       index=arms, columns=arms).round(3).to_string())
    print("\nMost common introduced words per editor (both reps):")
    for a in arms:
        print(f"  {a}: {(vocab[(a, 0)] + vocab[(a, 1)]).most_common(12)}")


if __name__ == "__main__":
    def add_args(p):
        p.add_argument("step", choices=["sample", "agent-tasks", "agent-collect", "analyze"])
        p.add_argument("--editors", nargs="*", help="analyze: only these editors (default: all)")
        p.add_argument("--reference", help="analyze: reference editor (default: study.reference)")
    cfg, args = cli(__doc__, add_args)
    if args.step == "analyze":
        analyze(cfg, args.editors, args.reference)
    else:
        {"sample": sample, "agent-tasks": agent_tasks, "agent-collect": agent_collect}[args.step](cfg)
