"""Third-party AI-text datasets through our pipeline (config thirdparty; decided with Ege 2026-11-06/07).

--prepare  thirdparty/docs.parquet, one row per document: doc_id, dataset, part, domain, label (human, edited,
           generated, unknown), base (the human document it derives from), group (the dataset's condition: editor,
           version, polish level), meta (JSON of the dataset's own fields), text, and sentences / sentence_labels (the
           dataset's own split and labels where it has one: OpAI-Bench, PASTED, AITDNA, our corpus). The nulls:
           null_dataset and null_domain mark the human documents of the one-null-per-dataset and per-domain nulls;
           scope_domain names a document's per-domain null (domains with fewer than min_null_docs null documents are
           pooled within their dataset); a document in a cross-fit group (xgroup) with fold f is tested only against
           null documents of that group in the other fold (fold -1: no cross-fitting). Corpus calib documents
           (dataset "corpus") are the null of the corpus variant.
--segment  thirdparty/sentences.parquet (doc_id, sent_idx, para_idx, text, n_words, label, segmenter): the datasets' own
           sentences ("own"; label 1 = AI, 0 = human), and the rest split by corpus.segmenter (segment.py, one
           description per dataset/domain) with SaT as its fallback (label -1, unknown).
--score D  for detector D of thirdparty.detectors: thirdparty/scores/D.parquet (doc_id, sent_idx, score), every sentence
           alone; and, with documents: true, thirdparty/docscores/D.parquet
           (doc_id, score). Scores are cached by text hash in thirdparty/score_parts/D/, so a preempted or repeated run
           scores only texts it has not scored. Higher scores are more AI-like for every detector.
"""
import glob
import importlib.util
import json
import urllib.request
import zipfile
import zlib

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from . import cli, path
from .sources import text_key
from .text import paragraphs, words

LABELS = {"human_written": "human", "ai_edited": "edited", "ai_generated": "generated"}
PASTED_FILES = {"dev": ("classification.zip", "classification/dev.json"),
                "test": ("classification.zip", "classification/test.json"), "ood": ("eval-OOD.zip", "eval-OOD/test.json")}
RAID_CELL = ["domain", "model", "decoding", "repetition_penalty"]
APT_FILES = [f"polished_texts_{level}_{polisher}.csv" for level in ("1", "5", "10", "20", "35", "50", "75", "extreme_minor",
             "minor", "slight_major", "major") for polisher in ("deepseek", "gpt", "llama", "llama2", "llama70b")]


def snapshot(c, kind="dataset"):
    from huggingface_hub import snapshot_download
    return snapshot_download(c["repo"] if kind == "dataset" else c["model"], repo_type=kind, revision=c["revision"],
                             allow_patterns=c.get("files"))   # the files downloaded, so it also works offline


def docs(d, **cols):
    """docs.parquet rows for the dataset frame d (text from d.text unless given); columns not given get defaults."""
    out = pd.DataFrame(dict(group="", xgroup="", null_dataset=False, null_domain=False, meta="{}", sentences=None,
                            sentence_labels=None, text=d["text"].to_numpy() if "text" in d else None), index=d.index)
    for k, v in cols.items():
        out[k] = v.to_numpy() if isinstance(v, pd.Series) else v
    return out.reset_index(drop=True)


def meta(d, cols):
    return [json.dumps(r, default=str) for r in d[cols].to_dict("records")]


def editlens(c, cfg):
    p = snapshot(c)
    read = lambda part: pd.read_parquet(glob.glob(f"{p}/data/{part}-*.parquet")[0])
    test_humans, out = set(read("test").query("text_type == 'human_written'").text_id), []
    for part in ("val", "test", "test_enron", "test_llama"):
        d = read(part)
        human = d.text_type.eq("human_written")
        if part == "test_llama":                        # its human rows are test's
            assert set(d[human].text_id) == test_humans
            d, human = d[~human], human[~human]
        cross = {"val": "editlens/val", "test_enron": "editlens/enron"}.get(part, "")
        out.append(docs(d, part=part, domain=d.source, label=d.text_type.map(LABELS), base=d.source_id, group=d.model,
                        null_dataset=human & (part == "val"), null_domain=human & bool(cross), xgroup=cross,
                        meta=meta(d, ["text_id", "cosine_score", "soft_ngrams_score"])))
    return pd.concat(out)


def fetch(url, f):
    if not f.exists():
        urllib.request.urlretrieve(url, f)
    return f


def apt_eval(c, cfg):
    p = snapshot(c)
    o, m = pd.read_csv(f"{p}/original.csv"), pd.read_csv(f"{p}/merged_apt_eval_dataset.csv")
    raw = path(cfg, cfg["thirdparty"]["raw_dir"], "apt_eval", "x").parent
    t = pd.read_csv(fetch(f"{c['github']}/merged_mgt_hwt_data.csv", raw / "merged_mgt_hwt_data.csv"))
    assert set(t[t.model == "human"].generation) == set(o.generation)   # its humans are the 300 originals
    ai = t[t.model != "human"].reset_index(drop=True)
    # the hub file renumbers its rows; the GitHub per-run files keep each polished text's original.csv id
    ids = pd.concat([pd.read_csv(fetch(f"{c['github']}/polished/{n}", raw / n)) for n in APT_FILES])
    source = m.generation.map(ids.groupby("generation").id.agg(set))
    assert source.notna().all() and source.map(len).eq(1).all()
    m["base"] = source.map(lambda x: str(next(iter(x))))
    level = m.polishing_degree.fillna(m.polishing_percent.map(lambda x: f"{x:g}%"))
    return pd.concat([
        docs(o, text=o.generation, part="original", domain=o.domain, label="human", base=o.id.astype(str),
             null_dataset=True, null_domain=True, xgroup="apt_eval"),
        docs(m, text=m.generation, part="polished", domain=m.domain, label="edited", base=m.base,
             group=m.polisher + "/" + level, xgroup="apt_eval",
             meta=meta(m, ["id", "polish_type", "polishing_degree", "polishing_percent", "polisher", "sem_similarity",
                           "levenshtein_distance", "jaccard_distance"])),
        docs(ai, text=ai.generation, part="threshold_ai", domain=ai.domain, label="generated",
             base="mixset/" + ai.id.astype(str), group=ai.model, xgroup="apt_eval")])


def opai(c, cfg):
    p, out = snapshot(c), []
    for part in ("dev", "test"):
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{p}/default/{part}/*.csv"))], ignore_index=True)
        v0 = d.version.eq("v0")                          # each generator's file repeats the human seed
        d = pd.concat([d[v0].drop_duplicates(["domain", "id"])] + ([d[~v0]] if part == "test" else []))
        human = d.version.eq("v0")
        out.append(docs(d, part=part, domain=d.domain, label=np.where(human, "human", "edited"),
                        base=d.domain + "/" + d.id.astype(str), group=np.where(human, "", d.generator + "/" + d.version),
                        null_dataset=human & (part == "dev"), null_domain=human & (part == "dev"),
                        sentences=d.sentences.map(json.loads), sentence_labels=d.sentence_labels.map(json.loads),
                        meta=meta(d, ["record_id", "generator", "version", "edit_operation", "ai_sentence_ratio"])))
    return pd.concat(out)


def pasted(c, cfg):
    p, out = snapshot(c), []
    for part, (z, name) in PASTED_FILES.items():
        d = pd.read_json(zipfile.ZipFile(f"{p}/{z}").open(name), lines=True)
        human_source = d.src.str.endswith("_human")
        if part == "dev":                                # only its unparaphrased human texts, the null
            d, human_source = d[human_source & ~d.is_para], human_source[human_source & ~d.is_para]
        human, sents = human_source & ~d.is_para, d.text.str.split(" </s> ")
        domain = d.src.str.replace(r"_(human|machine_.*|gpt4)$", "", regex=True)
        out.append(docs(d, text=sents.str.join(" "), part=part, domain=domain,
                        label=np.select([human, human_source], ["human", "edited"], "generated"),
                        base=domain + "/" + d["index"].astype(str), group=d.src,
                        null_dataset=human & (part == "dev"), null_domain=human & (part != "test"),
                        xgroup="pasted/ood" if part == "ood" else "", sentences=sents, sentence_labels=d.label,
                        meta=meta(d, ["index", "src", "is_para"])))
    return pd.concat(out)


def aitdna(c, cfg):
    p = snapshot(c)
    doc, sen, tok = (pd.read_parquet(f"{p}/{k}/test-00000-of-00001.parquet") for k in ("document", "sentence", "token"))
    md = pd.json_normalize(doc.metadata)
    md["ai_token_share"] = tok.data.map(lambda x: sum(t["author"] != "User" for t in x) / len(x))   # its Table 3's label
    return docs(md, text=doc.data.map(lambda x: x[0]["text"]), part="test", domain="aitdna",
                label=np.where(md.human_only, "human", "edited"), base=md.author,
                group=md.setting + "/" + md.task + "/" + md.model.fillna("none"), null_dataset=md.human_only,
                null_domain=md.human_only, xgroup="aitdna", sentences=sen.data.map(lambda x: [s["text"] for s in x]),
                sentence_labels=sen.data.map(lambda x: [int(s["author"] != "User") for s in x]),
                meta=meta(md, ["author", "model", "setting", "task", "temperature", "ai_token_share"]))


def peer_review(c, cfg):
    p = snapshot(c)
    h = pd.read_json(f"{p}/human-reviews.jsonl", lines=True).query("split in ['dev', 'test']")
    e = pd.concat([pd.read_json(f"{p}/{s}-subset-consolidated.jsonl", lines=True).assign(subset=s) for s in ("easy", "hard")])
    e = e.query("level == 'H-AI' and split == 'test'")
    key = lambda d: d.conference + "/" + d.paper_number.astype(str) + "/" + d.reviewer_id.astype(str)
    cols = ["conference", "paper_number", "reviewer_id", "generating_model", "prompt_id", "split"]
    return pd.concat([
        docs(h, text=h.review_text, part="human_" + h.split, domain="peer_review", label="human", base=key(h),
             null_dataset=h.split.eq("dev"), null_domain=h.split.eq("dev"), meta=meta(h, cols)),
        docs(e, text=e.review_text, part="polished_" + e.subset, domain="peer_review", label="edited", base=key(e),
             group=e.subset + "/" + e.generating_model + "/" + e.prompt_id, meta=meta(e, cols + ["subset"]))])


def raid(c, cfg):
    raw = path(cfg, cfg["thirdparty"]["raw_dir"], "raid", "x").parent
    t = pd.read_csv(raw / "train_none.csv", usecols=["id", "source_id"] + RAID_CELL + ["generation"])
    human = t.model.eq("human")
    ai = t[~human].groupby(RAID_CELL, group_keys=False).sample(c["per_cell"], random_state=cfg["seed"])
    h, test = t[human], pd.read_csv(raw / "test_none.csv")
    return pd.concat([
        docs(h, text=h.generation, part="train", domain=h.domain, label="human", base=h.id, null_dataset=True,
             null_domain=True, xgroup="raid", meta=meta(h, ["id"])),
        docs(ai, text=ai.generation, part="train", domain=ai.domain, label="generated", base=ai.source_id,
             group=ai.model + "/" + ai.decoding + "/" + ai.repetition_penalty, xgroup="raid", meta=meta(ai, ["id"] + RAID_CELL)),
        docs(test, text=test.generation, part="test", domain="unknown", label="unknown", base=test.id,
             meta=meta(test, ["id"]))])


def corpus(cfg):
    """Our corpus's calib documents, with their sentences, the null of the corpus variant."""
    from .pretrain import sentence_table   # imports transformers, which hangs on login nodes
    s = sentence_table(pd.read_parquet(path(cfg, "sentences.parquet"))).query("split == 'calib'")
    g = s.assign(t=s.sep + s.text).groupby("doc_id", sort=True)
    d = pd.DataFrame(dict(text=g.t.sum(), domain=g.source.first(), sentences=g.text.agg(list)))
    return docs(d, part="calib", domain=d.domain, label="human", base=d.index.astype(str), null_dataset=True,
                null_domain=True, sentences=d.sentences, sentence_labels=d.sentences.map(lambda x: [0] * len(x)))


LOADERS = {"editlens": editlens, "apt_eval": apt_eval, "opai": opai, "pasted": pasted, "aitdna": aitdna,
           "peer_review": peer_review, "raid": raid}


def assign_nulls(df, seed, min_null_docs):
    """fold (by a hash of seed, xgroup and base, so a document and every version of it share a fold; -1 outside a
    cross-fit group) and scope_domain (the dataset's domain, or its small_domains pool)."""
    df["fold"] = [zlib.crc32(repr((seed, x, b)).encode()) % 2 if x else -1 for x, b in zip(df.xgroup, df.base)]
    n_null = df[df.null_domain].groupby(["dataset", "domain"]).size()
    big = set(n_null[n_null >= min_null_docs].index)
    df["scope_domain"] = [f"{s}/{d}" if (s, d) in big else f"{s}/small_domains" for s, d in zip(df.dataset, df.domain)]
    return df


def prepare(cfg):
    c = cfg["thirdparty"]
    df = pd.concat([LOADERS[name](d, cfg).assign(dataset=name) for name, d in c["datasets"].items()]
                   + [corpus(cfg).assign(dataset="corpus")], ignore_index=True)
    df.insert(0, "doc_id", range(len(df)))
    df = assign_nulls(df.assign(base=df.base.astype(str)), cfg["seed"], c["min_null_docs"])
    df.to_parquet(path(cfg, "thirdparty", "docs.parquet"), index=False)
    print(df.groupby(["dataset", "part", "label"]).size().to_string())
    print(df[df.null_domain].groupby("scope_domain").size().to_string())


def segment(cfg):
    """Rewrites sentences.parquet after the datasets' own sentences and after each segmented dataset/domain, so scoring
    can start on finished datasets."""
    from .segment import Segmenter
    df, seg = pd.read_parquet(path(cfg, "thirdparty", "docs.parquet")), cfg["corpus"]["segmenter"]
    own = df[df.sentences.notna()]
    rows = [(d, i, 0, s, int(lab), "own") for d, ss, ls in zip(own.doc_id, own.sentences, own.sentence_labels)
            for i, (s, lab) in enumerate(zip(ss, ls))]

    def write():
        out = pd.DataFrame(rows, columns=["doc_id", "sent_idx", "para_idx", "text", "label", "segmenter"])
        out.insert(4, "n_words", out.text.map(lambda t: len(words(t))))
        f = path(cfg, "thirdparty", "sentences.parquet")
        out.sort_values(["doc_id", "sent_idx"]).to_parquet(f.with_suffix(".tmp"), index=False)
        f.with_suffix(".tmp").replace(f)                 # atomic, so a scoring job never reads half a file
        print(out.merge(df[["doc_id", "dataset"]]).groupby("dataset").segmenter.value_counts().unstack(fill_value=0),
              flush=True)
    write()
    rest, split = df[df.sentences.isna()], Segmenter(seg, path(cfg, "thirdparty", "segment_cache.jsonl"))
    for key, g in rest.groupby(rest.dataset + "/" + rest.domain):
        for start in tqdm(range(0, len(g), seg["batch_docs"]), desc=key):
            b = g.iloc[start:start + seg["batch_docs"]]
            for d, sents in zip(b.doc_id, split(key, "\n", [paragraphs(t) for t in b.text])):
                rows += [(d, i, p, s, -1, how) for i, (p, s, how) in enumerate(sents)]
        write()


@torch.no_grad()
def cached(score, texts, bs, parts, chunk=32768):
    """score(batch) for every text, cached by text hash in parts/*.parquet: only texts never scored are scored, in
    length-sorted chunks saved as each finishes. score returns a tensor, one value per text."""
    keys = [text_key(t) for t in texts]
    have = {}
    for f in sorted(parts.glob("*.parquet")):
        have.update(zip(*pd.read_parquet(f).to_dict("list").values()))
    todo = sorted({k: t for k, t in zip(keys, texts) if k not in have}.items(), key=lambda kt: len(kt[1]))
    for start in tqdm(range(0, len(todo), chunk)):
        part, vals = todo[start:start + chunk], []
        for i in range(0, len(part), bs):
            vals += score([t for _, t in part[i:i + bs]]).float().cpu().tolist()
        pd.DataFrame({"key": [k for k, _ in part], "score": vals}).to_parquet(parts / "tmp.parquet")
        (parts / "tmp.parquet").replace(parts / f"{part[0][0]}.parquet")
        have.update(zip([k for k, _ in part], vals))
    return [have[k] for k in keys]


def run_scorer(cfg, d):
    """One of our detect.py runs (its model.pt, or model_epoch<E>.pt with epoch E) as score(texts)."""
    from .detect import Detector, cuda, expand, load_trainable, sentence_batch, tokenizer
    run = expand(cfg)[d["run"]]
    tok, model = tokenizer(run), Detector(run).cuda().eval()
    load_trainable(model, torch.load(path(cfg, "detector", d["run"], f"model{'_epoch%d' % d['epoch'] if d.get('epoch') else ''}.pt")))

    def score(texts):
        with torch.autocast("cuda", torch.bfloat16):
            return model(**cuda(sentence_batch(tok, run, texts)))[:, 0].float()
    return score


def meld(d):
    """MELD scored by its own meld.py Scorer (canonical text, fp32, windows of 2,046 tokens, top-quantile pooling)."""
    p = snapshot(d, "model")
    spec = importlib.util.spec_from_file_location("meld", f"{p}/meld.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scorer = module.Scorer(p, device="cuda")

    def one(t):
        try:
            return scorer.score(t)["score"]
        except ValueError:                               # no token left after canonicalization
            return np.nan
    return lambda texts: torch.tensor([one(t) for t in texts])


def score(cfg, name):
    from .external import LOADERS as EXTERNAL
    d = next(x for x in cfg["thirdparty"]["detectors"] if x["name"] == name)
    sents = pd.read_parquet(path(cfg, "thirdparty", "sentences.parquet"), columns=["doc_id", "sent_idx", "text"])
    parts = lambda what: path(cfg, "thirdparty", "score_parts", name, what, "x").parent
    fn = run_scorer(cfg, d) if d["kind"] == "run" else meld(d) if d["kind"] == "meld" else EXTERNAL[d["kind"]](d)
    sents["score"] = cached(fn, sents.text.tolist(), d["batch_size"], parts("sentences"))
    if d.get("documents"):
        df = pd.read_parquet(path(cfg, "thirdparty", "docs.parquet"), columns=["doc_id", "text"])
        df["score"] = cached(fn, df.text.tolist(), d["doc_batch_size"], parts("documents"))
        df.drop(columns="text").to_parquet(path(cfg, "thirdparty", "docscores", f"{name}.parquet"), index=False)
    sents.drop(columns="text").to_parquet(path(cfg, "thirdparty", "scores", f"{name}.parquet"), index=False)
    print(f"{name}: {len(sents)} sentences, {sents.score.isna().sum()} without a score", flush=True)


def main():
    def args(p):
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--prepare", action="store_true")
        g.add_argument("--segment", action="store_true")
        g.add_argument("--score", metavar="DETECTOR", help="a thirdparty.detectors name")
    cfg, args = cli(__doc__, args)
    if args.prepare:
        prepare(cfg)
    elif args.segment:
        segment(cfg)
    else:
        score(cfg, args.score)


if __name__ == "__main__":
    main()
