"""Fresh human documents for the pretraining fresh-data control.

For each source in corpus.sources: the documents that follow the corpus's in the same seeded stream (documents whose
key is in sentences.parquet are skipped before segmentation), through corpus.py's repairs, segmenter, length filter
and duplicate filters, the latter also against every corpus document's sentences; kept until the source has
pretrain.fresh_multiple times the words of its train and val documents, or the stream runs out (then it says so).
The segmenter's reply cache starts as a copy of the corpus's (segment_cache.jsonl), so a document the corpus run
already segmented gets the same split. Writes fresh/sentences.parquet with corpus.py's columns, split "fresh", and
doc_id and sentence_id continuing after the corpus's.
"""
import shutil

import pandas as pd
from tqdm import tqdm

from . import cli, path
from .corpus import segmented
from .segment import Segmenter
from .sources import SOURCES
from .text import words


def take(docs, used_keys, seen, target, min_s, max_s):
    """(key, sents) from segmented docs, skipping used keys, lengths outside [min_s, max_s] and sentence signatures in
    seen (which grows), until the words reach target. Returns (kept, words)."""
    kept, n = [], 0
    for key, sents in docs:
        sig = tuple((p, s) for p, s, _ in sents)
        if str(key) in used_keys or not min_s <= len(sents) <= max_s or sig in seen:
            continue
        seen.add(sig)
        kept.append((key, sents))
        n += sum(len(words(s)) for _, s, _ in sents)
        if n >= target:
            break
    return kept, n


def main():
    cfg, _ = cli(__doc__)
    corpus = pd.read_parquet(path(cfg, "sentences.parquet"))
    cache = path(cfg, "fresh", "segment_cache.jsonl")
    if not cache.exists():
        shutil.copy(path(cfg, "segment_cache.jsonl"), cache)
    seg = cfg["corpus"]["segmenter"]
    segmenter = Segmenter(seg, cache)
    rows, doc_id, report = [], corpus.doc_id.max() + 1, []
    for name, overrides in cfg["corpus"]["sources"].items():
        c = {"seed": cfg["seed"], "data_dir": cfg["data_dir"], **cfg["corpus"]["defaults"], **(overrides or {})}
        load, sep = SOURCES[name]
        mine = corpus[corpus.source == name].sort_values(["doc_id", "sent_idx"])
        used = set(mine.doc_key.astype(str))
        seen = {tuple(zip(g.para_idx, g.text)) for _, g in mine.groupby("doc_id")}
        target = cfg["pretrain"]["fresh_multiple"] * mine.n_words[mine.split.isin(["train", "val"])].sum()
        docs = segmented(((k, t) for k, t in load(c) if str(k) not in used), name, sep, segmenter, seg["batch_docs"])
        kept, n = take(tqdm(docs, desc=name), used, seen, target, c["min_doc_sentences"], c["max_doc_sentences"])
        for key, sents in kept:
            rows += [dict(source=name, doc_id=doc_id, doc_key=str(key), split="fresh", sent_idx=i, para_idx=p, text=s,
                          n_words=len(words(s)), segmenter=how) for i, (p, s, how) in enumerate(sents)]
            doc_id += 1
        report.append((name, len(kept), n, target))
        print(f"{name}: {len(kept)} documents, {n:,} of {target:,} words" + ("" if n >= target else ", ran out"),
              flush=True)
    df = pd.DataFrame(rows)
    df.insert(0, "sentence_id", range(corpus.sentence_id.max() + 1, corpus.sentence_id.max() + 1 + len(df)))
    df.to_parquet(path(cfg, "fresh", "sentences.parquet"), index=False)
    print(pd.DataFrame(report, columns=["source", "documents", "words", "target"]).to_string(index=False))


if __name__ == "__main__":
    main()
