"""Load every source in corpus.sources, segment into sentences, keep eligible sentences, assign doc-level splits.

Writes sentences.parquet: sentence_id, source, doc_id, doc_key, split, sent_idx, text, n_words.
Within each source, docs come in a seeded random order and fill that source's splits in config order.
"""
import pandas as pd
from tqdm import tqdm

from . import cli, path
from .sources import SOURCES
from .text import sentences, words


def eligible(text, paragraph_sep, c):
    """Sentences of a document within [min_words, max_words]; None if their count is outside
    [min_doc_sentences, max_doc_sentences] (after keeping the first max_doc_sentences when truncate is set)."""
    sents = [s for s in sentences(text, paragraph_sep) if c["min_words"] <= len(words(s)) <= c["max_words"]]
    sents = sents[:c["max_doc_sentences"]] if c.get("truncate") else sents
    return sents if c["min_doc_sentences"] <= len(sents) <= c["max_doc_sentences"] else None


def main():
    cfg, _ = cli(__doc__)
    rows, doc_id = [], 0
    for name, overrides in cfg["corpus"]["sources"].items():
        c = {"seed": cfg["seed"], "data_dir": cfg["data_dir"], **cfg["corpus"]["defaults"], **overrides}
        load, paragraph_sep = SOURCES[name]
        docs = ((key, s) for key, text in load(c) if (s := eligible(text, paragraph_sep, c)))
        splits = [s for s, n in c["n_docs"].items() for _ in range(n)]
        for split, (key, sents) in zip(splits, tqdm(docs, total=len(splits), desc=name)):
            rows += [dict(source=name, doc_id=doc_id, doc_key=key, split=split, sent_idx=i, text=s, n_words=len(words(s)))
                     for i, s in enumerate(sents)]
            doc_id += 1
    df = pd.DataFrame(rows)
    df.insert(0, "sentence_id", range(len(df)))
    df.to_parquet(path(cfg, "sentences.parquet"), index=False)
    print(df.groupby(["source", "split"]).agg(docs=("doc_id", "nunique"), sentences=("sentence_id", "size")))


if __name__ == "__main__":
    main()
