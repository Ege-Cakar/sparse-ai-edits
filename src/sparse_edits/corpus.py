"""Stream C4, segment into sentences, keep eligible sentences, assign doc-level splits.

Writes sentences.parquet: sentence_id, doc_id, split, url, sent_idx, text, n_words.
Docs come from a seeded shuffled stream and fill the splits in config order, so splits are disjoint by doc.
"""
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from . import cli, path
from .text import sentences, words


def eligible_doc(doc, c):
    sents = [s for s in sentences(doc["text"]) if c["min_words"] <= len(words(s)) <= c["max_words"]]
    return (doc["url"], sents) if c["min_doc_sentences"] <= len(sents) <= c["max_doc_sentences"] else None


def main():
    cfg, _ = cli(__doc__)
    c = cfg["corpus"]
    stream = load_dataset(c["dataset"], c["config"], split=c["split"], streaming=True)
    stream = stream.shuffle(seed=cfg["seed"], buffer_size=c["shuffle_buffer"])
    docs = filter(None, (eligible_doc(d, c) for d in stream))
    splits = [s for s, n in c["n_docs"].items() for _ in range(n)]
    rows = [dict(doc_id=doc_id, split=split, url=url, sent_idx=i, text=s, n_words=len(words(s)))
            for doc_id, (split, (url, sents)) in enumerate(zip(splits, tqdm(docs, total=len(splits))))
            for i, s in enumerate(sents)]
    df = pd.DataFrame(rows)
    df.insert(0, "sentence_id", range(len(df)))
    df.to_parquet(path(cfg, "sentences.parquet"), index=False)
    print(df.groupby("split").agg(docs=("doc_id", "nunique"), sentences=("sentence_id", "size")))


if __name__ == "__main__":
    main()
