"""Load every source in corpus.sources, segment into sentences, keep documents of eligible length, assign doc-level splits.

Writes sentences.parquet: sentence_id, source, doc_id, doc_key, split, sent_idx, para_idx, text, n_words, segmenter.
Text is first repaired with ftfy (FTFY: encoding errors, HTML entities, ligatures, control characters, line breaks;
quote style and character width are left as written). Sentences come from the LLM in corpus.segmenter (segment.py),
which also decides which line breaks are paragraph breaks, in batches of documents; without one, from SaT within
paragraphs (text.indexed_sentences). segmenter records which split each sentence: "llm", "sat" or "punkt".
A document keeps every sentence, so it stays contiguous; documents with fewer than min_doc_sentences or more than
max_doc_sentences sentences are skipped whole, and so is a document whose paragraphs exactly repeat an
earlier one of the same source (e.g. one Enron email filed in several folders, one web page in several crawls).
Within each source, docs come in a seeded random order and fill that source's splits in config order, each with whole
documents until it has at least its n_sentences target.
"""
from itertools import islice

import pandas as pd
from ftfy import TextFixerConfig, fix_text
from tqdm import tqdm

from . import cli, path
from .segment import Segmenter
from .sources import SOURCES
from .text import indexed_sentences, paragraphs, words

FTFY = TextFixerConfig(uncurl_quotes=False, fix_character_width=False, explain=False)


def fill(docs, n_sentences):
    """Yield (split, key, sents) for whole documents assigned to each split in turn until the split has at least its
    target number of sentences, so a split overshoots its target by less than one document."""
    docs = iter(docs)
    for split, target in n_sentences.items():
        filled = 0
        while filled < target:
            if (doc := next(docs, None)) is None:
                raise ValueError(f"ran out of documents while filling {split} ({filled} of {target} sentences)")
            filled += len(doc[1])
            yield split, *doc


def unique(docs, signature):
    """Skip (key, x) documents whose signature(x) repeats an earlier one's; the first copy is kept. Applied to the
    paragraphs before splitting, so a copy is never split twice (the LLM may split it differently), and to the
    sentences after, since documents differing only in line wrapping become identical once it is undone."""
    seen = set()
    for key, x in docs:
        if (sig := signature(x)) not in seen:
            seen.add(sig)
            yield key, x


def segmented(docs, name, sep, segmenter, batch_docs):
    """(key, [(para_idx, sentence, segmenter)]) for each distinct (key, text), in order."""
    docs = unique(((key, paragraphs(fix_text(text, FTFY), sep)) for key, text in docs), tuple)
    if segmenter is None:
        yield from ((key, indexed_sentences(sep.join(paras), sep)) for key, paras in docs)
        return
    while batch := list(islice(docs, batch_docs)):
        yield from zip([key for key, _ in batch], segmenter(name, sep, [paras for _, paras in batch]))


def main():
    cfg, _ = cli(__doc__)
    rows, doc_id, seg = [], 0, cfg["corpus"].get("segmenter")
    segmenter = seg and Segmenter(seg, path(cfg, "segment_cache.jsonl"))
    for name, overrides in cfg["corpus"]["sources"].items():
        c = {"seed": cfg["seed"], "data_dir": cfg["data_dir"], **cfg["corpus"]["defaults"], **(overrides or {})}
        load, paragraph_sep = SOURCES[name]
        docs = ((key, s) for key, s in segmented(load(c), name, paragraph_sep, segmenter, seg and seg["batch_docs"])
                if c["min_doc_sentences"] <= len(s) <= c["max_doc_sentences"])
        docs = unique(docs, lambda sents: tuple((p, s) for p, s, _ in sents))
        for split, key, sents in tqdm(fill(docs, c["n_sentences"]), desc=name):
            rows += [dict(source=name, doc_id=doc_id, doc_key=key, split=split, sent_idx=i, para_idx=p, text=s,
                          n_words=len(words(s)), segmenter=how) for i, (p, s, how) in enumerate(sents)]
            doc_id += 1
    df = pd.DataFrame(rows)
    df.insert(0, "sentence_id", range(len(df)))
    df.to_parquet(path(cfg, "sentences.parquet"), index=False)
    print(df.groupby(["source", "split"]).agg(docs=("doc_id", "nunique"), sentences=("sentence_id", "size")))
    print(df.groupby("source").segmenter.value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
