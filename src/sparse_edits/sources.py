"""Human-written text sources. Each loader takes the source's merged config `c` (seed, data_dir, corpus.defaults,
per-source overrides) and yields (doc_key, text) in a seeded random order.

Sources with per-document dates keep only documents from c["max_year"] or earlier. SOURCES maps each name to
(loader, paragraph separator); newlines inside a paragraph are line wrapping.
"""
import hashlib
import re
from pathlib import Path

import gdown
import pandas as pd
from datasets import load_dataset

PERSUADE_DRIVE_ID = "13phHyDzIsb0MHyJr6q-B-qIa9P2tM135"   # training CSV linked from github.com/scrosseye/persuade_corpus_2.0
ELLIPSE_URL = "https://raw.githubusercontent.com/scrosseye/ELLIPSE-Corpus/main/ELLIPSE_Final_github_train.csv"
REVIEW_FIELDS = ["paper_summary", "main_review", "strength_weakness", "questions", "limitations", "review_summary"]


def stream(c, *args):
    return load_dataset(*args, split="train", streaming=True).shuffle(seed=c["seed"], buffer_size=c["shuffle_buffer"])


def shuffled(c, df):
    return df.sample(frac=1, random_state=c["seed"]).itertuples()


def arxiv_year(arxiv_id):
    """First-submission year from an arXiv id ('YYMM.NNNNN' or 'archive/YYMMNNN')."""
    yy = int(arxiv_id.split("/")[-1][:2])
    return 1900 + yy if yy >= 91 else 2000 + yy


def venue_year(venue):
    """Year in an OpenReview venue id such as 'ICLR_cc_2018_Conference'; None if absent."""
    m = re.search(r"_((?:19|20)\d\d)_", venue or "")
    return int(m.group(1)) if m else None


def c4(c):
    """Web pages from the April 2019 Common Crawl snapshot (Raffel et al. 2020)."""
    for d in stream(c, "allenai/c4", "en"):
        yield d["url"], d["text"]


def persuade(c):
    """PERSUADE 2.0 training essays: argumentative essays by US students in grades 6-12 (CC BY-NC-SA 4.0)."""
    f = Path(c["data_dir"], "raw", "persuade_corpus_2.0_train.csv")
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=PERSUADE_DRIVE_ID, output=str(f))
    for r in shuffled(c, pd.read_csv(f, usecols=["essay_id_comp", "full_text"]).drop_duplicates("essay_id_comp")):
        yield r.essay_id_comp, r.full_text


def ellipse(c):
    """ELLIPSE training essays: English language learners in grades 8-12, from state standardized tests
    (CC BY-NC-SA 4.0)."""
    for r in shuffled(c, pd.read_csv(ELLIPSE_URL, usecols=["text_id_kaggle", "full_text"])):
        yield str(r.text_id_kaggle), r.full_text


def xsum(c):
    """BBC news articles, 2010-2017, without their first sentence (Narayan et al. 2018)."""
    for d in stream(c, "EdinburghNLP/xsum"):
        yield d["id"], d["document"]


def arxiv_abstracts(c):
    """arXiv abstracts from a 2021 metadata snapshot (CC0), first submitted in c["max_year"] or earlier."""
    for d in stream(c, "gfissore/arxiv-abstracts-2021"):
        if arxiv_year(d["id"]) <= c["max_year"]:
            yield d["id"], d["abstract"]


def openreview(c):
    """Peer reviews from OpenReview venues held in c["max_year"] or earlier (CC BY-NC 4.0); one review per doc."""
    for d in stream(c, "nhop/OpenReview"):
        if (venue_year(d["venue"]) or 9999) <= c["max_year"]:
            for r in d["reviews"] or []:
                review = r["review"] or {}
                yield r["review_id"], "\n".join(review[k].removeprefix(f"{k}: ") for k in REVIEW_FIELDS if review.get(k))


def enron(c):
    """Enron emails, 2000-2002, cleaned bodies from AESLC (Zhang & Tetreault 2019)."""
    for d in stream(c, "Yale-LILY/aeslc"):
        yield hashlib.md5(d["email_body"].encode()).hexdigest()[:16], d["email_body"]


def pg19(c):
    """Project Gutenberg books published before 1919 (Rae et al. 2020); a passage from the middle of each book."""
    for d in stream(c, "emozilla/pg19"):
        mid = len(d["text"]) // 2
        yield d["url"], "\n\n".join(d["text"][mid - 20000:mid + 20000].split("\n\n")[1:-1])   # drop cut paragraphs


SOURCES = {f.__name__: (f, sep) for f, sep in [
    (c4, "\n"), (persuade, "\n\n"), (ellipse, "\n\n"), (xsum, "\n"), (arxiv_abstracts, "\n\n"), (openreview, "\n"),
    (enron, "\n"), (pg19, "\n\n")]}
