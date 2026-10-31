"""Human-written text sources. Each loader takes the source's merged config `c` (seed, data_dir, corpus.defaults,
per-source overrides) and yields (doc_key, text) in a seeded random order.

Sources with per-document dates keep only documents from c["max_year"] or earlier. Hub datasets and the ELLIPSE CSV are
read at the revisions pinned below (current on 2026-10-29). SOURCES maps each name to (loader, paragraph separator);
newlines inside a paragraph are line wrapping.
"""
import hashlib
import re
import urllib.request
from pathlib import Path

import gdown
import pandas as pd
from datasets import load_dataset

PERSUADE_DRIVE_ID = "13phHyDzIsb0MHyJr6q-B-qIa9P2tM135"   # training CSV linked from github.com/scrosseye/persuade_corpus_2.0
ELLIPSE_URL = ("https://raw.githubusercontent.com/scrosseye/ELLIPSE-Corpus/b5f4469611d2596810d31d1a60921e54ce97ad07/"
               "ELLIPSE_Final_github_train.csv")   # the last commit that changed the CSV (2024-11-14)
XSUM_URLS = ("https://raw.githubusercontent.com/EdinburghNLP/XSum/7b1d1ed4128e23bd96fa967ab1858523d29dfa96/"
             "XSum-Dataset/XSum-WebArxiveUrls.txt")   # archived BBC URL of every article, ending in its id
XSUM_SECTIONS = {"news", "newsbeat"}   # dropped: sport (incl. automated live commentary), newsround (video pages),
                                       # newyddion (Welsh), schoolreport
REVISIONS = {
    "HuggingFaceFW/fineweb-edu": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    "allenai/peS2o": "636a503e44a3ca1b58e01fb61eab0825cd574de0",
    "euclaise/writingprompts": "35f0aa359452ba8147b34d925684fccee26679cc",
    "EdinburghNLP/xsum": "7d4d486c2f8ef850b1a11aead99b894ff3dd7da9",
    "nhop/OpenReview": "751904d52705081496227083a41902bf44280ef9",
    "Yale-LILY/aeslc": "2305f2e63b68056f9b9037a3805c8c196e0d5581",
}
REVIEW_FIELDS = ["paper_summary", "main_review", "strength_weakness", "questions", "limitations", "review_summary"]


def stream(c, *args, **kwargs):
    return load_dataset(*args, split="train", streaming=True, **kwargs).shuffle(seed=c["seed"], buffer_size=c["shuffle_buffer"])


def hub(c, repo, **kwargs):
    """Train split of a Hub dataset at its pinned revision, streamed through the seeded shuffle buffer."""
    return stream(c, repo, revision=REVISIONS[repo], **kwargs)


def shuffled(c, df):
    return df.sample(frac=1, random_state=c["seed"]).itertuples()


def text_key(text):
    return hashlib.md5(text.encode()).hexdigest()[:16]


def venue_year(venue):
    """Year in an OpenReview venue id such as 'ICLR_cc_2018_Conference'; None if absent."""
    m = re.search(r"_((?:19|20)\d\d)_", venue or "")
    return int(m.group(1)) if m else None


def decode_qp(body):
    """Undo the quoted-printable transfer encoding left in some AESLC bodies, only in bodies that contain =20 (space),
    =09 (tab) or =3D (=): those codes and soft line breaks ("=" ending a line) are decoded, other codes left as is."""
    if not re.search(r"=(20|09|3D)", body):
        return body
    return re.sub(r"=(20|09|3D)", lambda m: {"20": " ", "09": "\t", "3D": "="}[m.group(1)], re.sub(r"=\r?\n", "", body))


def join_urls(text):
    """Remove the space PERSUADE's processing put after a period followed by a letter, inside web addresses only:
    every domain in the dataset is spaced ("www. npr. org/templates/story/story. php? storyId=95256794")."""
    text = re.sub(r"\bwww\. (?=\w)", "www.", text)
    text = re.sub(r"(?<=\w)\. (?=(?:com|org|net|gov|edu|php|html?|aspx?)\b)", ".", text)
    return re.sub(r"(\.(?:php|html?|aspx?))\? (?=\w)", r"\1?", text)


def detokenize(story):
    """Undo the WritingPrompts tokenization left in the stories: ``/'' quotes, split n't, spaces inside curly quotes."""
    story = re.sub(r"`` ?", '"', story).replace("''", '"')
    story = re.sub(r"(\w) n't\b", r"\1n't", story)
    return story.replace("“ ", "“").replace(" ”", "”")


def fineweb_edu(c):
    """Web pages from FineWeb-Edu (Common Crawl kept by a classifier of educational value, score >= 3; ODC-By), from
    crawls in c["max_year"] or earlier."""
    files = [f"data/CC-MAIN-{y}-*/*.parquet" for y in range(2013, c["max_year"] + 1)]   # the first crawl is 2013
    for d in hub(c, "HuggingFaceFW/fineweb-edu", data_files=files):
        yield d["id"], d["text"]


def pes2o(c):
    """Full-text open-access papers from peS2o v2 (S2ORC via Grobid; ODC-By), first published in c["max_year"] or
    earlier. Read as JSON because the repo's loading script no longer runs; files 10-19 hold the full texts."""
    files = [f"hf://datasets/allenai/peS2o@{REVISIONS['allenai/peS2o']}/data/v2/train-{i:05d}-of-00020.json.gz"
             for i in range(10, 20)]
    for d in stream(c, "json", data_files=files):
        if d["source"].startswith("s2orc") and int((d["created"] or "9999")[:4]) <= c["max_year"]:
            yield d["id"], d["text"]


def persuade(c):
    """PERSUADE 2.0 training essays: argumentative essays by US students in grades 6-12 (CC BY-NC-SA 4.0), with web
    addresses rejoined (join_urls)."""
    f = Path(c["data_dir"], "raw", "persuade_corpus_2.0_train.csv")
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=PERSUADE_DRIVE_ID, output=str(f))
    for r in shuffled(c, pd.read_csv(f, usecols=["essay_id_comp", "full_text"]).drop_duplicates("essay_id_comp")):
        yield r.essay_id_comp, join_urls(r.full_text)


def ellipse(c):
    """ELLIPSE training essays: English language learners in grades 8-12, from state standardized tests
    (CC BY-NC-SA 4.0)."""
    for r in shuffled(c, pd.read_csv(ELLIPSE_URL, usecols=["text_id_kaggle", "full_text"])):
        yield str(r.text_id_kaggle), r.full_text


def xsum(c):
    """BBC articles, 2010-2017, without their first sentence (Narayan et al. 2018), from the URL sections in
    XSUM_SECTIONS (looked up by article id in the dataset's URL list)."""
    urls = urllib.request.urlopen(XSUM_URLS).read().decode()
    section = {i: s for s, i in re.findall(r"bbc\.co(?:m|\.uk)(?::\d+)?/(\w+)/\S*?(\d+)/?$", urls, re.M)}
    for d in hub(c, "EdinburghNLP/xsum"):
        if section.get(d["id"]) in XSUM_SECTIONS:
            yield d["id"], d["document"]


def openreview(c):
    """Peer reviews from OpenReview venues held in c["max_year"] or earlier (CC BY-NC 4.0); one review per doc."""
    for d in hub(c, "nhop/OpenReview"):
        if (venue_year(d["venue"]) or 9999) <= c["max_year"]:
            for r in d["reviews"] or []:
                review = r["review"] or {}
                yield r["review_id"], "\n".join(review[k].removeprefix(f"{k}: ") for k in REVIEW_FIELDS if review.get(k))


def enron(c):
    """Enron emails, 2000-2002, cleaned bodies from AESLC (Zhang & Tetreault 2019), quoted-printable leftovers decoded
    (decode_qp)."""
    for d in hub(c, "Yale-LILY/aeslc"):
        yield text_key(d["email_body"]), decode_qp(d["email_body"])


def writingprompts(c):
    """Short stories from Reddit's r/WritingPrompts, collected for Fan et al. (2018), with the dataset's tokenization
    undone (`detokenize`); prompts dropped."""
    for d in hub(c, "euclaise/writingprompts"):
        yield text_key(d["story"]), detokenize(d["story"])


SOURCES = {f.__name__: (f, sep) for f, sep in [
    (fineweb_edu, "\n"), (pes2o, "\n\n"), (persuade, "\n\n"), (ellipse, "\n\n"), (xsum, "\n"), (openreview, "\n"),
    (enron, "\n"), (writingprompts, "\n")]}
