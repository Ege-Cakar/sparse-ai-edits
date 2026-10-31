import re
from functools import cache

import nltk
from rapidfuzz.distance import Levenshtein

WORD = re.compile(r"\w+(?:['’]\w+)*")
INVISIBLE = re.compile(r"[\s\u200b-\u200d\u2060\ufeff]*")   # whitespace and zero-width characters


def words(text):
    """Lowercased word tokens. Punctuation-only and case-only changes are not word changes."""
    return WORD.findall(text.lower())


SAT_MODEL = ("sat-12l-sm", "d70c72a9331b2d5a9e82baad00c64964a23a09bb")   # segment-any-text/<name>, revision
SAT_TOKENIZER = ("FacebookAI/xlm-roberta-base", "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089")


@cache
def _punkt():
    nltk.download("punkt_tab", quiet=True)


@cache
def _sat():
    """SaT at pinned model and tokenizer revisions, fp32 as in the splitter comparison, on the GPU when there is one."""
    import torch
    from huggingface_hub import snapshot_download
    from wtpsplit import SaT
    tokenizer = snapshot_download(SAT_TOKENIZER[0], revision=SAT_TOKENIZER[1], allow_patterns=["*.json", "*.model"])
    sat = SaT(SAT_MODEL[0], tokenizer_name_or_path=tokenizer, from_pretrained_kwargs=dict(revision=SAT_MODEL[1]))
    if torch.cuda.is_available():
        sat.to("cuda")   # moves the wrapped model in place
    return sat


def paragraphs(text, sep="\n"):
    """Paragraphs with visible text (not only whitespace or zero-width characters), with whitespace inside each (line
    wrapping) collapsed."""
    return [p for p in (" ".join(p.split()) for p in text.split(sep)) if not INVISIBLE.fullmatch(p)]


def sentences(text, paragraph_sep="\n"):
    """Punkt sentences within each paragraph. Used to check editor output; the corpus uses indexed_sentences (SaT)."""
    _punkt()
    return [s for p in paragraphs(text, paragraph_sep) for s in nltk.sent_tokenize(p)]


def indexed_sentences(text, paragraph_sep="\n"):
    """(paragraph index, sentence, segmenter) triples: SaT sentences within each paragraph; the index counts only
    paragraphs with visible text. segmenter is "sat", or "punkt" for paragraphs SaT failed on."""
    return [(i, s, how) for i, (sents, how) in enumerate(_split(paragraphs(text, paragraph_sep))) for s in sents]


def _split(paras):
    """(sentences, "sat" or "punkt") per paragraph. wtpsplit raised an AssertionError on one paragraph in the splitter
    comparison; after a failure, each paragraph is retried alone and only the failing ones go to Punkt."""
    try:
        return [([s for s in out if s], "sat") for out in _sat().split(paras, strip_whitespace=True)]
    except AssertionError:
        if len(paras) > 1:
            return [r for p in paras for r in _split([p])]
        _punkt()
        return [(nltk.sent_tokenize(paras[0]), "punkt")]


def word_diff(original, edited):
    """Word-level Levenshtein alignment.

    Returns (number of word insertions + deletions + substitutions,
             mask over the edited sentence's words: 1 = inserted or substituted).
    """
    ops = Levenshtein.editops(words(original), words(edited))
    mask = [0] * len(words(edited))
    for op in ops:
        if op.tag != "delete":
            mask[op.dest_pos] = 1
    return len(ops), mask
