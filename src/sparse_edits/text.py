import re
from functools import cache

import nltk
from rapidfuzz.distance import Levenshtein

WORD = re.compile(r"\w+(?:['’]\w+)*")


def words(text):
    """Lowercased word tokens. Punctuation-only and case-only changes are not word changes."""
    return WORD.findall(text.lower())


@cache
def _punkt():
    nltk.download("punkt_tab", quiet=True)


def sentences(text):
    """Punkt sentences within each newline-separated paragraph."""
    _punkt()
    return [s.strip() for para in text.split("\n") for s in nltk.sent_tokenize(para) if s.strip()]


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
