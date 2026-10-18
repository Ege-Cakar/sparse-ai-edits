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


def sentences(text, paragraph_sep="\n"):
    """Punkt sentences within each paragraph; whitespace inside a paragraph (line wrapping) is collapsed."""
    _punkt()
    return [s for para in text.split(paragraph_sep) for s in nltk.sent_tokenize(" ".join(para.split()))]


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
