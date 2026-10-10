import json
from io import StringIO

import pandas as pd
import pytest

from sparse_edits.edit_mlm import content_words, match_case, word_spans
from sparse_edits.text import word_diff, words


def test_word_spans_align_with_text_words():
    text = "The dog's ball, it's RED and 42 cats ran."
    spans, editable = word_spans(text)
    assert [text[a:b].lower() for a, b in spans] == words(text)
    assert [text[spans[i][0]:spans[i][1]] for i in editable] == ["The", "ball", "RED", "and", "cats", "ran"]


def test_urls_and_emails_are_not_editable():
    text = "Read more at https://www.ft.com/content today or mail info@brick.co.uk now."
    spans, editable = word_spans(text)
    assert [text[spans[i][0]:spans[i][1]] for i in editable] == ["Read", "more", "at", "today", "or", "mail", "now"]


def test_match_case_and_content_words():
    assert match_case("quick", "Fast") == "Quick" and match_case("quick", "FAST") == "QUICK"
    assert match_case("quick", "fast") == "quick"
    assert content_words("The cat sat on the mat with Bob") == {"with"}


@pytest.mark.slow
def test_infiller_changes_exactly_k_words():
    from sparse_edits.edit_mlm import Infiller, run
    sents = pd.DataFrame(dict(sentence_id=[0], doc_id=[0], sent_idx=[0], n_words=[13],
                              text=["The old house at the end of the street was finally sold last week."]))
    jobs = [(row, r, 0) for row in sents.itertuples() for r in (0.1, 0.3)]
    buf = StringIO()
    run(Infiller("roberta-base"), jobs, sents, {"name": "t"}, 0, buf)
    for out in map(json.loads, buf.getvalue().splitlines()):
        assert out["rejected"] is None
        assert word_diff(sents.text[0], out["text"])[0] == out["k_requested"] == len(out["positions"])
