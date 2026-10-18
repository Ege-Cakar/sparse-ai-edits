from sparse_edits.corpus import eligible
from sparse_edits.edit import clean, n_target, rejection
from sparse_edits.sources import arxiv_year, venue_year
from sparse_edits.text import sentences, word_diff, words


def test_words_ignore_case_and_punctuation():
    assert words("The dog's ball, it's RED!") == ["the", "dog's", "ball", "it's", "red"]


def test_word_diff_substitution_insertion_deletion():
    assert word_diff("The cat sat on the mat.", "The dog sat on a mat!") == (2, [0, 1, 0, 0, 1, 0])
    assert word_diff("I like it.", "I really like it a lot.") == (3, [0, 1, 0, 0, 1, 1])
    assert word_diff("I really like it.", "I like it.") == (1, [0, 0, 0])


def test_sentences_split_paragraphs_and_keep_abbreviations():
    assert sentences("Hello there. How are you?\nMr. Smith said hi.") == ["Hello there.", "How are you?",
                                                                         "Mr. Smith said hi."]


def test_edit_helpers():
    assert n_target(0.05, 10) == 1 and n_target(0.5, 21) == 10
    assert clean('  "A new sentence."  ', "A sentence.") == "A new sentence."
    assert clean('"Quoted," he said.', '"Quoted," she said.') == '"Quoted," he said.'
    assert rejection("The cat sat.", "The cat sat!") == "unchanged"
    assert rejection("The cat sat.", "The cat sat. The dog ran.") == "not_one_sentence"
    assert rejection("The cat sat.", "The dog sat.") is None


def test_sentences_join_wrapped_lines_within_paragraphs():
    text = "The first line is\nwrapped here. Second sentence.\n\nNew paragraph."
    assert sentences(text, "\n\n") == ["The first line is wrapped here.", "Second sentence.", "New paragraph."]


def test_eligible_filters_and_truncates():
    c = dict(min_words=2, max_words=5, min_doc_sentences=2, max_doc_sentences=3)
    text = "One. Two words. Three words here. Four words are here. Five words are right here."
    assert eligible(text, "\n", c) is None                                   # 4 eligible sentences > 3
    assert eligible(text, "\n", {**c, "truncate": True}) == ["Two words.", "Three words here.", "Four words are here."]


def test_dates_from_ids():
    assert arxiv_year("0704.0001") == 2007 and arxiv_year("1912.12345") == 2019 and arxiv_year("2001.00001") == 2020
    assert arxiv_year("math/9912001") == 1999 and arxiv_year("hep-th/0211159") == 2002
    assert venue_year("ICLR_cc_2018_Conference") == 2018 and venue_year("NoDaLiDa_2023_Conference") == 2023
    assert venue_year("unknown_venue") is None and venue_year(None) is None
