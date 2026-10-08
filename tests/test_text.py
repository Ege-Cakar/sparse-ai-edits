from sparse_edits.edit import clean, n_target, rejection
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
