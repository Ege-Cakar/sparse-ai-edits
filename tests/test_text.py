import asyncio
import re
import json
from types import SimpleNamespace

import httpx
import openai
import pandas as pd
import pytest

from ftfy import fix_text

from sparse_edits.corpus import FTFY, fill, unique
from sparse_edits.edit import clean, coverage, custom_id, documents, groups, n_target, parse_id, rejection, request, spent
from sparse_edits.segment import Segmenter, from_reply, resplit
from sparse_edits.sources import decode_qp, detokenize, join_urls, venue_year
from sparse_edits import segment, text
from sparse_edits.text import indexed_sentences, paragraphs, sentences, word_diff, words


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
    assert rejection("The cat sat.", "The cat sat. The dog ran.") == "sentence_count_changed"
    assert rejection('"Wait! Stop." he said.', '"Wait! Stop now." he said.') is None   # two Punkt sentences, kept two
    assert rejection('"Wait! Stop." he said.', '"Wait and stop." he said.') == "sentence_count_changed"
    assert rejection("The cat sat.", "The dog sat.") is None
    assert rejection("Thanks.", "I would appreciate your help with this.") == "too_long"


def test_request_shows_document_first_and_keeps_placeholders():
    sents = pd.DataFrame(dict(sentence_id=range(3), doc_id=0, para_idx=[0, 0, 1], source="persuade", n_words=[3, 4, 2],
                              text=["Dear Generic_Name, hi.", "I like your idea.", "Sincerely, STUDENT_NAME"]))
    doc = documents(sents)[0]
    assert doc == "Dear Generic_Name, hi. I like your idea.\n\nSincerely, STUDENT_NAME"
    body = request({"model": "m", "product": "ChatGPT"}, next(sents.itertuples()), doc)
    assert body["messages"][0]["content"].startswith("You are ChatGPT, a helpful AI assistant.")
    user = body["messages"][1]["content"]
    assert user.startswith(f"Text: {doc}\n\nEdit the sentence \"Dear Generic_Name, hi.\", please in the text above")
    assert user.endswith("Keep the placeholder Generic_Name exactly as written.")
    assert rejection("Dear Generic_Name, hi.", "Dear friend, hello.", ["Generic_Name"]) == "placeholder_changed"

    claude = request({"model": "m", "product": "ChatGPT", "api": "anthropic"}, next(sents.itertuples()), doc)
    head, ask = claude["messages"][0]["content"]
    assert claude["system"] == body["messages"][0]["content"] and head["text"] + ask["text"] == user
    assert head["cache_control"] == {"type": "ephemeral"} and "cache_control" not in ask and doc in head["text"]
    keyed = request({"model": "m", "product": "ChatGPT", "cache_key": True}, next(sents.itertuples()), doc)
    assert keyed["prompt_cache_key"] == "doc-0" and "prompt_cache_key" not in body


def test_spent_prices_the_usage_of_both_apis(tmp_path):
    out, price = tmp_path / "e.jsonl", {"input": 1, "cache_read": 0.1, "cache_write": 1.25, "output": 5}
    (tmp_path / "e.jsonl.usage").write_text(json.dumps({"usage": {
        "input_tokens": 1000, "cache_read_input_tokens": 2000, "cache_creation_input_tokens": 3000, "output_tokens": 4000}}))
    assert spent({"api": "anthropic", "price": price}, out) == pytest.approx(((1000 + 200 + 3750 + 20000) / 1e6, 1))
    (tmp_path / "e.jsonl.usage").write_text(json.dumps({"usage": {
        "prompt_tokens": 3000, "completion_tokens": 4000, "prompt_tokens_details": {"cached_tokens": 2000}}}))
    assert spent({"price": price}, out) == pytest.approx(((1000 + 200 + 20000) / 1e6, 1))


def test_run_batches_stops_before_a_batch_that_would_pass_the_cap(monkeypatch, tmp_path):
    from sparse_edits import edit
    sent = []
    monkeypatch.setattr(edit, "spent", lambda editor, out_path: (1.0, 100))   # $0.01 per request so far
    monkeypatch.setattr(edit, "groups", lambda *a: iter([[("a", {})] * 10] * 4))
    monkeypatch.setattr(edit, "submit", lambda client, editor, group: sent.append(len(group)) or f"b{len(sent)}")
    monkeypatch.setattr(edit, "collect", lambda *a: [])
    cfg = {"edit": {"batch_size": 10, "batch_mb": 1, "max_attempts": 3}}
    with pytest.raises(SystemExit):
        edit.run_batches(cfg, {"name": "e", "model": "m"}, None, None, [(0, None, 0, 1)], tmp_path / "e.jsonl", 1.25)
    assert sent == [10, 10] and json.loads((tmp_path / "e.jsonl.batches").read_text()) == ["b1", "b2"]


def test_run_batches_queues_at_most_max_queued_batches_and_carries_the_rest(monkeypatch, tmp_path):
    from sparse_edits import edit
    rounds = []
    monkeypatch.setattr(edit, "groups", lambda editor, by_id, docs, jobs, *a: iter([[(custom_id(*j), {})] for j in jobs]))
    monkeypatch.setattr(edit, "submit", lambda client, editor, group: rounds.append(group[0][0]) or "b")
    monkeypatch.setattr(edit, "collect", lambda *a: [(1, None, 0, 2)] if rounds == [custom_id(1, None, 0, 1)] else [])
    cfg = {"edit": {"batch_size": 1, "batch_mb": 1, "max_attempts": 3}}
    jobs = [(i, None, 0, 1) for i in range(1, 4)]
    edit.run_batches(cfg, {"name": "e", "model": "m", "batch_limits": {"max_queued": 1}}, None, None, jobs,
                     tmp_path / "e.jsonl")
    # one batch per round; the unsent jobs go first, then the resample of job 1
    assert rounds == [custom_id(*j) for j in jobs + [(1, None, 0, 2)]]


def test_run_editor_logs_usage_and_stops_before_a_chunk_that_would_pass_the_cap(monkeypatch, tmp_path):
    from sparse_edits import edit
    usage, calls = {"prompt_tokens": 1000, "completion_tokens": 0}, []   # $0.001 at $1/M input
    monkeypatch.setattr(edit.random, "uniform", lambda a, b: 0)

    async def create(**body):
        calls.append(body)
        if len(calls) == 1:   # no flex capacity: waited out and resent, not counted as an attempt
            raise openai.RateLimitError("busy", body={"code": "flex_unavailable"},
                                        response=httpx.Response(429, request=httpx.Request("POST", "http://x")))
        return SimpleNamespace(service_tier="flex", usage=SimpleNamespace(model_dump=lambda: usage),
                               choices=[SimpleNamespace(message=SimpleNamespace(content="Edited text here."))])
    monkeypatch.setattr(edit.openai, "AsyncOpenAI",
                        lambda **a: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    sents = pd.DataFrame(dict(sentence_id=range(3), doc_id=0, para_idx=0, source="xsum", n_words=2,
                              text=[f"Sentence {i}." for i in range(3)]))
    out = tmp_path / "e.jsonl"
    (tmp_path / "e.jsonl.usage").write_text(json.dumps({"usage": usage}) + "\n")   # the pilot's response
    cfg = {"edit": {"concurrency": 2, "chunk_size": 1, "max_attempts": 3}}
    with pytest.raises(SystemExit):   # $0.001 spent + $0.001 fits under $0.0025; the next chunk would not
        asyncio.run(edit.run_editor(cfg, {"name": "e", "model": "m", "product": "P", "price": {"input": 1}},
                                    documents(sents), [(row, None, 0) for row in sents.itertuples()], out, 0.0025))
    logged = [json.loads(line) for line in (tmp_path / "e.jsonl.usage").read_text().splitlines()]
    assert len(out.read_text().splitlines()) == 1
    assert logged[1] == dict(custom_id=custom_id(0, None, 0, 1), service_tier="flex", usage=usage)


def test_batch_ids_round_trip_and_groups_stay_under_limits():
    for job in [(12, None, 0, 3), (7, 0.25, 1, 2)]:
        assert parse_id(custom_id(*job)) == job and re.fullmatch(r"[A-Za-z0-9_-]+", custom_id(*job))
    sents = pd.DataFrame(dict(sentence_id=range(5), doc_id=[0, 0, 0, 1, 1], para_idx=0, source="xsum", n_words=2,
                              text=[f"Sentence {i}." for i in range(5)]))
    by_id, docs, jobs = sents.set_index("sentence_id", drop=False), documents(sents), [(i, None, 0, 1) for i in range(5)]
    assert [len(g) for g in groups({"model": "m", "product": "P"}, by_id, docs, jobs, 2, 10**6)] == [2, 2, 1]
    assert [len(g) for g in groups({"model": "m", "product": "P"}, by_id, docs, jobs, 10, 1)] == [1] * 5
    assert [c for g in groups({"model": "m", "product": "P"}, by_id, docs, jobs, 10, 10**6) for c, _ in g] == \
        [custom_id(*j) for j in jobs]


def test_coverage_counts_complete_passes(tmp_path):
    cfg = {"data_dir": str(tmp_path), "edit": {"splits": ["test"]}}
    sents = pd.DataFrame(dict(sentence_id=[0, 1, 2], source="s", split="test", n_words=[3, 4, 0]))   # 2 editable
    rec = lambda sid: json.dumps(dict(sentence_id=sid, editor="e", target_r=None, rep=0, text="x", attempts=1,
                                      rejected=None)) + "\n"
    (tmp_path / "edits").mkdir()
    (tmp_path / "edits" / "e.jsonl").write_text(rec(0))
    assert coverage(cfg, sents).loc[("e", "s", "test")].tolist() == [0, 0.5]
    (tmp_path / "edits" / "e.jsonl").write_text(rec(0) + rec(1) + rec(1))   # a repeated record counts once
    assert coverage(cfg, sents).loc[("e", "s", "test")].tolist() == [1, 1.0]


def test_sentences_join_wrapped_lines_within_paragraphs():
    text = "The first line is\nwrapped here. Second sentence.\n\nNew paragraph."
    assert sentences(text, "\n\n") == ["The first line is wrapped here.", "Second sentence.", "New paragraph."]
    assert paragraphs(text + "\n\n \n\n\u200b \u200b\n\nLast one.", "\n\n") == ["The first line is wrapped here. Second sentence.",
                                                                 "New paragraph.", "Last one."]   # blank and zero-width skipped


class FakeSaT:
    """Splits nothing (one sentence per paragraph) and fails on any batch containing "bad"."""
    def split(self, paras, strip_whitespace):
        if any("bad" in p for p in paras):
            raise AssertionError
        return ([p] for p in paras)


def test_indexed_sentences_fall_back_to_punkt_per_paragraph(monkeypatch):
    monkeypatch.setattr(text, "_sat", FakeSaT)
    assert indexed_sentences("One. Two.\n\nA bad one. Three.\n\nFour.", "\n\n") == \
        [(0, "One. Two.", "sat"), (1, "A bad one.", "punkt"), (1, "Three.", "punkt"), (2, "Four.", "sat")]


@pytest.mark.slow
def test_sat_keeps_every_character():
    doc = ("Demographic data District-level demographic data for Sri Lanka were obtained from the census (Department "
           'of Census and Statistics, 2012). They cover all 25 districts.\n"Wait! Stop." he said. "Now?"')
    out = indexed_sentences(doc)
    assert all(s and s == s.strip() and how == "sat" for _, s, how in out)
    for i, p in enumerate(paragraphs(doc)):   # sentences rejoin to the paragraph, up to the spaces between them
        assert "".join(s for q, s, _ in out if q == i).replace(" ", "") == p.replace(" ", "")


def test_llm_reply_may_change_only_breaks_spacing_and_line_end_hyphens():
    src = "Given a value\nn, it is the nth value. Next one.\nSee ( ADHD ) in- dex."
    reply = "Given a value n, it is the nth value.\nNext one.\n\nSee (ADHD) index."
    assert from_reply(src, reply, fix_spacing=True) == \
        [(True, "Given a value n, it is the nth value."), (False, "Next one."), (True, "See (ADHD) index.")]
    assert from_reply(src, reply, fix_spacing=False)[2] == (True, "See ( ADHD ) in- dex.")   # only the breaks used
    assert from_reply(src, reply.replace("value n", "number n"), True) is None
    assert from_reply("One. Two.", "One.\n\nTwo.", True) == [(True, "One."), (False, "Two.")]   # no line break there
    assert from_reply("the most pop-ular one.", "the most popular one.", True, inner_hyphens=True) == \
        [(True, "the most popular one.")]
    assert from_reply("the most pop-ular one.", "the most popular one.", True) is None
    assert from_reply("He said “hi” to the cat’s owner .", "He said \"hi\" to the cat's owner.", True) == \
        [(True, "He said “hi” to the cat’s owner.")]   # straightened quotes match; the original ones are kept


class FakeLLM:
    """Replies with REPLIES[chunk] for the chunk after "Text:"."""
    REPLIES = {"Wrapped\nline. Done.": "Wrapped line.\nDone.", "Bad\ntwo": "Reworded"}

    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=self)

    async def create(self, messages, **kwargs):
        reply = self.REPLIES[messages[0]["content"].split("Text:\n", 1)[1]]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply), finish_reason="stop")])

    async def close(self):
        pass


class ClauseSaT:
    """Splits after every period and comma."""
    def split(self, paras, strip_whitespace):
        return ([p for p in re.split(r"(?<=[.,]) ", x)] for x in paras)


def test_resplit_adds_only_breaks_after_sentence_ends(monkeypatch):
    monkeypatch.setattr(text, "_sat", ClauseSaT)
    sents = [(True, "Not life threatening. There were no arrests, he said", "llm"), (False, "Kept. As is", "sat")]
    assert resplit(sents) == [(True, "Not life threatening.", "llm+sat"),
                              (False, "There were no arrests, he said", "llm+sat"), (False, "Kept. As is", "sat")]


def test_segmenter_falls_back_to_sat_and_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(segment.openai, "AsyncOpenAI", FakeLLM)
    monkeypatch.setattr(text, "_sat", FakeSaT)
    cfg = dict(model="m", base_url="u", fix_spacing=[], rejoin_hyphens=[], max_chunk_chars=100, concurrency=2)
    seg = Segmenter(cfg, tmp_path / "cache.jsonl")
    out = seg("fineweb_edu", "\n", [["Wrapped", "line. Done."], ["Bad", "two"], ["Wrapped", "line. Done."]])
    assert out == [[(0, "Wrapped line.", "llm"), (0, "Done.", "llm")], [(0, "Bad", "sat"), (1, "two", "sat")], out[0]]
    assert len((tmp_path / "cache.jsonl").read_text().splitlines()) == 2   # the identical chunk shared one request
    assert Segmenter(cfg, tmp_path / "cache.jsonl").cache == seg.cache


def test_source_repairs():
    assert decode_qp("new role fo=\nr=20 Global =3D 5") == "new role for  Global = 5"
    assert decode_qp("a =\nb") == "a =\nb"   # no quoted-printable codes: left alone
    assert join_urls("NPR, www. npr. org/story. php? storyId=9. Next. Common") == \
        "NPR, www.npr.org/story.php?storyId=9. Next. Common"


def test_fill_assigns_whole_documents_until_each_target():
    docs = [(f"d{i}", ["s"] * n) for i, n in enumerate([3, 4, 2, 5, 3])]
    assert [(split, key) for split, key, _ in fill(docs, {"train": 6, "test": 4})] == \
        [("train", "d0"), ("train", "d1"), ("test", "d2"), ("test", "d3")]   # 7 >= 6, then 7 >= 4
    with pytest.raises(ValueError):
        list(fill(docs, {"train": 100}))


def test_unique_drops_exact_repeats():
    docs = [("a", ["Hi. Yo."]), ("b", ["Hi. Yo."]), ("c", ["Hi.", "Yo."])]
    assert [key for key, _ in unique(docs, tuple)] == ["a", "c"]   # same text in other paragraphs is a different doc


def test_ftfy_repairs_encoding_but_keeps_quote_style():
    assert fix_text("costs Â£490m &amp; more", FTFY) == "costs £490m & more"
    assert fix_text("“Hi,” she said – ｈｉ.", FTFY) == "“Hi,” she said – ｈｉ."


def test_detokenize_writingprompts():
    assert detokenize("`` I do n't know'' he said. `` Ca n't you?''") == '"I don\'t know" he said. "Can\'t you?"'
    assert detokenize("“ Hey! ” she said.") == "“Hey!” she said."


def test_venue_year():
    assert venue_year("ICLR_cc_2018_Conference") == 2018 and venue_year("NoDaLiDa_2023_Conference") == 2023
    assert venue_year("unknown_venue") is None and venue_year(None) is None
