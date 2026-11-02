import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from sparse_edits.detect import Docs, chunk_batch, expand, metrics, sentence_batch, sentence_weights


class Tok:
    """Whitespace tokenizer with the fast-tokenizer interface the detectors use; token 1 is BOS."""
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, truncation=False, max_length=None,
                 padding=False, return_tensors=None):
        if isinstance(text, list):
            out = [self(t, add_special_tokens) for t in text]
            if not padding:
                return SimpleNamespace(input_ids=[o.input_ids for o in out])
            n = max(len(o.input_ids) for o in out)
            return SimpleNamespace(input_ids=torch.tensor([o.input_ids + [0] * (n - len(o.input_ids)) for o in out]),
                                   attention_mask=torch.tensor([[1] * len(o.input_ids) + [0] * (n - len(o.input_ids))
                                                                for o in out]))
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        ids = [2 + len(text[a:b]) for a, b in spans]
        if add_special_tokens:
            ids, spans = [1] + ids, [(0, 0)] + spans
        n = max_length if truncation else None
        return SimpleNamespace(input_ids=ids[:n], offset_mapping=spans[:n])


CFG = {"edit": {"editors": [{"name": "e", "heldout": False}, {"name": "q", "heldout": True}]},
       "classifier": {}, "phase": {"thetas": [0.5]}}
SENTS = pd.DataFrame(dict(sentence_id=[10, 11, 12], doc_id=0, sent_idx=[0, 1, 2], para_idx=[0, 0, 1], source="xsum",
                          split="train", text=["A b.", "C d e.", "F."]))
VERSIONS = pd.DataFrame(dict(sentence_id=[11, 12, 11], doc_id=0, editor=["e", "e", "q"], text=["X y.", "Z.", "Q."]))


def test_versions_edit_only_accepted_positions_and_never_the_heldout_editor():
    docs = Docs(CFG, SENTS, VERSIONS, {0}, seed=0)
    assert docs.versions.editor.tolist() == ["human", "e"] and len(docs.versions.positions[1]) == round(3 ** 0.5)
    docs.versions["positions"] = [[], [0, 1]]   # sentence 10 has no accepted edit, so it stays human
    assert docs.sentences(1) == (["A b.", "X y.", "F."], [0, 1, 0], ["", " ", "\n"])
    assert docs.sentences(0)[1] == [0, 0, 0]


def test_chunks_keep_sentences_whole_and_cover_the_document():
    docs = Docs(CFG, SENTS, VERSIONS, {0}, seed=0)
    chunks, counts = docs.chunks(Tok(), budget=6)   # sentence costs (tokens + 1): 3, 4, 2
    assert [c for c in chunks if c[0] == 0] == [(0, 0, 1), (0, 1, 3)] and counts.sum() == 6


def test_chunks_pool_each_sentence_and_echo_pools_the_second_copy():
    docs = Docs(CFG, SENTS, VERSIONS, {0}, seed=0)
    enc = chunk_batch(Tok(), {"chunk_tokens": 100, "pooling": "mean"}, docs, [(0, 0, 3)])
    assert enc["spans"][0].tolist() == [-1, 0, 0, 1, 1, 1, 2] and enc["labels"][0].tolist() == [0, 0, 0]
    x = chunk_batch(Tok(), {"chunk_tokens": 100, "pooling": "last", "echo": True}, docs, [(0, 0, 3), (0, 2, 3)])
    assert x["input_ids"][0].tolist()[1:7] == x["input_ids"][0].tolist()[7:]   # the chunk, then the chunk again
    assert x["spans"][0].tolist() == [-1] * 7 + [-1, 0, -1, -1, 1, 2]   # each sentence's last token, second copy
    assert x["attention_mask"][1].tolist() == [1] * 3 + [0] * 10 and x["labels"][1].tolist() == [0, -1, -1]
    cut = chunk_batch(Tok(), {"chunk_tokens": 4, "pooling": "mean"}, docs, [(0, 0, 3)])   # truncation drops sentence 2
    assert cut["labels"][0].tolist() == [0, 0, -1]


def test_sentence_pooling_marks_the_right_tokens():
    b = sentence_batch(Tok(), {"max_tokens": 10, "pooling": "last"}, ["A b c.", "D."])
    assert b["spans"].tolist() == [[-1, -1, -1, 0], [-1, 0, -1, -1]]
    assert sentence_batch(Tok(), {"max_tokens": 10, "pooling": "mean"}, ["D."])["spans"].tolist() == [[0, 0]]


def test_weightings_and_metrics():
    df = pd.DataFrame(dict(sentence_id=[1, 1, 1, 2, 2], label=[0, 1, 1, 0, 1]))
    assert np.allclose(sentence_weights(df, "per_sentence"), [1, 0.5, 0.5, 1, 1])
    assert np.allclose(sentence_weights(df, "global"), [5 / 4, 5 / 6, 5 / 6, 5 / 4, 5 / 6])
    scored = pd.DataFrame(dict(score=list(range(100)) + [98.5, 50.0], label=[0] * 100 + [1, 1],
                               editor=["human"] * 100 + ["a", "b"], source="s"))
    m = metrics(scored, [0.01])   # the 99th human percentile is 99, so neither edit passes it
    assert m["pooled"]["tpr@0.01"] == 0 and m["editor"]["a"]["n_human"] == 100


def test_runs_expand_over_learning_rates_and_weightings():
    cfg = {"detector": {"defaults": {"epochs": 1}, "runs": {
        "enc": {"kind": "sentence", "lrs": [1e-5, 2e-5], "weightings": ["global", "per_sentence"]},
        "emb": {"kind": "embed"}}}}
    runs = expand(cfg)
    assert list(runs) == ["enc_lr1e-05_global", "enc_lr1e-05_per_sentence", "enc_lr2e-05_global",
                          "enc_lr2e-05_per_sentence", "emb"]
    assert runs["enc_lr2e-05_global"]["lr"] == 2e-5 and runs["emb"]["epochs"] == 1
