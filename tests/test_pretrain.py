import numpy as np
import pandas as pd

from sparse_edits.pretrain import documents, pack, sentence_table


def corpus():
    rows = [(0, "xsum", 0, "train", j, j // 2, f"t{j}") for j in range(4)] + \
           [(10 + j, "xsum", 1, "calib", j, 0, f"c{j}") for j in range(2)]
    sents = pd.DataFrame(rows, columns=["sentence_id", "source", "doc_id", "split", "sent_idx", "para_idx", "text"])
    sents.loc[0:3, "sentence_id"] = range(4)
    edits = {(ed, sid): (f"{ed}{sid}", r) for ed, r in [("a", 0.25), ("b", 1.5)] for sid in range(4)}
    return sentence_table(sents), edits


def test_separators_follow_paragraphs():
    s, _ = corpus()
    assert s[s.doc_id == 0].sep.tolist() == ["", " ", "\n", " "]   # xsum paragraphs are joined with "\n"


def test_human_mixture_and_all_edited_documents():
    s, edits = corpus()
    human = documents(s, edits, "human", "none", ["a", "b"], [0.5], np.random.default_rng(0))
    assert [d[1] for d in human] == [["t0", " t1", "\nt2", " t3"], ["c0", " c1"]] and all(
        w == 1 for d in human for w in d[2])
    mix = documents(s, edits, "mixture", "mask", ["a", "b"], [0.5], np.random.default_rng(0))
    edited = [j for j, w in enumerate(mix[0][2]) if w == 0]
    assert len(edited) == round(4 ** 0.5) and mix[1][2] == [1.0, 1.0]   # calib stays human
    assert all(mix[0][1][j].lstrip(" \n")[0] in "ab" for j in edited)
    full = documents(s, edits, "all_edited", "strength", ["a", "b"], [0.5], None)
    assert len(full) == 2 and full[0][1] == ["a0", " a1", "\na2", " a3"]   # one copy per editor, no calib
    assert full[0][2] == [0.75] * 4 and full[1][2] == [0.0] * 4           # 1 - min(actual_r, 1)
    rep = documents(s, edits, "repeat", "none", ["a", "b"], [0.5], None)
    assert [d[1] for d in rep] == [human[0][1]] * 2                         # the same passes, without the edits


class CharTok:
    """Each character is a token (its code point); EOS is 0."""
    eos_token_id = 0

    def __call__(self, texts, add_special_tokens):
        return type("Enc", (), {"input_ids": [[ord(ch) for ch in t] for t in texts]})


def test_pack_aligns_weights_and_sources_with_tokens():
    docs = [("xsum", ["ab", " c"], [0.5, 1.0]), ("enron", ["de"], [0.0])]
    ids, w, src, names = pack(docs, CharTok(), 3, [0, 1])
    flat_ids = [ord(ch) for ch in "ab c"] + [0] + [ord(ch) for ch in "de"] + [0]
    assert ids[:, :-1].flatten().tolist() == flat_ids[:6] and ids[0, -1] == ids[1, 0]   # windows overlap by one
    assert w[0].tolist() == [0.5, 0.5, 1.0, 1.0] and w[1].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert names == ["enron", "xsum"] and src[1].tolist() == [1, 1, 0, 0]
