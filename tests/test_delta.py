import numpy as np
import pandas as pd
import torch

from sparse_edits.delta import pairs, sentence_logprob, summary


def test_pairs_take_test_edits_with_their_nonempty_original():
    v = pd.DataFrame({"sentence_id": [1, 1, 2, 2, 3, 3], "source": "xsum",
                      "split": ["test", "test", "test", "test", "train", "train"],
                      "editor": ["human", "a", "human", "a", "human", "a"],
                      "text": ["one", "One!", " ", "two", "three", "Three!"]})
    p = pairs(v, 1, 0)
    assert p.text.tolist() == ["One!"] and p.original.tolist() == ["one"]   # 2 has an empty original, 3 is train


class CharTok:
    def __call__(self, texts, add_special_tokens):
        return type("Enc", (), {"input_ids": [[ord(ch) % 8 for ch in t] for t in texts]})


class Bigram(torch.nn.Module):
    """Logits that depend only on the previous token."""
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Embedding(8, 8)
        torch.nn.init.normal_(self.table.weight, generator=torch.Generator().manual_seed(0))

    def forward(self, input_ids, attention_mask):
        return type("Out", (), {"logits": self.table(input_ids)})


def test_sentence_logprob_is_the_mean_token_logprob_and_ignores_padding():
    m, tok = Bigram(), CharTok()
    texts = ["abc", "a", "abcdefg"]
    batched = sentence_logprob(m, tok, texts, 3, 0)
    alone = [sentence_logprob(m, tok, [t], 1, 0)[0] for t in texts]
    lp = m.table.weight.log_softmax(-1)
    ids = [0] + [ord(ch) % 8 for ch in "abc"]
    manual = np.mean([lp[a, b].item() for a, b in zip(ids[:-1], ids[1:])])
    assert np.allclose(batched, alone, atol=1e-6) and np.isclose(batched[0], manual, atol=1e-6)
    s = summary(pd.DataFrame({"source": ["x", "y"], "editor": ["a", "a"]}), np.array([-1.0, -2.0]), np.array([-1.5, -2.0]))
    assert np.isclose(s["pooled"]["delta"], 0.25) and s["source"]["x"]["delta"] == 0.5
