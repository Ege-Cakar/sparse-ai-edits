import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from sparse_edits.continued_pretrain import best_lr, calib_pair, document_nll, eval_windows, holdout, perplexity, summarize, train


class CharTok:
    """Each character is a token (its code point mod 8); EOS is 0."""
    eos_token_id = 0

    def __call__(self, texts, add_special_tokens):
        return type("Enc", (), {"input_ids": [[ord(ch) % 8 for ch in t] for t in texts]})


class Bigram(torch.nn.Module):
    """Logits that depend only on the previous token."""
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Embedding(8, 8)
        torch.nn.init.normal_(self.table.weight, generator=torch.Generator().manual_seed(0))

    def forward(self, input_ids):
        return type("Out", (), {"logits": self.table(input_ids)})


def test_holdout_takes_a_seeded_fraction_of_each_source():
    s = pd.DataFrame({"doc_id": np.repeat(range(30), 2), "source": ["a"] * 20 + ["b"] * 40})
    h = holdout(s, 0.1, 0)
    assert len(h & set(range(10))) == 1 and len(h & set(range(10, 30))) == 2 and h == holdout(s, 0.1, 0)


def test_document_nll_assigns_each_predicted_token_to_its_document():
    docs = [("xsum", ["ab", "c"], [1.0, 1.0]), ("enron", ["dea"], [1.0]), ("xsum", ["b"], [1.0])]
    ids, doc = eval_windows(docs, CharTok(), 3)
    stream, owner = [1, 2, 3, 0, 4, 5, 1, 0, 2, 0], [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]   # documents end with EOS
    m = Bigram()
    nll, n = document_nll(m, ids, doc, 3, 2)
    lp = m.table.weight.detach().log_softmax(-1)
    manual = np.zeros(3)
    for a, t, d in zip(stream[:-1], stream[1:], owner[1:]):   # a token is scored in the document it belongs to
        manual[d] -= lp[a, t].item()
    assert n.tolist() == [3, 4, 2] and np.allclose(nll, manual)
    ppl = perplexity(nll, n, ["xsum", "enron", "xsum"])
    assert np.isclose(ppl["pooled"], math.exp(manual.sum() / 9)) and np.isclose(ppl["enron"], math.exp(manual[1] / 4))


class Preempted(Exception):
    pass


def test_train_resumed_from_its_checkpoint_matches_an_uninterrupted_run(tmp_path):
    c = {"tokens_per_step": 6, "betas": [0.9, 0.95], "weight_decay": 0.0, "warmup": 0.25, "min_lr_frac": 0.1,
         "max_grad_norm": 1.0, "evals": 2, "ckpt_minutes": 60}   # 3-token windows, 2 a step, 4 steps, evals at 2 and 4
    g = torch.Generator().manual_seed(1)
    ids, w = torch.randint(1, 8, (8, 4), generator=g), torch.rand(8, 4, generator=g)
    whole = Bigram()
    (tmp_path / "whole").mkdir(), (tmp_path / "split").mkdir()
    run = {"base": {"model": "m"}}
    history, steps, _ = train(whole, ids, w, c, 0.1, tmp_path / "whole", run, lambda m: 1.0)
    calls = []

    def preempted_at_second_eval(m):
        calls.append(1)
        if len(calls) == 2:
            raise Preempted
        return 1.0
    with pytest.raises(Preempted):
        train(Bigram(), ids, w, c, 0.1, tmp_path / "split", run, preempted_at_second_eval)
    resumed, regridded = Bigram(), {"base": {"model": "m", "sweeps": {"human": [1e-4]}}}   # sweep grids never matter
    assert train(resumed, ids, w, c, 0.1, tmp_path / "split", regridded, lambda m: 1.0)[0] == history and steps == 4
    assert torch.equal(resumed.table.weight, whole.table.weight)
    with pytest.raises(SystemExit):   # a checkpoint from other settings is never resumed
        train(Bigram(), ids, w, c, 0.1, tmp_path / "split", {"base": {"model": "other"}}, lambda m: 1.0)


def test_summarize_averages_seeds_per_arm_and_keeps_the_base_model_first(tmp_path):
    def run(d, ppl, delta, arm=None):
        (tmp_path / d).mkdir(parents=True)
        m = {"test_ppl": {"512": {"pooled": ppl + 1}, "2048": {"pooled": ppl}}, "delta": {"pooled": {"delta": delta}}}
        (tmp_path / d / "metrics.json").write_text(json.dumps(m | ({"arm_name": arm} if arm else {})))
    run("p/base", 20.0, 0.3)
    run("p/ctx2048_human_seed0", 18.0, 0.2, "human"), run("p/ctx2048_human_seed1", 19.0, 0.4, "human")
    run("p/ctx2048_mixture-mask_seed0", 17.0, 0.1, "mixture-mask")
    run("p/sweep/ctx2048_lr0.001", 1.0, 9.0, "human")   # a sweep run, never in the summary
    t = summarize(tmp_path, ["mixture-mask", "human"]).set_index(["arm", "metric"])
    assert list(t.index.get_level_values("arm").unique()) == ["base", "mixture-mask", "human"]
    assert t.loc[("human", "ppl_native")].tolist()[1:] == [18.5, pytest.approx(0.5 ** 0.5), 2]
    assert t.loc[("human", "delta"), "mean"] == pytest.approx(0.3) and t.loc[("base", "ppl_512"), "seeds"] == 1


def test_rate_scales_each_steps_lr_by_its_mean_weight(tmp_path):
    c = {"tokens_per_step": 6, "betas": [0.9, 0.95], "weight_decay": 0.0, "warmup": 0.25, "min_lr_frac": 0.1,
         "max_grad_norm": 1.0, "evals": 1, "ckpt_minutes": 60}
    ids = torch.randint(1, 8, (8, 4), generator=torch.Generator().manual_seed(1))
    half, whole = Bigram(), Bigram()
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    train(half, ids, torch.full((8, 4), 0.5), c, 0.1, tmp_path / "a", {}, lambda m: 1.0, rate=True)
    train(whole, ids, torch.ones(8, 4), c, 0.05, tmp_path / "b", {}, lambda m: 1.0)   # weight 1/2 = half the lr
    assert torch.allclose(half.table.weight, whole.table.weight, atol=1e-6)


def test_best_lr_takes_the_lowest_held_out_perplexity_inside_the_grid(tmp_path):
    def sweep(name, ppl):
        for lr, x in ppl.items():
            (tmp_path / "sweep" / f"{name}_ctx2048_lr{lr:g}").mkdir(parents=True)
            (tmp_path / "sweep" / f"{name}_ctx2048_lr{lr:g}" / "metrics.json").write_text(json.dumps({"holdout_ppl": x}))
    sweep("human", {1e-4: 13.0, 2e-4: 12.0, 4e-4: 12.5})
    sweep("human-repeated", {1e-4: 14.0, 2e-4: 13.0, 4e-4: 12.5})
    assert best_lr([1e-4, 2e-4, 4e-4], tmp_path, "human", 2048) == 2e-4
    with pytest.raises(SystemExit):   # best at the edge
        best_lr([1e-4, 2e-4, 4e-4], tmp_path, "human-repeated", 2048)
    with pytest.raises(SystemExit):   # unfinished
        best_lr([1e-4, 2e-4, 4e-4, 8e-4], tmp_path, "human", 2048)


def test_calib_pair_draws_one_rewrite_per_document_and_matches_tokens_per_source():
    sents = pd.DataFrame({"sentence_id": range(5), "source": ["xsum", "xsum", "xsum", "enron", "xsum"],
                          "doc_id": [1, 1, 2, 3, 4], "split": ["calib"] * 4 + ["train"], "sent_idx": [0, 1, 0, 0, 0],
                          "para_idx": 0, "text": ["aaaa", "bbbb", "cc", "ddd", "eeee"]})
    rw = pd.DataFrame({"generator": ["a", "b", "a", "a", "b", "q", "a"], "doc_id": [1, 1, 2, 3, 3, 2, 9],
                       "source": ["xsum", "xsum", "xsum", "enron", "enron", "xsum", "xsum"],
                       "split": ["calib"] * 6 + ["test"], "sent_idx": 0, "para_idx": 0,
                       "text": ["a1", "b1", "a2a2a2a2", "a3a3a3a3a3", "b3b3b3b3b3", "q2", "t9"]})
    rewrites, human = calib_pair(sents, rw, ["a", "b"], CharTok(), np.random.default_rng(0))
    tokens = lambda docs, src: sum(len("".join(ts)) + 1 for s, ts, _ in docs if s == src)   # characters and EOS
    # xsum: rewrites 3 + 9 tokens, human 10 + 3 (doc 4 is train) -> human cut to at most 12, one document;
    # enron: the 11-token rewrite does not fit within the human document's 4 tokens
    assert [s for s, _, _ in rewrites] == ["xsum", "xsum"] and {"".join(ts) for _, ts, _ in rewrites} - {"a2a2a2a2"} \
        <= {"a1", "b1"}
    assert tokens(human, "enron") == 4 and tokens(human, "xsum") <= 12 and len(human) == 2
    assert calib_pair(sents, rw, ["a", "b"], CharTok(), np.random.default_rng(0)) == (rewrites, human)
