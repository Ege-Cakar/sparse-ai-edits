"""Paired Δ test: does a language model prefer AI-edited sentences to their human originals?

Pairs: delta.n_pairs test-split edit versions drawn with delta.seed from every editor in versions.parquet (the held-out
one included; sentences that are empty in either version are left out), each with its human original. Each sentence is
scored alone, after one prefix token (EOS for pretrain.py's models, which see documents separated by EOS), as its mean
per-token log-probability. Δ = mean over pairs of [log p(edited) - log p(original)]: positive means the model prefers
the AI-edited version. Reported pooled, per source and per editor, with standard errors and the mean log p of the
originals. pairs() and sentence_logprob() take any causal LM and its tokenizer.

For every run in pretrain/ with model.pt and no delta.json, writes pretrain/<run>/delta.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from . import cli, path
from .pretrain import llama


def pairs(versions, n, seed):
    """n edit versions of test-split sentences with their human original (column original)."""
    h = versions[versions.editor == "human"].set_index("sentence_id").text
    e = versions[(versions.split == "test") & (versions.editor != "human")]
    e = e.assign(original=h.loc[e.sentence_id].to_numpy())
    e = e[(e.text.str.strip() != "") & (e.original.str.strip() != "")]
    return e.sample(n, random_state=seed).reset_index(drop=True)


@torch.no_grad()
def sentence_logprob(model, tok, texts, bs, prefix_id):
    """Mean per-token log-probability of each text, scored alone after the token prefix_id, in input order."""
    model.eval()
    device = next(model.parameters()).device
    enc = [[prefix_id] + ids for ids in tok(list(texts), add_special_tokens=False).input_ids]
    out, order = np.empty(len(enc)), np.argsort([len(e) for e in enc])
    for i in range(0, len(enc), bs):
        idx = order[i:i + bs]
        width = max(len(enc[j]) for j in idx)
        ids = torch.full((len(idx), width), prefix_id, dtype=torch.long)
        mask = torch.zeros(len(idx), width)
        for k, j in enumerate(idx):
            ids[k, :len(enc[j])], mask[k, :len(enc[j])] = torch.tensor(enc[j]), 1
        ids, mask = ids.to(device), mask.to(device)
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
        lp = logits.log_softmax(-1).gather(-1, ids[:, 1:, None])[..., 0] * mask[:, 1:]
        out[idx] = (lp.sum(1) / mask[:, 1:].sum(1)).cpu().numpy()
    return out


def summary(p, lp_edit, lp_orig):
    """Δ pooled, per source and per editor (mean, standard error, pairs), and the originals' mean log p."""
    d = p.assign(delta=lp_edit - lp_orig)
    one = lambda g: {"delta": g.delta.mean(), "se": g.delta.std() / np.sqrt(len(g)), "pairs": len(g)}
    return {"pooled": one(d), "logp_original": float(np.mean(lp_orig)), "logp_edited": float(np.mean(lp_edit)),
            "source": {k: one(g) for k, g in d.groupby("source")}, "editor": {k: one(g) for k, g in d.groupby("editor")}}


def main():
    cfg, _ = cli(__doc__)
    c, d = cfg["pretrain"], cfg["delta"]
    tok = AutoTokenizer.from_pretrained(c["tokenizer"])
    p = pairs(pd.read_parquet(path(cfg, "versions.parquet"),
                              columns=["sentence_id", "source", "split", "editor", "text"]), d["n_pairs"], d["seed"])
    for run in sorted(Path(cfg["data_dir"], "pretrain").glob("*/model.pt")):
        out = run.parent / "delta.json"
        if out.exists():
            continue
        m = json.loads((run.parent / "metrics.json").read_text())
        model = llama(m["settings"], m["settings"]["size"]).cuda()
        model.load_state_dict(torch.load(run, map_location="cuda"))
        lp_o = sentence_logprob(model, tok, p.original, d["batch_size"], tok.eos_token_id)
        lp_e = sentence_logprob(model, tok, p.text, d["batch_size"], tok.eos_token_id)
        out.write_text(json.dumps(summary(p, lp_e, lp_o) | {"arm": m["arm"], "size": m["size"], "seed": m["seed"]},
                                  indent=1))
        print(f"{run.parent.name}: delta {np.mean(lp_e - lp_o):+.4f}", flush=True)


if __name__ == "__main__":
    main()
