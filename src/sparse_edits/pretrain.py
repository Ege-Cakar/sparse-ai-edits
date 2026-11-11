"""Small Llama-style LMs pretrained on the corpus's documents with per-token loss weights.

Arms (pretrain.arms) pair a data recipe with a weight:
- data human: the human documents of pretrain.train_splits.
- data mixture: those documents, each train or val document with theta drawn from phase.thetas, one training
  editor-wording drawn uniformly, and m = round(n^(1 - theta)) random sentences replaced by its accepted edit (a
  position without one stays human); calib documents have no edits and stay human.
- data all_edited: each train or val document once per training editor-wording, every sentence with an accepted edit
  replaced by it.
- data repeat: each train or val human document once per training editor-wording (the documents and passes of
  all_edited, without the edits).
An arm's passes (all_edited, repeat) caps the copies per document: all_edited then uses that many editor-wordings
drawn per document with the seed, repeat that many human copies.
- data fresh: the train and val human documents plus fresh/sentences.parquet's (fresh.py: new documents from the
  same sources, about pretrain.fresh_multiple times as many words where a source has them).
- weight none: 1 everywhere; mask: 0 on an edited sentence, 1 on a human one; strength: 1 - min(actual_r, 1).
A sentence's weight applies to its tokens (end-of-document tokens get 1). The loss is the weighted token cross-entropy,
divided by the tokens in the step times the mean weight of the arm's training tokens (weighted risk over the corpus).
Documents are tokenized sentence by sentence with their separators, shuffled with the seed, joined with EOS and cut
into seq_len + 1 windows. Perplexity is measured on pretrain.eval_split's human documents, pooled and per source.
Writes pretrain/<arm>_<size>_seed<seed>/: metrics.json and model.pt; runs that have metrics.json are skipped.
"""
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from . import cli, path
from .detect import training_editors
from .sources import SOURCES


def sentence_table(sents):
    """sents with each sentence's separator before it: "" first, " " within a paragraph, the source's between."""
    s = sents.sort_values(["doc_id", "sent_idx"]).reset_index(drop=True)
    new_doc = s.doc_id.ne(s.doc_id.shift())
    sep = np.where(s.para_idx.eq(s.para_idx.shift()), " ", s.source.map(lambda x: SOURCES[x][1]))
    return s.assign(sep=np.where(new_doc, "", sep))


def documents(s, edits, data, weight, editors, thetas, rng, passes=None):
    """[(source, texts, weights)] per training document. s: sentence_table rows; edits: (editor, sentence_id) ->
    (text, actual_r) for the training editors' accepted edits; passes: copies per document for repeat and
    all_edited (default: one per editor-wording)."""
    w_of = {"none": lambda r: 1.0, "mask": lambda r: 0.0, "strength": lambda r: 1.0 - min(r, 1.0)}[weight]
    out = []
    for _, g in s.groupby("doc_id", sort=True):
        human = [(sep + t, 1.0) for sep, t in zip(g.sep, g.text)]
        has_edits = g.split.iat[0] != "calib"
        if data == "fresh":   # calib documents are left out, as in repeat and all_edited
            versions = [human] if has_edits else []
        elif data == "human" or (data == "mixture" and not has_edits):
            versions = [human]
        elif data == "mixture":
            ed, n = editors[rng.integers(len(editors))], len(g)
            pos = set(rng.choice(n, round(n ** (1 - thetas[rng.integers(len(thetas))])), replace=False).tolist())
            versions = [[(sep + edits[ed, sid][0], w_of(edits[ed, sid][1])) if j in pos and (ed, sid) in edits else h
                         for j, (sep, sid, h) in enumerate(zip(g.sep, g.sentence_id, human))]]
        elif has_edits and data == "repeat":   # calib documents are not used by repeat and all_edited
            versions = [human] * (passes or len(editors))
        elif has_edits:
            eds = editors if passes is None else [editors[i] for i in rng.choice(len(editors), passes, replace=False)]
            versions = [[(sep + edits[ed, sid][0], w_of(edits[ed, sid][1])) if (ed, sid) in edits else h
                         for sep, sid, h in zip(g.sep, g.sentence_id, human)] for ed in eds]
        else:
            versions = []
        out += [(g.source.iat[0], [t for t, _ in v], [w for _, w in v]) for v in versions]
    return out


def pack(docs, tok, seq_len, order):
    """(ids, weights, sources) as (windows, seq_len + 1) arrays: documents in `order`, each followed by EOS."""
    texts = sorted({t for _, ts, _ in docs for t in ts})
    ids_of = dict(zip(texts, tok(texts, add_special_tokens=False).input_ids))
    srcs = sorted({src for src, _, _ in docs})
    ids, ws, ss = [], [], []
    for k in order:
        src, ts, w = docs[k]
        for t, x in zip(ts, w):
            ids += ids_of[t]; ws += [x] * len(ids_of[t])
        ids.append(tok.eos_token_id); ws.append(1.0)
        ss += [srcs.index(src)] * (len(ids) - len(ss))
    n = (len(ids) - 1) // seq_len
    win = lambda a, dt: torch.as_tensor(np.lib.stride_tricks.sliding_window_view(
        np.asarray(a[:n * seq_len + 1], dt), seq_len + 1)[::seq_len].copy())
    return win(ids, np.int64), win(ws, np.float32), win(ss, np.int64), srcs


def windows(s, edits, a, editors, thetas, seed, tok, seq_len):
    """Arm a's training windows (ids, weights) in random order, and its documents, drawn and ordered with the seed."""
    rng = np.random.default_rng(seed)
    docs = documents(s, edits, a["data"], a["weight"], editors, thetas, rng, a.get("passes"))
    return (*shuffled_windows(docs, tok, seq_len, rng), docs)


def shuffled_windows(docs, tok, seq_len, rng):
    """(ids, weights) of docs packed in a random order, the windows then shuffled."""
    ids, w, _, _ = pack(docs, tok, seq_len, rng.permutation(len(docs)))
    perm = rng.permutation(len(ids))
    return ids[perm], w[perm]


def load(cfg, fresh):
    """(sentences, sentence_table of pretrain.train_splits, the same plus fresh/sentences.parquet or None if not fresh,
    edits, editors): edits maps (editor, sentence_id) -> (text, actual_r) for the training editors' accepted train and
    val edits."""
    c = cfg["pretrain"]
    sents = pd.read_parquet(path(cfg, "sentences.parquet"))
    v = pd.read_parquet(path(cfg, "versions.parquet"), columns=["sentence_id", "split", "editor", "text", "actual_r"])
    editors = sorted(set(training_editors(cfg)) & set(v.editor[v.split.isin(["train", "val"])]))
    e = v[v.editor.isin(editors) & v.split.isin(["train", "val"])]
    edits = {(ed, sid): (t, r) for ed, sid, t, r in zip(e.editor, e.sentence_id, e.text, e.actual_r)}
    train = sents[sents.split.isin(c["train_splits"])]
    s_fresh = sentence_table(pd.concat([train, pd.read_parquet(path(cfg, "fresh", "sentences.parquet"))])) if fresh \
        else None
    return sents, sentence_table(train), s_fresh, edits, editors


def weighted_loss(logits, targets, w, denom):
    ce = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), reduction="none")
    return (ce * w.flatten()).sum() / denom


@torch.no_grad()
def perplexity(model, ids, src, names, bs):
    model.eval()
    tot, cnt = np.zeros(len(names)), np.zeros(len(names))
    for i in range(0, len(ids), bs):
        x, s = ids[i:i + bs].cuda(), src[i:i + bs, 1:].flatten().cuda()
        with torch.autocast("cuda", torch.bfloat16):
            logits = model(x[:, :-1]).logits
        ce = F.cross_entropy(logits.flatten(0, 1).float(), x[:, 1:].flatten(), reduction="none")
        tot += torch.zeros(len(names), device="cuda").index_add_(0, s, ce).cpu().numpy()
        cnt += torch.bincount(s, minlength=len(names)).cpu().numpy()
    model.train()
    return {"pooled": math.exp(tot.sum() / cnt.sum())} | {n: math.exp(t / c) for n, t, c in zip(names, tot, cnt)}


def llama(c, z):
    """The Llama-style model of size z (pretrain.sizes entry) under pretrain settings c."""
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=c["vocab_size"], hidden_size=z["d_model"], intermediate_size=z["d_ff"],
        num_hidden_layers=z["n_layers"], num_attention_heads=z["n_heads"], num_key_value_heads=z["n_heads"],
        max_position_embeddings=c["seq_len"], tie_word_embeddings=True))


def run(cfg, arm, size, seed, s, edits, editors, tok, test):
    c, a, z = cfg["pretrain"], cfg["pretrain"]["arms"][arm], cfg["pretrain"]["sizes"][size]
    out = path(cfg, "pretrain", f"{arm}_{size}_seed{seed}", "x").parent
    if (out / "metrics.json").exists():
        return print(f"{out.name}: done, skipped")
    ids, w, docs = windows(s, edits, a, editors, cfg["phase"]["thetas"], seed, tok, c["seq_len"])
    torch.manual_seed(seed)
    model = llama(c, z).cuda()
    n_emb = model.get_input_embeddings().weight.numel()
    n_body = sum(p.numel() for p in model.parameters()) - n_emb
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": c["weight_decay"]}, {"params": rest, "weight_decay": 0}],
                            lr=c["lr"], betas=c["betas"])
    bs, steps = c["batch_size"], len(ids) // c["batch_size"]
    warm = max(1, round(c["warmup"] * steps))
    lr_at = lambda t: min(1, (t + 1) / warm) * (c["min_lr_frac"] + (1 - c["min_lr_frac"]) * 0.5 * (
        1 + math.cos(math.pi * max(0, t - warm) / max(1, steps - warm))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    mean_w = w[:steps * bs, 1:].mean().item()
    denom = bs * c["seq_len"] * mean_w
    evals = {round(steps * (k + 1) / c["evals"]) for k in range(c["evals"])}
    history, t0 = [], time.time()
    for t in range(steps):
        x, wt = ids[t * bs:(t + 1) * bs].cuda(), w[t * bs:(t + 1) * bs, 1:].cuda()
        with torch.autocast("cuda", torch.bfloat16):
            logits = model(x[:, :-1]).logits
        loss = weighted_loss(logits, x[:, 1:], wt, denom)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c["max_grad_norm"])
        opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
        if t + 1 in evals:
            ppl = perplexity(model, *test, c["eval_batch_size"])
            history.append({"step": t + 1, "train_loss": loss.item(), "test_ppl": ppl["pooled"]})
            print(f"{out.name}: step {t + 1}/{steps}, loss {loss.item():.4f}, test ppl {ppl['pooled']:.2f}, "
                  f"{time.time() - t0:.0f}s", flush=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "metrics.json").write_text(json.dumps({
        "arm": arm, "size": size, "seed": seed, "documents": len(docs), "train_tokens": steps * bs * c["seq_len"],
        "mean_weight": mean_w, "steps": steps, "params_embedding": n_emb, "params_non_embedding": n_body,
        "test_ppl": ppl, "history": history, "settings": c | {"arm": a, "size": z}}, indent=1))


def main():
    def args(p):
        p.add_argument("--arms", nargs="+", help="default: every arm in pretrain.arms")
        p.add_argument("--sizes", nargs="+", help="default: every size in pretrain.sizes")
        p.add_argument("--seeds", nargs="+", type=int, help="default: pretrain.seeds")
    cfg, args = cli(__doc__, args)
    c = cfg["pretrain"]
    tok = AutoTokenizer.from_pretrained(c["tokenizer"], revision=c.get("tokenizer_revision"))
    arms = args.arms or list(c["arms"])
    sents, s, s_fresh, edits, editors = load(cfg, any(c["arms"][a]["data"] == "fresh" for a in arms))
    test_docs = documents(sentence_table(sents[sents.split == c["eval_split"]]), {}, "human", "none", [], [], None)
    test_ids, _, test_src, names = pack(test_docs, tok, c["seq_len"], range(len(test_docs)))
    print(f"editors {editors}; {len(test_ids)} test windows of {c['seq_len'] + 1} tokens", flush=True)
    for arm in arms:
        for size in args.sizes or c["sizes"]:
            for seed in args.seeds or c["seeds"]:
                run(cfg, arm, size, seed, s_fresh if c["arms"][arm]["data"] == "fresh" else s, edits, editors, tok,
                    (test_ids, test_src, names))


if __name__ == "__main__":
    main()
