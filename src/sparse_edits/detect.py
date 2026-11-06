"""Supervised detectors of AI-edited sentences: the model search of docs/review.md item 33.

Run kinds (detector.runs in the config; an entry expands over its lrs x weightings into run ids, listed by --list):
- embed: a frozen sentence-embedding model and a class-weighted logistic regression on its embeddings.
- sentence: a backbone and a linear head on the pooled tokens of one sentence (encoders fully fine-tuned, decoders
  with LoRA); weighted cross-entropy, weighting "global" (inverse class frequency) or "per_sentence" (each human
  sentence 1, each of its K edits 1/K).
- document: the same on document chunks of at most chunk_tokens tokens, cut at sentence boundaries (the remainder
  is the next chunk), one label per sentence, global class weights. Encoders pool each sentence's tokens; decoders
  (echo: true) read the chunk twice under their usual causal attention and pool each sentence's tokens in the second
  copy, which have seen the whole chunk (echo embeddings, Springer et al. 2024).
Data: train documents (a seeded search_frac of them, or the run's train_frac) and the training editors' accepted
edits, both wordings; document versions are, per document, one all-human version and, per training editor-wording and
theta, m = round(n^(1 - theta)) random positions given that editor's accepted edit (a position without one stays
human). Val:
the val split (document runs: a seeded val_frac of its documents, versions saved once to detector/doc_val.parquet),
training editors only; the held-out editor is never read. Writes detector/<run>/: metrics.json (val loss, AUROC,
TPR at detector.fpr; pooled, per editor and per source, and the run's settings), val_scores.parquet and, for neural
runs, model.pt (the trained weights), plus model_epoch<E>.pt and val_scores_epoch<E>.parquet, the best of them by the
end of each epoch E; resumes from detector/<run>/ckpt.pt after a preemption. --score SPLIT ...
scores every version of those splits with a sentence run's model.pt into scores/<run>.parquet (version_id, score), the
input of calibrate.py; with --epoch E, with model_epoch<E>.pt into scores/<run>_epoch<E>.parquet.
"""
import json
import shutil
import time
from bisect import bisect_right

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer, get_scheduler

from . import cli, path
from .sources import SOURCES


def training_editors(cfg):
    skip = set(cfg["classifier"].get("exclude_editors", []))
    return [e["name"] for e in cfg["edit"]["editors"]
            if not e["heldout"] and e.get("train_classifier", True) and e["name"] not in skip]


def expand(cfg):
    """Run id -> settings (detector.defaults, then the entry, then one lr and weighting)."""
    out = {}
    for name, r in cfg["detector"]["runs"].items():
        ws = r.get("weightings", ["global"])
        for lr in r.get("lrs", [None]):
            for w in ws:
                rid = name + (f"_lr{lr:g}" if lr else "") + (f"_{w}" if len(ws) > 1 else "")
                out[rid] = cfg["detector"]["defaults"] | r | {"name": rid, "lr": lr, "weighting": w}
    return out


def doc_sample(sents, split, frac, seed):
    docs = sents.loc[sents.split == split, "doc_id"].drop_duplicates()
    return set(docs.sample(frac=frac, random_state=seed) if frac < 1 else docs)


def sentence_examples(cfg, versions, split, docs):
    v = versions[(versions.split == split) & versions.doc_id.isin(docs)
                 & versions.editor.isin(["human"] + training_editors(cfg))]
    return v.assign(label=(v.editor != "human").astype(int)).reset_index(drop=True)


def class_weights(labels):
    """Inverse class frequency, scaled so the weights average 1."""
    labels = pd.Series(labels)
    return labels.map(len(labels) / (2 * labels.value_counts())).to_numpy()


def sentence_weights(df, how):
    if how == "global":
        return class_weights(df.label)
    k = df[df.label == 1].groupby("sentence_id").size()
    return np.where(df.label == 1, 1 / df.sentence_id.map(k).fillna(1), 1.0)


def metrics(df, fprs):
    """df: score (log-odds of edited), label, editor, source. Each TPR uses the human scores of the same rows."""
    def one(d):
        h, e = d.score[d.label == 0].to_numpy(), d.score[d.label == 1].to_numpy()
        if not len(h) or not len(e):
            return {"n_human": len(h), "n_edited": len(e)}
        return {"n_human": len(h), "n_edited": len(e), "auroc": roc_auc_score(d.label, d.score)} | {
            f"tpr@{f}": float((e > np.quantile(h, 1 - f, method="higher")).mean()) for f in fprs}
    bce = F.binary_cross_entropy_with_logits(torch.tensor(df.score.to_numpy(), dtype=torch.float64),
                                             torch.tensor(df.label.to_numpy(), dtype=torch.float64), reduction="none")
    human = df[df.label == 0]
    return {"val_loss": float(np.average(bce.numpy(), weights=class_weights(df.label))), "pooled": one(df),
            "editor": {e: one(pd.concat([human, g])) for e, g in df[df.label == 1].groupby("editor")},
            "source": {s: one(g) for s, g in df.groupby("source")}}


class Docs:
    """Documents as sentence lists, the training editors' accepted edits, and the versions to label (drawn with
    seed, or given)."""

    def __init__(self, cfg, sents, versions, doc_ids, seed, given=None):
        s = sents[sents.doc_id.isin(doc_ids)].sort_values(["doc_id", "sent_idx"])
        self.docs = dict(list(s.groupby("doc_id")))
        eds = training_editors(cfg)
        e = versions[versions.doc_id.isin(doc_ids) & versions.editor.isin(eds)]
        self.edit = dict(zip(zip(e.editor, e.sentence_id), e.text))
        if given is not None:
            self.versions = given.reset_index(drop=True)
            return
        rng, rows = np.random.default_rng(seed), []
        for d, g in self.docs.items():
            n = len(g)
            rows.append((d, "human", 0.0, []))
            rows += [(d, ed, th, sorted(rng.choice(n, round(n ** (1 - th)), replace=False).tolist()))
                     for ed in eds for th in cfg["phase"]["thetas"]]
        self.versions = pd.DataFrame(rows, columns=["doc_id", "editor", "theta", "positions"])

    def sentences(self, v):
        """(texts, labels, separator before each sentence) of version v."""
        d, ed, pos = self.versions.loc[v, ["doc_id", "editor", "positions"]]
        g, pos = self.docs[d], set(pos)
        edited = [j in pos and (ed, sid) in self.edit for j, sid in enumerate(g.sentence_id)]
        texts = [self.edit[(ed, sid)] if x else t for x, sid, t in zip(edited, g.sentence_id, g.text)]
        para = g.para_idx.to_numpy()
        seps = [""] + [" " if a == b else SOURCES[g.source.iat[0]][1] for a, b in zip(para[:-1], para[1:])]
        return texts, [int(x) for x in edited], seps

    def chunks(self, tok, budget):
        """(version, first sentence, end) per chunk, sentences kept whole, each chunk under budget tokens counting
        every sentence tokenized alone plus one for its separator; and the label counts (human, edited)."""
        out, counts = [], np.zeros(2, int)
        for v in range(len(self.versions)):
            texts, labels, _ = self.sentences(v)
            counts += np.bincount(labels, minlength=2)
            a, n = 0, 0
            for j, ids in enumerate(tok(texts, add_special_tokens=False).input_ids):
                if n and n + len(ids) + 1 > budget:
                    out.append((v, a, j))
                    a, n = j, 0
                n += len(ids) + 1
            out.append((v, a, len(texts)))
        return out, counts


def pad(seqs, value):
    return torch.nn.utils.rnn.pad_sequence([torch.as_tensor(s) for s in seqs], batch_first=True, padding_value=value)


def sentence_batch(tok, run, texts):
    """Inputs for single sentences; spans marks the tokens pooled (mean: all; cls: the first; last: the last)."""
    enc = tok(list(texts), truncation=True, max_length=run["max_tokens"], padding=True, return_tensors="pt")
    m = enc.attention_mask
    spans = torch.where(m.bool(), 0, -1)
    if run["pooling"] in ("cls", "last"):
        spans = torch.full_like(m, -1)
        spans[torch.arange(len(m)), 0 if run["pooling"] == "cls" else m.sum(1) - 1] = 0   # padding is on the right
    return dict(input_ids=enc.input_ids, attention_mask=m, spans=spans)


def chunk_batch(tok, run, docs, items):
    """Inputs and per-sentence labels (-1: padding, or cut off by truncation) for chunks (version, first, end). A token
    belongs to the sentence its last character falls in; pooling "last" keeps only each sentence's last token. With
    echo, the chunk is read twice and only the second copy is pooled."""
    ids, spans, labels = [], [], []
    for v, a, b in items:
        texts, labs, seps = (x[a:b] for x in docs.sentences(v))
        text, starts, ends = "", [], []
        for s, t in zip([""] + seps[1:], texts):
            starts.append(len(text) + len(s))
            text += s + t
            ends.append(len(text))

        def encode(special):
            enc = tok(text, add_special_tokens=special, return_offsets_mapping=True, truncation=True,
                      max_length=run["chunk_tokens"])
            sp = [j if e > s and j >= 0 and e <= ends[j] else -1
                  for j, (s, e) in zip((bisect_right(starts, e - 1) - 1 for _, e in enc.offset_mapping),
                                       enc.offset_mapping)]
            if run["pooling"] == "last":
                sp = [j if j >= 0 and (k + 1 == len(sp) or sp[k + 1] != j) else -1 for k, j in enumerate(sp)]
            return list(enc.input_ids), sp
        i, sp = encode(True)
        if run.get("echo"):
            again, sp = encode(False)
            i, sp = i + again, [-1] * len(i) + sp
        present = set(sp)
        ids.append(i), spans.append(sp), labels.append([lab if j in present else -1 for j, lab in enumerate(labs)])
    out = dict(input_ids=pad(ids, tok.pad_token_id), spans=pad(spans, -1), labels=pad(labels, -1))
    k = torch.arange(out["input_ids"].shape[1])
    return out | dict(attention_mask=(k[None] < torch.tensor([len(x) for x in ids])[:, None]).long())


class Detector(torch.nn.Module):
    """A backbone and a linear head on pooled spans: span s of a row averages the hidden states of its tokens; the
    score is logit(edited) - logit(human)."""

    def __init__(self, run):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(run["model"], revision=run.get("revision"),
                                                  dtype=torch.bfloat16 if run.get("lora") else torch.float32,
                                                  attn_implementation=run.get("attn", "sdpa"))
        if run.get("grad_ckpt"):
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if run.get("lora"):
            from peft import LoraConfig, get_peft_model
            self.backbone = get_peft_model(self.backbone, LoraConfig(**run["lora"]))
        self.head = torch.nn.Linear(self.backbone.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, spans):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        a = F.one_hot(spans + 1, int(spans.max()) + 2)[..., 1:].to(h.dtype)   # (batch, token, span)
        logits = self.head(((a.transpose(1, 2) @ h) / a.sum(1).clamp(min=1)[..., None]).float())
        return logits[..., 1] - logits[..., 0]


def cuda(d):
    return {k: v.cuda() for k, v in d.items()}


def load_trainable(model, state):
    """Load saved trainable weights (fit's weights()); stops unless every saved key is the model's and every trainable
    parameter was saved (base weights come from the backbone)."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    lost = sorted({k for k, p in model.named_parameters() if p.requires_grad} & set(missing))
    if unexpected or lost:
        raise SystemExit(f"saved weights do not match the model: unexpected {unexpected[:5]}, missing {lost[:5]}")


def fit(cfg, run, model, n, batch, evaluate, out, max_steps=None):
    """Weighted cross-entropy, AdamW, warmup then the schedule; batch(indices) -> (inputs, labels, weights). Evaluates
    eval_per_epoch times an epoch (evaluate(model) -> scored val rows) and keeps the weights (out/model.pt) and val
    scores (out/val_scores.parquet) with the best detector.select. The trainable weights, optimizer, schedule, step
    and eval history go to out/ckpt.pt every ckpt_every steps and after each eval. Returns the eval history, or None
    when stopped at max_steps."""
    bs, acc, key = run["batch_size"], run["grad_accum"], cfg["detector"]["select"]
    total = run["epochs"] * (n // (bs * acc))
    evals = {round(total * (k + 1) / (run["epochs"] * run["eval_per_epoch"]))
             for k in range(run["epochs"] * run["eval_per_epoch"])}
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=run["lr"], betas=run["betas"], eps=run["eps"], weight_decay=run["weight_decay"])
    sched = get_scheduler(run["scheduler"], opt, round(run["warmup"] * total), total)
    order = np.concatenate([np.random.default_rng([cfg["seed"], e]).permutation(n) for e in range(run["epochs"])])
    trainable = {k for k, p in model.named_parameters() if p.requires_grad}
    weights = lambda: {k: v for k, v in model.state_dict().items() if k in trainable}
    step, history, ckpt = 0, [], out / "ckpt.pt"
    if ckpt.exists():
        st = torch.load(ckpt, weights_only=False)
        if st["run"] != run:
            raise SystemExit(f"{ckpt} was saved with other settings ({st['run']}); move it away to start over")
        load_trainable(model, st["model"])
        opt.load_state_dict(st["opt"]), sched.load_state_dict(st["sched"])
        step, history = st["step"], st["history"]
        print(f"{run['name']}: resumed at step {step}/{total}")
    best = max((h[key] for h in history), default=-np.inf)
    if ckpt.exists() and (out / "val_scores.parquet").exists():   # from an eval the checkpoint may have missed
        best = max(best, metrics(pd.read_parquet(out / "val_scores.parquet"), cfg["detector"]["fpr"])["pooled"][key])
    model.train()
    t0 = time.time()
    while step < total and step != max_steps:
        for k in range(acc):
            i = (step * acc + k) * bs
            inputs, labels, w = batch(order[i:i + bs])
            with torch.autocast("cuda", torch.bfloat16):
                score = model(**cuda(inputs)).float()
            labels, w = labels.cuda(), w.cuda()
            m = labels >= 0
            loss = (w[m] * F.binary_cross_entropy_with_logits(score[m], labels[m].float(), reduction="none")).sum()
            (loss / w[m].sum() / acc).backward()
        torch.nn.utils.clip_grad_norm_(params, run["max_grad_norm"])
        opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
        step += 1
        if step % run["log_every"] == 0:
            print(f"{run['name']}: step {step}/{total}, loss {loss.item() / w[m].sum().item():.4f}, "
                  f"lr {sched.get_last_lr()[0]:.2e}, {time.time() - t0:.0f}s", flush=True)
        if step in evals:
            scored = evaluate(model)
            m = metrics(scored, cfg["detector"]["fpr"])
            history.append({"step": step, "val_loss": m["val_loss"]} | m["pooled"])
            print(f"{run['name']}: eval at step {step}: {json.dumps(history[-1])}", flush=True)
            if m["pooled"][key] >= best:   # temporary files first: a preemption never leaves half a file
                best = m["pooled"][key]
                torch.save(weights(), out / "model.tmp")
                scored.drop(columns=["text"], errors="ignore").to_parquet(out / "val_scores.tmp", index=False)
                (out / "model.tmp").replace(out / "model.pt"), (out / "val_scores.tmp").replace(out / "val_scores.parquet")
            if step % (total // run["epochs"]) == 0:   # the best weights by the end of this epoch, kept
                e = step // (total // run["epochs"])
                for a, b in [("model.pt", f"model_epoch{e}.pt"),
                             ("val_scores.parquet", f"val_scores_epoch{e}.parquet")]:
                    shutil.copyfile(out / a, out / f"{b}.tmp"), (out / f"{b}.tmp").replace(out / b)
            model.train()
        if step % run["ckpt_every"] == 0 or step in evals or step == max_steps:
            torch.save(dict(model=weights(), opt=opt.state_dict(), sched=sched.state_dict(), step=step,
                            history=history, run=run), out / "ckpt.tmp")
            (out / "ckpt.tmp").replace(ckpt)
    return history if step == total else None


@torch.no_grad()
def predict(model, n, batch, bs):
    """(row, span, score) for every labelled span of rows 0..n-1."""
    model.eval()
    out = []
    for i in range(0, n, bs):
        inputs, labels, _ = batch(np.arange(i, min(i + bs, n)))
        with torch.autocast("cuda", torch.bfloat16):
            s = model(**cuda(inputs)).float().cpu()
        r, j = (labels >= 0).nonzero(as_tuple=True)
        out += zip((r + i).tolist(), j.tolist(), s[r, j].tolist())
    return pd.DataFrame(out, columns=["row", "span", "score"])


def tokenizer(run):
    tok = AutoTokenizer.from_pretrained(run["model"], revision=run.get("revision"))
    tok.padding_side = "right"
    if run.get("pad_token"):
        tok.pad_token = run["pad_token"]
    return tok


def run_embed(cfg, run, train, val):
    """Embeddings standardized on train, then a class-weighted L2 logistic regression per C in run["Cs"]; the C with
    the best val detector.select is kept."""
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    m = SentenceTransformer(run["model"], revision=run.get("revision"), model_kwargs={"dtype": torch.bfloat16},
                            **run.get("st_kwargs", {}))
    m.max_seq_length = run["max_tokens"]
    emb = lambda t: m.encode(list(t), prompt=run.get("prompt"), prompt_name=run.get("prompt_name"),
                             batch_size=run["eval_batch_size"], normalize_embeddings=True, show_progress_bar=True)
    scale = StandardScaler().fit(x := emb(train.text))
    x, xv = scale.transform(x), scale.transform(emb(val.text))
    history, best, key = [], None, cfg["detector"]["select"]
    for c in run["Cs"]:
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=run["max_iter"]).fit(x, train.label)
        scored = val.assign(score=clf.decision_function(xv))
        mt = metrics(scored, cfg["detector"]["fpr"])
        history.append({"C": c, "val_loss": mt["val_loss"]} | mt["pooled"])
        print(f"{run['name']}: {json.dumps(history[-1])}", flush=True)
        best = scored if best is None or mt["pooled"][key] >= max(h[key] for h in history[:-1]) else best   # ties: later, as fit
    return best, history


def by_length(tok, run, texts):
    """Row order by token length (stable), for batches that carry little padding."""
    return np.argsort([len(x) for x in tok(list(texts), truncation=True, max_length=run["max_tokens"]).input_ids],
                      kind="stable")


def scores_of(model, run, tok, texts, order):
    """Scores of texts in their order, batched in `order` (by_length); padding is on the right and masked, so a
    sentence's score does not depend on its batch (up to floating-point rounding)."""
    t = texts[order]
    s = predict(model, len(t), lambda i: (sentence_batch(tok, run, t[i]), torch.zeros(len(i), 1), None),
                run["eval_batch_size"])
    assert (s.row.to_numpy() == np.arange(len(t))).all()          # one score per text, in batch order
    return s.score.to_numpy()[np.argsort(order)]


def run_sentence(cfg, run, train, val, out, max_steps):
    tok, model = tokenizer(run), Detector(run).cuda()
    texts, lab = train.text.to_numpy(), torch.tensor(train.label.to_numpy())
    w = torch.tensor(sentence_weights(train, run["weighting"]), dtype=torch.float32)
    vt = val.text.to_numpy()
    order = by_length(tok, run, vt)
    return fit(cfg, run, model, len(train), lambda i: (sentence_batch(tok, run, texts[i]), lab[i, None], w[i, None]),
               lambda mdl: val.assign(score=scores_of(mdl, run, tok, vt, order)), out, max_steps)


def score(cfg, run, versions, splits, out, epoch=None):
    """scores/<run>.parquet (version_id, score) for every version in splits, from the trained out/model.pt; with epoch
    E, scores/<run>_epoch<E>.parquet from out/model_epoch<E>.pt (the best weights by the end of epoch E)."""
    assert run["kind"] == "sentence", f"{run['name']}: only sentence runs keep a model to score with"
    tok, model = tokenizer(run), Detector(run).cuda()
    sfx = f"_epoch{epoch}" if epoch else ""
    load_trainable(model, torch.load(out / f"model{sfx}.pt"))
    v = versions[versions.split.isin(splits)].reset_index(drop=True)
    texts = v.text.to_numpy()
    s = scores_of(model, run, tok, texts, by_length(tok, run, texts))
    pd.DataFrame({"version_id": v.version_id, "score": s}).to_parquet(
        path(cfg, "scores", f"{run['name']}{sfx}.parquet"), index=False)
    print(f"{run['name']}: scored {len(v)} versions of {splits}")


def run_document(cfg, run, sents, versions, out, max_steps, smoke):
    d, tok = cfg["detector"], tokenizer(run)
    budget = run["chunk_tokens"] - 8   # room for special tokens
    train_frac = run.get("train_frac", d["search_frac"])
    train_docs, val_docs = (doc_sample(sents, sp, f, cfg["seed"]) for sp, f in [("train", train_frac),
                                                                                   ("val", d["val_frac"])])
    if smoke:
        train_docs, val_docs = (set(sorted(x)[:smoke]) for x in (train_docs, val_docs))
    saved = path(cfg, "detector", *(["smoke"] if smoke else []), "doc_val.parquet")
    train = Docs(cfg, sents, versions, train_docs, cfg["seed"])
    val = Docs(cfg, sents, versions, val_docs, cfg["seed"] + 1, pd.read_parquet(saved) if saved.exists() else None)
    if not saved.exists():
        val.versions.to_parquet(saved, index=False)
    (items, counts), (vitems, _) = train.chunks(tok, budget), val.chunks(tok, budget)
    w0, w1 = counts.sum() / (2 * counts)
    print(f"{run['name']}: {len(items)} train chunks from {len(train.versions)} versions, {counts} labels "
          f"(human, edited); {len(vitems)} val chunks", flush=True)

    def batch(docs, chunks, idx):
        x = chunk_batch(tok, run, docs, [chunks[i] for i in idx])
        labels = x.pop("labels")
        return x, labels, torch.where(labels == 1, w1, w0).float()

    def evaluate(model):
        s, rows = predict(model, len(vitems), lambda i: batch(val, vitems, i), run["eval_batch_size"]), []
        for r, g in s.groupby("row"):
            v, a, _ = vitems[r]
            labels, (doc, ed) = val.sentences(v)[1], val.versions.loc[v, ["doc_id", "editor"]]
            rows += [(doc, v, a + j, ed if labels[a + j] else "human", val.docs[doc].source.iat[0], labels[a + j], sc)
                     for j, sc in zip(g.span, g.score)]
        return pd.DataFrame(rows, columns=["doc_id", "version", "sent", "editor", "source", "label", "score"])
    return fit(cfg, run, Detector(run).cuda(), len(items), lambda i: batch(train, items, i), evaluate, out, max_steps)


def main():
    def args(p):
        p.add_argument("--run", help="run id (see --list)")
        p.add_argument("--list", action="store_true", help="print the run ids")
        p.add_argument("--smoke", type=int, metavar="N", help="quick check on N train and N val examples (documents "
                       "for document runs); outputs under detector/smoke/")
        p.add_argument("--max-steps", type=int, help="stop after this many optimizer steps, checkpointed")
        p.add_argument("--score", nargs="+", metavar="SPLIT", help="score these splits with the run's model.pt")
        p.add_argument("--epoch", type=int, help="with --score: use model_epoch<E>.pt, the best weights by epoch E")
    cfg, args = cli(__doc__, args)
    runs = expand(cfg)
    if args.list:
        return print("\n".join(runs))
    run, d = runs[args.run], cfg["detector"]
    out = path(cfg, "detector", *(["smoke"] if args.smoke else []), run["name"], "x").parent
    sents, versions = pd.read_parquet(path(cfg, "sentences.parquet")), pd.read_parquet(path(cfg, "versions.parquet"))
    if args.score:
        return score(cfg, run, versions, args.score, out, args.epoch)
    if run["kind"] == "document":
        history = run_document(cfg, run, sents, versions, out, args.max_steps, args.smoke)
    else:
        train, val = (sentence_examples(cfg, versions, sp, doc_sample(sents, sp, f, cfg["seed"]))
                      for sp, f in [("train", run.get("train_frac", d["search_frac"])), ("val", 1)])
        if args.smoke:
            train, val = (x.sample(args.smoke, random_state=cfg["seed"]).reset_index(drop=True) for x in (train, val))
        if run["kind"] == "embed":
            scored, history = run_embed(cfg, run, train, val)
            scored.drop(columns=["text"]).to_parquet(out / "val_scores.parquet", index=False)
        else:
            history = run_sentence(cfg, run, train, val, out, args.max_steps)
    if history is None:
        return print(f"{run['name']}: stopped at step {args.max_steps}; rerun to resume")
    best = metrics(pd.read_parquet(out / "val_scores.parquet"), d["fpr"])
    (out / "metrics.json").write_text(json.dumps(best | {"history": history, "run": run}, indent=1, default=str))
    print(f"{run['name']}: best {json.dumps(best['pooled'])}")


if __name__ == "__main__":
    main()
