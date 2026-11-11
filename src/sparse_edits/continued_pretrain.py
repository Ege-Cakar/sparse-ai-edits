"""Continued pretraining of released LMs on the corpus with LoRA and per-token loss weights (docs/review.md item 73).

Each base in continued.bases (a pinned revision, with its own tokenizer) trains a LoRA adapter on every arm of
pretrain.arms and continued.arms with pretrain.py's documents, weights, loss and data draws (pretrain.windows), 1 epoch,
at context length continued.seq_len ("native": the base's native_len), continued.tokens_per_step tokens a step, warmup
then cosine decay as in pretrain.py. An arm with rate: true also scales each step's learning rate by the mean weight of
its tokens, so a token of weight w moves the model about as much as w tokens of weight 1 (approximately, as Adam's
momentum averages over steps; without it, Adam's updates ignore the scale of the loss and the weights only shift
emphasis between the tokens of a step). Modes:
- base: evaluates each base model before training.
- sweep: the learning rates per base and context length. Each sweep in continued.sweeps (or the base's sweeps) trains
  its pretrain.arms arm at continued.sweep_seed without a held-out continued.sweep_holdout of each source's training
  documents, at every lr of its grid, scored by perplexity on the held-out documents.
- train: every arm and seed at the best lr of the arm's sweep (continued.arms' sweep, else human); stops if that sweep
  is unfinished or its best lr is at the edge of the grid.
- summary: per base, the pooled test perplexity at each context length and the pooled Δ of the base model and of every
  arm (mean and SD over seeds); printed and written to continued_pretrain/summary.csv.
Evaluation: summed NLL and tokens per document of the eval split's human documents at every context length of
continued.eval_seq_lens (perplexity pooled and per source), and delta.py's paired Δ test on the same pairs as the
from-scratch models.
Writes continued_pretrain/<base>/base/, <base>/ctx<L>_<arm>_seed<s>/ and <base>/sweep/<sweep>_ctx<L>_lr<lr>/, each
with metrics.json; evaluated models also test_nll.parquet (doc_id, source, seq_len, nll, tokens) and delta.parquet
(version_id, logp_edited, logp_original), trained ones adapter/ (peft's save_pretrained). kempner_requeue preempts
without warning, so a run saves its LoRA weights, optimizer, schedule, step, history and RNG states to ckpt.pt every
continued.ckpt_minutes and at each eval, and resumes from it. Runs with metrics.json are skipped.
"""
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import cli, path
from .delta import pairs, sentence_logprob, summary
from .detect import load_trainable
from .pretrain import documents, load, pack, sentence_table, shuffled_windows, weighted_loss

TRAIN_KEYS = ["tokens_per_step", "lora", "betas", "weight_decay", "warmup", "min_lr_frac", "max_grad_norm", "evals"]


def holdout(s, frac, seed):
    """doc_ids of a seeded frac of each source's documents in s."""
    return set(s.drop_duplicates("doc_id").groupby("source").sample(frac=frac, random_state=seed).doc_id)


def calib_pair(sents, rewrites, generators, tok, rng):
    """(rewrites, human) documents of the calib split: each calib document's rewrite by one of generators, drawn
    uniformly among those that wrote one, and the human documents. Per source, the side with more tokens keeps whole
    documents, in a random order, only while they fit within the other side's tokens."""
    r = rewrites[(rewrites.split == "calib") & rewrites.generator.isin(generators)]
    pick = {d: sorted(g)[rng.integers(len(g))] for d, g in r.groupby("doc_id").generator.unique().items()}
    sides = [documents(sentence_table(x), {}, "human", "none", [], [], None)
             for x in (r[r.generator == r.doc_id.map(pick)], sents[sents.split == "calib"])]
    n = [np.array([sum(map(len, tok(ts, add_special_tokens=False).input_ids)) + 1 for _, ts, _ in d]) for d in sides]
    out = ([], [])
    for src in sorted({d[0] for d in sides[1]}):
        idx = [np.flatnonzero([d[0] == src for d in side]) for side in sides]
        target = min(n[k][idx[k]].sum() for k in range(2))
        for k in range(2):
            order = rng.permutation(idx[k])
            out[k].extend(sides[k][j] for j in order[np.cumsum(n[k][order]) <= target])
    return out


def labelled(docs):
    """docs (pretrain.documents) labelled by their index, so pack's source array is the document of each token."""
    return [(f"{k:07d}", ts, ws) for k, (_, ts, ws) in enumerate(docs)]


def eval_windows(docs, tok, seq_len):
    """(ids, document index per token) of docs packed in order."""
    ids, _, doc, _ = pack(labelled(docs), tok, seq_len, range(len(docs)))
    return ids, doc


@torch.no_grad()
def document_nll(model, ids, doc, n_docs, bs):
    """Summed next-token NLL and number of predicted tokens per document; doc is the document of each token of ids."""
    model.eval()
    device = next(model.parameters()).device
    nll, n = (torch.zeros(n_docs, dtype=torch.float64, device=device) for _ in range(2))
    for i in range(0, len(ids), bs):
        x, d = ids[i:i + bs].to(device), doc[i:i + bs, 1:].flatten().to(device)
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=x[:, :-1]).logits
        ce = F.cross_entropy(logits.flatten(0, 1).float(), x[:, 1:].flatten(), reduction="none").double()
        nll.index_add_(0, d, ce), n.index_add_(0, d, torch.ones_like(ce))
    model.train()
    return nll.cpu().numpy(), n.cpu().numpy()


def perplexity(nll, n, sources):
    """exp(NLL per token), pooled and per source, from per-document sums."""
    t = pd.DataFrame({"nll": nll, "n": n, "source": sources}).groupby("source").sum()
    return {"pooled": math.exp(t.nll.sum() / t.n.sum())} | {k: math.exp(v) for k, v in (t.nll / t.n).items()}


def write_json(f, d):
    f.with_suffix(".tmp").write_text(json.dumps(d, indent=1))
    f.with_suffix(".tmp").replace(f)   # a preemption never leaves half a file


def train(model, ids, w, c, lr, out, settings, evaluate, rate=False):
    """1 epoch of AdamW on model's trainable parameters over windows ids with token weights w, c["tokens_per_step"]
    tokens a step, warmup then cosine decay, and with rate each step's lr times the mean weight of its tokens;
    evaluate(model) -> perplexity, c["evals"] times. Resumes from out/ckpt.pt (stops if it was saved with other
    settings). Returns (eval history, steps, mean weight of the trained tokens)."""
    device, seq_len = next(model.parameters()).device, ids.shape[1] - 1
    bs = c["tokens_per_step"] // seq_len
    steps = len(ids) // bs
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=c["betas"], weight_decay=c["weight_decay"])
    warm = max(1, round(c["warmup"] * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1, (t + 1) / warm) * (c["min_lr_frac"] + (
        1 - c["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * max(0, t - warm) / max(1, steps - warm)))))
    mean_w = w[:steps * bs, 1:].mean().item()
    denom = bs * seq_len * mean_w
    evals = {round(steps * (k + 1) / c["evals"]) for k in range(c["evals"])}
    trainable = {k for k, p in model.named_parameters() if p.requires_grad}
    step, history, ckpt = 0, [], out / "ckpt.pt"
    if ckpt.exists():
        st = torch.load(ckpt, weights_only=False)
        same = lambda x: x | {"base": {k: v for k, v in x["base"].items() if k != "sweeps"}}   # grids never matter
        if same(st["settings"]) != same(settings):
            raise SystemExit(f"{ckpt} was saved with other settings ({st['settings']}); move it away to start over")
        load_trainable(model, st["model"])
        opt.load_state_dict(st["opt"]), sched.load_state_dict(st["sched"]), torch.set_rng_state(st["rng"])
        if st["cuda_rng"] is not None:
            torch.cuda.set_rng_state(st["cuda_rng"])   # LoRA dropout masks continue rather than repeat
        step, history = st["step"], st["history"]
        print(f"{out.name}: resumed at step {step}/{steps}", flush=True)
    model.train()
    saved = t0 = time.time()
    while step < steps:
        x, wt = ids[step * bs:(step + 1) * bs].to(device), w[step * bs:(step + 1) * bs, 1:].to(device)
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=x[:, :-1]).logits
        loss = weighted_loss(logits, x[:, 1:], wt, denom)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, c["max_grad_norm"])
        if rate:
            for g in opt.param_groups:
                g["lr"] = sched.get_last_lr()[0] * wt.mean().item()
        opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
        step += 1
        if step in evals:
            history.append({"step": step, "train_loss": loss.item(), "ppl": evaluate(model)})
            print(f"{out.name}: step {step}/{steps}, loss {loss.item():.4f}, ppl {history[-1]['ppl']:.3f}, "
                  f"{time.time() - t0:.0f}s", flush=True)
        if step in evals or time.time() - saved > 60 * c["ckpt_minutes"]:
            torch.save(dict(model={k: v for k, v in model.state_dict().items() if k in trainable},
                            opt=opt.state_dict(), sched=sched.state_dict(), step=step, history=history,
                            settings=settings, rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state() if device.type == "cuda" else None), out / "ckpt.tmp")
            (out / "ckpt.tmp").replace(ckpt)
            saved = time.time()
    return history, steps, mean_w


def base_model(b):
    return AutoModelForCausalLM.from_pretrained(b["model"], revision=b["revision"], dtype=torch.bfloat16)


def fit(cfg, b, tok, a, seed, seq_len, lr, draw, out, evaluate):
    """A LoRA adapter on base b trained on arm a (a pretrain.arms or continued.arms entry) at seed, seq_len and lr, on
    the documents draw(rng) gives with the seed's rng (then packed and shuffled with it, as pretrain.windows), saved to
    out/adapter; returns the model and the run's settings and training record."""
    c, rng = cfg["continued"], np.random.default_rng(seed)
    docs = draw(rng)
    ids, w = shuffled_windows(docs, tok, seq_len, rng)
    if a["weight"] == "strength-mean":   # the strength arm's weights with every down-weighted token at their mean
        w[w < 1] = w[w < 1].mean()
    model = base_model(b)
    torch.manual_seed(seed)   # LoRA initialization and dropout
    model = get_peft_model(model, LoraConfig(**c["lora"])).cuda()
    settings = {k: c[k] for k in TRAIN_KEYS} | {"base": b, "arm": a, "seed": seed, "seq_len": seq_len, "lr": lr}
    history, steps, mean_w = train(model, ids, w, c, lr, out, settings, lambda m: evaluate(m)["pooled"],
                                   a.get("rate", False))
    model.save_pretrained(out / "adapter")
    return model, settings | {"documents": len(docs), "steps": steps, "mean_weight": mean_w, "history": history,
                              "train_tokens": steps * (c["tokens_per_step"] // seq_len) * seq_len,
                              "trainable_params": model.get_nb_trainable_parameters()[0]}


def score(cfg, model, tok, tests, table, p, out, info):
    """Evaluates model on the test documents (table: doc_id, source) at every context length and on the Δ pairs p;
    writes test_nll.parquet, delta.parquet and then metrics.json (info, perplexities, Δ summary)."""
    c, d = cfg["continued"], cfg["delta"]
    rows, ppl = [], {}
    for L, (ids, doc) in tests.items():
        nll, n = document_nll(model, ids, doc, len(table), c["eval_tokens"] // L)
        rows.append(table.assign(seq_len=L, nll=nll, tokens=n))
        ppl[L] = perplexity(nll, n, table.source)
    prefix = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    lp_o, lp_e = (sentence_logprob(model, tok, p[col], d["batch_size"], prefix) for col in ("original", "text"))
    pd.concat(rows).to_parquet(out / "test_nll.parquet", index=False)
    pd.DataFrame({"version_id": p.version_id, "logp_edited": lp_e, "logp_original": lp_o}).to_parquet(
        out / "delta.parquet", index=False)
    write_json(out / "metrics.json", info | {"test_ppl": ppl, "delta": summary(p, lp_e, lp_o)})
    (out / "ckpt.pt").unlink(missing_ok=True)


def best_lr(lrs, root, sweep, seq_len):
    """The lr of sweep (over lrs) with the lowest held-out perplexity at seq_len; stops if the sweep is unfinished or
    the best lr is at the edge of lrs."""
    fs = {lr: root / "sweep" / f"{sweep}_ctx{seq_len}_lr{lr:g}" / "metrics.json" for lr in lrs}
    if not all(f.exists() for f in fs.values()):
        raise SystemExit(f"{root.name} {sweep} ctx{seq_len}: the sweep is unfinished (--mode sweep)")
    ppl = {lr: json.loads(f.read_text())["holdout_ppl"] for lr, f in fs.items()}
    best = min(ppl, key=ppl.get)
    if best in (lrs[0], lrs[-1]):
        raise SystemExit(f"{root.name} {sweep} ctx{seq_len}: best lr {best:g} is at the edge of its grid: {ppl}")
    return best


def summarize(root, arms):
    """(base, arm, metric, mean, sd, seeds) over the runs under root: metric ppl_512 / ppl_native (pooled test
    perplexity at that context length) or delta (pooled Δ); arm "base" is the base model before training, the rest in
    the order of arms."""
    rows = []
    for f in root.glob("*/*/metrics.json"):   # sweep runs are one level deeper
        m = json.loads(f.read_text())
        run = {"base": f.parent.parent.name, "arm": m.get("arm_name", "base")}
        rows += [run | {"metric": "ppl_512" if L == "512" else "ppl_native", "value": v["pooled"]}
                 for L, v in m["test_ppl"].items()]
        rows.append(run | {"metric": "delta", "value": m["delta"]["pooled"]["delta"]})
    d = pd.DataFrame(rows).assign(arm=lambda x: pd.Categorical(x.arm, ["base"] + list(arms)))
    return (d.groupby(["base", "arm", "metric"], observed=True).value.agg(mean="mean", sd="std", seeds="count")
            .reset_index())


def main():
    def args(p):
        p.add_argument("--mode", choices=["base", "sweep", "train", "summary"], default="train")
        p.add_argument("--bases", nargs="+", help="default: every base in continued.bases")
        p.add_argument("--seq-len", help="training context length, 'native' or a number (default: continued.seq_len)")
        p.add_argument("--arms", nargs="+", help="default: every arm in pretrain.arms and continued.arms")
        p.add_argument("--seeds", nargs="+", type=int, help="default: pretrain.seeds")
    cfg, args = cli(__doc__, args)
    c, pc = cfg["continued"], cfg["pretrain"]
    all_arms = pc["arms"] | c["arms"]
    if args.mode == "summary":
        t = summarize(path(cfg, "continued_pretrain", "x").parent, all_arms)
        t.to_csv(path(cfg, "continued_pretrain", "summary.csv"), index=False)
        t["cell"] = [f"{m:.3f}" if n == 1 else f"{m:.3f} ± {sd:.3f}" for m, sd, n in zip(t["mean"], t.sd, t.seeds)]
        for base, g in t.groupby("base"):
            print(f"\n{base}\n" + g.pivot(index="arm", columns="metric", values="cell").to_string())
        return
    arms = args.arms or list(all_arms)
    sents, s, s_fresh, edits, editors = load(cfg, args.mode == "train" and any(
        all_arms[a]["data"] == "fresh" for a in arms))
    rewrites = pd.read_parquet(path(cfg, "generated", "sentences.parquet")) if args.mode == "train" and any(
        all_arms[a]["data"] in ("rewrites", "human_calib") for a in arms) else None
    test_sents = sentence_table(sents[sents.split == pc["eval_split"]])
    test = documents(test_sents, {}, "human", "none", [], [], None)
    table = pd.DataFrame({"doc_id": sorted(test_sents.doc_id.unique()), "source": [src for src, _, _ in test]})
    p = pairs(pd.read_parquet(path(cfg, "versions.parquet"),
                              columns=["version_id", "sentence_id", "source", "split", "editor", "text"]),
              cfg["delta"]["n_pairs"], cfg["delta"]["seed"])
    h = holdout(s, c["sweep_holdout"], c["sweep_seed"])
    held = documents(s[s.doc_id.isin(h)], {}, "human", "none", [], [], None)
    print(f"editors {editors}; {len(table)} test documents, {len(p)} Δ pairs, {len(held)} sweep documents held out",
          flush=True)
    for name in args.bases or c["bases"]:
        b, root = c["bases"][name], path(cfg, "continued_pretrain", name, "x").parent
        tok = AutoTokenizer.from_pretrained(b["model"], revision=b["revision"])
        length = lambda x: b["native_len"] if x == "native" else int(x)
        tests = {length(x): eval_windows(test, tok, length(x)) for x in c["eval_seq_lens"]}
        L, grids = length(args.seq_len or c["seq_len"]), c["sweeps"] | b.get("sweeps", {})

        def draw(a, s_):   # arm a's documents: pretrain.documents on sentence table s_, or a side of calib_pair
            if a["data"] in ("rewrites", "human_calib"):
                return lambda rng: calib_pair(sents, rewrites, c["rewrite_generators"], tok, rng)[
                    a["data"] == "human_calib"]
            weight = "strength" if a["weight"] == "strength-mean" else a["weight"]
            return lambda rng: documents(s_, edits, a["data"], weight, editors, cfg["phase"]["thetas"], rng,
                                         a.get("passes"))
        if args.mode == "base":
            if not (root / "base" / "metrics.json").exists():
                (root / "base").mkdir(exist_ok=True)
                score(cfg, base_model(b).cuda(), tok, tests, table, p, root / "base", {"base": b})
        elif args.mode == "sweep":
            hids, hdoc = eval_windows(held, tok, L)
            ev = lambda m: perplexity(*document_nll(m, hids, hdoc, len(held), c["eval_tokens"] // L),
                                      [src for src, _, _ in held])
            for sweep, lrs in grids.items():
                for lr in lrs:
                    out = root / "sweep" / f"{sweep}_ctx{L}_lr{lr:g}"
                    if (out / "metrics.json").exists():
                        continue
                    out.mkdir(parents=True, exist_ok=True)
                    model, info = fit(cfg, b, tok, pc["arms"][sweep], c["sweep_seed"], L, lr,
                                      draw(pc["arms"][sweep], s[~s.doc_id.isin(h)]), out, ev)
                    write_json(out / "metrics.json", info | {"holdout_ppl": ev(model)["pooled"]})
                    (out / "ckpt.pt").unlink(missing_ok=True)
                    del model
        else:
            for arm in arms:
                for seed in args.seeds or pc["seeds"]:
                    out, a = root / f"ctx{L}_{arm}_seed{seed}", all_arms[arm]
                    if (out / "metrics.json").exists():
                        continue
                    sweep = a.get("sweep", "human")
                    lr = best_lr(grids[sweep], root, sweep, L)
                    out.mkdir(exist_ok=True)
                    model, info = fit(cfg, b, tok, a, seed, L, lr, draw(a, s_fresh if a["data"] == "fresh" else s),
                                      out, lambda m: perplexity(*document_nll(
                                          m, *tests[L], len(table), c["eval_tokens"] // L), table.source))
                    score(cfg, model, tok, tests, table, p, out, info | {"arm_name": arm})
                    del model


if __name__ == "__main__":
    main()
