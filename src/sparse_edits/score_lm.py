"""Zero-shot sentence scores from one forward pass of an observer and a performer LM.

Each sentence is scored in isolation (no document context), for the versions in lm_scores.splits. --pair NAME picks
an observer/performer pair from lm_scores.pairs and writes scores/<score>_<NAME>.parquet (version_id, score):
  log_likelihood  mean token log-prob under the observer (higher = more AI-like)
  fast_detectgpt  analytic sampling discrepancy, sampling model = observer, scoring model = performer
                  (Bao et al. 2024, get_sampling_discrepancy_analytic; higher = more AI-like)
  binoculars      performer log-PPL / observer-to-performer cross-entropy (Hans et al. 2024; lower = more AI-like).
                  Both terms average over the same predicted positions (the paper's definition); the reference
                  code averages the cross-entropy over one extra position.
Length-sorted chunks of lm_scores.chunk versions are saved to lm_score_parts/<NAME>/ as they finish, so a preempted
run resumes at the first missing chunk.
"""
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import cli, path


@torch.no_grad()
def score_batch(texts, tok, observer, performer, max_tokens):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens).to(observer.device)
    y, mask = enc.input_ids[:, 1:, None], enc.attention_mask[:, 1:].float()
    lp_obs = observer(**enc).logits[:, :-1].float().log_softmax(-1)
    lp_perf = performer(**enc).logits[:, :-1].float().log_softmax(-1)
    ll_obs, ll_perf = lp_obs.gather(-1, y)[..., 0], lp_perf.gather(-1, y)[..., 0]
    mean_ref = (lp_obs.exp() * lp_perf).sum(-1)                      # E_{x~observer}[log p_performer(x)]
    var_ref = (lp_obs.exp() * lp_perf.square()).sum(-1) - mean_ref.square()
    total = lambda t: (t * mask).sum(1)
    return dict(log_likelihood=total(ll_obs) / mask.sum(1),
                fast_detectgpt=(total(ll_perf) - total(mean_ref)) / total(var_ref).sqrt(),
                binoculars=total(ll_perf) / total(mean_ref))         # ratio of two negated averages


def main():
    cfg, args = cli(__doc__, lambda p: p.add_argument("--pair", required=True, help="a key of lm_scores.pairs"))
    c = cfg["lm_scores"]
    pair, bs = c["pairs"][args.pair], c["batch_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(pair["observer"])
    tok.pad_token, tok.padding_side = tok.pad_token or tok.eos_token, "right"
    observer, performer = (AutoModelForCausalLM.from_pretrained(m, dtype=dtype).to(device).eval()
                           for m in (pair["observer"], pair["performer"]))

    versions = pd.read_parquet(path(cfg, "versions.parquet"), columns=["version_id", "split", "text"])
    versions = versions[versions.split.isin(c["splits"])].reset_index(drop=True)
    order = versions.text.str.len().argsort().to_numpy()            # length-sorted batches waste less padding
    parts = path(cfg, "lm_score_parts", args.pair, "x").parent
    for k, start in enumerate(tqdm(range(0, len(order), c["chunk"]))):
        f = parts / f"{k:04d}.parquet"
        if f.exists():
            continue
        idx = order[start:start + c["chunk"]]
        texts = versions.text.to_numpy()[idx]
        batches = [score_batch(list(texts[i:i + bs]), tok, observer, performer, c["max_tokens"])
                   for i in range(0, len(idx), bs)]
        pd.DataFrame({"version_id": versions.version_id.to_numpy()[idx]}
                     | {n: torch.cat([b[n] for b in batches]).cpu().numpy() for n in batches[0]}).to_parquet(
            f.with_suffix(".tmp"), index=False)
        f.with_suffix(".tmp").replace(f)
    scores = pd.concat([pd.read_parquet(f) for f in sorted(parts.glob("*.parquet"))])
    assert len(scores) == len(versions) and scores.version_id.is_unique
    for name in ["log_likelihood", "fast_detectgpt", "binoculars"]:
        scores[["version_id", name]].rename(columns={name: "score"}).to_parquet(
            path(cfg, "scores", f"{name}_{args.pair}.parquet"), index=False)
    print(f"{args.pair}: scored {len(scores)} versions of {c['splits']}")


if __name__ == "__main__":
    main()
