"""Zero-shot sentence scores from one forward pass of an observer and a performer LM.

Each sentence is scored in isolation (no document context). Only versions in `lm_scores.splits` are scored (zero-shot
scores need no training data; calibrate.py and phase.py use calib and test). Writes scores/<name>.parquet
(version_id, score):
  log_likelihood  mean token log-prob under the observer (higher = more AI-like)
  fast_detectgpt  analytic sampling discrepancy, sampling model = observer, scoring model = performer
                  (Bao et al. 2024, get_sampling_discrepancy_analytic; higher = more AI-like)
  binoculars      performer log-PPL / observer-to-performer cross-entropy (Hans et al. 2024; lower = more AI-like).
                  Both terms average over the same predicted positions (the paper's definition); the reference
                  code averages the cross-entropy over one extra position.
"""
import numpy as np
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
    cfg, _ = cli(__doc__)
    c = cfg["lm_scores"]
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(c["observer"])
    tok.pad_token, tok.padding_side = tok.pad_token or tok.eos_token, "right"
    observer, performer = (AutoModelForCausalLM.from_pretrained(m, dtype=dtype).to(device).eval()
                           for m in (c["observer"], c["performer"]))

    versions = pd.read_parquet(path(cfg, "versions.parquet"), columns=["version_id", "split", "text"])
    versions = versions[versions.split.isin(c["splits"])].reset_index(drop=True)
    order = versions.text.str.len().argsort().to_numpy()            # length-sorted batches waste less padding
    texts, bs = versions.text.to_numpy()[order], c["batch_size"]
    batches = [score_batch(list(texts[i:i + bs]), tok, observer, performer, c["max_tokens"])
               for i in tqdm(range(0, len(texts), bs))]
    for name in batches[0]:
        scores = np.empty(len(texts))
        scores[order] = torch.cat([b[name] for b in batches]).cpu().numpy()
        pd.DataFrame({"version_id": versions.version_id, "score": scores}).to_parquet(
            path(cfg, "scores", f"{name}.parquet"), index=False)


if __name__ == "__main__":
    main()
