"""Sentence scores from off-the-shelf AI-text classifiers, used as released (no training here).

Each sentence is scored in isolation, like score_lm.py, and only versions in `pretrained.splits` are scored.
Writes scores/<name>.parquet (version_id, score = logit(ai_label) - logit(other label); higher = more AI-like)
for each entry of `pretrained.detectors`.
"""
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import cli, path


@torch.no_grad()
def score_batch(texts, tok, model, ai, max_tokens):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens).to(model.device)
    logits = model(**enc).logits.float()
    return logits[:, ai] - logits[:, 1 - ai]


def main():
    cfg, args = cli(__doc__, lambda p: p.add_argument("--detectors", nargs="*", help="names to run (default: all)"))
    c = cfg["pretrained"]
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    versions = pd.read_parquet(path(cfg, "versions.parquet"), columns=["version_id", "split", "text"])
    versions = versions[versions.split.isin(c["splits"])].reset_index(drop=True)
    order = versions.text.str.len().argsort().to_numpy()            # length-sorted batches waste less padding
    texts, bs = versions.text.to_numpy()[order], c["batch_size"]
    for d in c["detectors"]:
        if args.detectors and d["name"] not in args.detectors:
            continue
        tok = AutoTokenizer.from_pretrained(d["model"])
        model = AutoModelForSequenceClassification.from_pretrained(d["model"]).to(device).eval()
        labels = model.config.id2label
        assert len(labels) == 2, f"{d['model']}: expected two labels, got {labels}"
        ai = {v: k for k, v in labels.items()}[d["ai_label"]]
        scores = np.empty(len(texts))
        scores[order] = torch.cat([score_batch(list(texts[i:i + bs]), tok, model, ai, c["max_tokens"]).cpu()
                                   for i in tqdm(range(0, len(texts), bs), desc=d["name"])]).numpy()
        pd.DataFrame({"version_id": versions.version_id, "score": scores}).to_parquet(
            path(cfg, "scores", f"{d['name']}.parquet"), index=False)


if __name__ == "__main__":
    main()
