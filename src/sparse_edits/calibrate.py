"""Convert every detector's scores to p-values against held-out human sentences of similar length.

Null: human sentences of the calib split from the same source (or, with calibrate.pool_sources, from every source). Writes pvalues/<detector>.parquet
(version_id, p) for all versions outside the calib split.

Also writes pvalues_calib/<detector>.parquet (version_id, p, fold) for the calib sentences themselves, cross-fitted:
calib documents are split in two folds (by a hash of seed and doc_id, the same for every detector), and each fold's
sentences get p-values against the other fold. These are human documents with realistic p-values, from which
phase.py estimates the within-document correlation. A separate random stream leaves pvalues/ unchanged.
"""
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import cli, path
from .stats import conformal_pvalues

HIGHER_IS_HUMAN = ("binoculars",)   # score-file prefixes (score_lm.py); all others are higher = more AI-like


def main():
    cfg, _ = cli(__doc__)
    versions = pd.read_parquet(path(cfg, "versions.parquet"),
                               columns=["version_id", "source", "doc_id", "split", "editor", "n_words"])
    for f in sorted(Path(cfg["data_dir"], "scores").glob("*.parquet")):
        df = versions.merge(pd.read_parquet(f), on="version_id")
        df["score"] *= -1 if f.stem.startswith(HIGHER_IS_HUMAN) else 1
        rng, out = np.random.default_rng(cfg["seed"]), []
        for source, g in df.groupby(null_groups(cfg, df)):
            null = g[(g.split == "calib") & (g.editor == "human")]
            test = g[g.split != "calib"]
            p = conformal_pvalues(null.score.to_numpy(), null.n_words.to_numpy(), test.score.to_numpy(),
                                  test.n_words.to_numpy(), cfg["calibrate"]["length_bins"], rng)
            out.append(pd.DataFrame({"version_id": test.version_id, "p": p}))
            print(f"{f.stem} / {source}: {len(null)} null sentences, {np.isnan(p).sum()} NaN p-values")
        pd.concat(out).to_parquet(path(cfg, "pvalues", f.name), index=False)
        cross_fit_calib(cfg, df).to_parquet(path(cfg, "pvalues_calib", f.name), index=False)


def null_groups(cfg, df):
    """The null each sentence is ranked against: its own source's calib sentences, or (calibrate.pool_sources) the
    calib sentences of every source pooled; length bins apply within either."""
    return pd.Series("pooled", index=df.index) if cfg["calibrate"].get("pool_sources") else df.source


def calib_fold(cfg, doc_ids):
    return np.array([zlib.crc32(repr((cfg["seed"], int(d))).encode()) % 2 for d in doc_ids])


def cross_fit_calib(cfg, df):
    """p-values of human calib sentences, each against the calib sentences of the other fold of its source."""
    rng, out = np.random.default_rng([cfg["seed"], 1]), []
    cal = df[(df.split == "calib") & (df.editor == "human")]
    for _, g in cal.groupby(null_groups(cfg, cal)):
        fold = calib_fold(cfg, g.doc_id)
        for k in (0, 1):
            null, test = g[fold != k], g[fold == k]
            p = conformal_pvalues(null.score.to_numpy(), null.n_words.to_numpy(), test.score.to_numpy(),
                                  test.n_words.to_numpy(), cfg["calibrate"]["length_bins"], rng)
            out.append(pd.DataFrame({"version_id": test.version_id, "p": p, "fold": k}))
    return pd.concat(out)


if __name__ == "__main__":
    main()
