"""Convert every detector's scores to p-values against held-out human sentences of similar length.

Null: human sentences of the calib split from the same source. Writes pvalues/<detector>.parquet
(version_id, p) for all versions outside the calib split.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import cli, path
from .stats import conformal_pvalues

HIGHER_IS_HUMAN = {"binoculars"}   # all other scores are oriented higher = more AI-like


def main():
    cfg, _ = cli(__doc__)
    versions = pd.read_parquet(path(cfg, "versions.parquet"),
                               columns=["version_id", "source", "split", "editor", "n_words"])
    for f in sorted(Path(cfg["data_dir"], "scores").glob("*.parquet")):
        df = versions.merge(pd.read_parquet(f), on="version_id")
        df["score"] *= -1 if f.stem in HIGHER_IS_HUMAN else 1
        rng, out = np.random.default_rng(cfg["seed"]), []
        for source, g in df.groupby("source"):
            null = g[(g.split == "calib") & (g.editor == "human")]
            test = g[g.split != "calib"]
            p = conformal_pvalues(null.score.to_numpy(), null.n_words.to_numpy(), test.score.to_numpy(),
                                  test.n_words.to_numpy(), cfg["calibrate"]["length_bins"], rng)
            out.append(pd.DataFrame({"version_id": test.version_id, "p": p}))
            print(f"{f.stem} / {source}: {len(null)} null sentences, {np.isnan(p).sum()} NaN p-values")
        pd.concat(out).to_parquet(path(cfg, "pvalues", f.name), index=False)


if __name__ == "__main__":
    main()
