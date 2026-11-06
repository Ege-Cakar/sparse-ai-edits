"""Ensemble detector: the mean of several detectors' scores, each standardized by the mean and standard deviation of its
scores on the val split's human sentences. Nothing is fit on calib or test, so calibrate.py's conformal p-values stay
valid. Reads scores/<run>.parquet for each run; writes scores/<name>.parquet (version_id, score) for the versions every
run scored. Scores are oriented higher = more AI-like, as every detect.py run is.
"""
import pandas as pd

from . import cli, path


def ensemble(scores, ref):
    """scores: one column per detector, indexed by version_id; ref: the version_ids that set each column's mean and sd."""
    r = scores[scores.index.isin(ref)]
    return ((scores - r.mean()) / r.std()).mean(axis=1)


def main():
    def args(p):
        p.add_argument("--name", required=True, help="output: scores/<name>.parquet")
        p.add_argument("--runs", nargs="+", required=True, help="detectors to combine: scores/<run>.parquet")
    cfg, args = cli(__doc__, args)
    v = pd.read_parquet(path(cfg, "versions.parquet"), columns=["version_id", "split", "editor"])
    scores = pd.concat([pd.read_parquet(path(cfg, "scores", f"{r}.parquet")).set_index("version_id").score.rename(r)
                        for r in args.runs], axis=1, join="inner")
    ref = v.version_id[(v.split == "val") & (v.editor == "human")]
    if not scores.index.isin(ref).any():
        raise SystemExit("none of the val split's human sentences was scored; score val first (detect.py --score)")
    s = ensemble(scores, ref)
    pd.DataFrame({"version_id": s.index, "score": s.to_numpy()}).to_parquet(
        path(cfg, "scores", f"{args.name}.parquet"), index=False)
    print(f"{args.name}: {len(s)} versions, {scores.index.isin(ref).sum()} val human sentences set the scales")


if __name__ == "__main__":
    main()
