"""Phase diagrams: Gaussian simulation at the same n, and empirical diagrams per detector.

--what simulate   phase/gaussian_n{n}.parquet: z ~ N(mu 1[edited], 1), mu = sqrt(2 r log n), over the (theta, r) grid.
--what empirical  phase/<detector>.parquet: test-split documents (fixed-n windows and/or whole docs). For each
                  (editor, target strength, theta), swap m = round(n^(1 - theta)) uniformly chosen sentences for
                  one fixed random edit of each, then evaluate with conformal p-values and, on the same draw,
                  with p-values recentered by an Efron empirical null fit on all z in the cell. theta = inf rows
                  have no edits (type I error). mu_eff = median z of edited test sentences.
--what plot       figures/: for window mode, Gaussian power next to empirical cells placed at
                  r_eff = max(mu_eff, 0)^2 / (2 log n), with the theoretical boundaries on both.
"""
from itertools import product
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm

from . import cli, path
from .stats import (almost_full_recovery_boundary, detection_boundary, empirical_null, evaluate,
                    exact_recovery_boundary, n_edited, null_thresholds, sample_truth)


def outcomes(p, truth, cfg):
    c = cfg["phase"]
    thresholds = null_thresholds(p.shape[1], c["alpha"], c["hc_alpha0"], c["null_mc_reps"], cfg["seed"])
    return evaluate(p, truth, thresholds, c["alpha"], c["hc_alpha0"], c["bh_q"])


def simulate(cfg, n, rng):
    c, rows = cfg["phase"], []
    for theta, r in product(c["thetas"], c["sim_r"]):
        truth = sample_truth(c["sim_docs"], n, n_edited(n, theta), rng)
        z = rng.standard_normal(truth.shape) + np.sqrt(2 * r * np.log(n)) * truth
        out = outcomes(norm.sf(z), truth, cfg)
        rows.append(dict(n=n, theta=theta, r=r, m=int(n_edited(n, theta)), **{k: v.mean() for k, v in out.items()}))
    return pd.DataFrame(rows)


def test_documents(versions, mode, n, rng):
    """(docs, n) sentence_id matrices of test-split documents: one random window of n consecutive sentences
    per doc with at least n (window), or whole docs grouped by length (whole)."""
    human = versions[(versions.split == "test") & (versions.editor == "human")].sort_values(["doc_id", "sent_idx"])
    docs = human.groupby("doc_id").sentence_id.apply(np.array)
    if mode == "window":
        windows = [d[s:s + n] for d in docs if len(d) >= n for s in [rng.integers(len(d) - n + 1)]]
        return [np.stack(windows)] if windows else []
    return [np.stack(list(g)) for _, g in docs.groupby(docs.map(len))]


def lookup(series, ids):
    return series.reindex(ids.ravel()).to_numpy().reshape(ids.shape)


def complete(*mats):
    """Drop documents (rows) with a missing p-value in any of the matrices."""
    ok = ~np.any([np.isnan(m).any(axis=1) for m in mats], axis=0)
    return tuple(m[ok] for m in mats)


def run_cell(cells, theta, cfg, rng):
    """cells: list of (human p, edited p) matrices with equal n. Returns one row per null type."""
    draws = []
    for h, a in cells:
        truth = sample_truth(*h.shape, n_edited(h.shape[1], theta), rng)
        draws.append((norm.isf(np.where(truth, a, h)), truth))
    fit = empirical_null(np.concatenate([z.ravel() for z, _ in draws]), cfg["phase"]["empirical_null_central"])
    rows = []
    for null, (delta0, sigma0) in [("conformal", (0.0, 1.0)), ("empirical", fit)]:
        if np.isnan(sigma0):
            print(f"theta={theta}: empirical null fit failed (non-concave); row skipped")
            continue
        outs = [outcomes(norm.sf((z - delta0) / sigma0), truth, cfg) for z, truth in draws]
        rows.append(dict(theta=theta, null=null, delta0=delta0, sigma0=sigma0, n_docs=sum(len(t) for _, t in draws),
                         m_mean=np.concatenate([t.sum(axis=1) for _, t in draws]).mean(),
                         **{k: np.concatenate([o[k] for o in outs]).mean() for k in outs[0]}))
    return rows


def empirical(cfg, detector, versions, rng):
    c = cfg["phase"]
    pv = versions.merge(pd.read_parquet(path(cfg, "pvalues", f"{detector}.parquet")), on="version_id")
    pv = pv[pv.split == "test"]
    human_p = pv[pv.editor == "human"].set_index("sentence_id").p
    edits = pv[pv.editor != "human"].sample(frac=1, random_state=cfg["seed"])
    edits = edits.drop_duplicates(["editor", "target_r", "sentence_id"])          # one random rep per sentence
    designs = [("window", n) for n in c["window_n"]] if "window" in c["modes"] else []
    designs += [("whole", np.nan)] if "whole" in c["modes"] else []
    rows = []
    for mode, n in designs:
        groups = test_documents(versions, mode, n, rng)
        null_cells = [complete(lookup(human_p, ids), lookup(human_p, ids)) for ids in groups]
        for row in run_cell(null_cells, np.inf, cfg, rng):
            rows.append(dict(mode=mode, n=n, editor="none", target_r=0.0, **row))
        for (editor, target_r), e in edits.groupby(["editor", "target_r"]):
            alt_p = e.set_index("sentence_id").p
            cells = [complete(lookup(human_p, ids), lookup(alt_p, ids)) for ids in groups]
            mu_eff = np.median(np.concatenate([norm.isf(a).ravel() for _, a in cells]))
            for theta in c["thetas"]:
                for row in run_cell(cells, theta, cfg, rng):
                    rows.append(dict(mode=mode, n=n, editor=editor, target_r=target_r, mu_eff=mu_eff,
                                     mean_actual_r=e.actual_r.mean(), **row))
    return pd.DataFrame(rows).assign(detector=detector)


SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
BOUNDARIES = [(detection_boundary, "detection", "-"), (almost_full_recovery_boundary, "almost full recovery", "--"),
              (exact_recovery_boundary, "exact recovery", ":")]


def draw_boundaries(ax, ymax):
    t = np.linspace(0.001, 0.999, 400)
    for f, label, style in BOUNDARIES:
        ax.plot(t, f(t), color="#0b0b0b", ls=style, lw=1.5, label=label,
                path_effects=[pe.withStroke(linewidth=3.5, foreground="white")])
    ax.set(xlim=(0, 1), ylim=(0, ymax), xlabel=r"$\vartheta$  ($\varepsilon = n^{-\vartheta}$)")


def plot(cfg):
    for f in sorted(Path(cfg["data_dir"], "phase").glob("*.parquet")):
        if f.stem.startswith("gaussian_"):
            continue
        emp = pd.read_parquet(f)
        emp = emp[(emp["mode"] == "window") & (emp["null"] == "conformal") & np.isfinite(emp.theta)]
        for (n, editor), e in emp.groupby(["n", "editor"]):
            n = int(n)
            sim = pd.read_parquet(path(cfg, "phase", f"gaussian_n{n}.parquet"))
            r_eff = np.maximum(e.mu_eff, 0) ** 2 / (2 * np.log(n))
            ymax = max(4.2, 1.05 * r_eff.max())
            for metric in cfg["phase"]["plot_metrics"]:
                fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, constrained_layout=True)
                grid = sim.pivot(index="r", columns="theta", values=metric)
                left.pcolormesh(grid.columns, grid.index, grid.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=1,
                                shading="nearest")
                dots = right.scatter(e.theta, r_eff, c=e[metric], cmap=SEQUENTIAL, vmin=0, vmax=1, s=70,
                                     edgecolors="#52514e", linewidths=0.5)
                for target_r, y in set(zip(e.target_r, r_eff)):
                    right.annotate(f"edit {target_r:g}", (0.995, y), ha="right", va="bottom", fontsize=7,
                                   color="#52514e")
                for ax in (left, right):
                    draw_boundaries(ax, ymax)
                left.set(title=f"Gaussian simulation, n = {n}", ylabel=r"$r$  ($\mu = \sqrt{2 r \log n}$)")
                right.set_title(f"{f.stem}, editor {editor}, n = {n}")
                fig.legend(*left.get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
                fig.colorbar(dots, ax=[left, right], label=metric)
                fig.savefig(path(cfg, "figures", f"{f.stem}_{editor}_n{n}_{metric}.png"), dpi=200)
                plt.close(fig)


def main():
    cfg, args = cli(__doc__, lambda p: p.add_argument("--what", choices=["simulate", "empirical", "plot"],
                                                      required=True))
    rng = np.random.default_rng(cfg["seed"])
    if args.what == "simulate":
        for n in cfg["phase"]["window_n"]:
            simulate(cfg, n, rng).to_parquet(path(cfg, "phase", f"gaussian_n{n}.parquet"), index=False)
    elif args.what == "empirical":
        versions = pd.read_parquet(path(cfg, "versions.parquet"),
                                   columns=["version_id", "sentence_id", "doc_id", "sent_idx", "split", "editor",
                                            "target_r", "actual_r"])
        for f in sorted(Path(cfg["data_dir"], "pvalues").glob("*.parquet")):
            empirical(cfg, f.stem, versions, rng).to_parquet(path(cfg, "phase", f.name), index=False)
    else:
        plot(cfg)


if __name__ == "__main__":
    main()
