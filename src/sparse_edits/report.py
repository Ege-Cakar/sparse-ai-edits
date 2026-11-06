"""Report figures from phase/ and phase_docs/ (written by phase.py), into figures_report/.

  phase_<detector>.png  Gaussian HC power at n = 16 next to the whole-mode test documents, each placed at its own
                        (theta_doc, r_doc) and binned: theta_doc = 1 - log m / log n for its m edited sentences,
                        r_doc = max(mu_doc, 0)^2 / (2 log n). Hexagon colour = share of its documents HC rejects.
                        r_doc uses the true edit positions, so it places a document; it is not an estimate.
  type1.png             rejection rate of each global test on human-only whole documents (theta = inf), per
                        detector, with Wilson intervals, under iid thresholds and, where phase.py produced them,
                        thresholds for the within-document correlation; the dashed line is the nominal level.
  power.png             HC rejection rate against theta per detector, by document length band (editors pooled),
                        with Wilson intervals; the grey line is the same detector's rate on human-only documents.
All use conformal p-values; phase and power figures use the correlated thresholds (null conformal_dependent)
when present, else the iid ones.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import cli, path
from .phase import SEQUENTIAL, draw_boundaries
from .stats import wilson

INK, INK_2 = "#0b0b0b", "#52514e"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]   # fixed order;
# adjacent pairs validated (bars, lines); the 3 length bands use slots 1-3, which also pass all-pairs
TESTS = {"reject_min_p": "min-p", "reject_hc": "HC", "reject_hc_plus": "HC+", "reject_sum": "Stouffer"}
BANDS = ([15, 31, 63, np.inf], ["16-31", "32-63", "64+"])


NULLS = {"conformal": "i.i.d. thresholds", "conformal_dependent": "correlated thresholds"}


def whole_docs(cfg, detector, null=None):
    """Whole-mode per-document rows under one null; by default the correlated thresholds when present."""
    d = pd.read_parquet(path(cfg, "phase_docs", f"{detector}.parquet"))
    d = d[d["mode"] == "whole"]
    null = null or ("conformal_dependent" if (d["null"] == "conformal_dependent").any() else "conformal")
    return d[d["null"] == null].assign(null_label=NULLS[null])


def doc_coordinates(d):
    """(theta_doc, r_doc) of edited documents; documents with one edited sentence sit at theta = 1 - 0 = 1."""
    m = np.round(d.eps_true * d.doc_n)
    keep = m >= 1
    d, m = d[keep], m[keep]
    log_n = np.log(d.doc_n)
    return d, 1 - np.log(m) / log_n, np.maximum(d.mu_doc, 0) ** 2 / (2 * log_n)


def phase_figure(cfg, detector, alpha):
    sim = pd.read_parquet(path(cfg, "phase", "gaussian_n16.parquet"))
    d, theta, r = doc_coordinates(whole_docs(cfg, detector).query("theta != inf"))
    ymax = 4.2
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, constrained_layout=True)
    grid = sim.pivot(index="r", columns="theta", values="reject_hc")
    left.pcolormesh(grid.columns, grid.index, grid.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=1, shading="nearest")
    hb = right.hexbin(theta, np.minimum(r, ymax - 0.01), C=d.reject_hc.astype(float), reduce_C_function=np.mean,
                      gridsize=(22, 14), extent=(0, 1, 0, ymax), mincnt=20, cmap=SEQUENTIAL, vmin=0, vmax=1,
                      edgecolors="white", linewidths=0.5)
    for ax in (left, right):
        draw_boundaries(ax, ymax)
    left.set(title="Gaussian simulation, n = 16", ylabel=r"$r$  ($\mu = \sqrt{2 r \log n}$)")
    at_zero = (r == 0).mean()
    right.set_title(f"{detector}, whole test documents (n >= 16)\n{d.null_label.iloc[0]}", fontsize=10)
    right.text(0.99, 0.97, f"{len(d):,} draws (document x editor x theta)\n{at_zero:.0%} at r = 0; "
               "hexagons with >= 20 draws",
               transform=right.transAxes, ha="right", va="top", fontsize=8, color=INK_2)
    fig.legend(*left.get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
    fig.colorbar(hb, ax=[left, right], label=f"HC rejection rate (level {alpha:g})")
    fig.savefig(path(cfg, "figures_report", f"phase_{detector}.png"), dpi=200)
    plt.close(fig)


def type1_figure(cfg, detectors, alpha, level):
    fig, axes = plt.subplots(1, len(detectors), figsize=(5 * len(detectors), 3.8), sharey=True,
                             constrained_layout=True, squeeze=False)
    for ax, det in zip(axes[0], detectors):
        nulls = [n for n in NULLS if len(whole_docs(cfg, det, n))]
        width = 0.8 / len(nulls)
        for j, null in enumerate(nulls):
            d = whole_docs(cfg, det, null).query("theta == inf")
            rate = np.array([d[t].mean() for t in TESTS])
            lo, hi = np.array([wilson(d[t].sum(), len(d), level) for t in TESTS]).T
            x = np.arange(len(TESTS)) + (j - (len(nulls) - 1) / 2) * width
            ax.bar(x, rate, width * 0.92, color=SERIES[j], label=NULLS[null])
            ax.errorbar(x, rate, yerr=[rate - lo, hi - rate], fmt="none", ecolor=INK, elinewidth=1, capsize=3)
        ax.axhline(alpha, color=INK_2, ls="--", lw=1)
        ax.annotate(f"nominal {alpha:g}", (len(TESTS) - 0.5, alpha), ha="right", va="bottom", fontsize=8,
                    color=INK_2)
        ax.set(xticks=range(len(TESTS)), xticklabels=TESTS.values(), title=f"{det} ({len(d):,} human documents)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0, 0].set_ylabel("rejection rate on human-only documents")
    fig.savefig(path(cfg, "figures_report", "type1.png"), dpi=200)
    plt.close(fig)


def power_figure(cfg, detectors, level):
    fig, axes = plt.subplots(1, len(detectors), figsize=(5 * len(detectors), 3.8), sharey=True,
                             constrained_layout=True, squeeze=False)
    for ax, det in zip(axes[0], detectors):
        d = whole_docs(cfg, det)
        null = d[np.isinf(d.theta)].reject_hc.mean()
        d = d[np.isfinite(d.theta)].assign(band=lambda x: pd.cut(x.doc_n, BANDS[0], labels=BANDS[1]))
        for k, (band, g) in enumerate(d.groupby("band", observed=True)):
            s = g.groupby("theta").reject_hc.agg(["sum", "size"])
            rate = s["sum"] / s["size"]
            lo, hi = wilson(s["sum"].to_numpy(), s["size"].to_numpy(), level)
            ax.fill_between(s.index, lo, hi, color=SERIES[k], alpha=0.15, lw=0)
            ax.plot(s.index, rate, color=SERIES[k], lw=2, marker="o", ms=4,
                    label=f"n {band} ({g.doc_n.size // g.theta.nunique() // g.editor.nunique():,} docs)")
        ax.axhline(null, color=INK_2, ls="--", lw=1, label="human-only documents")
        ax.set(title=f"{det}, {d.null_label.iloc[0]}", xlabel=r"$\vartheta$  (m = round($n^{1-\vartheta}$) edited sentences)",
               ylim=(0, None))
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, title="length band, editors pooled", title_fontsize=8)
    axes[0, 0].set_ylabel("HC rejection rate")
    fig.savefig(path(cfg, "figures_report", "power.png"), dpi=200)
    plt.close(fig)


def main():
    cfg, _ = cli(__doc__)
    c = cfg["phase"]
    detectors = sorted(f.stem for f in Path(cfg["data_dir"], "phase_docs").glob("*.parquet"))
    for det in detectors:
        phase_figure(cfg, det, c["alpha"])
    type1_figure(cfg, detectors, c["alpha"], c["ci_level"])
    power_figure(cfg, detectors, c["ci_level"])


if __name__ == "__main__":
    main()
