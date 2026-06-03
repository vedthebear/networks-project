#!/usr/bin/env python3
"""
plot_null_spearman.py -- cross-platform null figure in the spirit of Sam's
reddit spearman_null_distribution.png, but for BOTH platforms at once.

The statistic is the WITHIN-CORE Spearman rho (k-core vs clustering, k-core >= 2)
-- the bridging signal. On both platforms the real value lands far below its
degree-preserving null, i.e. central communities are bridges, not cliques.

Why a broken x-axis. Reddit's community graph is huge (35,776 nodes), so its
configuration-model null is extremely tight (std ~0.005); the real value sits
~130 null-standard-deviations away. A plain histogram would put the null in an
invisible sliver next to a vast empty gap. We therefore zoom each panel onto its
null cluster (so the histogram is actually visible) and break the axis to show
the real value, marked exactly as in Sam's figure. The break itself communicates
"the real value is nowhere near the null."

Reads (already produced in this branch):
    results/reddit_null_ensemble.csv   rho_core per null graph
    results/null_ensemble.csv          rho_core per null graph (Moltbook)
    results/reddit_real_summary.json   real rho_core (Reddit)
    results/real_summary.json          real rho_core (Moltbook)

Writes:
    figures/null_spearman_comparison.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE / "results"
FIGS = HERE / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

NULL_BLUE = "#6baed6"
BAND_BLUE = "#c6dbef"


def load():
    mb_null = pd.read_csv(R / "null_ensemble.csv")["rho_core"].to_numpy()
    rd_null = pd.read_csv(R / "reddit_null_ensemble.csv")["rho_core"].to_numpy()
    mb_real = json.loads((R / "real_summary.json").read_text())["rho_core"]
    rd_real = json.loads((R / "reddit_real_summary.json").read_text())["rho_core"]
    return rd_null, rd_real, mb_null, mb_real


def p_str(null, real):
    n = len(null)
    k = int((null <= real).sum())
    return f"p < {1/n:.3f}  (0/{n})" if k == 0 else f"p = {k/n:.3f}  ({k}/{n})"


def sigma_gap(null, real):
    return (real - null.mean()) / null.std()


def break_marks(ax_left, ax_right):
    """Diagonal break marks on the facing spines of two adjacent axes."""
    d = 0.5
    kw = dict(marker=[(-1, -d), (1, d)], markersize=9, linestyle="none",
              color="k", mec="k", mew=1, clip_on=False)
    ax_left.plot([1, 1], [0, 1], transform=ax_left.transAxes, **kw)
    ax_right.plot([0, 0], [0, 1], transform=ax_right.transAxes, **kw)


def draw_row(ax_real, ax_null, null, real, label, color, ymax):
    """Left axis = the real value; right axis = the null histogram. Broken between."""
    # --- right axis: the null distribution, zoomed in so it is actually visible
    rng = null.max() - null.min()
    pad = max(0.01, 0.25 * rng)
    ax_null.axvspan(null.min(), null.max(), color=BAND_BLUE, alpha=0.6,
                    label="null spread (300 graphs)")
    ax_null.hist(null, bins=30, color=NULL_BLUE, edgecolor="white")
    ax_null.axvline(null.mean(), color="#525252", lw=1.3, linestyle=":",
                    label=f"null mean = {null.mean():.2f}")
    ax_null.set_xlim(null.min() - pad, null.max() + pad)
    ax_null.spines["left"].set_visible(False)
    ax_null.tick_params(left=False, labelleft=False)
    ax_null.legend(fontsize=8, frameon=True, framealpha=0.9, loc="lower right")

    # --- left axis: the real value, on its own zoomed window
    ax_real.axvline(real, color=color, lw=2.6, linestyle="--")
    ax_real.set_xlim(real - 0.06, real + 0.06)
    ax_real.set_xticks([round(real, 2)])
    ax_real.spines["right"].set_visible(False)
    ax_real.set_ylabel("number of null graphs")
    ax_real.annotate(f"real = {real:.2f}", xy=(real, ymax * 0.93),
                     xytext=(real, ymax * 0.93), color=color, fontweight="bold",
                     ha="center", fontsize=10)

    for ax in (ax_real, ax_null):
        ax.set_ylim(0, ymax)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3, color="#cccccc")
    break_marks(ax_real, ax_null)

    # The paper frames this comparison by the raw gap below the null (the honest
    # cross-platform quantity), not by sigma. Keep gap + p; drop sigma.
    gap = real - null.mean()
    return (f"{label}:   within-core ρ = {real:.2f},  "
            f"{gap:+.2f} below its null   ({p_str(null, real)})")


def main():
    rd_null, rd_real, mb_null, mb_real = load()

    fig, axes = plt.subplots(
        2, 2, figsize=(11, 6.4),
        gridspec_kw={"width_ratios": [1, 2.4], "wspace": 0.06, "hspace": 0.65})
    fig.subplots_adjust(top=0.83, bottom=0.12, left=0.085, right=0.97)

    rd_ymax = np.histogram(rd_null, bins=30)[0].max() * 1.18
    mb_ymax = np.histogram(mb_null, bins=30)[0].max() * 1.18
    h_rd = draw_row(axes[0, 0], axes[0, 1], rd_null, rd_real, "Reddit (humans)",
                    "#2171b5", rd_ymax)
    h_mb = draw_row(axes[1, 0], axes[1, 1], mb_null, mb_real, "Moltbook (agents)",
                    "#d62728", mb_ymax)

    # row headers placed in the margins so they never collide with the legend
    fig.text(0.5, 0.855, h_rd, ha="center", fontsize=11, fontweight="bold")
    fig.text(0.5, 0.435, h_mb, ha="center", fontsize=11, fontweight="bold")

    fig.suptitle(
        "Central communities are bridges on BOTH platforms, far beyond chance\n"
        "within-core (k ≥ 2) k-core / clustering correlation vs degree-preserving null",
        fontsize=13, fontweight="bold", y=0.98)
    fig.supxlabel("within-core (k ≥ 2) Spearman ρ  (k-core vs clustering);  "
                  "negative = central communities bridge between groups",
                  fontsize=10, y=0.02)
    out = FIGS / "null_spearman_comparison.png"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  Reddit   real={rd_real:.3f}  null_mean={rd_null.mean():.3f}  "
          f"sigma={sigma_gap(rd_null, rd_real):.0f}")
    print(f"  Moltbook real={mb_real:.3f}  null_mean={mb_null.mean():.3f}  "
          f"sigma={sigma_gap(mb_null, mb_real):.0f}")


if __name__ == "__main__":
    main()
