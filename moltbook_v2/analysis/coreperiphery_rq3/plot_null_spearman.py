#!/usr/bin/env python3
"""
plot_null_spearman.py -- side-by-side null-distribution figure, in the style of
Sam's reddit spearman_null_distribution.png, but for BOTH platforms at once.

Sam's original (reddit/figures/rq4/spearman_null_distribution.png) shows the
configuration-model null distribution of the k-core vs clustering Spearman rho
as a histogram, with the real value marked by a vertical line: "is the observed
correlation beyond what degree alone forces?".

We already computed the same null ensembles for both platforms (Reddit and
Moltbook) in this branch, so we can put the two null distributions next to each
other. We use the WITHIN-CORE rho (k-core >= 2), which is the bridging signal:
on both platforms the real value lands far to the LEFT of its null, i.e. central
communities are bridges, not cliques, beyond what degree explains.

Reads (already produced by reddit_coreperiphery.py / moltbook_coreperiphery.py):
    results/reddit_null_ensemble.csv   rho_core per null graph
    results/null_ensemble.csv          rho_core per null graph (Moltbook)
    results/reddit_real_summary.json   real rho_core (Reddit)
    results/real_summary.json          real rho_core (Moltbook)

Writes:
    figures/null_spearman_comparison.png

Usage:
    python plot_null_spearman.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE / "results"
FIGS = HERE / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def left_tail_p(null_vals, real):
    """One-tailed p: fraction of null graphs at least as negative as the real
    value. Reported as < 1/N when none are (resolution floor of the ensemble)."""
    n = len(null_vals)
    k = int((null_vals <= real).sum())
    if k == 0:
        return f"< {1/n:.3f}  (0/{n})"
    return f"{k/n:.3f}  ({k}/{n})"


def panel(ax, null_vals, real, label, color):
    # Shaded band over the null's full spread. This stays visible even when the
    # null is razor-thin (Reddit's huge graph gives a very stable null), so the
    # gap between real and null is legible on both panels.
    ax.axvspan(float(null_vals.min()), float(null_vals.max()),
               color="#9ecae1", alpha=0.45, label="null spread (300 graphs)")
    ax.hist(null_vals, bins=30, color="#6baed6", edgecolor="white")
    ax.axvline(real, color=color, lw=2.5, linestyle="--",
               label=f"{label} (real) = {real:.2f}")
    ax.axvline(float(null_vals.mean()), color="#737373", lw=1.2, linestyle=":",
               label=f"null mean = {null_vals.mean():.2f}")
    ax.set_xlabel("within-core Spearman ρ (k-core vs clustering)")
    ax.set_ylabel("number of null graphs")
    p = left_tail_p(null_vals, real)
    ax.set_title(f"{label}    (p = {p})", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper center")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3, color="#cccccc")
    # own range: show the real line on the left and the null cloud, with margin
    lo = min(float(null_vals.min()), real) - 0.08
    hi = max(float(null_vals.max()), real) + 0.08
    ax.set_xlim(lo, hi)


def main():
    mb_null = pd.read_csv(R / "null_ensemble.csv")["rho_core"]
    rd_null = pd.read_csv(R / "reddit_null_ensemble.csv")["rho_core"]
    mb_real = json.loads((R / "real_summary.json").read_text())["rho_core"]
    rd_real = json.loads((R / "reddit_real_summary.json").read_text())["rho_core"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    panel(axes[0], rd_null, rd_real, "Reddit (humans)", "#2171b5")
    panel(axes[1], mb_null, mb_real, "Moltbook (agents)", "#d62728")

    fig.suptitle("Central communities are bridges on BOTH platforms, far beyond chance\n"
                 "within-core k-core / clustering correlation vs configuration-model null",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = FIGS / "null_spearman_comparison.png"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  Reddit   real rho_core = {rd_real:.3f}, null mean = {rd_null.mean():.3f}")
    print(f"  Moltbook real rho_core = {mb_real:.3f}, null mean = {mb_null.mean():.3f}")


if __name__ == "__main__":
    main()
