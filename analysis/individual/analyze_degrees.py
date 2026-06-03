"""
analyze.py — power-law analysis of the degree sequences.

For each distribution we:
  1. plot the complementary CDF (CCDF) on log-log axes  (Notes §3.3.1);
  2. fit a power law by maximum likelihood, choosing the lower cutoff k_min by
     KS minimization  (Clauset, Shalizi & Newman 2009, via the `powerlaw` pkg);
  3. compare the power law against lognormal and exponential alternatives with
     a likelihood-ratio test — "looks straight on a log-log plot" is not enough
     to claim scale-free, so this test is what the claim actually rests on.

We also compute projection-free concentration measures (Gini, top-k share) for
the Moltbook activity distributions — the "ghost town" signal.

Outputs (results/, committed):
    degree_ccdf_comparison.png   the headline figure (Reddit vs Moltbook)
    powerlaw_summary.csv         alpha, k_min, GoF, and alt-distribution tests
    concentration.csv            Gini + top-k shares
    ccdf_<name>.png              per-distribution CCDF with fit (for the SI)

Usage:
    python scripts/analyze.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import powerlaw

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DEG_DIR = SCRIPT_DIR / "data" / "degrees"   # written by extract_degrees.py
RESULTS = SCRIPT_DIR / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "font.family": "DejaVu Sans",
})

# Distributions to analyze: label -> (csv filename, value column, is the
# community-graph degree?)
DISTS = [
    ("Reddit subreddit degree",    "reddit_hyperlink_degree.csv", "degree", True),
    ("Moltbook submolt degree (core)", "moltbook_core_degree.csv", "degree", True),
    ("Moltbook posts per submolt", "moltbook_posts_per_submolt.csv", "value", False),
    ("Moltbook authors per submolt", "moltbook_authors_per_submolt.csv", "value", False),
    ("Moltbook posts per agent",   "moltbook_posts_per_agent.csv", "value", False),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_values(csv_name: str, col: str) -> np.ndarray:
    df = pd.read_csv(DEG_DIR / csv_name)
    v = df[col].to_numpy(dtype=float)
    return v[v > 0]                       # power-law fit needs positive values


def ccdf(values: np.ndarray):
    """Empirical CCDF: P(X >= x). Returns sorted unique x and P(X>=x)."""
    x = np.sort(values)
    n = len(x)
    # P(X >= x_i) using rank from the top
    xs, idx = np.unique(x, return_index=True)
    p = 1.0 - idx / n
    return xs, p


def gini(values: np.ndarray) -> float:
    x = np.sort(values)
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def top_k_share(values: np.ndarray, frac: float) -> float:
    x = np.sort(values)[::-1]
    k = max(1, int(round(frac * len(x))))
    return float(x[:k].sum() / x.sum())


def fit_powerlaw(values: np.ndarray) -> dict:
    """MLE power-law fit + comparison to lognormal & exponential."""
    fit = powerlaw.Fit(values, discrete=True, verbose=False)
    # Likelihood-ratio tests: R>0 favors power law, R<0 favors the alternative;
    # p is the significance of the sign.
    R_ln, p_ln = fit.distribution_compare(
        "power_law", "lognormal", normalized_ratio=True)
    R_exp, p_exp = fit.distribution_compare(
        "power_law", "exponential", normalized_ratio=True)
    return {
        "alpha": fit.power_law.alpha,
        "k_min": fit.power_law.xmin,
        "ks_distance": fit.power_law.D,
        "n_tail": int((values >= fit.power_law.xmin).sum()),
        "R_vs_lognormal": R_ln, "p_vs_lognormal": p_ln,
        "R_vs_exponential": R_exp, "p_vs_exponential": p_exp,
        "_fit": fit,
    }


def verdict(res: dict) -> str:
    """Plain-language read of whether the tail is power-law / scale-free."""
    pl_beats_exp = res["R_vs_exponential"] > 0 and res["p_vs_exponential"] < 0.10
    not_worse_than_ln = not (res["R_vs_lognormal"] < 0 and res["p_vs_lognormal"] < 0.10)
    if pl_beats_exp and not_worse_than_ln:
        return "heavy-tailed; power law plausible"
    if pl_beats_exp:
        return "heavy-tailed but lognormal fits at least as well"
    return "no clear power-law tail"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rows = []
    conc_rows = []
    fits = {}

    for label, csv_name, col, is_degree in DISTS:
        values = load_values(csv_name, col)
        res = fit_powerlaw(values)
        fits[label] = (values, res)
        rows.append({
            "distribution": label,
            "n": len(values),
            "max": int(values.max()),
            "mean": round(float(values.mean()), 2),
            "alpha": round(res["alpha"], 3),
            "k_min": round(res["k_min"], 1),
            "ks_distance": round(res["ks_distance"], 4),
            "n_tail": res["n_tail"],
            "R_vs_lognormal": round(res["R_vs_lognormal"], 2),
            "p_vs_lognormal": round(res["p_vs_lognormal"], 3),
            "R_vs_exponential": round(res["R_vs_exponential"], 2),
            "p_vs_exponential": round(res["p_vs_exponential"], 3),
            "verdict": verdict(res),
        })
        conc_rows.append({
            "distribution": label,
            "n": len(values),
            "gini": round(gini(values), 3),
            "top_1pct_share": round(top_k_share(values, 0.01), 3),
            "top_5pct_share": round(top_k_share(values, 0.05), 3),
            "top_10pct_share": round(top_k_share(values, 0.10), 3),
        })
        print(f"  {label:38s}  n={len(values):6d}  alpha={res['alpha']:.2f}  "
              f"k_min={res['k_min']:.0f}  -> {verdict(res)}")

    pd.DataFrame(rows).to_csv(RESULTS / "powerlaw_summary.csv", index=False)
    pd.DataFrame(conc_rows).to_csv(RESULTS / "concentration.csv", index=False)
    print(f"\n[done] summary tables -> {RESULTS}")

    make_comparison_figure(fits)
    make_individual_figures(fits)


def make_comparison_figure(fits: dict) -> None:
    """Headline: Reddit vs Moltbook community-graph degree, CCDF, log-log."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    palette = {
        "Reddit subreddit degree": ("#0072B2", "o"),
        "Moltbook submolt degree (core)": ("#D55E00", "s"),
        "Moltbook submolt degree (full)": ("#E69F00", "^"),
    }
    for label, (color, marker) in palette.items():
        if label not in fits:
            print(f"  [skip] {label} not found in fits; skipping")
            continue
        values, res = fits[label]
        xs, p = ccdf(values)
        ax.scatter(xs, p, s=12, color=color, marker=marker, alpha=0.55,
                   edgecolors="none", label=f"{label}  (α≈{res['alpha']:.2f})")
        # overlay the fitted power-law CCDF on its tail (k >= k_min)
        fit = res["_fit"]
        try:
            fit.power_law.plot_ccdf(ax=ax, color=color, linestyle="--", linewidth=1.5)
        except Exception:
            # if plotting the fitted CCDF fails for whatever reason, continue
            pass

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree  k  (number of connected communities)")
    ax.set_ylabel("P(degree ≥ k)")
    ax.set_title("Community-network degree distributions: Reddit vs. Moltbook\n"
                 "(log-log CCDF; dashed = maximum-likelihood power-law fit on the tail)")
    ax.legend(fontsize=9, frameon=False, loc="lower left")
    ax.grid(True, which="both", linestyle="--", alpha=0.3, color="#cccccc")
    fig.tight_layout()
    fig.savefig(RESULTS / "degree_ccdf_comparison.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"[fig] degree_ccdf_comparison.png")


def make_individual_figures(fits: dict) -> None:
    """Per-distribution CCDF with its own fit — for the SI / appendix."""
    for label, (values, res) in fits.items():
        fit = res["_fit"]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        fit.plot_ccdf(ax=ax, color="#333333", linewidth=0, marker="o",
                      markersize=3, label="data")
        fit.power_law.plot_ccdf(ax=ax, color="#D55E00", linestyle="--",
                                linewidth=1.8, label=f"power law α={res['alpha']:.2f}")
        ax.set_xlabel("value  x")
        ax.set_ylabel("P(X ≥ x)")
        ax.set_title(f"{label}\n(k_min={res['k_min']:.0f}, {verdict(res)})")
        ax.legend(fontsize=9, frameon=False)
        ax.grid(True, which="both", linestyle="--", alpha=0.3, color="#cccccc")
        fig.tight_layout()
        slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(RESULTS / f"ccdf_{slug}.png", bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
    print(f"[fig] {len(fits)} per-distribution CCDF figures")


if __name__ == "__main__":
    main()
