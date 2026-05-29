"""
RQ1 — Broadcast vs. Viral Diffusion (Method 2: Null Model Comparison)
======================================================================

Directed configuration model null ensemble: 500 random graphs that preserve
Reddit's in/out-degree sequences but randomize which nodes connect to which.
Edge weights are assigned by sampling from the real graph's empirical weight
distribution (95th-percentile-capped, matching Method 1), so the identical
IC spread formula P = 1-(1-β)^w is used — making this a direct apples-to-
apples comparison with Method 1.

Permutation test:
  p_depth  = fraction of null graphs with median_depth  >= Reddit's value
  p_width1 = fraction of null graphs with median_width1 <= Reddit's value
  (one-tailed; both test the viral direction)

Bonferroni correction applied for 12 simultaneous tests (6 β × 2 metrics);
corrected threshold = 0.05/12 ≈ 0.0042.

Usage:
    python reddit/rq1_null_model.py

Imports sir_run() and normalize_edge_weights() from rq1_diffusion_mode.py.
"""

from __future__ import annotations

import multiprocessing
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from graph_builder import load_graphs
from rq1_diffusion_mode import normalize_edge_weights, sir_run


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
METRICS_DIR = SCRIPT_DIR / "data" / "processed" / "metrics"
FIGURES_DIR = SCRIPT_DIR / "figures" / "rq1"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_NULL          = 500     # null graphs in ensemble
BETAS           = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
N_SEEDS_NULL    = 100     # seeds per null graph (vs 500 in Method 1)
N_RUNS_NULL     = 10      # runs per seed per null graph (vs 20 in Method 1)
MAX_DEPTH       = 45
RNG_SEED        = 1234    # different from Method 1 (42) to avoid correlation
N_WORKERS       = max(1, multiprocessing.cpu_count() - 1)
BONFERRONI_N    = 12      # 6 β × 2 metrics

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "DejaVu Sans",
})


# ---------------------------------------------------------------------------
# Null graph construction
# ---------------------------------------------------------------------------
def build_null_graph(in_seq: list[int], out_seq: list[int],
                     real_weights: list[float], rng: random.Random) -> nx.DiGraph:
    """One directed configuration model graph with sampled edge weights.

    Preserves degree sequence; randomizes topology and weight-degree correlations.
    Self-loops and multi-edges (artifacts of stub matching) are removed.
    """
    nx_seed = rng.randint(0, 2**31 - 1)
    H = nx.directed_configuration_model(in_seq, out_seq, seed=nx_seed)
    H = nx.DiGraph(H)
    H.remove_edges_from(list(nx.selfloop_edges(H)))
    for u, v in H.edges():
        H[u][v]["weight"] = rng.choice(real_weights)
    return H


# ---------------------------------------------------------------------------
# SIR on one null graph — worker-safe top-level function
# ---------------------------------------------------------------------------
def _worker(args: tuple) -> tuple[int, list[dict]]:
    """Build one null graph and run the full SIR sweep on it.

    Top-level function required for multiprocessing.Pool pickling.
    Returns (null_idx, list_of_per_beta_stat_dicts).
    """
    (null_idx, in_seq, out_seq, real_weights,
     betas, n_seeds, n_runs, base_seed, max_depth) = args

    rng = random.Random(base_seed + null_idx * 7919)  # prime offset for independence
    G_null = build_null_graph(in_seq, out_seq, real_weights, rng)

    nodes = list(G_null.nodes())
    if len(nodes) < n_seeds:
        seeds = nodes
    else:
        seeds = rng.sample(nodes, n_seeds)

    rows = []
    for beta in betas:
        depths, width1s, totals = [], [], []
        for seed_node in seeds:
            for _ in range(n_runs):
                result = sir_run(G_null, seed_node, beta, rng, max_depth=max_depth)
                if result["total_reached"] > 1:
                    w1 = (result["width_per_level"][1]
                          if len(result["width_per_level"]) > 1 else 0)
                    depths.append(result["depth"])
                    width1s.append(w1)
                    totals.append(result["total_reached"])

        n_total = len(seeds) * n_runs
        n_active = len(depths)
        rows.append({
            "null_idx":       null_idx,
            "beta":           beta,
            "active_frac":    n_active / n_total if n_total > 0 else 0.0,
            "median_depth":   float(np.median(depths)) if depths else np.nan,
            "median_width_1": float(np.median(width1s)) if width1s else np.nan,
            "mean_depth":     float(np.mean(depths))   if depths else np.nan,
            "mean_width_1":   float(np.mean(width1s))  if width1s else np.nan,
            "n_active":       n_active,
        })
    return null_idx, rows


# ---------------------------------------------------------------------------
# Ensemble runner
# ---------------------------------------------------------------------------
def build_null_ensemble(in_seq: list[int], out_seq: list[int],
                        real_weights: list[float], n_null: int,
                        betas: list[float], n_seeds: int, n_runs: int,
                        rng_seed: int, max_depth: int,
                        n_workers: int) -> pd.DataFrame:
    """Run the full null ensemble in parallel. Returns long-format DataFrame."""
    args_list = [
        (i, in_seq, out_seq, real_weights, betas, n_seeds, n_runs, rng_seed, max_depth)
        for i in range(n_null)
    ]

    all_rows = []
    if n_workers > 1:
        with multiprocessing.Pool(processes=n_workers) as pool:
            for null_idx, rows in tqdm(
                pool.imap_unordered(_worker, args_list),
                total=n_null, desc=f"Null ensemble ({n_workers} workers)"
            ):
                all_rows.extend(rows)
    else:
        for args in tqdm(args_list, desc="Null ensemble (sequential)"):
            _, rows = _worker(args)
            all_rows.extend(rows)

    return pd.DataFrame(all_rows).sort_values(["null_idx", "beta"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------
def permutation_test(real_summary: pd.DataFrame,
                     null_df: pd.DataFrame) -> pd.DataFrame:
    """One-tailed permutation tests per β for depth (viral = deeper) and width_1 (viral = narrower).

    p_depth  = P(null median_depth  >= real median_depth)   — right tail
    p_width1 = P(null median_width1 <= real median_width1)  — left tail
    """
    bonferroni_threshold = 0.05 / BONFERRONI_N
    rows = []

    real_idx = real_summary.set_index("beta")

    for beta in BETAS:
        null_sub = null_df[null_df["beta"] == beta]["median_depth"].dropna()
        null_w1  = null_df[null_df["beta"] == beta]["median_width_1"].dropna()

        real_depth  = real_idx.loc[beta, "active_median_depth"]
        real_width1 = real_idx.loc[beta, "active_median_width_1"]

        p_depth  = (null_sub >= real_depth).mean()  if len(null_sub) > 0 else np.nan
        p_width1 = (null_w1  <= real_width1).mean() if len(null_w1)  > 0 else np.nan

        sig_depth  = p_depth  < bonferroni_threshold if not np.isnan(p_depth)  else False
        sig_width1 = p_width1 < bonferroni_threshold if not np.isnan(p_width1) else False

        rows.append({
            "beta":                  beta,
            "real_median_depth":     real_depth,
            "null_depth_mean":       null_sub.mean(),
            "null_depth_std":        null_sub.std(),
            "p_depth":               p_depth,
            "sig_depth":             sig_depth,
            "real_median_width1":    real_width1,
            "null_width1_mean":      null_w1.mean(),
            "null_width1_std":       null_w1.std(),
            "p_width1":              p_width1,
            "sig_width1":            sig_width1,
            "bonferroni_threshold":  bonferroni_threshold,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _save(fig, name: str) -> None:
    fig.savefig(FIGURES_DIR / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_null_depth_distribution(null_df: pd.DataFrame,
                                  real_summary: pd.DataFrame) -> None:
    betas = sorted(null_df["beta"].unique())
    ncols = 3
    nrows = -(-len(betas) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()
    palette = sns.color_palette("colorblind")
    real_idx = real_summary.set_index("beta")

    for i, beta in enumerate(betas):
        ax = axes[i]
        vals = null_df[null_df["beta"] == beta]["median_depth"].dropna()
        ax.hist(vals, bins=30, color=palette[0], alpha=0.8, edgecolor="white")
        real_val = real_idx.loc[beta, "active_median_depth"]
        ax.axvline(real_val, color="red", linewidth=2, linestyle="--",
                   label=f"Reddit = {real_val:.1f}")
        ax.set_title(f"β = {beta}")
        ax.set_xlabel("Null median depth")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Null Distribution of Median Cascade Depth vs. Reddit (red line)",
                 fontweight="bold", y=1.02)
    _save(fig, "null_depth_distribution_by_beta.png")


def plot_null_width1_distribution(null_df: pd.DataFrame,
                                   real_summary: pd.DataFrame) -> None:
    betas = sorted(null_df["beta"].unique())
    ncols = 3
    nrows = -(-len(betas) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()
    palette = sns.color_palette("colorblind")
    real_idx = real_summary.set_index("beta")

    for i, beta in enumerate(betas):
        ax = axes[i]
        vals = null_df[null_df["beta"] == beta]["median_width_1"].dropna()
        ax.hist(vals, bins=30, color=palette[1], alpha=0.8, edgecolor="white")
        real_val = real_idx.loc[beta, "active_median_width_1"]
        ax.axvline(real_val, color="red", linewidth=2, linestyle="--",
                   label=f"Reddit = {real_val:.1f}")
        ax.set_title(f"β = {beta}")
        ax.set_xlabel("Null median width at hop 1")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Null Distribution of Median Hop-1 Width vs. Reddit (red line)",
                 fontweight="bold", y=1.02)
    _save(fig, "null_width1_distribution_by_beta.png")


def plot_permutation_test_summary(ptest: pd.DataFrame) -> None:
    palette = sns.color_palette("colorblind")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    threshold = ptest["bonferroni_threshold"].iloc[0]
    betas_str = [str(b) for b in ptest["beta"]]

    for ax, col, label, color in [
        (axes[0], "p_depth",  "p-value: depth (viral = deeper)", palette[0]),
        (axes[1], "p_width1", "p-value: width₁ (viral = narrower)", palette[1]),
    ]:
        bars = ax.bar(betas_str, ptest[col], color=color, alpha=0.8, edgecolor="white")
        ax.axhline(0.05, color="orange", linestyle="--", linewidth=1.5, label="p=0.05 (uncorrected)")
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5,
                   label=f"p={threshold:.4f} (Bonferroni)")
        for bar, val in zip(bars, ptest[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xlabel("β")
        ax.set_ylabel("p-value")
        ax.set_title(label)
        ax.set_ylim(0, max(1.0, ptest[col].max() + 0.1))
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4, color="#cccccc")

    fig.suptitle("Permutation Test: Is Reddit's Viral Pattern Statistically Non-Random?",
                 fontweight="bold", y=1.02)
    _save(fig, "permutation_test_summary.png")


def plot_real_vs_null_scatter(null_df: pd.DataFrame,
                               real_summary: pd.DataFrame) -> None:
    betas = sorted(null_df["beta"].unique())
    ncols = 3
    nrows = -(-len(betas) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()
    palette = sns.color_palette("colorblind")
    real_idx = real_summary.set_index("beta")

    for i, beta in enumerate(betas):
        ax = axes[i]
        sub = null_df[null_df["beta"] == beta].dropna(subset=["median_depth", "median_width_1"])
        ax.scatter(sub["median_depth"], sub["median_width_1"],
                   alpha=0.4, s=15, color=palette[0], label="Null graphs")
        rd = real_idx.loc[beta, "active_median_depth"]
        rw = real_idx.loc[beta, "active_median_width_1"]
        ax.scatter([rd], [rw], color="red", s=100, zorder=5, marker="*", label="Reddit")
        ax.set_xlabel("Median depth")
        ax.set_ylabel("Median width at hop 1")
        ax.set_title(f"β = {beta}")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Joint Null Distribution: Depth vs. Width₁ (Reddit = red star)",
                 fontweight="bold", y=1.02)
    _save(fig, "real_vs_null_scatter.png")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("RQ1 — Method 2: Null Model Comparison")
    print("=" * 72)

    print("\n[1/6] Loading graphs and Method 1 results...")
    graphs = load_graphs()
    G_raw = graphs["weighted"]
    G_norm, cap, _ = normalize_edge_weights(G_raw)
    real_summary = pd.read_csv(METRICS_DIR / "rq1_broadcast_score.csv")
    print(f"      graph: {G_norm.number_of_nodes():,} nodes, {G_norm.number_of_edges():,} edges")
    print(f"      weight cap: {cap:.2f}")

    in_seq  = [d for _, d in G_norm.in_degree()]
    out_seq = [d for _, d in G_norm.out_degree()]
    real_weights = [d["weight"] for _, _, d in G_norm.edges(data=True)]

    null_csv = METRICS_DIR / "rq1_null_ensemble.csv"
    if null_csv.exists():
        print(f"\n[2/6] Loading cached null ensemble from {null_csv.name}...")
        null_df = pd.read_csv(null_csv)
        print(f"      loaded {len(null_df):,} rows")
    else:
        print(f"\n[2/6] Building null ensemble: {N_NULL} graphs × {N_SEEDS_NULL} seeds × "
              f"{N_RUNS_NULL} runs × {len(BETAS)} β = "
              f"{N_NULL * N_SEEDS_NULL * N_RUNS_NULL * len(BETAS):,} simulations")
        print(f"      parallelism: {N_WORKERS} workers")
        null_df = build_null_ensemble(
            in_seq, out_seq, real_weights,
            N_NULL, BETAS, N_SEEDS_NULL, N_RUNS_NULL,
            RNG_SEED, MAX_DEPTH, N_WORKERS,
        )
        null_df.to_csv(null_csv, index=False)
        print(f"      saved {len(null_df):,} rows → rq1_null_ensemble.csv")

    print("\n[3/6] Running permutation tests...")
    ptest = permutation_test(real_summary, null_df)
    ptest.to_csv(METRICS_DIR / "rq1_permutation_test.csv", index=False)
    print("\n      Results (Bonferroni threshold = {:.4f}):".format(
        0.05 / BONFERRONI_N))
    print(ptest[["beta", "real_median_depth", "null_depth_mean",
                  "p_depth", "sig_depth",
                  "real_median_width1", "null_width1_mean",
                  "p_width1", "sig_width1"]].to_string(index=False))

    print("\n[4/6] Generating figures...")
    plot_null_depth_distribution(null_df, real_summary)
    plot_null_width1_distribution(null_df, real_summary)
    plot_permutation_test_summary(ptest)
    plot_real_vs_null_scatter(null_df, real_summary)
    print(f"      4 figures saved → {FIGURES_DIR}")

    print("\n[5/6] Sanity checks...")
    assert len(null_df) == N_NULL * len(BETAS), \
        f"Expected {N_NULL * len(BETAS)} rows, got {len(null_df)}"
    assert ptest["p_depth"].dropna().between(0, 1).all(), "p_depth out of [0,1]"
    assert ptest["p_width1"].dropna().between(0, 1).all(), "p_width1 out of [0,1]"
    assert not null_df["median_depth"].isna().all(), "all null depths are NaN"
    print("      ✓ all sanity checks passed")

    print("\n[6/6] Final RQ1 conclusion:")
    sig_depth_any  = ptest["sig_depth"].any()
    sig_width1_any = ptest["sig_width1"].any()
    if sig_depth_any and sig_width1_any:
        print("      ✅ Reddit's viral diffusion pattern is statistically significant.")
        print("         Both depth (deeper) and width₁ (narrower) pass Bonferroni correction")
        print("         at at least one β. Viral structure is non-random — not explained by")
        print("         degree distribution alone. RQ2 relay-node reframing is supported.")
    elif sig_depth_any or sig_width1_any:
        print("      ⚠  Partial significance: one metric passes Bonferroni, the other does not.")
        print("         Interpret with caution; viral pattern is suggestive but mixed.")
    else:
        print("      ❌ Reddit's viral pattern is NOT statistically distinguishable from")
        print("         configuration model null. The degree distribution alone explains")
        print("         the observed cascade shapes. Topology beyond degree is not driving")
        print("         the viral pattern.")

    print("\nDone.")


if __name__ == "__main__":
    main()
