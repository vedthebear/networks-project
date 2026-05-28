"""
RQ1 — Broadcast vs. Viral Diffusion (Method 1: SIR Simulation)
===============================================================

Independent Cascade model (discrete-time SIR with gamma=1 fixed) on the Reddit
hyperlink network. Sweeps beta across six values, samples 500 random seeds,
runs 20 Monte Carlo realizations per (beta, seed). Produces per-cascade
depth/width/reach measurements, two complementary broadcast scores, survival
curves, and 8 publication-quality figures.

Method 2 (configuration-model null comparison + permutation test) is OUT OF
SCOPE for this file — that produces the p-value and lives in a separate module.

Usage:
    python reddit/rq1_diffusion_mode.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_builder import load_graphs


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
METRICS_DIR = SCRIPT_DIR / "data" / "processed" / "metrics"
FIGURES_DIR = SCRIPT_DIR / "figures" / "rq1"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BETAS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
N_SEEDS = 500
N_RUNS = 20
RNG_SEED = 42
MAX_DEPTH = 45
WEIGHT_CAP_PERCENTILE = 95


# ---------------------------------------------------------------------------
# Plot style (per reddit/CLAUDE.md visualization standards)
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
# Weight normalization (95th-percentile cap)
# ---------------------------------------------------------------------------
def normalize_edge_weights(G: nx.DiGraph, percentile: int = WEIGHT_CAP_PERCENTILE
                           ) -> tuple[nx.DiGraph, float, float]:
    """Cap edge weights at a percentile to prevent saturation in IC spread.

    Raw Reddit hyperlink counts have a heavy tail — a few subreddit pairs
    link hundreds of times. With p_spread = 1 - (1-beta)^w, those edges go
    deterministic at moderate beta. We cap weights at the percentile cutoff
    of the edge-weight distribution.

    Returns the normalized graph, the cap value used, and the fraction of
    edges whose weight was reduced.
    """
    weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    cap = float(np.percentile(weights, percentile))
    n_affected = int((weights > cap).sum())
    frac_affected = n_affected / len(weights)

    H = G.copy()
    for u, v, d in H.edges(data=True):
        if d["weight"] > cap:
            d["weight"] = cap
    return H, cap, frac_affected


# ---------------------------------------------------------------------------
# SIR simulation (Independent Cascade)
# ---------------------------------------------------------------------------
def sir_run(G: nx.DiGraph, seed: str, beta: float, rng: random.Random,
            max_depth: int = MAX_DEPTH) -> dict:
    """One Independent Cascade run from one seed.

    Each infected node gets a single time step to infect each susceptible
    out-neighbor with probability 1 - (1 - beta)^weight. Synchronous updates.
    Cascade terminates when no new infections occur or when max_depth is hit.
    """
    infected = {seed}
    recovered = set()
    width_per_level = [1]
    depth = 0
    hit_depth_cap = False

    while infected:
        if depth >= max_depth:
            hit_depth_cap = True
            break

        newly_infected = set()
        for u in infected:
            for v in G.successors(u):
                if v in recovered or v in infected:
                    continue
                # If v is already in newly_infected from another u this step,
                # still draw the coin (IC gives each u→v pair an independent
                # attempt); we just don't re-add. Drawing keeps the rng stream
                # deterministic regardless of iteration order.
                w = G[u][v]["weight"]
                p_spread = 1.0 - (1.0 - beta) ** w
                if rng.random() < p_spread:
                    newly_infected.add(v)

        recovered |= infected
        infected = newly_infected
        if infected:
            depth += 1
            width_per_level.append(len(infected))

    return {
        "depth": depth,
        "width_per_level": width_per_level,
        "total_reached": sum(width_per_level),
        "hit_depth_cap": hit_depth_cap,
    }


def measure_all_cascades(G: nx.DiGraph, betas: list[float], n_seeds: int,
                         n_runs: int, rng_seed: int,
                         max_depth: int = MAX_DEPTH) -> pd.DataFrame:
    """Full beta x seed x run sweep. Returns long-format DataFrame."""
    rng = random.Random(rng_seed)
    nodes = list(G.nodes())
    seeds = rng.sample(nodes, n_seeds)

    rows = []
    width_per_level_records = []  # collected for survival curves later

    total = len(betas) * n_seeds * n_runs
    with tqdm(total=total, desc="SIR sweep") as bar:
        for beta in betas:
            for seed_node in seeds:
                for run_idx in range(n_runs):
                    result = sir_run(G, seed_node, beta, rng, max_depth=max_depth)
                    width_1 = (result["width_per_level"][1]
                               if len(result["width_per_level"]) > 1 else 0)
                    rows.append({
                        "beta": beta,
                        "seed": seed_node,
                        "run": run_idx,
                        "depth": result["depth"],
                        "width_1": width_1,
                        "total_reached": result["total_reached"],
                        "hit_depth_cap": result["hit_depth_cap"],
                    })
                    width_per_level_records.append(
                        (beta, result["width_per_level"])
                    )
                    bar.update(1)

    df = pd.DataFrame(rows)
    df.attrs["width_per_level_records"] = width_per_level_records
    return df


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------
def summarize_broadcast_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Both broadcast scores per beta, for all cascades and active-only.

    Many seeds in the Reddit network have out-degree 0 or 1, so the median
    across ALL runs is uninformative (dominated by zero-spread cascades).
    We report active-only stats (total_reached > 1) as the primary measure,
    plus the full-sample fraction-active for context.
    """
    out = []
    for beta, sub in df.groupby("beta"):
        active = sub[sub["total_reached"] > 1]
        frac_active = len(active) / len(sub)

        # All-sample medians (for completeness; mostly 0 due to zero-spread seeds)
        med_depth_all = sub["depth"].median()
        med_width1_all = sub["width_1"].median()

        # Active-cascade stats — the informative measure
        if len(active) > 0:
            med_depth = active["depth"].median()
            med_width_1 = active["width_1"].median()
            med_total = active["total_reached"].median()
            ratio = (med_width_1 / med_depth) if med_depth > 0 else np.nan
            per_run = active["width_1"] / active["total_reached"]
            reach_score = per_run.median()
            mean_depth = active["depth"].mean()
            mean_width_1 = active["width_1"].mean()
            mean_total = active["total_reached"].mean()
        else:
            med_depth = med_width_1 = med_total = ratio = reach_score = np.nan
            mean_depth = mean_width_1 = mean_total = np.nan

        out.append({
            "beta": beta,
            # Active-cascade primary stats
            "active_frac": frac_active,
            "active_median_depth": med_depth,
            "active_median_width_1": med_width_1,
            "active_median_total_reached": med_total,
            "active_mean_depth": mean_depth,
            "active_mean_width_1": mean_width_1,
            "active_mean_total_reached": mean_total,
            "broadcast_score_ratio": ratio,
            "broadcast_score_reach": reach_score,
            # All-sample context
            "all_median_depth": med_depth_all,
            "all_median_width_1": med_width1_all,
            "frac_hit_cap": sub["hit_depth_cap"].mean(),
            "n_runs": len(sub),
            "n_active": len(active),
        })
    return pd.DataFrame(out).sort_values("beta").reset_index(drop=True)


def compute_survival_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Empirical survival S(d) = P(cascade depth >= d), per beta."""
    rows = []
    for beta, sub in df.groupby("beta"):
        max_d = int(sub["depth"].max())
        depths = sub["depth"].values
        n = len(depths)
        for d in range(0, max_d + 2):
            rows.append({
                "beta": beta,
                "depth": d,
                "survival": (depths >= d).sum() / n,
            })
    return pd.DataFrame(rows)


def compute_mean_width_per_level(df: pd.DataFrame,
                                  G: nx.DiGraph | None = None,
                                  betas: list | None = None,
                                  n_seeds: int = N_SEEDS,
                                  n_runs: int = N_RUNS,
                                  rng_seed: int = RNG_SEED,
                                  max_depth: int = MAX_DEPTH) -> pd.DataFrame:
    """Mean width at each depth level per beta, from a small replay sample.

    The full width_per_level is not stored in the CSV (only depth and width_1
    are). We replay a 50-seed sample to recover per-level width profiles.
    If G is None, returns an approximation using only width_1 (level-1 bar)
    and depth (endpoint bar) from the CSV — useful as a fallback.
    """
    if G is not None and betas is not None:
        rng = random.Random(rng_seed + 9999)  # different seed from main sweep
        sample_seeds = rng.sample(list(G.nodes()), min(50, n_seeds))
        rows = []
        by_beta_level: dict = {}
        for beta in betas:
            for seed_node in sample_seeds:
                for _ in range(n_runs):
                    result = sir_run(G, seed_node, beta, rng, max_depth=max_depth)
                    for level, w in enumerate(result["width_per_level"]):
                        by_beta_level.setdefault((beta, level), []).append(w)
        for (beta, level), widths in by_beta_level.items():
            rows.append({"beta": beta, "level": level, "mean_width": float(np.mean(widths))})
        return pd.DataFrame(rows).sort_values(["beta", "level"]).reset_index(drop=True)

    # Fallback: reconstruct from CSV columns (width_1 at level 1, depth as proxy)
    rows = []
    for beta, sub in df.groupby("beta"):
        active = sub[sub["total_reached"] > 1]
        rows.append({"beta": beta, "level": 0, "mean_width": 1.0})
        if len(active) > 0:
            rows.append({"beta": beta, "level": 1, "mean_width": active["width_1"].mean()})
            rows.append({"beta": beta, "level": active["depth"].mean(),
                         "mean_width": 1.0})  # proxy endpoint
    return pd.DataFrame(rows).sort_values(["beta", "level"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _save(fig, name: str) -> None:
    fig.savefig(FIGURES_DIR / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_depth_histogram(df: pd.DataFrame) -> None:
    g = sns.displot(
        data=df, x="depth", col="beta", col_wrap=3,
        kind="hist", bins=30, height=3, aspect=1.3,
        facet_kws={"sharey": False},
    )
    g.set_titles("β = {col_name}")
    g.set_axis_labels("Cascade depth (hops)", "Count")
    g.fig.suptitle("Cascade Depth Distribution by β", y=1.02, fontweight="bold")
    _save(g.fig, "cascade_depth_histogram_by_beta.png")


def plot_width_hop1_histogram(df: pd.DataFrame) -> None:
    g = sns.displot(
        data=df, x="width_1", col="beta", col_wrap=3,
        kind="hist", bins=40, height=3, aspect=1.3,
        facet_kws={"sharey": False},
    )
    g.set_titles("β = {col_name}")
    g.set_axis_labels("Width at hop 1 (newly infected at level 1)", "Count")
    g.fig.suptitle("Hop-1 Width Distribution by β", y=1.02, fontweight="bold")
    _save(g.fig, "cascade_width_hop1_histogram_by_beta.png")


def plot_total_reach_loglog(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for beta in sorted(df["beta"].unique()):
        reaches = np.sort(df.loc[df["beta"] == beta, "total_reached"].values)
        # CCDF: P(X >= x)
        n = len(reaches)
        ccdf = 1.0 - np.arange(n) / n
        ax.loglog(reaches, ccdf, label=f"β = {beta}", linewidth=1.5)
    ax.set_xlabel("Total reach (unique subreddits infected)")
    ax.set_ylabel("P(reach ≥ x)")
    ax.set_title("Total Cascade Reach — Complementary CDF (log-log)")
    ax.legend(title="β", loc="lower left")
    ax.grid(True, which="both", linestyle="--", alpha=0.4, color="#cccccc")
    _save(fig, "cascade_total_reach_loglog_by_beta.png")


def plot_mean_width_per_level(mwpl: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for beta in sorted(mwpl["beta"].unique()):
        sub = mwpl[mwpl["beta"] == beta]
        ax.plot(sub["level"], sub["mean_width"], marker="o",
                label=f"β = {beta}", linewidth=1.5, markersize=4)
    ax.set_yscale("log")
    ax.set_xlabel("Cascade level (hops from seed)")
    ax.set_ylabel("Mean width (newly infected at this level)")
    ax.set_title("Mean Cascade Width per Level — Broadcast vs. Viral Signature")
    ax.legend(title="β")
    ax.grid(True, which="both", linestyle="--", alpha=0.4, color="#cccccc")
    _save(fig, "mean_width_per_level_by_beta.png")


def plot_broadcast_scores(summary: pd.DataFrame) -> None:
    palette = sns.color_palette("colorblind")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: both scores (active cascades only)
    ax1 = axes[0]
    ax1.plot(summary["beta"], summary["broadcast_score_ratio"],
             marker="o", color=palette[0], linewidth=2, label="Ratio (width₁/depth)")
    ax1.plot(summary["beta"], summary["broadcast_score_reach"],
             marker="s", color=palette[3], linewidth=2, linestyle="--",
             label="Reach (width₁/total)")
    ax1.set_xlabel("β (spread probability)")
    ax1.set_ylabel("Broadcast score (active cascades)")
    ax1.set_title("Broadcast Scores vs. β\n(Active cascades only: total_reached > 1)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4, color="#cccccc")

    # Right: fraction of seeds that produced any spread
    ax2 = axes[1]
    ax2.plot(summary["beta"], summary["active_frac"] * 100,
             marker="o", color=palette[1], linewidth=2)
    ax2.set_xlabel("β (spread probability)")
    ax2.set_ylabel("% of cascades with total_reached > 1")
    ax2.set_title("Cascade Activation Rate vs. β\n(How often spread occurs at all)")
    ax2.grid(True, linestyle="--", alpha=0.4, color="#cccccc")

    fig.suptitle("Broadcast Scores and Activation Rate by β", fontweight="bold", y=1.02)
    _save(fig, "broadcast_score_vs_beta.png")


def plot_depth_vs_width1_hexbin(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    hb = ax.hexbin(
        df["depth"], df["width_1"], gridsize=40, mincnt=1,
        cmap="viridis", bins="log",
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("log10(count)")
    ax.set_xlabel("Cascade depth (hops)")
    ax.set_ylabel("Width at hop 1")
    ax.set_title("Joint Distribution: Depth vs. Hop-1 Width (all runs)")
    ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
    _save(fig, "depth_vs_width1_hexbin.png")


def plot_survival_curves(survival: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for beta in sorted(survival["beta"].unique()):
        sub = survival[survival["beta"] == beta]
        ax.plot(sub["depth"], sub["survival"], marker="o",
                label=f"β = {beta}", linewidth=1.5, markersize=4)
    ax.set_xlabel("Cascade depth d")
    ax.set_ylabel("P(cascade reaches depth ≥ d)")
    ax.set_title("Cascade Survival Curves")
    ax.legend(title="β")
    ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
    _save(fig, "survival_curves_by_beta.png")


def plot_edge_weight_distribution(G_raw: nx.DiGraph, cap: float) -> None:
    weights = np.array([d["weight"] for _, _, d in G_raw.edges(data=True)])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(weights, bins=np.logspace(0, np.log10(weights.max() + 1), 50),
            color=sns.color_palette("colorblind")[0], alpha=0.85, edgecolor="white")
    ax.axvline(cap, color="red", linestyle="--", linewidth=2,
               label=f"95th-percentile cap = {cap:.1f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Edge weight (hyperlink count)")
    ax.set_ylabel("Number of edges")
    ax.set_title("Raw Edge-Weight Distribution with Normalization Cap")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4, color="#cccccc")
    _save(fig, "edge_weight_distribution_with_cap.png")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("RQ1 — Broadcast vs. Viral Diffusion (Method 1: SIR Simulation)")
    print("=" * 72)

    print("\n[1/6] Loading graphs from cache (or building if missing)...")
    graphs = load_graphs()
    G_raw = graphs["weighted"]
    print(f"      weighted graph: {G_raw.number_of_nodes():,} nodes, "
          f"{G_raw.number_of_edges():,} edges")

    print(f"\n[2/6] Normalizing edge weights at {WEIGHT_CAP_PERCENTILE}th percentile...")
    G, cap, frac_affected = normalize_edge_weights(G_raw)
    print(f"      cap value: {cap:.2f}")
    print(f"      fraction of edges affected: {frac_affected:.4%}")

    pd.DataFrame([{
        "weight_cap_percentile": WEIGHT_CAP_PERCENTILE,
        "cap_value": cap,
        "frac_edges_affected": frac_affected,
        "n_edges_total": G.number_of_edges(),
    }]).to_csv(METRICS_DIR / "rq1_edge_weight_normalization.csv", index=False)

    cascade_csv = METRICS_DIR / "rq1_cascade_runs.csv"
    if cascade_csv.exists():
        print(f"\n[3/6] Loading existing simulation results from {cascade_csv.name}...")
        df = pd.read_csv(cascade_csv)
        print(f"      loaded {len(df):,} rows")
    else:
        print(f"\n[3/6] Running SIR sweep: {len(BETAS)} β × {N_SEEDS} seeds × "
              f"{N_RUNS} runs = {len(BETAS) * N_SEEDS * N_RUNS:,} simulations")
        df = measure_all_cascades(G, BETAS, N_SEEDS, N_RUNS, RNG_SEED, MAX_DEPTH)
        df.to_csv(cascade_csv, index=False)
        print(f"      saved {len(df):,} rows → rq1_cascade_runs.csv")

    print("\n[4/6] Computing aggregates...")
    summary = summarize_broadcast_scores(df)
    summary.to_csv(METRICS_DIR / "rq1_broadcast_score.csv", index=False)
    print("      broadcast scores per β:")
    print(summary.to_string(index=False))

    survival = compute_survival_curves(df)
    survival.to_csv(METRICS_DIR / "rq1_survival_curves.csv", index=False)
    print(f"      saved {len(survival):,} survival rows")

    print("      computing mean-width-per-level profile (50-seed replay)...")
    mwpl = compute_mean_width_per_level(df, G=G, betas=BETAS)

    print("\n[5/6] Generating figures...")
    plot_depth_histogram(df)
    plot_width_hop1_histogram(df)
    plot_total_reach_loglog(df)
    plot_mean_width_per_level(mwpl)
    plot_broadcast_scores(summary)
    plot_depth_vs_width1_hexbin(df)
    plot_survival_curves(survival)
    plot_edge_weight_distribution(G_raw, cap)
    print(f"      8 figures saved → {FIGURES_DIR}")

    print("\n[6/6] Sanity checks...")
    assert (df["depth"] >= 0).all(), "negative depth found"
    assert (df["width_1"] >= 0).all(), "negative width_1 found"
    assert (df["total_reached"] > 0).all(), "zero-reach run found (should include seed)"
    cap_hit = df["hit_depth_cap"].sum()
    if cap_hit > 0:
        print(f"      ⚠ {cap_hit} runs hit max_depth={MAX_DEPTH}; review before publishing")
    else:
        print(f"      ✓ no runs hit the depth cap")
    print("      ✓ all sanity checks passed")

    print("\nDone.")


if __name__ == "__main__":
    main()
