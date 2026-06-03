"""
RQ4 — Core-Periphery Structure
==============================================================================

Does a community's GLOBAL structural position (k-core number) predict its
LOCAL connection behavior (clustering coefficient)?

Hypothesis: core communities bridge across many disconnected niches (low
clustering); peripheral communities sit in tight cliques (high clustering)
=> a NEGATIVE k-core <-> clustering correlation, stronger than a degree-matched
configuration-model null would produce.

This is the deliberate contrast to RQ1. RQ1's diffusion shape was fully
degree-explained (the null reproduced it). RQ4 targets features the null
CANNOT reproduce: random rewiring preserves degree but destroys triangles, so
clustering / core-periphery organization is exactly where Reddit can beat the
null.

Metrics (all on the undirected projection of the `weighted` graph):
  - k-core number              nx.core_number       (global embeddedness)
  - clustering coefficient     nx.clustering        (local cohesion / bridging)
  - Spearman(k-core, cluster)  scipy.stats          (the central test)
  - rich-club coefficient      nx.rich_club_*       (elite-core cohesion)

Clustering is reported UNWEIGHTED (primary, topological) and WEIGHTED
(robustness, using RQ1's 95th-pct weight cap). Null ensemble = 500 directed
configuration-model graphs, projected identically to the real graph.

Usage:
    python reddit/rq4_core_periphery.py

Imports `load_graphs` from graph_builder, `normalize_edge_weights` and
`build_null_graph` from the RQ1 modules (construction only — no RQ1 sim output
is reused).
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
from scipy.stats import spearmanr
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from graph_builder import load_graphs
from rq1_diffusion_mode import normalize_edge_weights
from rq1_null_model import build_null_graph


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
METRICS_DIR = SCRIPT_DIR / "data" / "processed" / "metrics"
FIGURES_DIR = SCRIPT_DIR / "figures" / "rq4"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_NULL            = 500
RNG_SEED          = 2024                       # distinct from RQ1 (42, 1234)
N_WORKERS         = max(1, multiprocessing.cpu_count() - 1)
RICHCLUB_MIN_NODES = 10        # only report rich-club at k where > this many nodes exceed k
RICHCLUB_CI       = (2.5, 97.5)  # percentile band for the null rich-club curve

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
# Undirected projection
# ---------------------------------------------------------------------------
def build_undirected(G: nx.DiGraph) -> nx.Graph:
    """Undirected projection of a directed graph.

    An undirected edge exists iff a link runs in EITHER direction. Reciprocal
    weights are SUMMED (A->B + B->A). Self-loops are dropped (k-core, clustering
    and rich-club are undefined / ill-behaved on self-loops).

    The summed weight is stored on every edge; unweighted metrics simply ignore
    it, so this one projection serves both the unweighted and weighted passes.
    """
    UG = nx.Graph()
    UG.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        if u == v:
            continue
        w = d.get("weight", 1.0)
        if UG.has_edge(u, v):
            UG[u][v]["weight"] += w
        else:
            UG.add_edge(u, v, weight=w)
    return UG


# ---------------------------------------------------------------------------
# Per-graph structural statistics
# ---------------------------------------------------------------------------
def richclub_on_ks(UG: nx.Graph, ks: list[int]) -> dict[int, float]:
    """Rich-club coefficient phi(k) restricted to the requested k values.

    nx.rich_club_coefficient(normalized=False) returns {k: phi} for k up to
    max_degree-1; we extract the requested ks (NaN where undefined).
    """
    try:
        phi = nx.rich_club_coefficient(UG, normalized=False)
    except Exception:
        return {k: np.nan for k in ks}
    return {k: float(phi.get(k, np.nan)) for k in ks}


def select_richclub_ks(UG: nx.Graph) -> list[int]:
    """Choose the k range for the rich-club curve: k >= 1 with enough nodes
    above the threshold to be stable (avoids the unstable extreme-degree tail).
    """
    degs = np.array([d for _, d in UG.degree()])
    max_deg = int(degs.max()) if len(degs) else 0
    ks = []
    for k in range(1, max_deg):
        if int((degs > k).sum()) >= RICHCLUB_MIN_NODES:
            ks.append(k)
    return ks


def compute_graph_stats(UG: nx.Graph, ks: list[int],
                        with_weighted: bool) -> dict:
    """k-core, clustering (unweighted; weighted optional), Spearman, rich-club."""
    kcore = nx.core_number(UG)
    clustering_uw = nx.clustering(UG)              # topological
    nodes = list(UG.nodes())

    kc = np.array([kcore[n] for n in nodes], dtype=float)
    cu = np.array([clustering_uw[n] for n in nodes], dtype=float)

    rho_uw, p_uw = spearmanr(kc, cu)

    stats = {
        "nodes": nodes,
        "kcore": kc,
        "clustering_uw": cu,
        "spearman_rho_uw": float(rho_uw),
        "spearman_p_uw": float(p_uw),
        "mean_clustering_uw": float(np.mean(cu)),
        "richclub": richclub_on_ks(UG, ks),
    }

    if with_weighted:
        clustering_w = nx.clustering(UG, weight="weight")
        cw = np.array([clustering_w[n] for n in nodes], dtype=float)
        rho_w, p_w = spearmanr(kc, cw)
        stats.update({
            "clustering_w": cw,
            "spearman_rho_w": float(rho_w),
            "spearman_p_w": float(p_w),
            "mean_clustering_w": float(np.mean(cw)),
        })

    return stats


# ---------------------------------------------------------------------------
# Null worker — build one config-model graph, project, compute stats
# ---------------------------------------------------------------------------
def _worker(args: tuple) -> dict:
    """One null graph: directed config model -> undirected projection -> stats.

    Top-level function for multiprocessing.Pool pickling. Reuses ONLY the null
    construction from RQ1 (build_null_graph); no RQ1 simulation output is used.
    """
    (null_idx, in_seq, out_seq, real_weights, ks, base_seed) = args

    rng = random.Random(base_seed + null_idx * 7919)  # prime offset for independence
    G_null = build_null_graph(in_seq, out_seq, real_weights, rng)
    UG = build_undirected(G_null)
    s = compute_graph_stats(UG, ks, with_weighted=True)

    row = {
        "null_idx":            null_idx,
        "spearman_rho_uw":     s["spearman_rho_uw"],
        "mean_clustering_uw":  s["mean_clustering_uw"],
        "spearman_rho_w":      s["spearman_rho_w"],
        "mean_clustering_w":   s["mean_clustering_w"],
        "richclub":            s["richclub"],
    }
    return row


def build_null_ensemble(in_seq, out_seq, real_weights, ks,
                        n_null, rng_seed, n_workers) -> tuple[pd.DataFrame, dict]:
    """Run the null ensemble in parallel. Returns (scalar_df, richclub_by_k)."""
    args_list = [
        (i, in_seq, out_seq, real_weights, ks, rng_seed)
        for i in range(n_null)
    ]

    rows = []
    if n_workers > 1:
        with multiprocessing.Pool(processes=n_workers) as pool:
            for row in tqdm(pool.imap_unordered(_worker, args_list),
                            total=n_null, desc=f"Null ensemble ({n_workers} workers)"):
                rows.append(row)
    else:
        for args in tqdm(args_list, desc="Null ensemble (sequential)"):
            rows.append(_worker(args))

    # rich-club: gather phi per k across nulls
    richclub_by_k = {k: [] for k in ks}
    for row in rows:
        for k in ks:
            richclub_by_k[k].append(row["richclub"].get(k, np.nan))

    scalar_df = pd.DataFrame(
        [{kk: vv for kk, vv in row.items() if kk != "richclub"} for row in rows]
    ).sort_values("null_idx").reset_index(drop=True)
    return scalar_df, richclub_by_k


# ---------------------------------------------------------------------------
# Permutation tests
# ---------------------------------------------------------------------------
def permutation_test(real: dict, null_df: pd.DataFrame) -> pd.DataFrame:
    """Permutation tests vs. the null ensemble.

    We do NOT pre-assume the sign of the k-core <-> clustering correlation
    (the data, not the original hypothesis, decides). For the Spearman tests
    the tail is chosen by the OBSERVED sign — "is Reddit's correlation more
    extreme, in the direction it actually points, than the null?":
        real_rho >= 0  ->  p = P(null_rho >= real_rho)   (right tail)
        real_rho <  0  ->  p = P(null_rho <= real_rho)   (left tail)
    A two-sided p = P(|null_rho| >= |real_rho|) is also reported.

    For clustering magnitude the test is right-tailed (Reddit more clustered
    than a triangle-destroying random graph):
        p = P(null_clust >= real_clust)
    """
    rows = []

    def clean(null_vals):
        nv = np.asarray(null_vals, dtype=float)
        return nv[~np.isnan(nv)]

    def directional_p(real_val, nv):
        if len(nv) == 0:
            return np.nan, np.nan, "none"
        if real_val >= 0:
            return float((nv >= real_val).mean()), "right", \
                   float((np.abs(nv) >= abs(real_val)).mean())
        return float((nv <= real_val).mean()), "left", \
               float((np.abs(nv) >= abs(real_val)).mean())

    # Spearman stats — direction decided by observed sign
    for name, real_val, col in [
        ("spearman_rho_uw", real["spearman_rho_uw"], "spearman_rho_uw"),
        ("spearman_rho_w",  real["spearman_rho_w"],  "spearman_rho_w"),
    ]:
        nv = clean(null_df[col].values)
        p_dir, tail, p_two = directional_p(real_val, nv)
        rows.append({
            "statistic": name, "real_value": real_val,
            "null_mean": float(np.mean(nv)) if len(nv) else np.nan,
            "null_std":  float(np.std(nv))  if len(nv) else np.nan,
            "tail": tail, "p_value": p_dir, "p_two_sided": p_two,
        })

    # Clustering magnitude — right tail
    for name, real_val, col in [
        ("mean_clustering_uw", real["mean_clustering_uw"], "mean_clustering_uw"),
        ("mean_clustering_w",  real["mean_clustering_w"],  "mean_clustering_w"),
    ]:
        nv = clean(null_df[col].values)
        p = float((nv >= real_val).mean()) if len(nv) else np.nan
        rows.append({
            "statistic": name, "real_value": real_val,
            "null_mean": float(np.mean(nv)) if len(nv) else np.nan,
            "null_std":  float(np.std(nv))  if len(nv) else np.nan,
            "tail": "right", "p_value": p, "p_two_sided": np.nan,
        })
    return pd.DataFrame(rows)


def build_richclub_curve(real_richclub: dict[int, float],
                         richclub_by_k: dict[int, list],
                         ks: list[int]) -> pd.DataFrame:
    rows = []
    for k in ks:
        nv = np.asarray(richclub_by_k[k], dtype=float)
        nv = nv[~np.isnan(nv)]
        phi_real = real_richclub.get(k, np.nan)
        null_mean = float(np.mean(nv)) if len(nv) else np.nan
        lo = float(np.percentile(nv, RICHCLUB_CI[0])) if len(nv) else np.nan
        hi = float(np.percentile(nv, RICHCLUB_CI[1])) if len(nv) else np.nan
        rows.append({
            "k":             k,
            "phi_real":      phi_real,
            "phi_null_mean": null_mean,
            "phi_null_lo":   lo,
            "phi_null_hi":   hi,
            "phi_normalized": (phi_real / null_mean) if (null_mean and not np.isnan(null_mean) and null_mean != 0) else np.nan,
            "above_null_band": bool(phi_real > hi) if not (np.isnan(phi_real) or np.isnan(hi)) else False,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _save(fig, name: str) -> None:
    fig.savefig(FIGURES_DIR / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_kcore_vs_clustering(node_df: pd.DataFrame, rho: float, p: float) -> None:
    palette = sns.color_palette("colorblind")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(node_df["kcore"], node_df["clustering_uw"],
               alpha=0.15, s=12, color=palette[0], edgecolors="none",
               label="Subreddits")
    # median clustering per k-core as a trend line
    trend = node_df.groupby("kcore")["clustering_uw"].median()
    ax.plot(trend.index, trend.values, color=palette[3], linewidth=2,
            marker="o", markersize=3, label="Median clustering per k-core")
    ax.set_xlabel("k-core number (global embeddedness)")
    ax.set_ylabel("Local clustering coefficient")
    ax.set_title(f"k-core vs. Clustering   (Spearman ρ = {rho:.3f}, p = {p:.1e})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
    _save(fig, "kcore_vs_clustering_scatter.png")


def plot_spearman_null(null_df: pd.DataFrame, real_rho: float, p: float) -> None:
    palette = sns.color_palette("colorblind")
    fig, ax = plt.subplots(figsize=(10, 6))
    vals = null_df["spearman_rho_uw"].dropna()
    ax.hist(vals, bins=30, color=palette[0], alpha=0.8, edgecolor="white",
            label="Null graphs")
    ax.axvline(real_rho, color="red", linewidth=2, linestyle="--",
               label=f"Reddit ρ = {real_rho:.3f}")
    ax.set_xlabel("Spearman ρ (k-core vs. clustering)")
    ax.set_ylabel("Count")
    ax.set_title(f"Null Distribution of Spearman ρ   (p = {p:.3f})")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, color="#cccccc")
    _save(fig, "spearman_null_distribution.png")


def plot_clustering_null(null_df: pd.DataFrame, real_clust: float, p: float) -> None:
    palette = sns.color_palette("colorblind")
    fig, ax = plt.subplots(figsize=(10, 6))
    vals = null_df["mean_clustering_uw"].dropna()
    ax.hist(vals, bins=30, color=palette[1], alpha=0.8, edgecolor="white",
            label="Null graphs")
    ax.axvline(real_clust, color="red", linewidth=2, linestyle="--",
               label=f"Reddit = {real_clust:.4f}")
    ax.set_xlabel("Mean local clustering coefficient")
    ax.set_ylabel("Count")
    ax.set_title(f"Reddit Clustering vs. Null   (p = {p:.3f})")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, color="#cccccc")
    _save(fig, "clustering_real_vs_null.png")


def plot_richclub_curve(curve: pd.DataFrame) -> None:
    palette = sns.color_palette("colorblind")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(curve["k"], curve["phi_real"], color="red", linewidth=2,
            marker="o", markersize=3, label="Reddit φ(k)")
    ax.plot(curve["k"], curve["phi_null_mean"], color=palette[0], linewidth=1.5,
            label="Null mean φ(k)")
    ax.fill_between(curve["k"], curve["phi_null_lo"], curve["phi_null_hi"],
                    color=palette[0], alpha=0.25, label="Null 95% band")
    ax.set_xlabel("Degree threshold k")
    ax.set_ylabel("Rich-club coefficient φ(k)")
    ax.set_title("Rich-Club Coefficient: Reddit vs. Null")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
    _save(fig, "richclub_curve_vs_null.png")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("RQ4 — Core-Periphery Structure")
    print("=" * 72)

    print("\n[1/6] Loading graph and building undirected projection...")
    graphs = load_graphs()
    G = graphs["weighted"]
    # Weighted-pass weights use RQ1's 95th-pct cap, for project consistency.
    G_norm, cap, frac = normalize_edge_weights(G)
    UG = build_undirected(G_norm)
    print(f"      directed: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"      undirected projection: {UG.number_of_nodes():,} nodes, "
          f"{UG.number_of_edges():,} edges")
    print(f"      weight cap (weighted pass): {cap:.2f} ({frac:.2%} of edges)")

    print("\n[2/6] Computing real-network statistics...")
    ks = select_richclub_ks(UG)
    real = compute_graph_stats(UG, ks, with_weighted=True)
    print(f"      Spearman ρ (unweighted) = {real['spearman_rho_uw']:.4f} "
          f"(p = {real['spearman_p_uw']:.2e})")
    print(f"      Spearman ρ (weighted)   = {real['spearman_rho_w']:.4f}")
    print(f"      mean clustering (uw)    = {real['mean_clustering_uw']:.4f}")
    print(f"      mean clustering (w)     = {real['mean_clustering_w']:.4f}")
    print(f"      rich-club k range       = [{ks[0]}, {ks[-1]}] ({len(ks)} values)")

    # Per-node CSV (this kcore column is the RQ5 dependency)
    node_df = pd.DataFrame({
        "node":           real["nodes"],
        "kcore":          real["kcore"].astype(int),
        "clustering_uw":  real["clustering_uw"],
        "clustering_w":   real["clustering_w"],
        "degree":         [UG.degree(n) for n in real["nodes"]],
    })
    node_df.to_csv(METRICS_DIR / "rq4_node_metrics.csv", index=False)
    print(f"      saved per-node metrics ({len(node_df):,} rows) → rq4_node_metrics.csv")

    # ---- null ensemble (with cache) ----
    null_csv = METRICS_DIR / "rq4_null_distribution.csv"
    richclub_csv = METRICS_DIR / "rq4_richclub_curve.csv"
    if null_csv.exists() and richclub_csv.exists():
        print(f"\n[3/6] Loading cached null ensemble from {null_csv.name}...")
        null_df = pd.read_csv(null_csv)
        richclub_curve = pd.read_csv(richclub_csv)
    else:
        print(f"\n[3/6] Building null ensemble: {N_NULL} config-model graphs "
              f"({N_WORKERS} workers)")
        in_seq  = [d for _, d in G.in_degree()]
        out_seq = [d for _, d in G.out_degree()]
        real_weights = [d["weight"] for _, _, d in G_norm.edges(data=True)]
        null_df, richclub_by_k = build_null_ensemble(
            in_seq, out_seq, real_weights, ks, N_NULL, RNG_SEED, N_WORKERS)
        null_df.to_csv(null_csv, index=False)
        richclub_curve = build_richclub_curve(real["richclub"], richclub_by_k, ks)
        richclub_curve.to_csv(richclub_csv, index=False)
        print(f"      saved {len(null_df):,} null rows → {null_csv.name}")

    print("\n[4/6] Permutation tests...")
    ptest = permutation_test(real, null_df)
    ptest.to_csv(METRICS_DIR / "rq4_global_stats.csv", index=False)
    print(ptest.to_string(index=False))

    print("\n[5/6] Generating figures...")
    plot_kcore_vs_clustering(node_df, real["spearman_rho_uw"], real["spearman_p_uw"])
    p_rho = float(ptest.loc[ptest["statistic"] == "spearman_rho_uw", "p_value"].iloc[0])
    p_cl = float(ptest.loc[ptest["statistic"] == "mean_clustering_uw", "p_value"].iloc[0])
    plot_spearman_null(null_df, real["spearman_rho_uw"], p_rho)
    plot_clustering_null(null_df, real["mean_clustering_uw"], p_cl)
    plot_richclub_curve(richclub_curve)
    print(f"      4 figures saved → {FIGURES_DIR}")

    print("\n[6/6] Sanity checks...")
    assert len(node_df) == UG.number_of_nodes(), "node_df row count mismatch"
    assert node_df["kcore"].notna().all(), "kcore has NaN"
    assert ptest["p_value"].dropna().between(0, 1).all(), "p-value out of [0,1]"
    null_mean_clust = null_df["mean_clustering_uw"].mean()
    print(f"      null mean clustering ≈ {null_mean_clust:.5f} (expected near 0)")
    print("      ✓ all sanity checks passed")

    # ---- plain-language conclusion ----
    print("\nConclusion:")
    rho = real["spearman_rho_uw"]
    n_above = int(richclub_curve["above_null_band"].sum())
    sig_rho = p_rho < 0.05
    sig_cl = p_cl < 0.05
    print(f"  Spearman ρ = {rho:.3f} ({'negative' if rho < 0 else 'positive'}); "
          f"vs null p = {p_rho:.3f} → {'beats null' if sig_rho else 'NOT distinguishable from null'}")
    print(f"  Mean clustering vs null p = {p_cl:.3f} → "
          f"{'beats null' if sig_cl else 'NOT distinguishable'}")
    print(f"  Rich-club: real φ(k) exceeds the null 95% band at {n_above}/{len(richclub_curve)} k values")

    print("\nDone.")


if __name__ == "__main__":
    main()
