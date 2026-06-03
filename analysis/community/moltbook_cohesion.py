#!/usr/bin/env python3
"""
moltbook_cohesion.py -- RQ1 (cohesion): is the Moltbook community network more
clustered/cohesive than chance?

Mirrors Sam's Reddit RQ4 method (reddit/rq4_core_periphery.py):
  1. Take the undirected community graph.
  2. Measure its cohesion: local clustering, global transitivity, giant-
     component share (plus density / mean degree as descriptors).
  3. Build an ensemble of degree-preserving CONFIGURATION-MODEL null graphs
     (same degree sequence, edges rewired at random; self-loops & multi-edges
     dropped). The config model preserves degree -> preserves m -> preserves
     density, so density is a DESCRIPTOR, not a test target. Clustering,
     transitivity and giant-component share are NOT preserved -> those are the
     real null tests.
  4. Permutation test: is the real value more extreme than the null ensemble?
     Report the null-relative ratio (real / null-mean) and a one-tailed p-value
     -- exactly the "5.3x null, p<0.002" style Sam used for Reddit.

Graph: the Moltbook shared-agent ACTIVE-CORE graph (submolts with >=5 authors,
edges where >=2 agents post in both). Built from data/tables/membership.csv.

Outputs:
  results/cohesion_real.json        real-graph metrics
  results/cohesion_null.csv         per-null metrics (the ensemble)
  results/cohesion_summary.csv      real, null-mean, ratio, p-value per metric
  figures/clustering_vs_null.png    headline: real clustering vs null distribution
  figures/structure_vs_null.png     transitivity + giant component vs null
"""
import json, random
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
TAB = HERE.parents[1] / "data" / "data" / "tables"   # repo root/data/data/tables
RESULTS = HERE / "results"; FIGS = HERE / "figures"
N_NULL = 500
SEED = 168
MIN_AUTHORS = 5      # "active" submolt threshold
MIN_SHARED = 2       # edge requires >=2 shared agents (robust; drops spam)

# --------------------------------------------------------------------------
# Build the real undirected community graph from the bipartite membership list
# --------------------------------------------------------------------------
def build_core_graph():
    mem = pd.read_csv(TAB / "membership.csv")
    authors_per = mem.groupby("submolt")["agent"].nunique()
    active = set(authors_per[authors_per >= MIN_AUTHORS].index)
    mem = mem[mem.submolt.isin(active)]
    sub_agents = defaultdict(set); agent_subs = defaultdict(set)
    for a, s in zip(mem.agent, mem.submolt):
        sub_agents[s].add(a); agent_subs[a].add(s)
    shared = defaultdict(int)
    for subs in agent_subs.values():
        for a, b in combinations(sorted(subs), 2):
            shared[(a, b)] += 1
    G = nx.Graph(); G.add_nodes_from(sub_agents)
    for (a, b), sh in shared.items():
        if sh >= MIN_SHARED:
            G.add_edge(a, b, weight=sh)
    return G

# --------------------------------------------------------------------------
# Cohesion metrics (computed on real and on every null)
# --------------------------------------------------------------------------
def cohesion_metrics(G):
    n = G.number_of_nodes(); m = G.number_of_edges()
    giant = max((len(c) for c in nx.connected_components(G)), default=0)
    return {
        "n": n, "m": m,
        "density": nx.density(G),
        "mean_degree": (2 * m / n) if n else 0.0,
        "mean_clustering": nx.average_clustering(G),     # local, unweighted
        "transitivity": nx.transitivity(G),             # global
        "giant_frac": giant / n if n else 0.0,
    }

# --------------------------------------------------------------------------
# Null models
#   degree_preserving_null : Maslov-Sneppen double-edge-swap. Preserves the
#     EXACT degree sequence AND edge count, keeps the graph simple. The right
#     degree-preserving null for a dense graph (configuration_model loses ~23%
#     of edges to multi-edge collapse here, which would confound the test).
#   er_null : Erdos-Renyi G(n, m) -- same n and m, edges placed at random.
#     This is the random-graph null taught in the lecture notes (section 4);
#     its clustering is ~ density, a clean "no structure" reference.
# --------------------------------------------------------------------------
def degree_preserving_null(G, seed):
    g = G.copy()
    m = g.number_of_edges()
    try:
        nx.double_edge_swap(g, nswap=3 * m, max_tries=30 * m, seed=seed)
    except nx.NetworkXAlgorithmError:
        pass  # rare: not enough swappable edges; partial randomization is fine
    return g

def er_null(n, m, seed):
    return nx.gnm_random_graph(n, m, seed=seed)

def main():
    RESULTS.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)
    G = build_core_graph()
    real = cohesion_metrics(G)
    print("=== Moltbook active-core community graph ===")
    for k, v in real.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    (RESULTS / "cohesion_real.json").write_text(json.dumps(real, indent=2))

    # Two null ensembles
    n, m = G.number_of_nodes(), G.number_of_edges()
    rng = random.Random(SEED)
    dp_rows, er_rows = [], []
    print(f"\nbuilding {N_NULL} degree-preserving + {N_NULL} Erdos-Renyi nulls...")
    for i in range(N_NULL):
        dp_rows.append(cohesion_metrics(degree_preserving_null(G, rng.randint(0, 2**31 - 1))))
        er_rows.append(cohesion_metrics(er_null(n, m, rng.randint(0, 2**31 - 1))))
    null_df = pd.DataFrame(dp_rows)      # primary null = degree-preserving
    er_df = pd.DataFrame(er_rows)
    null_df.to_csv(RESULTS / "cohesion_null.csv", index=False)
    er_df.to_csv(RESULTS / "cohesion_null_er.csv", index=False)

    # Compare real vs null for the metrics the null does NOT preserve
    test_metrics = ["mean_clustering", "transitivity", "giant_frac"]
    desc_metrics = ["density", "mean_degree"]   # preserved by construction
    summ = []
    for metric in test_metrics + desc_metrics:
        rv = real[metric]
        dp = null_df[metric].values; er = er_df[metric].values
        dp_mean, dp_std = float(dp.mean()), float(dp.std())
        er_mean = float(er.mean())
        p_hi = (np.sum(dp >= rv) + 1) / (len(dp) + 1)   # vs degree-preserving null
        z = (rv - dp_mean) / dp_std if dp_std else float("inf")
        summ.append({
            "metric": metric, "real": rv,
            "dp_null_mean": dp_mean, "dp_null_std": dp_std,
            "ratio_vs_dp_null": rv / dp_mean if dp_mean else float("inf"),
            "er_null_mean": er_mean,
            "ratio_vs_er_null": rv / er_mean if er_mean else float("inf"),
            "z_score": z, "p_value_real_ge_dp_null": p_hi,
            "is_null_test": metric in test_metrics,
        })
    summ_df = pd.DataFrame(summ)
    summ_df.to_csv(RESULTS / "cohesion_summary.csv", index=False)
    print("\n=== real vs nulls (degree-preserving / Erdos-Renyi) ===")
    for r in summ:
        tag = "TEST" if r["is_null_test"] else "descriptor"
        print(f"  {r['metric']:16s} real={r['real']:.4f}  dp-null={r['dp_null_mean']:.4f} "
              f"({r['ratio_vs_dp_null']:.2f}x, p={r['p_value_real_ge_dp_null']:.4f})  "
              f"er-null={r['er_null_mean']:.4f} ({r['ratio_vs_er_null']:.2f}x)  [{tag}]")

    # ---- Figure 1: clustering vs null distributions (the headline) ----
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(er_df["mean_clustering"], bins=30, color="#d9d9d9",
            edgecolor="white", label="Erdos-Renyi null (no structure)")
    ax.hist(null_df["mean_clustering"], bins=30, color="#9ecae1",
            edgecolor="white", label="degree-preserving null")
    ax.axvline(real["mean_clustering"], color="#d62728", lw=2.5,
               label=f"Moltbook (real) = {real['mean_clustering']:.3f}")
    ratio = real["mean_clustering"] / null_df["mean_clustering"].mean()
    ax.set_xlabel("mean local clustering coefficient")
    ax.set_ylabel("number of null graphs")
    ax.set_title(f"Moltbook community clustering vs degree-matched null\n"
                 f"real is {ratio:.1f}x the degree-preserving null mean")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "clustering_vs_null.png", dpi=300)
    plt.close(fig)

    # ---- Figure 2: transitivity + giant component vs null ----
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric, label in zip(axes, ["transitivity", "giant_frac"],
                                 ["global transitivity", "giant-component share"]):
        ax.hist(null_df[metric], bins=30, color="#a1d99b", edgecolor="white",
                label="null")
        ax.axvline(real[metric], color="#d62728", lw=2.5,
                   label=f"real = {real[metric]:.3f}")
        ax.set_xlabel(label); ax.set_ylabel("# null graphs"); ax.legend(fontsize=8)
    fig.suptitle("Moltbook structure vs degree-matched configuration-model null")
    fig.tight_layout(); fig.savefig(FIGS / "structure_vs_null.png", dpi=300)
    plt.close(fig)
    print(f"\nfigures -> {FIGS}")

if __name__ == "__main__":
    main()
