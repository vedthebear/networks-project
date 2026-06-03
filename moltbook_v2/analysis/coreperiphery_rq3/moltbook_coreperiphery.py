#!/usr/bin/env python3
"""
moltbook_coreperiphery.py -- RQ3: are CENTRAL Moltbook communities bridges
(low clustering) or cliques (high clustering)? Does a community's GLOBAL
position predict its LOCAL connectivity, beyond chance?

This mirrors Sam's Reddit RQ4 (reddit/rq4_core_periphery.py) as closely as
possible so the two platforms can be compared directly:
  - global position  := k-core number      (nx.core_number)
  - local behavior   := local clustering    (nx.clustering)
  - relationship     := Spearman rho(k-core, clustering), reported BOTH on the
    full graph and WITHIN the core (k-core >= 2), because the full-graph value
    is mechanically dragged positive by degree-1 leaf communities (Sam's exact
    observation on Reddit).
  - rich-club coefficient: do the highest-degree communities interlink?
  - significance: a degree-preserving CONFIGURATION-MODEL null ensemble
    (same as Sam). Directional permutation test for the correlation (left tail
    for a negative rho), right-tail for clustering magnitude.

Graph object: the FULL Moltbook shared-agent community graph (ALL submolts,
periphery included) -- this carries the leaf fringe that the dense active-core
graph throws away, so it is the fair structural analogue of Reddit's sparse
subreddit graph. We sweep the edge threshold min_shared in {2,3,5,10} for
robustness and run the full null ensemble on the primary threshold.

Reddit reference (Sam): full-graph rho ~= +0.78 (leaf-driven), within-core
rho ~= -0.15 (central = bridges), clustering ~5.3x its null.
"""
import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
TAB = HERE.parents[1] / "data" / "tables"
RESULTS = HERE / "results"; FIGS = HERE / "figures"
PRIMARY_MIN_SHARED = 3      # primary threshold (full graph, density ~Reddit-like)
SWEEP = [2, 3, 5, 10]
N_NULL = 500
SEED = 168
RICHCLUB_CI = (2.5, 97.5)

# --------------------------------------------------------------------------
def build_graph(min_authors, min_shared):
    mem = pd.read_csv(TAB / "membership.csv")
    if min_authors > 0:
        ac = mem.groupby("submolt")["agent"].nunique()
        mem = mem[mem.submolt.isin(set(ac[ac >= min_authors].index))]
    sub_ag = defaultdict(set); ag_sub = defaultdict(set)
    for a, s in zip(mem.agent, mem.submolt):
        sub_ag[s].add(a); ag_sub[a].add(s)
    shared = defaultdict(int)
    for subs in ag_sub.values():
        for x, y in combinations(sorted(subs), 2):
            shared[(x, y)] += 1
    G = nx.Graph(); G.add_nodes_from(sub_ag)
    for (x, y), sh in shared.items():
        if sh >= min_shared:
            G.add_edge(x, y)
    return G

def graph_stats(G, richclub_ks):
    kcore = nx.core_number(G)
    clust = nx.clustering(G)
    nodes = list(G.nodes())
    kc = np.array([kcore[n] for n in nodes], float)
    cl = np.array([clust[n] for n in nodes], float)
    core = kc >= 2
    rho_full = spearmanr(kc, cl).statistic
    rho_core = spearmanr(kc[core], cl[core]).statistic if core.sum() > 10 else np.nan
    # rich-club at selected k
    rc = {}
    try:
        full_rc = nx.rich_club_coefficient(G, normalized=False)
        rc = {k: full_rc.get(k, np.nan) for k in richclub_ks}
    except Exception:
        rc = {k: np.nan for k in richclub_ks}
    return {
        "mean_clustering": float(cl.mean()),
        "rho_full": float(rho_full),
        "rho_core": float(rho_core),
        "richclub": rc,
        "kc": kc, "cl": cl,
    }

def config_null(G, seed):
    deg = [d for _, d in G.degree()]
    g = nx.configuration_model(deg, seed=seed)
    g = nx.Graph(g)
    g.remove_edges_from(nx.selfloop_edges(g))
    return g

def main():
    RESULTS.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)

    # ---- robustness sweep: real rho across thresholds (fast, no null) ----
    sweep_rows = []
    for s in SWEEP:
        G = build_graph(0, s)
        st = graph_stats(G, [])
        sweep_rows.append({
            "min_shared": s, "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
            "density": nx.density(G), "mean_clustering": st["mean_clustering"],
            "rho_full": st["rho_full"], "rho_core": st["rho_core"],
        })
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(RESULTS / "sweep.csv", index=False)
    print("=== robustness sweep (full graph, real values) ===")
    print(sweep_df.to_string(index=False))

    # ---- primary threshold: full null ensemble ----
    G = build_graph(0, PRIMARY_MIN_SHARED)
    n, m = G.number_of_nodes(), G.number_of_edges()
    # rich-club k thresholds: a handful spanning the degree range
    degs = sorted(set(d for _, d in G.degree()))
    rc_ks = [k for k in [2, 3, 5, 8, 12, 20, 35, 60] if k <= max(degs)]
    real = graph_stats(G, rc_ks)
    print(f"\n=== PRIMARY graph: full, min_shared={PRIMARY_MIN_SHARED} ===")
    print(f"nodes={n} edges={m} density={nx.density(G):.4f} "
          f"mean_clustering={real['mean_clustering']:.3f} "
          f"rho_full={real['rho_full']:.3f} rho_core={real['rho_core']:.3f}")

    rng = np.random.default_rng(SEED)
    null_rows = []; rc_null = {k: [] for k in rc_ks}
    print(f"\nbuilding {N_NULL} configuration-model nulls...")
    for i in range(N_NULL):
        gnull = config_null(G, int(rng.integers(0, 2**31 - 1)))
        st = graph_stats(gnull, rc_ks)
        null_rows.append({"mean_clustering": st["mean_clustering"],
                          "rho_full": st["rho_full"], "rho_core": st["rho_core"]})
        for k in rc_ks:
            rc_null[k].append(st["richclub"][k])
    null_df = pd.DataFrame(null_rows)
    null_df.to_csv(RESULTS / "null_ensemble.csv", index=False)

    # ---- permutation tests (Sam's directional logic) ----
    def directional_p(real_val, nv):
        nv = nv[~np.isnan(nv)]
        if real_val >= 0:
            return float((nv >= real_val).mean())
        return float((nv <= real_val).mean())
    tests = []
    for name in ["rho_full", "rho_core"]:
        nv = null_df[name].values
        tests.append({"statistic": name, "real": real[name],
                      "null_mean": float(np.nanmean(nv)), "null_std": float(np.nanstd(nv)),
                      "p_value": directional_p(real[name], nv),
                      "tail": "left" if real[name] < 0 else "right"})
    nv = null_df["mean_clustering"].values
    tests.append({"statistic": "mean_clustering", "real": real["mean_clustering"],
                  "null_mean": float(np.nanmean(nv)), "null_std": float(np.nanstd(nv)),
                  "p_value": float((nv >= real["mean_clustering"]).mean()),
                  "tail": "right",
                  })
    tests_df = pd.DataFrame(tests)
    tests_df["ratio_real_over_null"] = tests_df["real"] / tests_df["null_mean"]
    tests_df.to_csv(RESULTS / "permutation_tests.csv", index=False)
    print("\n=== permutation tests vs configuration-model null ===")
    for t in tests:
        print(f"  {t['statistic']:16s} real={t['real']:+.3f}  null={t['null_mean']:+.3f}"
              f" (std {t['null_std']:.3f})  p={t['p_value']:.4f}  [{t['tail']} tail]")

    # ---- rich-club curve ----
    rc_rows = []
    for k in rc_ks:
        nv = np.asarray(rc_null[k], float); nv = nv[~np.isnan(nv)]
        phi = real["richclub"][k]
        nm = float(np.mean(nv)) if len(nv) else np.nan
        rc_rows.append({"k": k, "phi_real": phi, "phi_null_mean": nm,
                        "phi_null_lo": float(np.percentile(nv, RICHCLUB_CI[0])) if len(nv) else np.nan,
                        "phi_null_hi": float(np.percentile(nv, RICHCLUB_CI[1])) if len(nv) else np.nan,
                        "phi_normalized": phi / nm if nm else np.nan})
    pd.DataFrame(rc_rows).to_csv(RESULTS / "richclub.csv", index=False)

    # save real summary
    (RESULTS / "real_summary.json").write_text(json.dumps({
        "graph": f"full, min_shared={PRIMARY_MIN_SHARED}", "nodes": n, "edges": m,
        "density": nx.density(G), "mean_clustering": real["mean_clustering"],
        "rho_full": real["rho_full"], "rho_core": real["rho_core"],
    }, indent=2))

    # ================= FIGURES =================
    # Fig 1: k-core vs clustering scatter (the structure, real graph)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(real["kc"], real["cl"], s=8, alpha=0.35, color="#3182bd")
    # mean clustering per k-core level
    dfp = pd.DataFrame({"kc": real["kc"], "cl": real["cl"]})
    trend = dfp[dfp.kc >= 2].groupby("kc")["cl"].mean()
    ax.plot(trend.index, trend.values, color="#d62728", lw=2, marker="o", ms=3,
            label="mean clustering within core (k≥2)")
    ax.set_xlabel("k-core number (global position →)")
    ax.set_ylabel("local clustering coefficient")
    ax.set_title(f"Moltbook: central communities are bridges, not cliques\n"
                 f"within-core ρ = {real['rho_core']:.2f} (p={tests[1]['p_value']:.3f})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "kcore_vs_clustering.png", dpi=300); plt.close(fig)

    # Fig 2: within-core rho vs null distribution (the null test)
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.hist(null_df["rho_core"], bins=30, color="#9ecae1", edgecolor="white",
            label="configuration-model null")
    ax.axvline(real["rho_core"], color="#d62728", lw=2.5,
               label=f"Moltbook (real) = {real['rho_core']:.2f}")
    ax.set_xlabel("within-core Spearman ρ (k-core vs clustering)")
    ax.set_ylabel("number of null graphs")
    ax.set_title("Is the bridging real, or just mechanical?\n"
                 "real ρ vs degree-preserving null")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "rho_vs_null.png", dpi=300); plt.close(fig)

    # Fig 3: rich-club curve
    rc = pd.DataFrame(rc_rows)
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.plot(rc.k, rc.phi_real, color="#d62728", marker="o", label="Moltbook (real)")
    ax.plot(rc.k, rc.phi_null_mean, color="#636363", ls="--", label="null mean")
    ax.fill_between(rc.k, rc.phi_null_lo, rc.phi_null_hi, color="#cccccc", alpha=0.5,
                    label="null 95% band")
    ax.set_xlabel("degree threshold k"); ax.set_ylabel("rich-club coefficient φ(k)")
    ax.set_title("Do the most-connected communities interlink? (rich club)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "richclub.png", dpi=300); plt.close(fig)
    print(f"\nfigures -> {FIGS}")

if __name__ == "__main__":
    main()
