#!/usr/bin/env python3
"""
reddit_coreperiphery.py -- the SAME core-periphery test, run on the Reddit
subreddit hyperlink graph, so Moltbook and Reddit are compared with identical
code (identical null, identical statistics). This makes the null-relative
within-core correlation directly comparable across platforms.

Reddit graph: SNAP soc-redditHyperlinks-body.tsv -> UNDIRECTED projection
(edge if a hyperlink runs in either direction), matching the undirected
projection Sam's RQ4 uses. We then run the exact same:
  - k-core + clustering per node
  - Spearman rho(k-core, clustering): full + within-core (k>=2)
  - degree-preserving configuration-model null + permutation test
as moltbook_coreperiphery.py.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
TSV = HERE.parents[2] / "degree_distribution" / "data" / "reddit" / "soc-redditHyperlinks-body.tsv"
RESULTS = HERE / "results"
N_NULL = 300
SEED = 168

def build_reddit_undirected():
    df = pd.read_csv(TSV, sep="\t", usecols=["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"])
    G = nx.Graph()
    G.add_edges_from(zip(df.SOURCE_SUBREDDIT, df.TARGET_SUBREDDIT))
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def graph_stats(G):
    kcore = nx.core_number(G)
    clust = nx.clustering(G)
    nodes = list(G.nodes())
    kc = np.array([kcore[n] for n in nodes], float)
    cl = np.array([clust[n] for n in nodes], float)
    core = kc >= 2
    return {
        "mean_clustering": float(cl.mean()),
        "rho_full": float(spearmanr(kc, cl).statistic),
        "rho_core": float(spearmanr(kc[core], cl[core]).statistic) if core.sum() > 10 else np.nan,
    }

def config_null(G, seed):
    deg = [d for _, d in G.degree()]
    g = nx.configuration_model(deg, seed=seed)
    g = nx.Graph(g)
    g.remove_edges_from(nx.selfloop_edges(g))
    return g

def main():
    RESULTS.mkdir(exist_ok=True)
    print("loading Reddit hyperlink graph...")
    G = build_reddit_undirected()
    real = graph_stats(G)
    print(f"Reddit undirected: nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
          f"density={nx.density(G):.6f}")
    print(f"  mean_clustering={real['mean_clustering']:.3f} "
          f"rho_full={real['rho_full']:.3f} rho_core={real['rho_core']:.3f}")

    rng = np.random.default_rng(SEED)
    rows = []
    print(f"building {N_NULL} configuration-model nulls...")
    for i in range(N_NULL):
        st = graph_stats(config_null(G, int(rng.integers(0, 2**31 - 1))))
        rows.append(st)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{N_NULL}")
    null_df = pd.DataFrame(rows)
    null_df.to_csv(RESULTS / "reddit_null_ensemble.csv", index=False)

    def directional_p(rv, nv):
        nv = nv[~np.isnan(nv)]
        return float((nv <= rv).mean()) if rv < 0 else float((nv >= rv).mean())

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
                  "p_value": float((nv >= real["mean_clustering"]).mean()), "tail": "right"})
    td = pd.DataFrame(tests)
    td["ratio_real_over_null"] = td["real"] / td["null_mean"]
    td.to_csv(RESULTS / "reddit_permutation_tests.csv", index=False)
    (RESULTS / "reddit_real_summary.json").write_text(json.dumps({
        "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
        "density": nx.density(G), **real}, indent=2))
    print("\n=== Reddit permutation tests vs configuration-model null ===")
    for t in tests:
        print(f"  {t['statistic']:16s} real={t['real']:+.3f}  null={t['null_mean']:+.3f}"
              f" (std {t['null_std']:.3f})  p={t['p_value']:.4f}  [{t['tail']} tail]")

if __name__ == "__main__":
    main()
