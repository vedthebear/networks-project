"""
analyze_interaction.py -- Section 1 of "Who Controls the Agent Internet?"

The agent-to-agent reply graph -- the cleanest map of influence on
MoltBook. A directed edge replier -> recipient with weight = #replies.

Computes:
  - in / out / total degree (weighted)
  - PageRank
  - betweenness centrality (sampled on big graphs)
  - eigenvector centrality (on undirected projection)
  - HITS hub and authority scores
  - Louvain communities + modularity
  - degree distribution (log-log)
  - top-k bar charts per centrality

Outputs (in figures/):
  agent_centralities.csv     -- one row per agent, all centrality columns
  agent_communities.csv      -- agent -> community membership
  interaction_summary.csv    -- headline stats
  interaction_top_agents.png -- bar charts of top-k by each centrality
  interaction_degree_dist.png-- log-log degree distribution
  interaction_community_sizes.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from collections import Counter


def build_global_reply_graph(reply_df: pd.DataFrame) -> nx.DiGraph:
    """Aggregate per-submolt reply weights into one global directed graph."""
    agg = (reply_df
           .groupby(["replier", "recipient"], as_index=False)["weight"]
           .sum())
    return nx.from_pandas_edgelist(
        agg, source="replier", target="recipient",
        edge_attr="weight", create_using=nx.DiGraph,
    )


def compute_centralities(G: nx.DiGraph, betweenness_sample: int = 500) -> pd.DataFrame:
    """One row per agent, columns for each centrality."""
    n = G.number_of_nodes()
    if n == 0:
        return pd.DataFrame()

    print(f"  centralities on {n:,} nodes / {G.number_of_edges():,} edges ...")

    cents = {
        "in_degree":  dict(G.in_degree(weight="weight")),
        "out_degree": dict(G.out_degree(weight="weight")),
        "pagerank":   nx.pagerank(G, weight="weight"),
    }

    # Betweenness scales as O(nm); sample on larger graphs.
    k = min(betweenness_sample, n)
    print(f"  betweenness (sampling k={k}) ...")
    cents["betweenness"] = nx.betweenness_centrality(G, k=k, weight="weight", seed=42)

    # Eigenvector centrality on undirected projection (more numerically stable).
    UG = G.to_undirected()
    try:
        cents["eigenvector"] = nx.eigenvector_centrality_numpy(UG, weight="weight")
    except Exception as e:
        print(f"    eigenvector failed: {e}")
        cents["eigenvector"] = {x: float("nan") for x in G.nodes}

    # HITS: hubs (point at lots of authorities) and authorities (pointed at by hubs)
    try:
        h, a = nx.hits(G, max_iter=300)
        cents["hub_score"] = h
        cents["authority_score"] = a
    except Exception as e:
        print(f"    HITS failed: {e}")

    return (pd.DataFrame(cents)
              .rename_axis("agent")
              .reset_index())


def detect_communities(G: nx.DiGraph):
    """Louvain on the undirected projection."""
    UG = G.to_undirected()
    communities = list(nx.community.louvain_communities(UG, seed=42, weight="weight"))
    modularity = nx.community.modularity(UG, communities, weight="weight")

    rows = []
    for i, comm in enumerate(communities):
        for agent in comm:
            rows.append({"agent": agent, "community": i, "community_size": len(comm)})
    return pd.DataFrame(rows), modularity


def plot_top_agents(cent_df: pd.DataFrame, out_path: Path, top_k: int = 20):
    metrics = [m for m in ("pagerank", "in_degree", "betweenness", "authority_score")
               if m in cent_df.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 7))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        top = cent_df.nlargest(top_k, metric)
        ax.barh(range(len(top)), top[metric].values)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["agent"].astype(str), fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(metric)
        ax.set_title(f"Top {top_k} by {metric}")
    fig.suptitle("Who controls the conversation? (top agents per centrality)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_degree_distribution(G: nx.DiGraph, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for which, vals in (("in", [d for _, d in G.in_degree()]),
                        ("out", [d for _, d in G.out_degree()])):
        cnt = Counter(vals)
        xs = sorted(k for k in cnt if k > 0)
        ys = [cnt[x] for x in xs]
        ax.loglog(xs, ys, "o", alpha=0.6, label=f"{which}-degree")
    ax.set_xlabel("degree k")
    ax.set_ylabel("count")
    ax.set_title("Reply-graph degree distribution (log-log)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_community_sizes(comm_df: pd.DataFrame, out_path: Path):
    if comm_df.empty:
        return
    sizes = comm_df.groupby("community")["agent"].count().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(range(len(sizes)), sizes.values)
    ax.set_xlabel("community rank")
    ax.set_ylabel("size (#agents)")
    ax.set_yscale("log")
    ax.set_title("Louvain community sizes (global reply graph)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", type=Path,
                        help="folder containing agent_reply_edges.csv")
    parser.add_argument("--out", default="figures", type=Path)
    parser.add_argument("--betweenness-k", default=500, type=int,
                        help="sample size for betweenness (use 0 for exact)")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)

    reply_path = args.data / "agent_reply_edges.csv"
    if not reply_path.exists():
        raise SystemExit(f"missing {reply_path}; run fetch_data.py first")

    reply = pd.read_csv(reply_path)
    G = build_global_reply_graph(reply)
    print(f"Global reply graph: {G.number_of_nodes():,} agents, "
          f"{G.number_of_edges():,} edges")

    cent = compute_centralities(G, betweenness_sample=args.betweenness_k or G.number_of_nodes())
    cent_path = args.out / "agent_centralities.csv"
    cent.to_csv(cent_path, index=False)
    print(f"  wrote {cent_path}")

    comm, modularity = detect_communities(G)
    comm_path = args.out / "agent_communities.csv"
    comm.to_csv(comm_path, index=False)
    print(f"  wrote {comm_path}  (modularity = {modularity:.4f}, "
          f"#communities = {comm['community'].nunique()})")

    plot_top_agents(cent, args.out / "interaction_top_agents.png")
    plot_degree_distribution(G, args.out / "interaction_degree_dist.png")
    plot_community_sizes(comm, args.out / "interaction_community_sizes.png")

    UG = G.to_undirected()
    summary = {
        "agents":         G.number_of_nodes(),
        "edges":          G.number_of_edges(),
        "density":        nx.density(G),
        "avg_clustering": nx.average_clustering(UG),
        "reciprocity":    nx.reciprocity(G) if G.number_of_edges() else float("nan"),
        "assortativity":  nx.degree_assortativity_coefficient(G),
        "communities":    comm["community"].nunique() if not comm.empty else 0,
        "modularity":     modularity,
    }
    pd.Series(summary).to_csv(args.out / "interaction_summary.csv", header=False)
    print("\nHeadline stats:")
    for k, v in summary.items():
        print(f"  {k:>16}: {v}")


if __name__ == "__main__":
    main()
