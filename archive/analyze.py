"""
analyze.py -- Network analysis for the Math 168 Moltbook vs Reddit project.

Loads the edge-list CSVs produced by fetch_data.py and computes the
standard descriptive network statistics + plots.

Three networks are analyzed:

  1. Submolt-overlap network (community-to-community)
       Nodes = submolts, edges weighted by shared agents / Jaccard.
       Tells you which communities are "close" in membership.

  2. Bipartite agent x submolt
       Nodes = agents + submolts. Useful for cross-community
       participation centrality and as the substrate for #1.

  3. Reply graph, per submolt (within-community)
       Nodes = agents in that submolt, directed weighted edges
       replier -> recipient. This is the one we run centrality
       and community detection on.

Usage
-----
After running fetch_data.py once for Moltbook:
    python analyze.py

Side-by-side comparison with Reddit (assumes you have a parallel
collector that writes the same three CSVs into a different folder):
    python analyze.py --moltbook-data data --reddit-data reddit_data

Outputs go to figures/ as PNGs, and per-platform per-submolt metrics
go to figures/<platform>_per_submolt_metrics.csv.
"""

import argparse
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #

def load_edges(data_dir: Path) -> dict:
    """Return DataFrames for the three edge files; missing ones become None."""
    files = {
        "agent_submolt": data_dir / "agent_submolt_edges.csv",
        "overlap":       data_dir / "submolt_overlap_edges.csv",
        "reply":         data_dir / "agent_reply_edges.csv",
    }
    out = {}
    for key, path in files.items():
        if path.exists():
            df = pd.read_csv(path)
            out[key] = df if not df.empty else None
            print(f"  loaded {path.name}: {len(df)} rows")
        else:
            out[key] = None
            print(f"  missing {path.name}")
    return out


# --------------------------------------------------------------------------- #
# Stats                                                                       #
# --------------------------------------------------------------------------- #

def graph_stats(G: nx.Graph, name: str = "") -> dict:
    """Basic descriptive stats."""
    if G.number_of_nodes() == 0:
        return {"name": name, "nodes": 0}

    UG = G.to_undirected() if G.is_directed() else G
    components = list(nx.connected_components(UG))
    largest = max(components, key=len)

    stats = {
        "name":                   name,
        "nodes":                  G.number_of_nodes(),
        "edges":                  G.number_of_edges(),
        "density":                nx.density(G),
        "avg_degree":             sum(dict(G.degree()).values()) / G.number_of_nodes(),
        "n_components":           len(components),
        "largest_component_frac": len(largest) / G.number_of_nodes(),
        "avg_clustering":         nx.average_clustering(UG),
    }
    if G.is_directed():
        try:
            stats["reciprocity"] = nx.reciprocity(G)
        except Exception:
            stats["reciprocity"] = float("nan")
    try:
        stats["assortativity"] = nx.degree_assortativity_coefficient(G)
    except Exception:
        stats["assortativity"] = float("nan")
    return stats


def per_submolt_metrics(reply_df: pd.DataFrame, min_edges: int = 5) -> pd.DataFrame:
    """For each submolt, build the within-community reply graph and compute metrics."""
    if reply_df is None or reply_df.empty:
        return pd.DataFrame()

    rows = []
    for submolt, sub_df in reply_df.groupby("submolt"):
        if len(sub_df) < min_edges:
            continue

        G = nx.from_pandas_edgelist(
            sub_df, source="replier", target="recipient",
            edge_attr="weight", create_using=nx.DiGraph,
        )
        UG = G.to_undirected()

        # Louvain community detection on the undirected projection.
        try:
            communities = list(nx.community.louvain_communities(UG, seed=42))
            modularity = nx.community.modularity(UG, communities)
        except Exception:
            communities, modularity = [], float("nan")

        try:
            assort = nx.degree_assortativity_coefficient(G)
        except Exception:
            assort = float("nan")
        try:
            recip = nx.reciprocity(G)
        except Exception:
            recip = float("nan")

        rows.append({
            "submolt":        submolt,
            "agents":         G.number_of_nodes(),
            "edges":          G.number_of_edges(),
            "density":        nx.density(G),
            "avg_clustering": nx.average_clustering(UG),
            "reciprocity":    recip,
            "communities":    len(communities),
            "modularity":     modularity,
            "assortativity":  assort,
        })

    return pd.DataFrame(rows).sort_values("agents", ascending=False)


def top_agents_per_submolt(reply_df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Top-k agents by PageRank within each submolt's reply graph."""
    if reply_df is None or reply_df.empty:
        return pd.DataFrame()

    rows = []
    for submolt, sub_df in reply_df.groupby("submolt"):
        if len(sub_df) < 5:
            continue
        G = nx.from_pandas_edgelist(
            sub_df, source="replier", target="recipient",
            edge_attr="weight", create_using=nx.DiGraph,
        )
        try:
            pr = nx.pagerank(G, weight="weight")
        except Exception:
            continue
        for agent, score in sorted(pr.items(), key=lambda kv: -kv[1])[:k]:
            rows.append({"submolt": submolt, "agent": agent, "pagerank": score})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #

def plot_degree_distributions(graphs: dict, out_path: Path, title: str):
    """Log-log degree distribution for one or more graphs on one axis."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, G in graphs.items():
        if G is None or G.number_of_nodes() == 0:
            continue
        degs = [d for _, d in G.degree()]
        cnt = Counter(degs)
        xs = sorted(cnt)
        ys = [cnt[x] for x in xs]
        ax.loglog(xs, ys, "o", alpha=0.65, label=f"{label} (n={G.number_of_nodes()})")
    ax.set_xlabel("degree k")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_per_submolt_scatter(per_sm: pd.DataFrame, out_path: Path, name: str):
    """Density vs modularity scatter, one dot per submolt, sized by #agents."""
    if per_sm.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = (per_sm["agents"] / per_sm["agents"].max()) * 300 + 20
    ax.scatter(per_sm["density"], per_sm["modularity"], s=sizes, alpha=0.6)
    for _, r in per_sm.iterrows():
        ax.annotate(r["submolt"], (r["density"], r["modularity"]),
                    fontsize=7, alpha=0.8)
    ax.set_xlabel("density")
    ax.set_ylabel("modularity (Louvain)")
    ax.set_title(f"{name}: per-community structure")
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_comparison(moltbook_per_sm: pd.DataFrame, reddit_per_sm: pd.DataFrame,
                    out_path: Path):
    """Side-by-side distributions of per-community metrics."""
    if moltbook_per_sm.empty and reddit_per_sm.empty:
        return
    metrics = ["density", "avg_clustering", "modularity",
               "reciprocity", "assortativity"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))

    for ax, metric in zip(axes, metrics):
        data, labels = [], []
        if not moltbook_per_sm.empty and metric in moltbook_per_sm:
            d = moltbook_per_sm[metric].dropna()
            if len(d):
                data.append(d.values); labels.append("Moltbook")
        if not reddit_per_sm.empty and metric in reddit_per_sm:
            d = reddit_per_sm[metric].dropna()
            if len(d):
                data.append(d.values); labels.append("Reddit")
        if data:
            ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_title(metric)

    fig.suptitle("Per-community metric distributions: Moltbook vs Reddit")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- #
# Per-platform driver                                                         #
# --------------------------------------------------------------------------- #

def analyze_platform(data_dir: Path, name: str, out_dir: Path) -> dict:
    print(f"\n=== {name} ({data_dir}) ===")
    edges = load_edges(data_dir)

    results = {"name": name, "edges": edges}

    # ---- Submolt-overlap network ----------------------------------------
    if edges["overlap"] is not None:
        G_overlap = nx.from_pandas_edgelist(
            edges["overlap"], source="submolt_a", target="submolt_b",
            edge_attr=["shared_agents", "jaccard"],
        )
        s = graph_stats(G_overlap, f"{name}_submolt_overlap")
        results["overlap_stats"] = s
        results["overlap_graph"] = G_overlap
        print("Submolt-overlap network:", {k: round(v, 4) if isinstance(v, float) else v
                                            for k, v in s.items()})

    # ---- Bipartite agent x submolt --------------------------------------
    G_bip = None
    if edges["agent_submolt"] is not None:
        df = edges["agent_submolt"]
        G_bip = nx.Graph()
        G_bip.add_nodes_from(df["agent_id"].unique(), bipartite="agent")
        G_bip.add_nodes_from(df["submolt"].unique(), bipartite="submolt")
        G_bip.add_weighted_edges_from(
            zip(df["agent_id"], df["submolt"], df["weight"])
        )
        s = graph_stats(G_bip, f"{name}_agent_submolt_bipartite")
        results["bipartite_stats"] = s
        print("Bipartite agent x submolt:", {k: round(v, 4) if isinstance(v, float) else v
                                              for k, v in s.items()})

    # ---- Per-submolt within-community metrics ---------------------------
    per_sm = per_submolt_metrics(edges["reply"])
    results["per_submolt"] = per_sm
    if not per_sm.empty:
        per_sm_path = out_dir / f"{name}_per_submolt_metrics.csv"
        per_sm.to_csv(per_sm_path, index=False)
        print(f"  wrote {per_sm_path} ({len(per_sm)} submolts)")

    # Top agents by PageRank, per submolt.
    tops = top_agents_per_submolt(edges["reply"], k=5)
    if not tops.empty:
        tops_path = out_dir / f"{name}_top_agents_by_submolt.csv"
        tops.to_csv(tops_path, index=False)
        print(f"  wrote {tops_path}")

    # ---- Plots -----------------------------------------------------------
    graphs_for_dd = {}
    if results.get("overlap_graph") is not None:
        graphs_for_dd["submolt-overlap"] = results["overlap_graph"]
    # one large reply graph (all submolts pooled) for a global degree dist
    if edges["reply"] is not None:
        G_reply_all = nx.from_pandas_edgelist(
            edges["reply"], source="replier", target="recipient",
            edge_attr="weight", create_using=nx.DiGraph,
        )
        graphs_for_dd["reply (all submolts)"] = G_reply_all
        results["reply_graph"] = G_reply_all

    if graphs_for_dd:
        plot_degree_distributions(
            graphs_for_dd, out_dir / f"{name}_degree_distribution.png",
            title=f"{name}: degree distributions",
        )

    if not per_sm.empty:
        plot_per_submolt_scatter(per_sm, out_dir / f"{name}_per_submolt_scatter.png",
                                 name=name)

    return results


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--moltbook-data", default="data", type=Path,
                        help="folder with Moltbook edge CSVs (default: data)")
    parser.add_argument("--reddit-data", default=None, type=Path,
                        help="folder with Reddit edge CSVs (same schema). "
                             "If given, produces side-by-side comparison plots.")
    parser.add_argument("--out", default="figures", type=Path,
                        help="output folder for figures + CSVs (default: figures)")
    args = parser.parse_args()

    args.out.mkdir(exist_ok=True)

    moltbook = analyze_platform(args.moltbook_data, "moltbook", args.out)

    if args.reddit_data is not None:
        if not args.reddit_data.exists():
            print(f"\nReddit data folder not found: {args.reddit_data}")
            return
        reddit = analyze_platform(args.reddit_data, "reddit", args.out)

        # Side-by-side comparison ----------------------------------------
        print("\n=== Moltbook vs Reddit ===")
        plot_comparison(
            moltbook.get("per_submolt", pd.DataFrame()),
            reddit.get("per_submolt", pd.DataFrame()),
            args.out / "comparison_per_community_metrics.png",
        )

        # Joint degree-distribution overlay
        joint = {}
        if moltbook.get("reply_graph") is not None:
            joint["Moltbook reply"] = moltbook["reply_graph"]
        if reddit.get("reply_graph") is not None:
            joint["Reddit reply"] = reddit["reply_graph"]
        if joint:
            plot_degree_distributions(
                joint, args.out / "comparison_degree_distribution.png",
                title="Reply-graph degree distribution: Moltbook vs Reddit",
            )

        # Headline stats table
        rows = []
        for plat in (moltbook, reddit):
            for key in ("overlap_stats", "bipartite_stats"):
                s = plat.get(key)
                if s:
                    rows.append({"platform": plat["name"], "graph": key, **s})
        if rows:
            summary = pd.DataFrame(rows)
            path = args.out / "summary_stats.csv"
            summary.to_csv(path, index=False)
            print(f"  wrote {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
