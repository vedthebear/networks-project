"""
extract_degrees.py — build degree sequences and concentration tables for both
platforms, ready for power-law analysis.

Outputs (to data/degrees/, git-ignored):
    reddit_hyperlink_degree.csv        node, degree   (undirected projection)
    moltbook_core_degree.csv           node, degree   (>=5 authors, >=2 shared)
    moltbook_full_degree.csv           node, degree   (all submolts)
    moltbook_posts_per_submolt.csv     node, value
    moltbook_authors_per_submolt.csv   node, value
    moltbook_posts_per_agent.csv       node, value

The Moltbook community-graph degrees already exist in moltbook_v2/graphs/
(node tables produced by build_shared_agent_graph.py); we just copy the degree
column. The Reddit degree is computed here from the raw hyperlink edge list.

Usage:
    python scripts/extract_degrees.py
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DD_DIR = SCRIPT_DIR.parent                       # degree_distribution/
OUT_DIR = DD_DIR / "data" / "degrees"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REDDIT_TSV = DD_DIR / "data" / "reddit" / "soc-redditHyperlinks-body.tsv"
V2_GRAPHS = DD_DIR.parent / "moltbook_v2" / "graphs"
V2_TABLES = DD_DIR.parent / "moltbook_v2" / "data" / "tables"


# ---------------------------------------------------------------------------
# Reddit: undirected projection of the subreddit hyperlink graph
# ---------------------------------------------------------------------------
def extract_reddit() -> None:
    if not REDDIT_TSV.exists():
        raise FileNotFoundError(
            f"Missing {REDDIT_TSV}. Run scripts/fetch_reddit.py first.")
    print(f"[reddit] reading {REDDIT_TSV.name} ...")
    df = pd.read_csv(REDDIT_TSV, sep="\t",
                     usecols=["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"])
    print(f"[reddit] {len(df):,} hyperlink rows")

    # Undirected projection: edge if a link runs in EITHER direction; drop
    # self-loops. Matches the projection used in our RQ4 core-periphery work.
    UG = nx.Graph()
    src = df["SOURCE_SUBREDDIT"].to_numpy()
    tgt = df["TARGET_SUBREDDIT"].to_numpy()
    edges = ((a, b) for a, b in zip(src, tgt) if a != b)
    UG.add_edges_from(edges)

    deg = pd.DataFrame(
        [(n, d) for n, d in UG.degree()], columns=["node", "degree"])
    out = OUT_DIR / "reddit_hyperlink_degree.csv"
    deg.to_csv(out, index=False)
    print(f"[reddit] {UG.number_of_nodes():,} nodes, {UG.number_of_edges():,} "
          f"undirected edges -> {out.name}")


# ---------------------------------------------------------------------------
# Moltbook: community-graph degrees (already computed) + activity tables
# ---------------------------------------------------------------------------
def _copy_degree(node_csv: Path, out_name: str) -> None:
    df = pd.read_csv(node_csv)
    df = df.rename(columns={"submolt": "node"})[["node", "degree"]]
    df.to_csv(OUT_DIR / out_name, index=False)
    print(f"[moltbook] {len(df):,} nodes -> {out_name}")


def extract_moltbook() -> None:
    _copy_degree(V2_GRAPHS / "shared_agent_nodes_core.csv",
                 "moltbook_core_degree.csv")
    _copy_degree(V2_GRAPHS / "shared_agent_nodes_full.csv",
                 "moltbook_full_degree.csv")

    # Activity / size distributions (projection-free concentration signal)
    submolts = pd.read_csv(V2_TABLES / "submolts.csv")
    (submolts.rename(columns={"submolt": "node", "n_posts": "value"})
             [["node", "value"]]
             .to_csv(OUT_DIR / "moltbook_posts_per_submolt.csv", index=False))
    (submolts.rename(columns={"submolt": "node", "n_authors": "value"})
             [["node", "value"]]
             .to_csv(OUT_DIR / "moltbook_authors_per_submolt.csv", index=False))
    print(f"[moltbook] {len(submolts):,} submolts -> posts/authors per submolt")

    agents = pd.read_csv(V2_TABLES / "agents.csv")
    (agents.rename(columns={"agent": "node", "n_posts": "value"})
           [["node", "value"]]
           .to_csv(OUT_DIR / "moltbook_posts_per_agent.csv", index=False))
    print(f"[moltbook] {len(agents):,} agents -> posts per agent")


def main() -> None:
    extract_reddit()
    extract_moltbook()
    print("\n[done] degree sequences in", OUT_DIR)


if __name__ == "__main__":
    main()
