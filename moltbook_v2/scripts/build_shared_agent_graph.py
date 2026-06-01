#!/usr/bin/env python3
"""
build_shared_agent_graph.py -- project the agent-submolt bipartite graph onto
submolts to get the shared-agent community network.

Construction (the textbook bipartite projection, Newman ch.6 / HW2):
  - Bipartite graph: agents on one side, submolts on the other; an agent links
    to a submolt if it posted there (data/tables/membership.csv).
  - Projection onto submolts: submolts A and B get an UNDIRECTED edge if at
    least `min_shared` agents posted in both. Edge weight = Jaccard overlap
    of their agent sets:  |A∩B| / |A∪B|.  We also store the raw shared count.

This is the Moltbook community network we compare against Reddit's subreddit
hyperlink graph.

Outputs (graphs/):
  shared_agent_edges.csv     src, dst, shared, jaccard
  shared_agent_nodes.csv     submolt, degree, weighted_degree, n_authors, clustering
  shared_agent_stats.json    summary network statistics

Usage:
  python moltbook_v2/scripts/build_shared_agent_graph.py --min-shared 2
"""
import argparse, json
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "data" / "tables"
OUT = ROOT / "graphs"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-shared", type=int, default=2,
                    help="min agents shared for an edge (2 reduces noise from one-off crossovers)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    mem = pd.read_csv(TAB / "membership.csv")
    # agent -> set of submolts ; submolt -> set of agents
    sub_agents = defaultdict(set)
    for agent, submolt in zip(mem.agent, mem.submolt):
        sub_agents[submolt].add(agent)

    # Build edges by iterating over agents' submolt sets (efficient co-occurrence count).
    agent_subs = defaultdict(set)
    for agent, submolt in zip(mem.agent, mem.submolt):
        agent_subs[agent].add(submolt)

    shared = defaultdict(int)   # (a,b) sorted -> #shared agents
    for agent, subs in agent_subs.items():
        if len(subs) < 2:
            continue
        for a, b in combinations(sorted(subs), 2):
            shared[(a, b)] += 1

    G = nx.Graph()
    G.add_nodes_from(sub_agents.keys())
    rows = []
    for (a, b), sh in shared.items():
        if sh < args.min_shared:
            continue
        union = len(sub_agents[a] | sub_agents[b])
        jac = sh / union if union else 0.0
        G.add_edge(a, b, shared=sh, weight=jac)
        rows.append({"src": a, "dst": b, "shared": sh, "jaccard": jac})

    edges = pd.DataFrame(rows)
    edges.to_csv(OUT / "shared_agent_edges.csv", index=False)

    # node metrics
    clustering = nx.clustering(G, weight="weight")
    node_rows = []
    for n in G.nodes():
        node_rows.append({
            "submolt": n,
            "degree": G.degree(n),
            "weighted_degree": sum(d["weight"] for _, _, d in G.edges(n, data=True)),
            "n_authors": len(sub_agents.get(n, ())),
            "clustering": clustering.get(n, 0.0),
        })
    nodes = pd.DataFrame(node_rows).sort_values("degree", ascending=False)
    nodes.to_csv(OUT / "shared_agent_nodes.csv", index=False)

    # summary stats
    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G, weight="weight"),
        "min_shared": args.min_shared,
    }
    if G.number_of_edges():
        comps = list(nx.connected_components(G))
        big = max(comps, key=len)
        stats["n_components"] = len(comps)
        stats["largest_component_frac"] = len(big) / G.number_of_nodes()
        deg = dict(G.degree())
        stats["max_degree"] = max(deg.values())
        stats["mean_degree"] = sum(deg.values()) / len(deg)
    (OUT / "shared_agent_stats.json").write_text(json.dumps(stats, indent=2))

    print("=== SHARED-AGENT SUBMOLT GRAPH ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if G.number_of_nodes():
        print("\ntop hubs by degree:")
        print(nodes.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
