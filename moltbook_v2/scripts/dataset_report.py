#!/usr/bin/env python3
"""
dataset_report.py -- one-glance health report on the scraped Moltbook dataset.

Reports breadth (submolts covered), how many are actually ACTIVE (since empty
tail submolts contribute no graph edges), agent crossover (the fuel for the
shared-agent projection), and the resulting graph size at a couple of
edge thresholds. Writes data/tables/dataset_report.json + prints a summary.
"""
import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "data" / "tables"

def graph_size(sub_agents, agent_subs, min_shared):
    shared = defaultdict(int)
    for subs in agent_subs.values():
        if len(subs) < 2:
            continue
        for a, b in combinations(sorted(subs), 2):
            shared[(a, b)] += 1
    G = nx.Graph()
    G.add_nodes_from(sub_agents)
    for (a, b), sh in shared.items():
        if sh >= min_shared:
            G.add_edge(a, b)
    n = G.number_of_nodes()
    frac = 0.0
    if G.number_of_edges():
        frac = len(max(nx.connected_components(G), key=len)) / n
    # isolate the non-trivial core (drop degree-0 nodes)
    core = G.subgraph([x for x in G if G.degree(x) > 0]).copy()
    return {
        "min_shared": min_shared,
        "nodes_total": n,
        "edges": G.number_of_edges(),
        "connected_nodes": core.number_of_nodes(),
        "largest_component_frac_all": round(frac, 3),
    }

def main():
    posts = pd.read_csv(TAB / "posts.csv")
    mem = pd.read_csv(TAB / "membership.csv")
    subs = pd.read_csv(TAB / "submolts.csv")

    sub_authors = posts.groupby("submolt")["author"].nunique()
    active1 = (sub_authors >= 1).sum()
    active5 = (sub_authors >= 5).sum()
    active10 = (sub_authors >= 10).sum()

    sub_agents = defaultdict(set); agent_subs = defaultdict(set)
    for a, s in zip(mem.agent, mem.submolt):
        sub_agents[s].add(a); agent_subs[a].add(s)
    crossover = sum(1 for s in agent_subs.values() if len(s) > 1)

    rep = {
        "posts": int(len(posts)),
        "agents": int(posts.author.nunique()),
        "submolts_with_any_post": int(posts.submolt.nunique()),
        "submolts_active_5plus_authors": int(active5),
        "submolts_active_10plus_authors": int(active10),
        "agents_in_multiple_submolts": int(crossover),
        "graph_min_shared_1": graph_size(sub_agents, agent_subs, 1),
        "graph_min_shared_2": graph_size(sub_agents, agent_subs, 2),
    }
    (TAB / "dataset_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()
