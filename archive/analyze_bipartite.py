"""
analyze_bipartite.py -- Section 3 of "Who Controls the Agent Internet?"

Bipartite agent-submolt analysis.

The agents x submolts incidence matrix B is the substrate; from it we
build two one-mode projections:

  P_agent  = B B^T   -- agents linked if they co-participate in submolts
                         (weight = sum_j B[i,j] * B[k,j])
  P_submolt= B^T B   -- submolts linked if they share active agents
                         (weight = #shared agents, weighted by activity)

We also compute the Latapy bipartite clustering coefficient for each
agent (a generalization of the triangle-based clustering coefficient
to bipartite graphs) and run Louvain community detection on both
projections.

Outputs (figures/):
  incidence_matrix.npz                     -- sparse B
  incidence_agents.csv / incidence_submolts.csv
  agent_agent_projection.csv               -- weighted edges
  submolt_submolt_projection.csv           -- weighted edges
  agent_projection_communities.csv
  submolt_projection_communities.csv
  bipartite_clustering.csv
  bipartite_summary.csv
  incidence_heatmap.png
  bipartite_projection_sizes.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse


def build_incidence(df: pd.DataFrame):
    agents = sorted(df["agent_id"].unique().tolist())
    submolts = sorted(df["submolt"].unique().tolist())
    a_idx = {a: i for i, a in enumerate(agents)}
    s_idx = {s: j for j, s in enumerate(submolts)}

    rows = df["agent_id"].map(a_idx).values
    cols = df["submolt"].map(s_idx).values
    data = df["weight"].astype(float).values

    B = sparse.csr_matrix((data, (rows, cols)),
                          shape=(len(agents), len(submolts)))
    return B, agents, submolts


def project_agent_agent(B: sparse.csr_matrix, agents, min_weight: float = 1.0) -> pd.DataFrame:
    """A = B B^T, dropped self-loops, upper-triangular as edge list."""
    A = B @ B.T
    A.setdiag(0)
    A.eliminate_zeros()
    A = sparse.triu(A, k=1).tocoo()
    rows = []
    for i, j, w in zip(A.row, A.col, A.data):
        if w >= min_weight:
            rows.append({"agent_a": agents[i], "agent_b": agents[j], "weight": float(w)})
    return pd.DataFrame(rows)


def project_submolt_submolt(B: sparse.csr_matrix, submolts) -> pd.DataFrame:
    """S = B^T B, similar treatment."""
    S = B.T @ B
    S.setdiag(0)
    S.eliminate_zeros()
    S = sparse.triu(S, k=1).tocoo()
    rows = []
    for i, j, w in zip(S.row, S.col, S.data):
        rows.append({"submolt_a": submolts[i], "submolt_b": submolts[j],
                     "weight": float(w)})
    return pd.DataFrame(rows)


def latapy_clustering(Gb, agent_nodes):
    """Bipartite clustering coefficient (Latapy 2008)."""
    try:
        from networkx.algorithms.bipartite.cluster import latapy_clustering as lc
        return lc(Gb, agent_nodes)
    except Exception as e:
        print(f"  Latapy clustering failed ({e}); trying networkx generic ...")
        try:
            from networkx.algorithms.bipartite.cluster import clustering as bc
            return bc(Gb, agent_nodes)
        except Exception as e2:
            print(f"  bipartite clustering unavailable ({e2})")
            return {}


def communities(G: nx.Graph):
    if G.number_of_edges() == 0:
        return [], float("nan")
    comms = list(nx.community.louvain_communities(G, seed=42, weight="weight"))
    mod = nx.community.modularity(G, comms, weight="weight")
    return comms, mod


def plot_incidence_heatmap(B: sparse.csr_matrix, out_path: Path,
                           agents: list, submolts: list, max_agents: int = 200):
    """Heatmap. If too many agents, take the top-N most active."""
    if B.shape[0] <= max_agents:
        M = B.toarray()
        ylabel = f"agents (n={B.shape[0]})"
    else:
        totals = np.asarray(B.sum(axis=1)).ravel()
        idx = np.argsort(-totals)[:max_agents]
        M = B[idx].toarray()
        ylabel = f"top {max_agents} agents by total activity"

    fig, ax = plt.subplots(figsize=(min(14, 0.25 * len(submolts) + 4), 10))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(submolts)))
    ax.set_xticklabels(submolts, rotation=90, fontsize=7)
    ax.set_xlabel("submolts")
    ax.set_ylabel(ylabel)
    ax.set_title("Agent x Submolt incidence (activity weight)")
    fig.colorbar(im, ax=ax, label="weight")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_projection_community_sizes(comms_a, comms_s, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, comms, label in ((axes[0], comms_a, "agent projection"),
                             (axes[1], comms_s, "submolt projection")):
        sizes = sorted((len(c) for c in comms), reverse=True)
        if sizes:
            ax.bar(range(len(sizes)), sizes)
        ax.set_xlabel("community rank")
        ax.set_ylabel("size")
        ax.set_yscale("log")
        ax.set_title(f"Louvain community sizes: {label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", type=Path)
    parser.add_argument("--out", default="figures", type=Path)
    parser.add_argument("--agent-projection-min-weight", default=2.0, type=float,
                        help="filter the agent-agent projection to keep size manageable")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)

    in_path = args.data / "agent_submolt_edges.csv"
    if not in_path.exists():
        raise SystemExit(f"missing {in_path}; run fetch_data.py first")

    df = pd.read_csv(in_path)
    print(f"Agent x Submolt edges: {len(df):,}")

    B, agents, submolts = build_incidence(df)
    print(f"Incidence matrix B: {B.shape[0]:,} agents x {B.shape[1]} submolts "
          f"(nnz={B.nnz:,})")
    sparse.save_npz(args.out / "incidence_matrix.npz", B)
    pd.DataFrame({"agent_id": agents}).to_csv(args.out / "incidence_agents.csv",
                                              index=False)
    pd.DataFrame({"submolt": submolts}).to_csv(args.out / "incidence_submolts.csv",
                                               index=False)

    # ----- Agent x Agent projection ----------------------------------------
    print("\nProjecting onto agents (B @ B.T) ...")
    agent_proj_df = project_agent_agent(B, agents, args.agent_projection_min_weight)
    agent_proj_df.to_csv(args.out / "agent_agent_projection.csv", index=False)
    print(f"  agent-projection edges (min_weight={args.agent_projection_min_weight}): "
          f"{len(agent_proj_df):,}")

    Ga = nx.from_pandas_edgelist(agent_proj_df, source="agent_a", target="agent_b",
                                 edge_attr="weight") if not agent_proj_df.empty else nx.Graph()
    comms_a, mod_a = communities(Ga)
    rows = [{"agent": a, "community": i}
            for i, c in enumerate(comms_a) for a in c]
    pd.DataFrame(rows).to_csv(args.out / "agent_projection_communities.csv", index=False)
    print(f"  agent communities: {len(comms_a)} (modularity={mod_a:.4f})")

    # ----- Submolt x Submolt projection ------------------------------------
    print("\nProjecting onto submolts (B.T @ B) ...")
    submolt_proj_df = project_submolt_submolt(B, submolts)
    submolt_proj_df.to_csv(args.out / "submolt_submolt_projection.csv", index=False)
    print(f"  submolt-projection edges: {len(submolt_proj_df):,}")

    Gs = nx.from_pandas_edgelist(submolt_proj_df, source="submolt_a", target="submolt_b",
                                 edge_attr="weight") if not submolt_proj_df.empty else nx.Graph()
    comms_s, mod_s = communities(Gs)
    rows = [{"submolt": s, "community": i}
            for i, c in enumerate(comms_s) for s in c]
    pd.DataFrame(rows).to_csv(args.out / "submolt_projection_communities.csv", index=False)
    print(f"  submolt communities: {len(comms_s)} (modularity={mod_s:.4f})")

    # ----- Bipartite clustering --------------------------------------------
    print("\nBipartite clustering (Latapy) ...")
    Gb = nx.Graph()
    Gb.add_nodes_from(agents, bipartite=0)
    Gb.add_nodes_from(submolts, bipartite=1)
    Gb.add_weighted_edges_from(
        zip(df["agent_id"], df["submolt"], df["weight"].astype(float)),
    )
    cc = latapy_clustering(Gb, agents)
    if cc:
        cc_df = (pd.DataFrame({"agent": list(cc.keys()),
                               "bipartite_clustering": list(cc.values())})
                 .sort_values("bipartite_clustering", ascending=False))
        cc_df.to_csv(args.out / "bipartite_clustering.csv", index=False)
        print(f"  mean bipartite clustering: {cc_df['bipartite_clustering'].mean():.4f}")

    # ----- Plots -----------------------------------------------------------
    plot_incidence_heatmap(B, args.out / "incidence_heatmap.png", agents, submolts)
    plot_projection_community_sizes(comms_a, comms_s,
                                    args.out / "bipartite_projection_sizes.png")

    summary = {
        "agents":                        B.shape[0],
        "submolts":                      B.shape[1],
        "incidence_nnz":                 B.nnz,
        "agent_projection_edges":        len(agent_proj_df),
        "submolt_projection_edges":      len(submolt_proj_df),
        "agent_communities":             len(comms_a),
        "agent_projection_modularity":   mod_a,
        "submolt_communities":           len(comms_s),
        "submolt_projection_modularity": mod_s,
        "mean_bipartite_clustering":     (float(np.mean(list(cc.values())))
                                          if cc else float("nan")),
    }
    pd.Series(summary).to_csv(args.out / "bipartite_summary.csv", header=False)
    print("\nHeadline stats:")
    for k, v in summary.items():
        print(f"  {k:>34}: {v}")


if __name__ == "__main__":
    main()
