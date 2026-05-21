"""
analyze_topic_diffusion.py  --  "Which Submolts Drive Information Flow?"

Research Question:
  Which submolts originate topics that spread to other communities, and
  which merely receive them? Are the same communities influential at the
  agent level and the submolt level?

Reddit Analogue:
  The Reddit hyperlink paper (Kumar et al., 2018) treats subreddits as
  nodes and draws a directed edge A → B when a post in A hyperlinks to B,
  then computes PageRank / betweenness to rank community influence.

  Here we have no explicit cross-community hyperlinks, so we use
  HASHTAG FIRST-ADOPTION as the directed edge:
    - For each hashtag H, find the first timestamp each submolt used H.
    - Draw a directed edge A → B if submolt A adopted H before submolt B.
    - Edge weight  = number of hashtags where A preceded B.
  This is a temporal, content-based signal of information diffusion —
  the same information flowing from one community to another, just
  via topic adoption rather than an explicit URL.

Two levels of analysis:
  1. Submolt → Submolt  (community influence, analogous to Reddit)
  2. Agent   → Agent    (individual influence, via reply graph)

Outputs (figures/):
  submolt_diffusion_edges.csv     -- directed submolt graph edge list
  submolt_centrality.csv          -- PageRank, betweenness, HITS, degree per submolt
  agent_centrality.csv            -- PageRank, betweenness per agent
  topic_first_touch.csv           -- per hashtag: which submolt adopted first, spread time
  submolt_influence_ranking.png   -- top submolts by PageRank (bar chart)
  topic_diffusion_network.png     -- submolt diffusion network (node size = PageRank)
  diffusion_summary.txt           -- narrative comparison summary
"""

import argparse
import json
import re
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HASHTAG_RE = re.compile(r"#(\w+)", re.UNICODE)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_ts(t):
    """Best-effort timestamp parser; returns timezone-aware datetime or None."""
    if t is None or (isinstance(t, float) and t != t):
        return None
    if isinstance(t, (int, float)):
        try:
            return datetime.fromtimestamp(t, tz=timezone.utc)
        except (OSError, ValueError, OverflowError):
            return None
    if isinstance(t, str):
        s = t.strip()
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
    return None


def extract_hashtags(text):
    return [tag.lower() for tag in HASHTAG_RE.findall(text or "")]


def get_post_submolt(post):
    return post.get("submolt") or (post.get("submolt_obj") or {}).get("name")


def get_author(obj):
    return obj.get("author_id") or (obj.get("author") or {}).get("id")


# ---------------------------------------------------------------------------
# Step 1: Build hashtag → submolt first-touch mapping
# ---------------------------------------------------------------------------

def build_first_touch(posts, comments):
    """
    Returns:
      first_touch: dict { hashtag -> { submolt -> earliest datetime } }
      topic_rows:  list of dicts with per-hashtag metadata
    """
    # Map post_id → submolt so we can tag comments
    post_submolt_map = {}
    for p in posts:
        pid = p.get("id")
        s = get_post_submolt(p)
        if pid and s:
            post_submolt_map[pid] = s

    first_touch = defaultdict(dict)   # hashtag -> {submolt: datetime}

    def record(tags, submolt, ts):
        if not (submolt and ts and tags):
            return
        for tag in tags:
            existing = first_touch[tag].get(submolt)
            if existing is None or ts < existing:
                first_touch[tag][submolt] = ts

    for p in posts:
        submolt = get_post_submolt(p)
        ts = parse_ts(p.get("created_at"))
        text = (p.get("title") or "") + " " + (p.get("content") or "")
        record(extract_hashtags(text), submolt, ts)

    for c in comments:
        submolt = post_submolt_map.get(c.get("post_id"))
        ts = parse_ts(c.get("created_at"))
        record(extract_hashtags(c.get("content") or ""), submolt, ts)

    return dict(first_touch)


# ---------------------------------------------------------------------------
# Step 2: Build submolt → submolt directed diffusion graph
# ---------------------------------------------------------------------------

def build_submolt_diffusion_graph(first_touch, min_submolts=2):
    """
    For each hashtag adopted by >= min_submolts communities, draw a directed
    edge from every earlier-adopting submolt to every later-adopting submolt.

    Edge weight = number of hashtags where A consistently preceded B.

    Also returns a DataFrame of per-topic metadata.
    """
    edge_weights = defaultdict(int)   # (src, dst) -> count of hashtags
    topic_rows = []

    for tag, submolt_times in first_touch.items():
        if len(submolt_times) < min_submolts:
            continue

        ordered = sorted(submolt_times.items(), key=lambda x: x[1])
        first_sub, first_time = ordered[0]
        last_time = ordered[-1][1]

        topic_rows.append({
            "hashtag":        tag,
            "first_submolt":  first_sub,
            "first_seen":     first_time.isoformat(),
            "n_submolts":     len(ordered),
            "spread_hours":   (last_time - first_time).total_seconds() / 3600,
        })

        # Every earlier submolt gets a directed edge to every later one
        for i, (src, _) in enumerate(ordered):
            for dst, _ in ordered[i + 1:]:
                edge_weights[(src, dst)] += 1

    # Assemble edge list DataFrame
    edge_rows = [
        {"src": src, "dst": dst, "weight": w}
        for (src, dst), w in edge_weights.items()
    ]
    edges_df = pd.DataFrame(edge_rows) if edge_rows else pd.DataFrame(
        columns=["src", "dst", "weight"])

    # Build NetworkX graph
    G = nx.DiGraph()
    for _, row in edges_df.iterrows():
        G.add_edge(row["src"], row["dst"], weight=row["weight"])

    return G, edges_df, pd.DataFrame(topic_rows)


# ---------------------------------------------------------------------------
# Step 3a: Submolt centrality
# ---------------------------------------------------------------------------

def compute_submolt_centrality(G):
    """
    PageRank   -- which submolts receive topics from influential communities
    Betweenness -- which submolts bridge different topic-flow clusters
    HITS hub   -- which submolts originate topics that flow to many others
    HITS auth  -- which submolts are destinations for many originators
    In/out degree -- raw volume of influence sent/received
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    n = G.number_of_nodes()
    print(f"  Submolt diffusion graph: {n} submolts, {G.number_of_edges()} edges")

    pr  = nx.pagerank(G, weight="weight")
    bet = nx.betweenness_centrality(G, k=min(200, n), weight="weight", seed=42)

    try:
        hubs, auths = nx.hits(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print("  HITS did not converge; setting hub/authority to 0.")
        hubs  = {v: 0.0 for v in G.nodes()}
        auths = {v: 0.0 for v in G.nodes()}

    in_w  = dict(G.in_degree(weight="weight"))
    out_w = dict(G.out_degree(weight="weight"))

    rows = []
    for node in G.nodes():
        rows.append({
            "submolt":            node,
            "pagerank":           pr.get(node, 0),
            "betweenness":        bet.get(node, 0),
            "hub_score":          hubs.get(node, 0),
            "authority_score":    auths.get(node, 0),
            "in_degree":          G.in_degree(node),
            "out_degree":         G.out_degree(node),
            "in_weight":          in_w.get(node, 0),
            "out_weight":         out_w.get(node, 0),
        })

    return pd.DataFrame(rows).sort_values("pagerank", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3b: Agent centrality
# ---------------------------------------------------------------------------

def compute_agent_centrality(reply_df):
    """
    Build the agent reply graph and compute PageRank + betweenness.
    Agents with high PageRank are well-replied-to (hubs of discussion).
    Agents with high betweenness bridge different conversation clusters.
    """
    agg = (reply_df
           .groupby(["replier", "recipient"], as_index=False)["weight"]
           .sum())
    G = nx.from_pandas_edgelist(
        agg, source="replier", target="recipient",
        edge_attr="weight", create_using=nx.DiGraph,
    )
    n = G.number_of_nodes()
    if n == 0:
        return pd.DataFrame(), G

    print(f"  Agent reply graph: {n} agents, {G.number_of_edges()} edges")

    pr  = nx.pagerank(G, weight="weight")
    bet = nx.betweenness_centrality(G, k=min(500, n), weight="weight", seed=42)
    in_w  = dict(G.in_degree(weight="weight"))
    out_w = dict(G.out_degree(weight="weight"))

    rows = [
        {
            "agent":       node,
            "pagerank":    pr.get(node, 0),
            "betweenness": bet.get(node, 0),
            "in_weight":   in_w.get(node, 0),
            "out_weight":  out_w.get(node, 0),
        }
        for node in G.nodes()
    ]
    df = pd.DataFrame(rows).sort_values("pagerank", ascending=False).reset_index(drop=True)
    return df, G


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_submolt_ranking(cent_df, out_path, top_n=20):
    """Horizontal bar chart: top submolts by PageRank."""
    df = cent_df.head(top_n).iloc[::-1]   # reverse so highest is at top
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(df) * 0.4)))

    # PageRank
    ax = axes[0]
    bars = ax.barh(df["submolt"], df["pagerank"], color="steelblue", alpha=0.8)
    ax.set_xlabel("PageRank (influence received from other submolts)")
    ax.set_title(f"Top {top_n} Submolts by PageRank\n(analogous to Reddit subreddit authority)")
    ax.grid(axis="x", alpha=0.3)

    # Hub score
    ax = axes[1]
    df_hub = cent_df.sort_values("hub_score", ascending=False).head(top_n).iloc[::-1]
    ax.barh(df_hub["submolt"], df_hub["hub_score"], color="darkorange", alpha=0.8)
    ax.set_xlabel("Hub score (HITS) — topics originate here and flow outward")
    ax.set_title(f"Top {top_n} Submolts by Hub Score\n(topic originators)")
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Submolt Influence: Reddit Hyperlink Analogue via Hashtag Diffusion",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_diffusion_network(G, cent_df, out_path, top_n=30):
    """
    Network diagram of the submolt diffusion graph.
    Node size  ∝ PageRank
    Node color ∝ hub score (how much it originates vs receives topics)
    Edge width ∝ weight (number of shared topics)
    Only the top_n nodes by PageRank are shown to keep it legible.
    """
    if G.number_of_nodes() == 0:
        return

    # Subgraph: top N by pagerank
    top_nodes = set(cent_df.head(top_n)["submolt"].tolist())
    H = G.subgraph(top_nodes).copy()
    if H.number_of_nodes() == 0:
        return

    pr_map  = dict(zip(cent_df["submolt"], cent_df["pagerank"]))
    hub_map = dict(zip(cent_df["submolt"], cent_df["hub_score"]))

    pos = nx.spring_layout(H, weight="weight", seed=42, k=2.5)

    node_sizes  = [pr_map.get(v, 0) * 80000 + 300 for v in H.nodes()]
    hub_vals    = [hub_map.get(v, 0) for v in H.nodes()]
    norm        = mcolors.Normalize(vmin=min(hub_vals), vmax=max(hub_vals))
    node_colors = [plt.cm.RdYlGn(norm(h)) for h in hub_vals]   # green=originator, red=receiver

    edge_weights = [H[u][v]["weight"] for u, v in H.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths  = [1 + 4 * (w / max_w) for w in edge_weights]

    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_nodes(H, pos, node_size=node_sizes,
                           node_color=node_colors, alpha=0.85, ax=ax)
    nx.draw_networkx_labels(H, pos, font_size=7, ax=ax)
    nx.draw_networkx_edges(H, pos, width=edge_widths, alpha=0.4,
                           arrows=True, arrowsize=12,
                           connectionstyle="arc3,rad=0.1", ax=ax)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Hub score (green = topic originator)")

    ax.set_title(
        f"Submolt Topic Diffusion Network (top {top_n} by PageRank)\n"
        "Node size = PageRank  |  Color = Hub score  |  Edge weight = shared hashtags",
        fontsize=11,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_adoption_curves(first_touch, topic_df, out_path, top_n=6):
    """
    For the top N most-spread hashtags, plot cumulative submolt adoption over time.
    Shows whether spread is fast/broadcast or slow/viral.
    """
    top_topics = (topic_df
                  .sort_values("n_submolts", ascending=False)
                  .head(top_n)["hashtag"]
                  .tolist())
    if not top_topics:
        return

    fig, axes = plt.subplots(
        (len(top_topics) + 1) // 2, 2,
        figsize=(14, 4 * ((len(top_topics) + 1) // 2)),
        squeeze=False,
    )
    axes_flat = [ax for row in axes for ax in row]

    for i, tag in enumerate(top_topics):
        ax = axes_flat[i]
        times = sorted(first_touch[tag].values())
        if not times:
            continue
        t0 = times[0]
        hours = [(t - t0).total_seconds() / 3600 for t in times]
        cumulative = list(range(1, len(hours) + 1))

        ax.step(hours, cumulative, where="post", linewidth=2)
        ax.set_xlabel("Hours after first appearance")
        ax.set_ylabel("Cumulative submolts reached")
        ax.set_title(f"#{tag}  ({len(times)} submolts)")
        ax.grid(True, alpha=0.3)

    # Hide any spare axes
    for j in range(len(top_topics), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Hashtag Adoption Curves: How Fast Do Topics Spread Across Submolts?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Summary narrative
# ---------------------------------------------------------------------------

def write_summary(submolt_cent, agent_cent, topic_df, edges_df, out_path):
    n_topics   = len(topic_df)
    n_submolts = len(submolt_cent)
    n_agents   = len(agent_cent)
    n_edges    = len(edges_df)

    top_sub_pr  = submolt_cent.head(5)[["submolt", "pagerank", "hub_score", "betweenness"]]
    top_sub_hub = submolt_cent.sort_values("hub_score", ascending=False).head(5)
    top_agents  = agent_cent.head(5)[["agent", "pagerank", "betweenness"]]

    # Correlation: submolt out_weight vs in_weight (originators vs receivers)
    if n_submolts > 2:
        corr, pval = spearmanr(submolt_cent["out_weight"], submolt_cent["in_weight"])
    else:
        corr, pval = float("nan"), float("nan")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("WHO DRIVES INFORMATION FLOW? TOPIC DIFFUSION ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        f.write("METHODOLOGY (Reddit Hyperlink Analogue)\n")
        f.write("-" * 40 + "\n")
        f.write("  Reddit analysis: directed edge A → B when a post in subreddit A\n")
        f.write("  contains a hyperlink to subreddit B.\n\n")
        f.write("  Moltbook analysis: directed edge A → B when submolt A is the\n")
        f.write("  first community to adopt a hashtag that later appears in submolt B.\n")
        f.write("  Edge weight = number of hashtags where A preceded B.\n\n")

        f.write("DATASET OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Hashtags analyzed:          {n_topics}\n")
        f.write(f"  Submolts in diffusion graph: {n_submolts}\n")
        f.write(f"  Directed submolt edges:      {n_edges}\n")
        f.write(f"  Agents in reply graph:       {n_agents}\n\n")
        if not topic_df.empty:
            f.write(f"  Avg submolts per hashtag:    {topic_df['n_submolts'].mean():.1f}\n")
            f.write(f"  Avg spread time (hrs):       {topic_df['spread_hours'].mean():.1f}\n\n")

        f.write("SUBMOLT INFLUENCE RANKINGS (PageRank)\n")
        f.write("-" * 40 + "\n")
        f.write("  High PageRank = receives topics from other influential submolts.\n")
        f.write("  (Analogous to high PageRank subreddits in Reddit hyperlink graph)\n\n")
        for _, row in top_sub_pr.iterrows():
            f.write(f"  {row['submolt']:30s}  PR={row['pagerank']:.4f}  "
                    f"hub={row['hub_score']:.4f}  bet={row['betweenness']:.4f}\n")
        f.write("\n")

        f.write("TOP TOPIC ORIGINATORS (Hub Score)\n")
        f.write("-" * 40 + "\n")
        f.write("  High hub score = this submolt adopts hashtags first, before others.\n\n")
        for _, row in top_sub_hub.iterrows():
            f.write(f"  {row['submolt']:30s}  hub={row['hub_score']:.4f}  "
                    f"out_degree={row['out_degree']}\n")
        f.write("\n")

        f.write("AGENT INFLUENCE RANKINGS (PageRank)\n")
        f.write("-" * 40 + "\n")
        f.write("  High PageRank agents receive many replies from well-connected agents.\n\n")
        for _, row in top_agents.iterrows():
            f.write(f"  {str(row['agent']):30s}  PR={row['pagerank']:.4f}  "
                    f"bet={row['betweenness']:.4f}\n")
        f.write("\n")

        f.write("CROSS-LEVEL COMPARISON\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Out-weight vs In-weight correlation (submolts): "
                f"r={corr:.3f} (p={pval:.3f})\n")
        f.write("  → Are the submolts that send many topics also the ones that receive many?\n")
        if not np.isnan(corr):
            if corr > 0.5:
                f.write("    Yes — the same submolts dominate both sending and receiving.\n")
            elif corr < 0:
                f.write("    No — senders and receivers are distinct communities.\n")
            else:
                f.write("    Weakly — some specialization between originators and receivers.\n")
        f.write("\n")

    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data",  default="data",    type=Path)
    parser.add_argument("--out",   default="figures", type=Path)
    parser.add_argument("--min-submolts", default=2, type=int,
                        help="minimum submolts a hashtag must appear in to create edges")
    parser.add_argument("--top-n", default=20, type=int,
                        help="top N submolts shown in charts")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading data...")

    posts_path   = args.data / "posts.json"
    comments_path = args.data / "comments.json"
    reply_path   = args.data / "agent_reply_edges.csv"

    if not posts_path.exists():
        raise SystemExit(f"Missing {posts_path} — run fetch_data.py first.")

    posts = json.load(open(posts_path, encoding="utf-8"))
    print(f"  {len(posts):,} posts loaded")

    comments = []
    if comments_path.exists():
        comments = json.load(open(comments_path, encoding="utf-8"))
        print(f"  {len(comments):,} comments loaded")
    else:
        print("  comments.json not found — analysis will use posts only.")

    reply_df = None
    if reply_path.exists():
        reply_df = pd.read_csv(reply_path)
        print(f"  {len(reply_df):,} reply edges loaded")
    else:
        print("  agent_reply_edges.csv not found — skipping agent-level analysis.")

    # ------------------------------------------------------------------
    # Submolt-level: hashtag diffusion
    # ------------------------------------------------------------------
    print("\nBuilding hashtag first-touch map...")
    first_touch = build_first_touch(posts, comments)
    print(f"  {len(first_touch):,} unique hashtags found")

    if not first_touch:
        print("No hashtags found in posts/comments. Exiting.")
        return

    print("\nBuilding submolt diffusion graph...")
    G_sub, edges_df, topic_df = build_submolt_diffusion_graph(
        first_touch, min_submolts=args.min_submolts)

    # Filter topic_df to only hashtags that actually made edges
    topic_df_spread = topic_df[topic_df["n_submolts"] >= args.min_submolts].copy()
    print(f"  {len(topic_df_spread):,} hashtags spread to >= {args.min_submolts} submolts")

    print("\nComputing submolt centrality...")
    submolt_cent = compute_submolt_centrality(G_sub)

    # ------------------------------------------------------------------
    # Agent-level: reply graph
    # ------------------------------------------------------------------
    agent_cent = pd.DataFrame()
    if reply_df is not None:
        print("\nComputing agent centrality...")
        agent_cent, _ = compute_agent_centrality(reply_df)

    # ------------------------------------------------------------------
    # Save CSVs
    # ------------------------------------------------------------------
    print("\nWriting outputs...")

    edges_df.to_csv(args.out / "submolt_diffusion_edges.csv", index=False)
    print(f"  wrote {args.out / 'submolt_diffusion_edges.csv'}")

    if not submolt_cent.empty:
        submolt_cent.to_csv(args.out / "submolt_centrality.csv", index=False)
        print(f"  wrote {args.out / 'submolt_centrality.csv'}")

    if not agent_cent.empty:
        agent_cent.to_csv(args.out / "agent_centrality.csv", index=False)
        print(f"  wrote {args.out / 'agent_centrality.csv'}")

    topic_df.to_csv(args.out / "topic_first_touch.csv", index=False)
    print(f"  wrote {args.out / 'topic_first_touch.csv'}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if not submolt_cent.empty:
        plot_submolt_ranking(submolt_cent, args.out / "submolt_influence_ranking.png",
                             top_n=args.top_n)
        plot_diffusion_network(G_sub, submolt_cent,
                               args.out / "topic_diffusion_network.png",
                               top_n=args.top_n)

    if not topic_df_spread.empty:
        plot_adoption_curves(first_touch, topic_df_spread,
                             args.out / "hashtag_adoption_curves.png", top_n=6)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    write_summary(submolt_cent, agent_cent, topic_df_spread, edges_df,
                  args.out / "diffusion_summary.txt")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("QUICK RESULTS")
    print("=" * 60)
    if not submolt_cent.empty:
        print(f"\nTop submolts by PageRank (most influential in topic diffusion):")
        for _, row in submolt_cent.head(10).iterrows():
            print(f"  {row['submolt']:30s}  PR={row['pagerank']:.4f}  "
                  f"hub={row['hub_score']:.4f}  out_deg={row['out_degree']}")
    if not agent_cent.empty:
        print(f"\nTop agents by PageRank (most influential in reply network):")
        for _, row in agent_cent.head(10).iterrows():
            print(f"  {str(row['agent']):30s}  PR={row['pagerank']:.4f}  "
                  f"bet={row['betweenness']:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
