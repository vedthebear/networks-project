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

  Here we use HASHTAG FIRST-ADOPTION as the directed edge:
    - For each hashtag H, find the first timestamp each submolt used H.
    - Draw a directed edge A → B if submolt A adopted H before submolt B.
    - Edge weight = number of hashtags where A preceded B.

Supports two data formats automatically:
  v2 (preferred): data/data/tables/posts.csv  +  data/graphs/shared_agent_edges_core.csv
  v1 (fallback):  data/posts.json  +  data/comments.json  +  data/agent_reply_edges.csv

Outputs (figures/):
  submolt_diffusion_edges.csv     -- directed submolt->submolt edge list
  submolt_centrality.csv          -- PageRank, betweenness, HITS, degree per submolt
  agent_centrality.csv            -- PageRank, betweenness per agent (if reply data exists)
  topic_first_touch.csv           -- per hashtag: first submolt, spread time
  submolt_influence_ranking.png   -- bar charts: top submolts by PageRank + hub score
  topic_diffusion_network.png     -- network diagram (node size=PageRank, color=hub score)
  hashtag_adoption_curves.png     -- cumulative submolt adoption over time per topic
  diffusion_summary.txt           -- narrative summary
"""

import argparse
import json
import re
import warnings
from collections import defaultdict
from datetime import timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

HASHTAG_RE = re.compile(r"#(\w+)", re.UNICODE)


# ---------------------------------------------------------------------------
# Data loading — auto-detects v2 CSV or v1 JSON
# ---------------------------------------------------------------------------

def load_posts_df(data_dir: Path) -> tuple[pd.DataFrame, str]:
    """
    Return (posts_df, source_label).
    posts_df always has columns: submolt, created_at, title, content
    Tries v2 CSV paths first, falls back to v1 JSON.
    """
    # v2: data/data/tables/posts.csv  (moltbook_v2 files land here)
    for csv_rel in ("data/tables/posts.csv", "tables/posts.csv"):
        csv_path = data_dir / csv_rel
        if csv_path.exists():
            print(f"  Loading v2 CSV: {csv_path}")
            df = pd.read_csv(csv_path, dtype=str, low_memory=False)
            # Normalise column names
            df = df.rename(columns={"post_id": "id"})
            if "content" not in df.columns:
                df["content"] = ""
            df["title"]   = df["title"].fillna("")
            df["content"] = df["content"].fillna("")
            return df, "v2_csv"

    # v1: data/posts.json
    json_path = data_dir / "posts.json"
    if json_path.exists():
        print(f"  Loading v1 JSON: {json_path}")
        raw = json.load(open(json_path, encoding="utf-8"))
        rows = []
        for p in raw:
            author = p.get("author_id") or (p.get("author") or {}).get("name") or ""
            submolt = (p.get("submolt") or p.get("submolt_name") or
                       (p.get("submolt_obj") or {}).get("name") or "")
            rows.append({
                "id":         p.get("id", ""),
                "author":     author,
                "submolt":    submolt,
                "created_at": p.get("created_at", ""),
                "title":      p.get("title") or "",
                "content":    p.get("content") or "",
            })
        return pd.DataFrame(rows), "v1_json"

    raise SystemExit(
        f"No posts data found under {data_dir}.\n"
        "Expected: data/data/tables/posts.csv  OR  data/posts.json"
    )


def load_comments_df(data_dir: Path) -> pd.DataFrame:
    """Load v1 comments.json; returns empty DataFrame if not available (v2 has no comments)."""
    path = data_dir / "comments.json"
    if not path.exists():
        return pd.DataFrame(columns=["submolt", "created_at", "title", "content"])

    raw = json.load(open(path, encoding="utf-8"))

    def _flatten(lst):
        out = []
        for c in lst:
            replies = c.pop("replies", None) or []
            out.append(c)
            out.extend(_flatten(replies))
        return out

    raw = _flatten(raw)
    rows = []
    for c in raw:
        author = c.get("author_id") or (c.get("author") or {}).get("id") or ""
        rows.append({
            "id":         c.get("id", ""),
            "post_id":    c.get("post_id", ""),
            "author":     author,
            "created_at": c.get("created_at", ""),
            "title":      "",
            "content":    c.get("content") or "",
        })
    return pd.DataFrame(rows)


def load_shared_agent_graph(data_dir: Path) -> nx.Graph | None:
    """
    Load the pre-built shared-agent undirected submolt graph (v2 only).
    Returns None if not available.
    """
    for rel in ("graphs/shared_agent_edges_core.csv",
                "../graphs/shared_agent_edges_core.csv"):
        p = data_dir / rel
        if p.exists():
            df = pd.read_csv(p)
            G = nx.from_pandas_edgelist(df, source="src", target="dst",
                                        edge_attr=["shared", "jaccard"],
                                        create_using=nx.Graph)
            print(f"  Loaded pre-built shared-agent graph: "
                  f"{G.number_of_nodes()} submolts, {G.number_of_edges()} edges")
            return G
    return None


def load_reply_df(data_dir: Path) -> pd.DataFrame | None:
    """Load agent_reply_edges.csv; returns None if not available."""
    path = data_dir / "agent_reply_edges.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


# ---------------------------------------------------------------------------
# Hashtag first-touch (vectorized for 1M+ rows)
# ---------------------------------------------------------------------------

def build_first_touch(posts_df: pd.DataFrame,
                      comments_df: pd.DataFrame | None = None) -> dict:
    """
    For each hashtag, find the earliest timestamp each submolt used it.
    Returns: { hashtag -> { submolt -> datetime (tz-aware) } }
    Uses vectorised pandas operations — handles 1M+ rows efficiently.
    """
    frames = [posts_df]
    if comments_df is not None and not comments_df.empty:
        frames.append(comments_df)

    df = pd.concat(frames, ignore_index=True)
    df = df[df["submolt"].notna() & (df["submolt"] != "")].copy()

    # Parse timestamps once
    df["ts"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])

    # Extract hashtags (from title + content combined)
    text = df["title"].fillna("") + " " + df["content"].fillna("")
    df["hashtags"] = text.str.findall(r"#(\w+)", re.IGNORECASE)
    df["hashtags"] = df["hashtags"].apply(
        lambda tags: list({t.lower() for t in tags}) if isinstance(tags, list) else []
    )

    # Explode so one row per (submolt, ts, hashtag)
    df = df[["submolt", "ts", "hashtags"]].explode("hashtags").dropna(subset=["hashtags"])
    df = df[df["hashtags"].str.len() > 0]

    if df.empty:
        return {}

    # First touch per (hashtag, submolt)
    ft = (df.groupby(["hashtags", "submolt"])["ts"]
            .min()
            .reset_index()
            .rename(columns={"hashtags": "hashtag"}))

    # Build result dict
    first_touch = defaultdict(dict)
    for row in ft.itertuples(index=False):
        ts = row.ts
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        first_touch[row.hashtag][row.submolt] = ts

    return dict(first_touch)


# ---------------------------------------------------------------------------
# Submolt diffusion graph
# ---------------------------------------------------------------------------

def build_submolt_diffusion_graph(first_touch: dict,
                                  min_submolts: int = 2) -> tuple:
    """
    Directed edge A → B for each hashtag adopted by A before B.
    Weight = number of hashtags where A preceded B.
    """
    edge_weights = defaultdict(int)
    topic_rows = []

    for tag, submolt_times in first_touch.items():
        if len(submolt_times) < min_submolts:
            continue
        ordered = sorted(submolt_times.items(), key=lambda x: x[1])
        first_sub, first_time = ordered[0]
        last_time = ordered[-1][1]

        spread_secs = (last_time - first_time).total_seconds()
        topic_rows.append({
            "hashtag":       tag,
            "first_submolt": first_sub,
            "first_seen":    first_time.isoformat(),
            "n_submolts":    len(ordered),
            "spread_hours":  spread_secs / 3600,
        })

        for i, (src, _) in enumerate(ordered):
            for dst, _ in ordered[i + 1:]:
                edge_weights[(src, dst)] += 1

    edge_rows = [{"src": s, "dst": d, "weight": w}
                 for (s, d), w in edge_weights.items()]
    edges_df = pd.DataFrame(edge_rows) if edge_rows else pd.DataFrame(
        columns=["src", "dst", "weight"])

    G = nx.DiGraph()
    for _, row in edges_df.iterrows():
        G.add_edge(row["src"], row["dst"], weight=row["weight"])

    return G, edges_df, pd.DataFrame(topic_rows)


# ---------------------------------------------------------------------------
# Centrality
# ---------------------------------------------------------------------------

def compute_submolt_centrality(G: nx.DiGraph) -> pd.DataFrame:
    n = G.number_of_nodes()
    if n == 0:
        return pd.DataFrame()

    print(f"  Diffusion graph: {n:,} submolts, {G.number_of_edges():,} edges")

    pr  = nx.pagerank(G, weight="weight")
    bet = nx.betweenness_centrality(G, k=min(200, n), weight="weight", seed=42)

    try:
        hubs, auths = nx.hits(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print("  HITS did not converge — setting hub/authority to 0.")
        hubs  = {v: 0.0 for v in G.nodes()}
        auths = {v: 0.0 for v in G.nodes()}

    in_w  = dict(G.in_degree(weight="weight"))
    out_w = dict(G.out_degree(weight="weight"))

    rows = [{
        "submolt":         n,
        "pagerank":        pr.get(n, 0),
        "betweenness":     bet.get(n, 0),
        "hub_score":       hubs.get(n, 0),
        "authority_score": auths.get(n, 0),
        "in_degree":       G.in_degree(n),
        "out_degree":      G.out_degree(n),
        "in_weight":       in_w.get(n, 0),
        "out_weight":      out_w.get(n, 0),
    } for n in G.nodes()]

    return pd.DataFrame(rows).sort_values("pagerank", ascending=False).reset_index(drop=True)


def compute_agent_centrality(reply_df: pd.DataFrame) -> tuple[pd.DataFrame, nx.DiGraph]:
    agg = reply_df.groupby(["replier", "recipient"], as_index=False)["weight"].sum()
    G   = nx.from_pandas_edgelist(agg, "replier", "recipient",
                                  edge_attr="weight", create_using=nx.DiGraph)
    n = G.number_of_nodes()
    if n == 0:
        return pd.DataFrame(), G

    print(f"  Agent reply graph: {n:,} agents, {G.number_of_edges():,} edges")
    pr  = nx.pagerank(G, weight="weight")
    bet = nx.betweenness_centrality(G, k=min(500, n), weight="weight", seed=42)
    in_w  = dict(G.in_degree(weight="weight"))
    out_w = dict(G.out_degree(weight="weight"))

    rows = [{
        "agent":       node,
        "pagerank":    pr.get(node, 0),
        "betweenness": bet.get(node, 0),
        "in_weight":   in_w.get(node, 0),
        "out_weight":  out_w.get(node, 0),
    } for node in G.nodes()]

    df = pd.DataFrame(rows).sort_values("pagerank", ascending=False).reset_index(drop=True)
    return df, G


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_submolt_ranking(cent_df: pd.DataFrame, out_path: Path, top_n: int = 20):
    df_pr  = cent_df.head(top_n).iloc[::-1]
    df_hub = cent_df.sort_values("hub_score", ascending=False).head(top_n).iloc[::-1]
    if df_pr.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, top_n * 0.4)))

    axes[0].barh(df_pr["submolt"], df_pr["pagerank"], color="steelblue", alpha=0.85)
    axes[0].set_xlabel("PageRank")
    axes[0].set_title(f"Top {top_n} Submolts by PageRank\n(topic influence received)")
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].barh(df_hub["submolt"], df_hub["hub_score"], color="darkorange", alpha=0.85)
    axes[1].set_xlabel("HITS Hub Score")
    axes[1].set_title(f"Top {top_n} Submolts by Hub Score\n(topic originators)")
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle("Submolt Influence via Hashtag Diffusion (Reddit Hyperlink Analogue)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_diffusion_network(G: nx.DiGraph, cent_df: pd.DataFrame,
                           out_path: Path, top_n: int = 40):
    if G.number_of_nodes() == 0:
        return

    top_nodes = set(cent_df.head(top_n)["submolt"].tolist())
    H = G.subgraph(top_nodes).copy()
    if H.number_of_nodes() == 0:
        return

    pr_map  = dict(zip(cent_df["submolt"], cent_df["pagerank"]))
    hub_map = dict(zip(cent_df["submolt"], cent_df["hub_score"]))

    pos         = nx.spring_layout(H, weight="weight", seed=42, k=2.5)
    node_sizes  = [pr_map.get(v, 0) * 80_000 + 200 for v in H.nodes()]
    hub_vals    = [hub_map.get(v, 0) for v in H.nodes()]
    norm        = mcolors.Normalize(vmin=min(hub_vals), vmax=max(hub_vals))
    node_colors = [plt.cm.RdYlGn(norm(h)) for h in hub_vals]
    edge_ws     = [H[u][v]["weight"] for u, v in H.edges()]
    max_w       = max(edge_ws) if edge_ws else 1
    edge_widths = [0.5 + 4 * (w / max_w) for w in edge_ws]

    fig, ax = plt.subplots(figsize=(16, 12))
    nx.draw_networkx_nodes(H, pos, node_size=node_sizes,
                           node_color=node_colors, alpha=0.85, ax=ax)
    nx.draw_networkx_labels(H, pos, font_size=6, ax=ax)
    nx.draw_networkx_edges(H, pos, width=edge_widths, alpha=0.35,
                           arrows=True, arrowsize=10,
                           connectionstyle="arc3,rad=0.1", ax=ax)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Hub score (green=originator, red=receiver)")
    ax.set_title(
        f"Submolt Topic Diffusion Network (top {top_n} by PageRank)\n"
        "Node size=PageRank  |  Color=Hub score  |  Edge weight=shared hashtags",
        fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_adoption_curves(first_touch: dict, topic_df: pd.DataFrame,
                         out_path: Path, top_n: int = 6):
    top_tags = (topic_df.sort_values("n_submolts", ascending=False)
                        .head(top_n)["hashtag"].tolist())
    if not top_tags:
        return

    cols = min(2, len(top_tags))
    rows = (len(top_tags) + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), squeeze=False)
    axes_flat  = [ax for row in axes for ax in row]

    for i, tag in enumerate(top_tags):
        ax    = axes_flat[i]
        times = sorted(first_touch[tag].values())
        t0    = times[0]
        hours = [(t - t0).total_seconds() / 3600 for t in times]
        ax.step(hours, range(1, len(hours) + 1), where="post", lw=2)
        ax.set_xlabel("Hours after first appearance")
        ax.set_ylabel("Submolts reached")
        ax.set_title(f"#{tag}  ({len(times)} submolts)")
        ax.grid(True, alpha=0.3)

    for j in range(len(top_tags), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Hashtag Adoption Curves: How Fast Do Topics Spread Across Submolts?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(submolt_cent, agent_cent, topic_df, edges_df, source, out_path):
    n_topics   = len(topic_df)
    n_submolts = len(submolt_cent)
    n_agents   = len(agent_cent) if not agent_cent.empty else 0
    n_edges    = len(edges_df)

    corr, pval = (float("nan"), float("nan"))
    if n_submolts > 2:
        try:
            corr, pval = spearmanr(submolt_cent["out_weight"],
                                   submolt_cent["in_weight"])
        except Exception:
            pass

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("WHO DRIVES INFORMATION FLOW? TOPIC DIFFUSION ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Data source: {source}\n\n")

        f.write("METHODOLOGY\n")
        f.write("-" * 40 + "\n")
        f.write("  Reddit:   directed edge A -> B when a post in subreddit A\n")
        f.write("            contains a hyperlink to subreddit B.\n")
        f.write("  Moltbook: directed edge A -> B when submolt A is the first\n")
        f.write("            community to adopt a hashtag later seen in submolt B.\n")
        f.write("  Edge weight = number of hashtags where A preceded B.\n\n")

        f.write("DATASET\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Hashtags analyzed:           {n_topics:,}\n")
        f.write(f"  Submolts in diffusion graph: {n_submolts:,}\n")
        f.write(f"  Directed submolt edges:      {n_edges:,}\n")
        f.write(f"  Agents in reply graph:       {n_agents:,}\n")
        if not topic_df.empty:
            f.write(f"  Avg submolts per hashtag:    {topic_df['n_submolts'].mean():.1f}\n")
            f.write(f"  Avg spread time (hrs):       {topic_df['spread_hours'].mean():.1f}\n")
        f.write("\n")

        f.write("TOP SUBMOLTS BY PAGERANK (most influential in diffusion)\n")
        f.write("-" * 40 + "\n")
        for _, row in submolt_cent.head(15).iterrows():
            f.write(f"  {row['submolt']:35s}  PR={row['pagerank']:.5f}  "
                    f"hub={row['hub_score']:.5f}  bet={row['betweenness']:.5f}\n")
        f.write("\n")

        f.write("TOP TOPIC ORIGINATORS (hub score)\n")
        f.write("-" * 40 + "\n")
        for _, row in submolt_cent.sort_values("hub_score", ascending=False).head(15).iterrows():
            f.write(f"  {row['submolt']:35s}  hub={row['hub_score']:.5f}  "
                    f"out_deg={row['out_degree']:4d}  "
                    f"out_weight={int(row['out_weight']):5d}\n")
        f.write("\n")

        if n_agents > 0:
            f.write("TOP AGENTS BY PAGERANK\n")
            f.write("-" * 40 + "\n")
            for _, row in agent_cent.head(10).iterrows():
                f.write(f"  {str(row['agent']):35s}  PR={row['pagerank']:.5f}  "
                        f"bet={row['betweenness']:.5f}\n")
            f.write("\n")

        f.write("CROSS-LEVEL\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Out-weight vs In-weight Spearman r = {corr:.3f}  (p={pval:.3f})\n")
        if not np.isnan(corr):
            if corr > 0.5:
                f.write("  -> Same submolts dominate both sending and receiving topics.\n")
            elif corr < 0:
                f.write("  -> Senders and receivers are distinct communities.\n")
            else:
                f.write("  -> Weak specialisation between originators and receivers.\n")
        f.write("\n")

        top_spread = topic_df.sort_values("spread_hours").head(5)
        f.write("FASTEST-SPREADING HASHTAGS\n")
        f.write("-" * 40 + "\n")
        for _, row in top_spread.iterrows():
            f.write(f"  #{row['hashtag']:30s}  {row['n_submolts']:3d} submolts  "
                    f"spread={row['spread_hours']:.1f}h  "
                    f"origin={row['first_submolt']}\n")

    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data",         default="data",    type=Path)
    parser.add_argument("--out",          default="figures", type=Path)
    parser.add_argument("--min-submolts", default=2,         type=int,
                        help="min submolts a hashtag must reach to create edges")
    parser.add_argument("--top-n",        default=20,        type=int,
                        help="top N submolts shown in charts")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading posts...")
    posts_df, source = load_posts_df(args.data)
    print(f"  {len(posts_df):,} posts | "
          f"{posts_df['submolt'].nunique():,} submolts | source={source}")

    print("Loading comments...")
    comments_df = load_comments_df(args.data)
    print(f"  {len(comments_df):,} comments")

    print("Loading reply graph...")
    reply_df = load_reply_df(args.data)
    if reply_df is not None:
        print(f"  {len(reply_df):,} reply edges")
    else:
        print("  agent_reply_edges.csv not found — skipping agent-level analysis")

    print("Loading pre-built shared-agent graph (if available)...")
    G_shared = load_shared_agent_graph(args.data)

    # ------------------------------------------------------------------
    # Hashtag first-touch
    # ------------------------------------------------------------------
    print(f"\nExtracting hashtag first-touch across submolts "
          f"(this may take a moment for large datasets)...")
    first_touch = build_first_touch(posts_df, comments_df if not comments_df.empty else None)
    print(f"  {len(first_touch):,} unique hashtags found")

    if not first_touch:
        print("No hashtags found in data. Exiting.")
        return

    # ------------------------------------------------------------------
    # Submolt diffusion graph
    # ------------------------------------------------------------------
    print("\nBuilding submolt diffusion graph...")
    G_diff, edges_df, topic_df = build_submolt_diffusion_graph(
        first_touch, min_submolts=args.min_submolts)

    topic_spread = topic_df[topic_df["n_submolts"] >= args.min_submolts].copy()
    print(f"  {len(topic_spread):,} hashtags spread to >= {args.min_submolts} submolts")
    print(f"  Diffusion graph: {G_diff.number_of_nodes():,} nodes, "
          f"{G_diff.number_of_edges():,} edges")

    # ------------------------------------------------------------------
    # Centrality
    # ------------------------------------------------------------------
    print("\nComputing submolt centrality...")
    submolt_cent = compute_submolt_centrality(G_diff)

    agent_cent = pd.DataFrame()
    if reply_df is not None:
        print("\nComputing agent centrality...")
        agent_cent, _ = compute_agent_centrality(reply_df)

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    print("\nWriting outputs...")

    edges_df.to_csv(args.out / "submolt_diffusion_edges.csv", index=False)
    print(f"  wrote {args.out / 'submolt_diffusion_edges.csv'}")

    if not submolt_cent.empty:
        submolt_cent.to_csv(args.out / "submolt_centrality.csv", index=False)
        print(f"  wrote {args.out / 'submolt_centrality.csv'}")
        plot_submolt_ranking(submolt_cent, args.out / "submolt_influence_ranking.png",
                             top_n=args.top_n)
        plot_diffusion_network(G_diff, submolt_cent,
                               args.out / "topic_diffusion_network.png",
                               top_n=args.top_n)

    if not agent_cent.empty:
        agent_cent.to_csv(args.out / "agent_centrality.csv", index=False)
        print(f"  wrote {args.out / 'agent_centrality.csv'}")

    topic_df.to_csv(args.out / "topic_first_touch.csv", index=False)
    print(f"  wrote {args.out / 'topic_first_touch.csv'}")

    if not topic_spread.empty:
        plot_adoption_curves(first_touch, topic_spread,
                             args.out / "hashtag_adoption_curves.png", top_n=6)

    write_summary(submolt_cent, agent_cent, topic_spread, edges_df,
                  source, args.out / "diffusion_summary.txt")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("RESULTS SNAPSHOT")
    print("=" * 65)
    if not submolt_cent.empty:
        print(f"\nTop submolts by PageRank:")
        for _, row in submolt_cent.head(10).iterrows():
            print(f"  {row['submolt']:35s}  PR={row['pagerank']:.5f}  "
                  f"hub={row['hub_score']:.5f}")
        print(f"\nTop topic originators (hub score):")
        for _, row in (submolt_cent
                       .sort_values("hub_score", ascending=False)
                       .head(10).iterrows()):
            print(f"  {row['submolt']:35s}  hub={row['hub_score']:.5f}  "
                  f"out_weight={int(row['out_weight'])}")
    if not agent_cent.empty:
        print(f"\nTop agents by PageRank:")
        for _, row in agent_cent.head(10).iterrows():
            print(f"  {str(row['agent']):35s}  PR={row['pagerank']:.5f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
