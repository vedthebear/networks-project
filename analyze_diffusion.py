"""
analyze_diffusion.py -- Section 2 of "Who Controls the Agent Internet?"

Information diffusion: how does a topic spread through the agent
population over time, and what does the cascade look like on the
reply graph?

For each top topic (hashtags preferred, fall back to high-frequency
non-stopword terms):
  1. find every agent's first-touch timestamp with that topic
  2. construct a diffusion cascade -- each adopter is parented to
     the most-recently-adopted neighbor in the reply graph (or is a
     root, if no prior adopter is connected)
  3. compute structural virality (Goel, Anderson, Hofman, Watts 2015)
     = average pairwise distance in the largest cascade tree.
     Low virality -> "broadcast" spread (one source, many adopters);
     high virality -> "viral" spread (deep peer-to-peer chains).

Outputs (figures/):
  diffusion_summary.csv           -- per-topic metrics
  diffusion_cascade_<topic>.csv   -- parent/child edges per topic
  diffusion_adoption_curves.png   -- cumulative adoption over time
  diffusion_virality_scatter.png  -- adopters vs structural virality
"""

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


HASHTAG_RE = re.compile(r"#(\w+)", re.UNICODE)
WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z']{4,}\b")  # words >= 5 chars

STOPWORDS = set("""
about above after again against among because before being below between
could would should their these those there where which while since rather
through about would could should they them then than just been will more
some into very over also like only most when most what about being doing
having around right thing think really maybe still even though might thats
something nothing everything anyone everyone someone people other another
""".split())


def parse_ts(t):
    """Best-effort timestamp parser."""
    if t is None or (isinstance(t, float) and t != t):  # nan
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


def extract_topics(posts, top_k=8, min_count=20):
    """Top hashtags if available, else top high-frequency content words."""
    hashtag_counts = Counter()
    word_counts = Counter()
    for p in posts:
        text = (p.get("title") or "") + " " + (p.get("content") or "")
        for tag in HASHTAG_RE.findall(text):
            hashtag_counts[tag.lower()] += 1
        for w in WORD_RE.findall(text.lower()):
            if w not in STOPWORDS:
                word_counts[w] += 1

    hashtags = [t for t, c in hashtag_counts.most_common() if c >= min_count]
    if len(hashtags) >= top_k:
        return hashtags[:top_k], "hashtag"

    words = [w for w, c in word_counts.most_common() if c >= min_count]
    return words[:top_k], "keyword"


def first_touch_per_agent(topic, posts, comments):
    """agent_id -> earliest datetime they engaged with the topic."""
    pat = re.compile(r"\b" + re.escape(topic) + r"\b", re.IGNORECASE)
    first = {}

    def maybe_record(text, author, ts):
        if not (text and author and ts):
            return
        if pat.search(text):
            existing = first.get(author)
            if existing is None or ts < existing:
                first[author] = ts

    for p in posts:
        author = p.get("author_id") or (p.get("author") or {}).get("id")
        ts = parse_ts(p.get("created_at"))
        text = (p.get("title") or "") + " " + (p.get("content") or "")
        maybe_record(text, author, ts)

    for c in comments:
        author = c.get("author_id") or (c.get("author") or {}).get("id")
        ts = parse_ts(c.get("created_at"))
        text = c.get("content") or ""
        maybe_record(text, author, ts)

    return first


def build_cascade(first_touch, reply_graph: nx.DiGraph) -> nx.DiGraph:
    """
    Cascade tree construction.

    Walk adopters in chronological order. For each new adopter A:
      - candidates = neighbors of A in the reply graph (in either direction)
                     who have already adopted
      - parent = most-recently-adopted candidate; otherwise A is a root
    """
    sorted_agents = sorted(first_touch.items(), key=lambda kv: kv[1])
    adopted = set()
    cascade = nx.DiGraph()

    for agent, ts in sorted_agents:
        cascade.add_node(agent, adopted_at=ts.isoformat())
        if reply_graph.has_node(agent):
            neighbors = (set(reply_graph.predecessors(agent)) |
                         set(reply_graph.successors(agent)))
        else:
            neighbors = set()
        candidates = neighbors & adopted
        if candidates:
            parent = max(candidates, key=lambda x: first_touch[x])
            cascade.add_edge(parent, agent)
        adopted.add(agent)

    return cascade


def structural_virality(cascade: nx.DiGraph) -> float:
    """Goel et al. (2015): average pairwise distance in the largest
    weakly-connected component of the cascade tree."""
    if cascade.number_of_nodes() < 2:
        return float("nan")
    UG = cascade.to_undirected()
    comps = list(nx.connected_components(UG))
    if not comps:
        return float("nan")
    biggest = UG.subgraph(max(comps, key=len))
    if biggest.number_of_nodes() < 2:
        return float("nan")
    return nx.average_shortest_path_length(biggest)


def cascade_metrics(topic, cascade: nx.DiGraph, first_touch: dict) -> dict:
    if cascade.number_of_nodes() == 0:
        return {"topic": topic, "adopters": 0}
    UG = cascade.to_undirected()
    components = list(nx.connected_components(UG))
    largest = max(components, key=len) if components else set()
    times = sorted(first_touch.values())
    half_idx = len(times) // 2

    return {
        "topic":               topic,
        "adopters":            cascade.number_of_nodes(),
        "n_cascades":          len(components),
        "largest_cascade":     len(largest),
        "structural_virality": structural_virality(cascade),
        "first_adopt":         times[0].isoformat() if times else None,
        "last_adopt":          times[-1].isoformat() if times else None,
        "time_to_half_hours":  ((times[half_idx] - times[0]).total_seconds() / 3600
                                if len(times) > 1 else float("nan")),
    }


def plot_adoption_curves(topics_data, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    plotted = 0
    for topic, ft in topics_data.items():
        if not ft:
            continue
        times = sorted(ft.values())
        ax.plot(times, range(1, len(times) + 1), label=topic, alpha=0.85)
        plotted += 1
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("time")
    ax.set_ylabel("cumulative adopters")
    ax.set_title("Topic diffusion across MoltBook agents")
    ax.legend(fontsize=8, loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_virality_scatter(summary_df, out_path):
    if summary_df.empty:
        return
    df = summary_df.dropna(subset=["structural_virality"])
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["adopters"], df["structural_virality"], alpha=0.7,
               s=60 + df["largest_cascade"])
    for _, r in df.iterrows():
        ax.annotate(r["topic"], (r["adopters"], r["structural_virality"]),
                    fontsize=8, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("adopters (log)")
    ax.set_ylabel("structural virality (avg pairwise distance)")
    ax.set_title("Broadcast vs viral: how each topic spread")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", type=Path)
    parser.add_argument("--out", default="figures", type=Path)
    parser.add_argument("--top-k-topics", default=8, type=int)
    parser.add_argument("--min-adopters", default=30, type=int,
                        help="skip topics with fewer first-touchers than this")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)

    posts_path = args.data / "posts.json"
    comments_path = args.data / "comments.json"
    reply_path = args.data / "agent_reply_edges.csv"
    for p in (posts_path, comments_path, reply_path):
        if not p.exists():
            raise SystemExit(f"missing {p}; run fetch_data.py first")

    posts = json.load(open(posts_path, encoding="utf-8"))
    comments = json.load(open(comments_path, encoding="utf-8"))
    reply_df = pd.read_csv(reply_path)

    G_reply = nx.from_pandas_edgelist(
        reply_df.groupby(["replier", "recipient"], as_index=False)["weight"].sum(),
        source="replier", target="recipient",
        edge_attr="weight", create_using=nx.DiGraph,
    )
    print(f"Reply graph: {G_reply.number_of_nodes():,} agents, "
          f"{G_reply.number_of_edges():,} edges")

    topics, source = extract_topics(posts, top_k=args.top_k_topics)
    print(f"Top {len(topics)} {source}-topics: {topics}")

    summary_rows = []
    topics_data = {}
    for topic in topics:
        ft = first_touch_per_agent(topic, posts, comments)
        if len(ft) < args.min_adopters:
            print(f"  skip '{topic}' ({len(ft)} adopters < {args.min_adopters})")
            continue

        topics_data[topic] = ft
        cascade = build_cascade(ft, G_reply)
        m = cascade_metrics(topic, cascade, ft)
        summary_rows.append(m)
        print(f"  '{topic}': {m['adopters']} adopters, "
              f"{m['n_cascades']} cascades, "
              f"largest={m['largest_cascade']}, "
              f"virality={m['structural_virality']:.3f}")

        if cascade.number_of_edges() > 0:
            edges_df = nx.to_pandas_edgelist(cascade, source="parent", target="child")
            safe = re.sub(r"[^a-zA-Z0-9_-]", "_", topic)[:40]
            edges_df.to_csv(args.out / f"diffusion_cascade_{safe}.csv", index=False)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(args.out / "diffusion_summary.csv", index=False)
        print(f"  wrote {args.out / 'diffusion_summary.csv'}")
        plot_adoption_curves(topics_data, args.out / "diffusion_adoption_curves.png")
        plot_virality_scatter(summary_df, args.out / "diffusion_virality_scatter.png")
    else:
        print("No topics passed the min-adopters threshold.")


if __name__ == "__main__":
    main()
