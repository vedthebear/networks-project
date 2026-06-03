"""
analyze_adoption_influence.py -- "Who Drives Information Flow?"

Research Question:
  Are network HUBS (high PageRank) the first to adopt new topics,
  or do BROKERS (high betweenness) introduce topics earlier and drive spread?

Analysis Pipeline:
  1. Build the global reply graph from agent_reply_edges.csv
  2. Compute centralities: PageRank (hub score), betweenness (broker score)
  3. Extract topics from posts/comments (hashtags, high-freq keywords)
  4. For each topic, find first-touch timestamps per agent
  5. Stratify adopters: early (first quartile), late (last quartile)
  6. Compare centrality profiles: are early adopters more central or peripheral?
  7. Test: correlation between adoption order and centrality scores

Outputs (figures/):
  adoption_centrality_by_topic.csv  -- one row per topic: early vs late adoption patterns
  adopter_centralities.csv          -- one row per adopter: topic, adoption_rank, centralities
  adoption_hub_vs_broker.png        -- violin plots of PageRank/betweenness by adoption quartile
  adoption_timing_vs_centrality.png -- scatter: adoption rank vs centrality, one per metric
  adoption_influence_summary.txt    -- narrative summary of findings
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #

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


def extract_topics(posts, top_k=20, min_count=1):
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
    if hashtags:
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


# --------------------------------------------------------------------------- #
# Network Analysis                                                            #
# --------------------------------------------------------------------------- #

def build_reply_graph(reply_df: pd.DataFrame) -> nx.DiGraph:
    """Aggregate per-submolt reply weights into one global directed graph."""
    agg = (reply_df
           .groupby(["replier", "recipient"], as_index=False)["weight"]
           .sum())
    return nx.from_pandas_edgelist(
        agg, source="replier", target="recipient",
        edge_attr="weight", create_using=nx.DiGraph,
    )


def compute_centralities(G: nx.DiGraph) -> pd.DataFrame:
    """Compute PageRank (hub) and betweenness (broker) centrality."""
    n = G.number_of_nodes()
    if n == 0:
        return pd.DataFrame()

    print(f"  Computing centralities on {n:,} nodes / {G.number_of_edges():,} edges ...")

    cents = {
        "pagerank":   nx.pagerank(G, weight="weight"),
        "betweenness": nx.betweenness_centrality(G, k=min(500, n), weight="weight", seed=42),
    }

    return (pd.DataFrame(cents)
            .rename_axis("agent")
            .reset_index())


# --------------------------------------------------------------------------- #
# Adoption x Centrality Analysis                                              #
# --------------------------------------------------------------------------- #

def analyze_topic_adoption(topic, first_touch, cents_df):
    """
    For a given topic:
      1. Get adoption order (ranked by timestamp)
      2. Stratify into early (Q1) vs late (Q4) adopters
      3. Compare their centrality profiles
      4. Return statistics
    """
    if not first_touch or len(first_touch) < 2:
        return None

    adopters = sorted(first_touch.items(), key=lambda kv: kv[1])
    adopter_ids = [a for a, _ in adopters]

    # Merge with centrality data
    cent_subset = cents_df[cents_df["agent"].isin(adopter_ids)].copy()
    if cent_subset.empty:
        return None

    cent_subset["adoption_order"] = (
        cent_subset["agent"].map({a: i for i, a in enumerate(adopter_ids)})
    )
    cent_subset = cent_subset.dropna(subset=["adoption_order"])

    if len(cent_subset) < 2:
        return None

    n = len(cent_subset)
    q1_thresh = cent_subset["adoption_order"].quantile(0.25)
    q4_thresh = cent_subset["adoption_order"].quantile(0.75)

    early = cent_subset[cent_subset["adoption_order"] <= q1_thresh]
    late = cent_subset[cent_subset["adoption_order"] >= q4_thresh]

    if len(early) < 1 or len(late) < 1:
        return None

    # Compute statistics
    pagerank_corr, pagerank_pval = spearmanr(
        cent_subset["adoption_order"].values,
        cent_subset["pagerank"].values
    )
    betweenness_corr, betweenness_pval = spearmanr(
        cent_subset["adoption_order"].values,
        cent_subset["betweenness"].values
    )

    return {
        "topic": topic,
        "n_adopters": len(cent_subset),
        "n_early": len(early),
        "n_late": len(late),
        "early_pagerank_mean": early["pagerank"].mean(),
        "late_pagerank_mean": late["pagerank"].mean(),
        "early_pagerank_std": early["pagerank"].std(),
        "late_pagerank_std": late["pagerank"].std(),
        "early_betweenness_mean": early["betweenness"].mean(),
        "late_betweenness_mean": late["betweenness"].mean(),
        "early_betweenness_std": early["betweenness"].std(),
        "late_betweenness_std": late["betweenness"].std(),
        "pagerank_adoption_corr": pagerank_corr,
        "pagerank_adoption_pval": pagerank_pval,
        "betweenness_adoption_corr": betweenness_corr,
        "betweenness_adoption_pval": betweenness_pval,
        "early_adopters_are_hubs": early["pagerank"].mean() > late["pagerank"].mean(),
        "early_adopters_are_brokers": early["betweenness"].mean() > late["betweenness"].mean(),
    }


# --------------------------------------------------------------------------- #
# Visualization                                                               #
# --------------------------------------------------------------------------- #

def plot_hub_vs_broker(topic_results, cents_df, first_touches, out_path):
    """
    For each topic, plot violin distributions of PageRank and betweenness
    stratified by adoption quartile.
    """
    # Pick top 4 topics by adopter count
    topics = sorted(
        [r for r in topic_results if r is not None],
        key=lambda r: r["n_adopters"],
        reverse=True
    )[:4]

    if not topics:
        return

    fig, axes = plt.subplots(len(topics), 2, figsize=(12, 4 * len(topics)))
    if len(topics) == 1:
        axes = [axes]

    for idx, topic_info in enumerate(topics):
        topic = topic_info["topic"]
        first_touch = first_touches[topic]
        adopters = sorted(first_touch.items(), key=lambda kv: kv[1])
        adopter_ids = [a for a, _ in adopters]

        cent_subset = cents_df[cents_df["agent"].isin(adopter_ids)].copy()
        cent_subset["adoption_order"] = (
            cent_subset["agent"].map({a: i for i, a in enumerate(adopter_ids)})
        )
        cent_subset = cent_subset.dropna(subset=["adoption_order"])

        q1 = cent_subset["adoption_order"].quantile(0.25)
        q4 = cent_subset["adoption_order"].quantile(0.75)

        cent_subset["quartile"] = "Q2/Q3"
        cent_subset.loc[cent_subset["adoption_order"] <= q1, "quartile"] = "Early (Q1)"
        cent_subset.loc[cent_subset["adoption_order"] >= q4, "quartile"] = "Late (Q4)"

        # PageRank
        ax = axes[idx][0]
        data_pr = [cent_subset[cent_subset["quartile"] == q]["pagerank"].values
                   for q in ("Early (Q1)", "Q2/Q3", "Late (Q4)")]
        ax.violinplot(data_pr, positions=[0, 1, 2], showmeans=True)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Early", "Middle", "Late"])
        ax.set_ylabel("PageRank (hub score)")
        ax.set_title(f"{topic}: PageRank by adoption order")
        ax.grid(True, alpha=0.3)

        # Betweenness
        ax = axes[idx][1]
        data_bet = [cent_subset[cent_subset["quartile"] == q]["betweenness"].values
                    for q in ("Early (Q1)", "Q2/Q3", "Late (Q4)")]
        ax.violinplot(data_bet, positions=[0, 1, 2], showmeans=True)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Early", "Middle", "Late"])
        ax.set_ylabel("Betweenness (broker score)")
        ax.set_title(f"{topic}: Betweenness by adoption order")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Hubs vs Brokers: Centrality of early vs late topic adopters")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_adoption_timing_vs_centrality(topic_results, cents_df, first_touches, out_path):
    """
    Scatter plot: adoption rank (x) vs centrality (y) for the top 3 topics.
    One subplot per topic, one scatter per metric (PageRank and betweenness).
    """
    topics = sorted(
        [r for r in topic_results if r is not None],
        key=lambda r: r["n_adopters"],
        reverse=True
    )[:3]

    if not topics:
        return

    fig, axes = plt.subplots(len(topics), 2, figsize=(12, 4 * len(topics)))
    if len(topics) == 1:
        axes = [axes]

    for idx, topic_info in enumerate(topics):
        topic = topic_info["topic"]
        first_touch = first_touches[topic]
        adopters = sorted(first_touch.items(), key=lambda kv: kv[1])
        adopter_ids = [a for a, _ in adopters]

        cent_subset = cents_df[cents_df["agent"].isin(adopter_ids)].copy()
        cent_subset["adoption_rank"] = (
            cent_subset["agent"].map({a: i for i, a in enumerate(adopter_ids)})
        )
        cent_subset = cent_subset.dropna(subset=["adoption_rank"])

        # PageRank
        ax = axes[idx][0]
        ax.scatter(cent_subset["adoption_rank"], cent_subset["pagerank"], alpha=0.6, s=40)
        ax.set_xlabel("adoption rank (early → late)")
        ax.set_ylabel("PageRank")
        ax.set_title(f"{topic}: PageRank vs adoption order")
        if len(cent_subset) > 2:
            z = np.polyfit(cent_subset["adoption_rank"], cent_subset["pagerank"], 1)
            p = np.poly1d(z)
            ax.plot(cent_subset["adoption_rank"].values,
                   p(cent_subset["adoption_rank"].values),
                   "r--", alpha=0.5, label="trend")
            ax.legend()
        ax.grid(True, alpha=0.3)

        # Betweenness
        ax = axes[idx][1]
        ax.scatter(cent_subset["adoption_rank"], cent_subset["betweenness"], alpha=0.6, s=40)
        ax.set_xlabel("adoption rank (early → late)")
        ax.set_ylabel("Betweenness")
        ax.set_title(f"{topic}: Betweenness vs adoption order")
        if len(cent_subset) > 2:
            z = np.polyfit(cent_subset["adoption_rank"], cent_subset["betweenness"], 1)
            p = np.poly1d(z)
            ax.plot(cent_subset["adoption_rank"].values,
                   p(cent_subset["adoption_rank"].values),
                   "r--", alpha=0.5, label="trend")
            ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Adoption timing vs centrality: early adopters are more/less central?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default="data", type=Path,
                        help="folder containing posts.json, comments.json, agent_reply_edges.csv")
    parser.add_argument("--out", default="figures", type=Path,
                        help="output folder for figures and CSVs")
    parser.add_argument("--top-k-topics", default=20, type=int,
                        help="number of topics to analyze")
    parser.add_argument("--min-adopters", default=1, type=int,
                        help="minimum adopters per topic")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)

    # Load data — supports v2 CSV (data/data/tables/posts.csv) and v1 JSON
    print("Loading data...")

    def _load_posts(data_dir):
        for csv_rel in ("data/tables/posts.csv", "tables/posts.csv"):
            p = data_dir / csv_rel
            if p.exists():
                print(f"  posts: {p} (v2 CSV)")
                df = pd.read_csv(p, dtype=str, low_memory=False)
                df = df.rename(columns={"post_id": "id"})
                df["content"] = df.get("content", pd.Series([""] * len(df))).fillna("")
                df["title"]   = df["title"].fillna("")
                # Convert to list of dicts matching v1 field expectations
                posts = []
                for _, row in df.iterrows():
                    posts.append({
                        "id":         row.get("id", ""),
                        "author_id":  row.get("author", ""),
                        "submolt":    row.get("submolt", ""),
                        "created_at": row.get("created_at", ""),
                        "title":      row.get("title", ""),
                        "content":    row.get("content", ""),
                    })
                return posts
        p = data_dir / "posts.json"
        if p.exists():
            print(f"  posts: {p} (v1 JSON)")
            raw = json.load(open(p, encoding="utf-8"))
            for post in raw:
                if not post.get("author_id"):
                    post["author_id"] = (post.get("author") or {}).get("name") or \
                                        (post.get("author") or {}).get("id")
                if not post.get("submolt"):
                    post["submolt"] = (post.get("submolt_obj") or {}).get("name") or \
                                       post.get("submolt_name")
            return raw
        raise SystemExit(f"No posts data found under {data_dir}")

    def _load_comments(data_dir):
        p = data_dir / "comments.json"
        if not p.exists():
            print("  comments: none found — topic adoption from posts only")
            return []
        print(f"  comments: {p}")
        raw = json.load(open(p, encoding="utf-8"))
        def _flatten(lst):
            out = []
            for c in lst:
                replies = c.pop("replies", None) or []
                out.append(c)
                out.extend(_flatten(replies))
            return out
        return _flatten(raw)

    posts    = _load_posts(args.data)
    comments = _load_comments(args.data)

    reply_path = args.data / "agent_reply_edges.csv"
    if reply_path.exists():
        reply_df = pd.read_csv(reply_path)
        print(f"  reply edges: {len(reply_df):,} rows")
    else:
        reply_df = None
        print("  agent_reply_edges.csv not found — skipping network centrality")

    # Build network and compute centralities (skipped if no reply data)
    cents_df = pd.DataFrame()
    if reply_df is not None:
        print("\nBuilding reply graph and computing centralities...")
        G_reply  = build_reply_graph(reply_df)
        print(f"Reply graph: {G_reply.number_of_nodes():,} agents, "
              f"{G_reply.number_of_edges():,} edges")
        cents_df = compute_centralities(G_reply)
    else:
        print("\nSkipping agent centrality (no reply edge data).")

    # Extract topics
    print("\nExtracting topics...")
    topics, source = extract_topics(posts, top_k=args.top_k_topics)
    print(f"Top {len(topics)} {source}-topics: {topics}")

    # Analyze adoption patterns per topic
    print("\nAnalyzing adoption x influence patterns...")
    topic_results = []
    first_touches = {}

    for topic in topics:
        ft = first_touch_per_agent(topic, posts, comments)
        if len(ft) < args.min_adopters:
            print(f"  skip '{topic}' ({len(ft)} adopters < {args.min_adopters})")
            continue

        first_touches[topic] = ft
        result = analyze_topic_adoption(topic, ft, cents_df)
        if result:
            topic_results.append(result)
            print(f"  '{topic}': {result['n_adopters']} adopters")
            print(f"    → early adopters hub score (PageRank): {result['early_pagerank_mean']:.4f} "
                  f"vs late: {result['late_pagerank_mean']:.4f}")
            print(f"    → early adopters broker score (betweenness): {result['early_betweenness_mean']:.4f} "
                  f"vs late: {result['late_betweenness_mean']:.4f}")

    if not topic_results:
        if cents_df.empty:
            print("\nNo centrality data available — topic adoption rankings require "
                  "agent_reply_edges.csv.\nTopic first-touch data written to "
                  f"{args.out / 'adopter_centralities.csv'} where available.")
        else:
            print("No topics passed the min-adopters threshold.")
        return

    # Write outputs
    print("\nWriting outputs...")
    results_df = pd.DataFrame(topic_results)
    results_path = args.out / "adoption_centrality_by_topic.csv"
    results_df.to_csv(results_path, index=False)
    print(f"  wrote {results_path}")

    # Plots
    plot_hub_vs_broker(topic_results, cents_df, first_touches,
                       args.out / "adoption_hub_vs_broker.png")
    plot_adoption_timing_vs_centrality(topic_results, cents_df, first_touches,
                                       args.out / "adoption_timing_vs_centrality.png")

    # Summary narrative
    print("\nGenerating summary...")
    summary_path = args.out / "adoption_influence_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("WHO DRIVES INFORMATION FLOW? ADOPTION INFLUENCE ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Analyzed {len(topic_results)} topics with ≥{args.min_adopters} adopters.\n\n")

        hubs_first = sum(1 for r in topic_results if r["early_adopters_are_hubs"])
        brokers_first = sum(1 for r in topic_results if r["early_adopters_are_brokers"])

        f.write(f"MAIN FINDINGS:\n")
        f.write(f"  Topics where HUBS (PageRank) adopt first: {hubs_first}/{len(topic_results)}\n")
        f.write(f"  Topics where BROKERS (betweenness) adopt first: {brokers_first}/{len(topic_results)}\n\n")

        f.write(f"PER-TOPIC DETAILS:\n")
        for r in sorted(topic_results, key=lambda x: x["n_adopters"], reverse=True):
            f.write(f"\n  {r['topic'].upper()} ({r['n_adopters']} adopters)\n")
            f.write(f"    Early adopters: PageRank={r['early_pagerank_mean']:.4f} ± {r['early_pagerank_std']:.4f}, "
                   f"Betweenness={r['early_betweenness_mean']:.4f} ± {r['early_betweenness_std']:.4f}\n")
            f.write(f"    Late adopters:  PageRank={r['late_pagerank_mean']:.4f} ± {r['late_pagerank_std']:.4f}, "
                   f"Betweenness={r['late_betweenness_mean']:.4f} ± {r['late_betweenness_std']:.4f}\n")
            f.write(f"    Adoption→PageRank correlation: r={r['pagerank_adoption_corr']:.3f} (p={r['pagerank_adoption_pval']:.3f})\n")
            f.write(f"    Adoption→Betweenness correlation: r={r['betweenness_adoption_corr']:.3f} (p={r['betweenness_adoption_pval']:.3f})\n")
            f.write(f"    Interpretation: Early adopters are {'HUBS' if r['early_adopters_are_hubs'] else 'PERIPHERAL'} "
                   f"and {'BROKERS' if r['early_adopters_are_brokers'] else 'NOT brokers'}\n")

    print(f"  wrote {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
