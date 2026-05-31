from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .graphs import (
    load_moltbook_graph,
    load_reddit_temporal_graph,
    summarize_graph,
    summarize_reddit_adjacency_graph_if_small,
    summarize_reddit_adjacency_snapshots,
    summarize_reddit_threads,
)
from .storage import Store
from .util import ensure_dir, json_dumps, utc_now, write_json


def write_analysis_report(
    store: Store,
    reports_dir: Path,
    *,
    reddit_threads_dataset: str = "reddit_threads",
    reddit_adjacency_dataset: str = "snap_reddit",
    start: str = "",
    end: str = "",
) -> dict[str, Any]:
    ensure_dir(reports_dir)
    report = {
        "created_at": utc_now(),
        "window": {"start": start, "end": end},
        "moltbook_actor_graph": summarize_graph(load_moltbook_graph(store, start=start, end=end)),
        "moltbook_comment_on_post_graph": summarize_graph(
            load_moltbook_graph(store, start=start, end=end, edge_type="comment_on_post")
        ),
        "moltbook_reply_graph": summarize_graph(
            load_moltbook_graph(store, start=start, end=end, edge_type="reply_to_comment")
        ),
        "moltbook_uncapped_actor_graph": summarize_graph(
            load_moltbook_graph(store, start=start, end=end, exclude_capped=True)
        ),
        "reddit_temporal_actor_graph": summarize_graph(load_reddit_temporal_graph(store, start=start, end=end)),
        "reddit_reply_adjacency_graph": summarize_reddit_adjacency_graph_if_small(
            store, dataset=reddit_adjacency_dataset, network_type="reply"
        ),
        "reddit_chain_adjacency_graph": summarize_reddit_adjacency_graph_if_small(
            store, dataset=reddit_adjacency_dataset, network_type="chain"
        ),
        "reddit_adjacency_snapshots": summarize_reddit_adjacency_snapshots(store, reddit_adjacency_dataset),
        "reddit_thread_graphs": summarize_reddit_threads(store, reddit_threads_dataset),
        "moltbook_coverage": coverage_summary(store),
        "moltbook_submolts": submolt_summary(store),
        "moltbook_runs": run_summary(store),
    }
    write_json(reports_dir / "network_analysis_summary.json", report, pretty=True)
    (reports_dir / "network_analysis_report.md").write_text(
        render_markdown_report(report),
        encoding="utf-8",
    )
    write_csv_table(
        reports_dir / "moltbook_top_agents.csv",
        report["moltbook_actor_graph"].get("top_pagerank", []),
        ["node", "value"],
    )
    return report


def coverage_summary(store: Store) -> dict[str, Any]:
    rows = store.query("SELECT * FROM coverage")
    if not rows:
        return {
            "posts_with_coverage_rows": 0,
            "avg_coverage_ratio": None,
            "capped_posts": 0,
            "unknown_expected_posts": 0,
            "observed_comments_sum": 0,
            "expected_comments_sum": 0,
        }
    ratios = [
        float(row["coverage_ratio"])
        for row in rows
        if row["coverage_ratio"] is not None
    ]
    observed = sum(int(row["observed_comment_count"] or 0) for row in rows)
    expected = sum(int(row["expected_comment_count"] or 0) for row in rows if row["expected_comment_count"] is not None)
    return {
        "posts_with_coverage_rows": len(rows),
        "avg_coverage_ratio": sum(ratios) / len(ratios) if ratios else None,
        "capped_posts": sum(1 for row in rows if int(row["capped_flag"] or 0) == 1),
        "fully_observed_posts": sum(1 for row in rows if int(row["capped_flag"] or 0) == 0),
        "unknown_expected_posts": sum(1 for row in rows if row["expected_comment_count"] is None),
        "observed_comments_sum": observed,
        "expected_comments_sum": expected,
    }


def submolt_summary(store: Store) -> list[dict[str, Any]]:
    rows = store.query(
        """
        SELECT
            submolt,
            COUNT(DISTINCT post_id) AS posts,
            COUNT(DISTINCT author_key) AS post_authors
        FROM posts
        GROUP BY submolt
        ORDER BY posts DESC
        LIMIT 50
        """
    )
    return [dict(row) for row in rows]


def run_summary(store: Store) -> dict[str, Any]:
    rows = store.query(
        """
        SELECT status, COUNT(*) AS runs
        FROM crawl_runs
        GROUP BY status
        ORDER BY runs DESC
        """
    )
    latest = store.query(
        """
        SELECT run_id, source, target, started_at, ended_at, status, notes
        FROM crawl_runs
        ORDER BY run_id DESC
        LIMIT 8
        """
    )
    errors = store.query(
        """
        SELECT scope, COUNT(*) AS errors
        FROM crawl_errors
        GROUP BY scope
        ORDER BY errors DESC
        """
    )
    return {
        "runs_by_status": [dict(row) for row in rows],
        "latest_runs": [dict(row) for row in latest],
        "errors_by_scope": [dict(row) for row in errors],
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    mb = report["moltbook_actor_graph"]
    attention = report["moltbook_comment_on_post_graph"]
    replies = report["moltbook_reply_graph"]
    uncapped = report["moltbook_uncapped_actor_graph"]
    rt = report["reddit_thread_graphs"]
    red_temporal = report["reddit_temporal_actor_graph"]
    red_reply = report["reddit_reply_adjacency_graph"]
    red_chain = report["reddit_chain_adjacency_graph"]
    red_adj = report["reddit_adjacency_snapshots"]
    cov = report["moltbook_coverage"]
    lines = [
        "# Moltbook and Reddit Network Analysis",
        "",
        f"Generated: `{report['created_at']}`",
        "",
        "## Moltbook Actor Graph",
        "",
        _kv("Nodes", mb.get("nodes")),
        _kv("Directed edges", mb.get("directed_edges")),
        _kv("Directed density", mb.get("directed_density")),
        _kv("Largest weak component share", mb.get("largest_weak_component_share")),
        _kv("Largest strong component share", mb.get("largest_strong_component_share")),
        _kv("Reciprocity", mb.get("reciprocity")),
        _kv("Weighted reciprocity", mb.get("weighted_reciprocity")),
        _kv("Average local clustering", mb.get("avg_local_clustering_lcc")),
        _kv("Weighted in-degree Gini", mb.get("weighted_in_gini")),
        _kv("Top 1% inbound share", mb.get("top_1pct_weighted_in_share")),
        _kv("Top 5% inbound share", mb.get("top_5pct_weighted_in_share")),
        "",
        "Graph variants:",
        "",
        _kv("Comment-on-post nodes", attention.get("nodes")),
        _kv("Comment-on-post edges", attention.get("directed_edges")),
        _kv("Reply-to-comment nodes", replies.get("nodes")),
        _kv("Reply-to-comment edges", replies.get("directed_edges")),
        _kv("Uncapped/fully observed nodes", uncapped.get("nodes")),
        _kv("Uncapped/fully observed edges", uncapped.get("directed_edges")),
        "",
        "Top PageRank agents:",
        "",
    ]
    for row in mb.get("top_pagerank", [])[:10]:
        lines.append(f"- `{row['node']}`: {row['value']:.6g}")
    lines.extend(
        [
            "",
            "## Moltbook Coverage",
            "",
            _kv("Posts with coverage rows", cov.get("posts_with_coverage_rows")),
            _kv("Average coverage ratio", cov.get("avg_coverage_ratio")),
            _kv("Capped or under-observed posts", cov.get("capped_posts")),
            _kv("Unknown expected-comment posts", cov.get("unknown_expected_posts")),
            _kv("Observed comments across coverage rows", cov.get("observed_comments_sum")),
            _kv("Expected comments across coverage rows", cov.get("expected_comments_sum")),
            "",
            "## Reddit Thread Graphs",
            "",
            _kv("Dataset", rt.get("dataset")),
            _kv("Thread count", rt.get("thread_count")),
            _kv("Average nodes/thread", rt.get("avg_nodes_per_thread")),
            _kv("Average edges/thread", rt.get("avg_edges_per_thread")),
            _kv("Average density", rt.get("avg_density")),
            "",
            "## Reddit Temporal Actor Graph",
            "",
            _kv("Nodes", red_temporal.get("nodes")),
            _kv("Directed edges", red_temporal.get("directed_edges")),
            _kv("Reciprocity", red_temporal.get("reciprocity")),
            _kv("Weighted in-degree Gini", red_temporal.get("weighted_in_gini")),
            "",
            "## Reddit SNAP Reply/Chain Graphs",
            "",
            _kv("Dataset", red_adj.get("dataset")),
            _kv("Reply graph skipped", red_reply.get("skipped")),
            _kv("Reply graph nodes", red_reply.get("nodes")),
            _kv("Reply graph directed edges", red_reply.get("directed_edges")),
            _kv("Reply graph self-loop edges", red_reply.get("self_loop_edges")),
            _kv("Chain graph skipped", red_chain.get("skipped")),
            _kv("Chain graph nodes", red_chain.get("nodes")),
            _kv("Chain graph directed edges", red_chain.get("directed_edges")),
            _kv("Chain graph self-loop edges", red_chain.get("self_loop_edges")),
            _kv("Chain weighted in-degree Gini", red_chain.get("weighted_in_gini")),
            "",
            "## Interpretation Notes",
            "",
            "- Compare Moltbook actor metrics to Reddit temporal actor metrics when using a user-reply edge stream.",
            "- Compare Moltbook post/thread shapes to Reddit thread graphs when using Reddit Threads.",
            "- Treat coverage metrics as a warning light: low coverage means comment caps or endpoint limits still need deeper hydration.",
            "- For a fair final paper, compare matched time windows and matched sample sizes, then bootstrap results.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv_table(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def _kv(label: str, value: Any) -> str:
    if isinstance(value, float):
        return f"- **{label}:** {value:.6g}"
    return f"- **{label}:** {value}"
