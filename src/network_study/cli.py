from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .adjacency import export_moltbook_snap_adjacency
from .analysis import write_analysis_report
from .config import StudyConfig, load_config
from .moltbook import MoltbookClient, MoltbookCrawler
from .reddit import (
    import_reddit_adjacency_snapshots,
    import_reddit_threads,
    import_temporal_reply_edges,
    resolve_dataset_root,
)
from .storage import Store
from .util import ensure_dir, write_json


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(Path(args.config) if getattr(args, "config", None) else None)
    if getattr(args, "db", None):
        config.database_path = Path(args.db)
    store = Store(config.database_path)
    try:
        if args.command == "init":
            store.init_schema()
            print(f"Initialized database: {config.database_path}")
            return 0
        store.init_schema()
        return dispatch(args, config, store)
    finally:
        store.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="network-study",
        description="Longitudinal Moltbook scraper and Reddit network comparison toolkit.",
    )
    parser.add_argument("--config", default="", help="Path to study config JSON.")
    parser.add_argument("--db", default="", help="Override SQLite database path.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Create or migrate the SQLite database.")

    probe = sub.add_parser("probe-moltbook", help="Probe Moltbook endpoint behavior and schemas.")
    probe.add_argument("--out", default="", help="Optional JSON report path.")

    parent_probe = sub.add_parser("probe-moltbook-parents", help="Probe whether Moltbook exposes reply parent data.")
    parent_probe.add_argument("--out", default="", help="Optional JSON report path.")
    parent_probe.add_argument("--post-limit", type=int, default=25)
    parent_probe.add_argument("--comment-limit", type=int, default=100)
    parent_probe.add_argument("--comment-sorts", nargs="*", default=None)

    crawl = sub.add_parser("crawl-moltbook", help="Run one Moltbook crawl pass.")
    crawl.add_argument("--sorts", nargs="*", default=None, help="Post sorts to crawl.")
    crawl.add_argument("--limit", type=int, default=0, help="Posts per page.")
    crawl.add_argument("--max-pages", type=int, default=0, help="Max post pages per sort/submolt.")
    crawl.add_argument("--no-comments", action="store_true", help="Skip comment hydration.")
    crawl.add_argument("--comment-sorts", nargs="*", default=None, help="Comment sorts to fetch.")
    crawl.add_argument("--comment-limit", type=int, default=0, help="Comments per page.")
    crawl.add_argument("--comment-max-pages", type=int, default=0, help="Max comment pages per sort.")
    crawl.add_argument("--submolts", action="store_true", help="Crawl known submolts separately.")
    crawl.add_argument("--submolt", default="", help="Crawl a single submolt.")
    crawl.add_argument("--resume", action="store_true", help="Skip submolt/sort units already completed.")

    hydrate = sub.add_parser("hydrate-recent", help="Rehydrate comments for recent Moltbook posts.")
    hydrate.add_argument("--limit", type=int, default=0, help="Recent posts to hydrate.")
    hydrate.add_argument("--comment-sorts", nargs="*", default=None)
    hydrate.add_argument("--comment-limit", type=int, default=0)
    hydrate.add_argument("--comment-max-pages", type=int, default=0)

    loop = sub.add_parser("run-study", help="Run repeated Moltbook crawl passes.")
    loop.add_argument("--iterations", type=int, default=48)
    loop.add_argument("--sleep-minutes", type=float, default=30)
    loop.add_argument("--crawl-submolts", action="store_true")
    loop.add_argument("--strict", action="store_true", help="Stop the study on the first failed iteration.")

    temporal = sub.add_parser("run-temporal-study", help="Run bounded new-post snapshots for longitudinal analysis.")
    temporal.add_argument("--iterations", type=int, default=48)
    temporal.add_argument("--interval-minutes", type=float, default=30)
    temporal.add_argument("--active-minutes", type=float, default=22)
    temporal.add_argument("--submolt-count", type=int, default=6)
    temporal.add_argument("--include-global", action="store_true")
    temporal.add_argument("--sorts", nargs="*", default=None, help="Post sorts; default is new only.")
    temporal.add_argument("--limit", type=int, default=50)
    temporal.add_argument("--max-pages", type=int, default=1)
    temporal.add_argument("--no-comments", action="store_true")
    temporal.add_argument("--comment-sorts", nargs="*", default=None)
    temporal.add_argument("--comment-limit", type=int, default=100)
    temporal.add_argument("--comment-max-pages", type=int, default=1)

    status = sub.add_parser("status", help="Show crawl health, database size, and current counts.")
    status.add_argument("--raw", action="store_true", help="Also count raw response files and bytes.")

    imp = sub.add_parser("import-reddit-threads", help="Import SNAP/TUDataset Reddit Threads.")
    imp.add_argument("--path", required=True, help="Folder or zip containing reddit_edges.json.")
    imp.add_argument("--dataset", default="reddit_threads")
    imp.add_argument("--limit", type=int, default=0)

    temp = sub.add_parser("import-reddit-temporal", help="Import generic Reddit user-reply CSV.")
    temp.add_argument("--path", required=True)
    temp.add_argument("--dataset", default="reddit_temporal_reply")
    temp.add_argument("--source-col", default="source")
    temp.add_argument("--target-col", default="target")
    temp.add_argument("--time-col", default="timestamp")
    temp.add_argument("--delimiter", default=",")
    temp.add_argument("--limit", type=int, default=0)

    adj = sub.add_parser("import-reddit-adjacency", help="Import SNAP reply_networks/chain_networks JSON files.")
    adj.add_argument("--reply-dir", default="")
    adj.add_argument("--chain-dir", default="")
    adj.add_argument("--dataset", default="snap_reddit")
    adj.add_argument("--limit-files", type=int, default=0)
    adj.add_argument("--summary-only", action="store_true", help="Store per-snapshot Reddit metrics without storing every edge row.")

    export_adj = sub.add_parser("export-moltbook-adjacency", help="Export Moltbook graphs in SNAP adjacency JSON shape.")
    export_adj.add_argument("--out-dir", default="")
    export_adj.add_argument("--network", choices=["all", "direct", "chain", "reply", "attention"], default="all")
    export_adj.add_argument("--bins", type=int, default=11)
    export_adj.add_argument("--start", default="")
    export_adj.add_argument("--end", default="")
    export_adj.add_argument("--node-label", choices=["key", "display"], default="key")
    export_adj.add_argument("--exclude-capped", action="store_true")
    export_adj.add_argument("--include-attention", action="store_true")

    analyze = sub.add_parser("analyze", help="Build comparable network reports.")
    analyze.add_argument("--reports-dir", default="")
    analyze.add_argument("--reddit-threads-dataset", default="reddit_threads")
    analyze.add_argument("--reddit-adjacency-dataset", default="snap_reddit")
    analyze.add_argument("--start", default="")
    analyze.add_argument("--end", default="")
    return parser


def dispatch(args, config: StudyConfig, store: Store) -> int:
    if args.command == "probe-moltbook":
        crawler = make_crawler(config, store)
        run_id = store.begin_run("moltbook", "probe", {})
        try:
            report = crawler.probe(run_id)
            out = Path(args.out) if args.out else config.reports_dir / "moltbook_endpoint_probe.json"
            write_json(out, report, pretty=True)
            store.end_run(run_id)
            print(f"Wrote probe report: {out}")
            return 0
        except Exception as exc:
            store.end_run(run_id, "failed", str(exc))
            raise

    if args.command == "probe-moltbook-parents":
        crawler = make_crawler(config, store)
        run_id = store.begin_run("moltbook", "parent-probe", vars(args))
        try:
            report = crawler.probe_parent_structure(
                run_id=run_id,
                post_limit=args.post_limit,
                comment_limit=args.comment_limit,
                sorts=args.comment_sorts or config.moltbook.comment_sorts,
            )
            out = Path(args.out) if args.out else config.reports_dir / "moltbook_parent_probe.json"
            write_json(out, report, pretty=True)
            store.end_run(run_id)
            print(f"Wrote parent probe report: {out}")
            return 0
        except Exception as exc:
            store.end_run(run_id, "failed", str(exc))
            raise

    if args.command == "crawl-moltbook":
        result = run_moltbook_crawl(args, config, store)
        print(result)
        return 0

    if args.command == "hydrate-recent":
        crawler = make_crawler(config, store)
        run_id = store.begin_run("moltbook", "hydrate-recent", vars(args))
        try:
            total = crawler.hydrate_recent_posts(
                run_id=run_id,
                limit=args.limit or config.moltbook.hydrate_recent_posts,
                sorts=args.comment_sorts or config.moltbook.comment_sorts,
                comment_limit=args.comment_limit or config.moltbook.comment_limit,
                comment_max_pages=args.comment_max_pages or config.moltbook.comment_max_pages,
            )
            store.end_run(run_id)
            print(f"Hydrated {total} comments across recent posts.")
            return 0
        except Exception as exc:
            store.end_run(run_id, "failed", str(exc))
            raise

    if args.command == "run-study":
        for idx in range(args.iterations):
            print(f"Study iteration {idx + 1}/{args.iterations}")
            try:
                run_moltbook_crawl(args, config, store, crawl_submolts=args.crawl_submolts)
            except Exception as exc:
                store.record_error(
                    run_id=None,
                    source="moltbook",
                    scope="study-iteration",
                    context={"iteration": idx + 1, "iterations": args.iterations},
                    error=str(exc),
                )
                if args.strict or not config.moltbook.safety.continue_on_error:
                    raise
                print(f"[moltbook] warning: iteration failed; continuing: {exc}", flush=True)
            if idx + 1 < args.iterations:
                time.sleep(max(0.0, args.sleep_minutes) * 60.0)
        return 0

    if args.command == "run-temporal-study":
        run_temporal_study(args, config, store)
        return 0

    if args.command == "status":
        print_status(config, store, include_raw=args.raw)
        return 0

    if args.command == "import-reddit-threads":
        work = ensure_dir(config.raw_dir / "reddit_imports")
        root = resolve_dataset_root(Path(args.path), work)
        run_id = store.begin_run("reddit", "import-reddit-threads", vars(args))
        try:
            result = import_reddit_threads(
                store,
                root,
                dataset_name=args.dataset,
                limit=args.limit or None,
            )
            store.end_run(run_id)
            print(result)
            return 0
        except Exception as exc:
            store.end_run(run_id, "failed", str(exc))
            raise

    if args.command == "import-reddit-temporal":
        run_id = store.begin_run("reddit", "import-reddit-temporal", vars(args))
        try:
            result = import_temporal_reply_edges(
                store,
                Path(args.path),
                dataset_name=args.dataset,
                source_col=args.source_col,
                target_col=args.target_col,
                time_col=args.time_col,
                delimiter=args.delimiter,
                limit=args.limit or None,
            )
            store.end_run(run_id)
            print(result)
            return 0
        except Exception as exc:
            store.end_run(run_id, "failed", str(exc))
            raise

    if args.command == "import-reddit-adjacency":
        run_id = store.begin_run("reddit", "import-reddit-adjacency", vars(args))
        try:
            result = import_reddit_adjacency_snapshots(
                store,
                reply_dir=Path(args.reply_dir) if args.reply_dir else None,
                chain_dir=Path(args.chain_dir) if args.chain_dir else None,
                dataset_name=args.dataset,
                limit_files=args.limit_files or None,
                store_edges=not args.summary_only,
            )
            store.end_run(run_id)
            print(result)
            return 0
        except Exception as exc:
            store.end_run(run_id, "failed", str(exc))
            raise

    if args.command == "export-moltbook-adjacency":
        out_dir = Path(args.out_dir) if args.out_dir else config.reports_dir / "moltbook_adjacency"
        result = export_moltbook_snap_adjacency(
            store,
            out_dir,
            network=args.network,
            bins=args.bins,
            start=args.start,
            end=args.end,
            node_label=args.node_label,
            exclude_capped=args.exclude_capped,
            include_attention=args.include_attention,
        )
        print(result)
        return 0

    if args.command == "analyze":
        out_dir = Path(args.reports_dir) if args.reports_dir else config.reports_dir
        report = write_analysis_report(
            store,
            out_dir,
            reddit_threads_dataset=args.reddit_threads_dataset,
            reddit_adjacency_dataset=args.reddit_adjacency_dataset,
            start=args.start,
            end=args.end,
        )
        print(f"Wrote analysis report to {out_dir}")
        print(f"Moltbook nodes: {report['moltbook_actor_graph']['nodes']}")
        print(f"Reddit threads: {report['reddit_thread_graphs']['thread_count']}")
        return 0

    raise ValueError(f"Unknown command: {args.command}")


def make_crawler(config: StudyConfig, store: Store) -> MoltbookCrawler:
    ensure_dir(config.raw_dir)
    return MoltbookCrawler(MoltbookClient(config.moltbook), store, config.raw_dir)


def run_moltbook_crawl(args, config: StudyConfig, store: Store, crawl_submolts: bool | None = None) -> dict[str, int]:
    crawler = make_crawler(config, store)
    run_id = store.begin_run("moltbook", "crawl", vars(args))
    total = {"submolts": 0, "posts": 0, "comments": 0}
    try:
        total["submolts"] = crawler.crawl_submolts(run_id, limit=250)
        sorts = getattr(args, "sorts", None) or config.moltbook.post_sorts
        limit = getattr(args, "limit", 0) or config.moltbook.default_limit
        max_pages = getattr(args, "max_pages", 0) or config.moltbook.max_pages
        comment_sorts = getattr(args, "comment_sorts", None) or config.moltbook.comment_sorts
        comment_limit = getattr(args, "comment_limit", 0) or config.moltbook.comment_limit
        comment_max_pages = getattr(args, "comment_max_pages", 0) or config.moltbook.comment_max_pages
        hydrate = not getattr(args, "no_comments", False) and config.moltbook.hydrate_comments
        selected_submolt = getattr(args, "submolt", "") or ""
        resume = bool(getattr(args, "resume", False))
        should_crawl_submolts = bool(crawl_submolts if crawl_submolts is not None else getattr(args, "submolts", False))
        submolts = [selected_submolt] if selected_submolt else [""]
        if should_crawl_submolts and not selected_submolt:
            rows = store.query("SELECT name FROM submolts ORDER BY name")
            submolts = [str(row["name"]) for row in rows if row["name"]]
        for submolt in submolts:
            for sort in sorts:
                progress_key = f"posts:{submolt or 'global'}:{sort}"
                if resume:
                    prior = store.query("SELECT status FROM crawl_progress WHERE progress_key = ?", (progress_key,))
                    if prior and prior[0]["status"] == "completed":
                        print(f"[moltbook] resume: skipping completed {progress_key}", flush=True)
                        continue
                result = crawler.crawl_posts(
                    run_id=run_id,
                    sort=sort,
                    limit=limit,
                    max_pages=max_pages,
                    submolt=submolt,
                    submolt_query_param=config.moltbook.submolt_query_param,
                    hydrate_comments=hydrate,
                    comment_sorts=comment_sorts,
                    comment_limit=comment_limit,
                    comment_max_pages=comment_max_pages,
                )
                total["posts"] += result["posts"]
                total["comments"] += result["comments"]
        store.end_run(run_id)
        return total
    except Exception as exc:
        store.end_run(run_id, "failed", str(exc))
        raise


def run_temporal_study(args, config: StudyConfig, store: Store) -> None:
    crawler = make_crawler(config, store)
    sorts = getattr(args, "sorts", None) or ["new"]
    comment_sorts = getattr(args, "comment_sorts", None) or ["new"]
    hydrate = not getattr(args, "no_comments", False) and config.moltbook.hydrate_comments

    for idx in range(args.iterations):
        iteration_started = time.monotonic()
        active_deadline = iteration_started + max(0.0, args.active_minutes) * 60.0
        run_id = store.begin_run(
            "moltbook",
            "temporal-study",
            {**vars(args), "iteration": idx + 1},
        )
        total = {"submolts": 0, "posts": 0, "comments": 0, "skipped_for_budget": 0}
        try:
            total["submolts"] = crawler.crawl_submolts(run_id, limit=250)
            submolts = select_temporal_submolts(
                store,
                iteration=idx,
                count=args.submolt_count,
                include_global=args.include_global,
            )
            if not submolts:
                submolts = [""]
            for submolt in submolts:
                if time.monotonic() >= active_deadline:
                    total["skipped_for_budget"] += 1
                    continue
                for sort in sorts:
                    if time.monotonic() >= active_deadline:
                        total["skipped_for_budget"] += 1
                        break
                    result = crawler.crawl_posts(
                        run_id=run_id,
                        sort=sort,
                        limit=args.limit,
                        max_pages=args.max_pages,
                        submolt=submolt,
                        submolt_query_param=config.moltbook.submolt_query_param,
                        hydrate_comments=hydrate,
                        comment_sorts=comment_sorts,
                        comment_limit=args.comment_limit,
                        comment_max_pages=args.comment_max_pages,
                        deadline_monotonic=active_deadline,
                    )
                    total["posts"] += result["posts"]
                    total["comments"] += result["comments"]
            status = "completed" if total["skipped_for_budget"] == 0 else "partial"
            store.end_run(run_id, status, json_like(total))
            print(f"Temporal iteration {idx + 1}/{args.iterations}: {total}", flush=True)
        except Exception as exc:
            store.record_error(
                run_id=run_id,
                source="moltbook",
                scope="temporal-study",
                context={"iteration": idx + 1, "totals": total},
                error=str(exc),
            )
            store.end_run(run_id, "failed", str(exc))
            if not config.moltbook.safety.continue_on_error:
                raise
            print(f"[moltbook] warning: temporal iteration failed; continuing: {exc}", flush=True)

        if idx + 1 < args.iterations:
            elapsed = time.monotonic() - iteration_started
            sleep_for = max(0.0, args.interval_minutes * 60.0 - elapsed)
            if sleep_for:
                time.sleep(sleep_for)


def select_temporal_submolts(store: Store, *, iteration: int, count: int, include_global: bool) -> list[str]:
    rows = store.query(
        """
        SELECT name, COALESCE(platform_post_count, 0) AS platform_post_count
        FROM submolts
        WHERE name <> ''
        ORDER BY platform_post_count DESC, name
        """
    )
    names = [str(row["name"]) for row in rows if row["name"]]
    if count <= 0 or not names:
        selected: list[str] = []
    else:
        count = min(count, len(names))
        offset = (iteration * count) % len(names)
        selected = [names[(offset + idx) % len(names)] for idx in range(count)]
    if include_global:
        selected.insert(0, "")
    return selected


def print_status(config: StudyConfig, store: Store, *, include_raw: bool = False) -> None:
    db_path = config.database_path
    print(f"Database: {db_path}")
    if db_path.exists():
        stat = db_path.stat()
        print(f"Database bytes: {stat.st_size}")
        print(f"Database updated: {stat.st_mtime}")
    for table in [
        "crawl_runs",
        "crawl_errors",
        "crawl_progress",
        "agents",
        "submolts",
        "posts",
        "post_snapshots",
        "comments",
        "comment_snapshots",
        "edges",
        "coverage",
        "raw_responses",
        "reddit_adjacency_snapshots",
    ]:
        try:
            row = store.query(f"SELECT COUNT(*) AS n FROM {table}")[0]
            print(f"{table}: {row['n']}")
        except Exception:
            pass
    print("\nRecent runs:")
    for row in store.query("SELECT * FROM crawl_runs ORDER BY run_id DESC LIMIT 8"):
        print(dict(row))
    print("\nRecent errors:")
    for row in store.query(
        """
        SELECT error_id, run_id, scope, endpoint, occurred_at, substr(error, 1, 240) AS error
        FROM crawl_errors
        ORDER BY error_id DESC
        LIMIT 8
        """
    ):
        print(dict(row))
    print("\nRecent progress:")
    for row in store.query(
        """
        SELECT progress_key, status, updated_at, substr(state_json, 1, 220) AS state
        FROM crawl_progress
        ORDER BY updated_at DESC
        LIMIT 8
        """
    ):
        print(dict(row))
    if include_raw:
        total_files = 0
        total_bytes = 0
        raw_root = config.raw_dir / "moltbook"
        if raw_root.exists():
            for path in raw_root.rglob("*"):
                if path.is_file():
                    total_files += 1
                    total_bytes += path.stat().st_size
        print(f"\nRaw Moltbook files: {total_files}")
        print(f"Raw Moltbook bytes: {total_bytes}")


def json_like(value: dict[str, int]) -> str:
    return ", ".join(f"{key}={val}" for key, val in sorted(value.items()))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
