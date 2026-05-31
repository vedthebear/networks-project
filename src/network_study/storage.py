from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .util import ensure_dir, json_dumps, normalize_timestamp, stable_hash, utc_now


SCHEMA_VERSION = 3


class Store:
    def __init__(self, path: Path):
        self.path = path
        ensure_dir(path.parent)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS crawl_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                status TEXT NOT NULL,
                config_json TEXT NOT NULL DEFAULT '{}',
                notes TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS raw_responses (
                response_hash TEXT PRIMARY KEY,
                run_id INTEGER,
                source TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                query_json TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                headers_json TEXT NOT NULL,
                body_path TEXT NOT NULL,
                body_bytes INTEGER NOT NULL,
                FOREIGN KEY(run_id) REFERENCES crawl_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS endpoint_probes (
                probe_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                endpoint TEXT NOT NULL,
                query_json TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                ok INTEGER NOT NULL,
                item_count INTEGER NOT NULL DEFAULT 0,
                keys_json TEXT NOT NULL DEFAULT '[]',
                sample_json TEXT NOT NULL DEFAULT '{}',
                error TEXT NOT NULL DEFAULT '',
                fetched_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES crawl_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS crawl_errors (
                error_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                source TEXT NOT NULL,
                scope TEXT NOT NULL,
                endpoint TEXT NOT NULL DEFAULT '',
                query_json TEXT NOT NULL DEFAULT '{}',
                context_json TEXT NOT NULL DEFAULT '{}',
                error TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES crawl_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS crawl_progress (
                progress_key TEXT PRIMARY KEY,
                run_id INTEGER,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                scope TEXT NOT NULL,
                subject_key TEXT NOT NULL DEFAULT '',
                state_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'running',
                updated_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES crawl_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS agents (
                agent_key TEXT PRIMARY KEY,
                platform TEXT NOT NULL,
                display_name TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                raw_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS submolts (
                submolt_key TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                display_name TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                subscribers INTEGER,
                platform_post_count INTEGER,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                raw_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS posts (
                post_id TEXT PRIMARY KEY,
                platform TEXT NOT NULL,
                author_key TEXT NOT NULL,
                author_name TEXT NOT NULL,
                submolt TEXT NOT NULL DEFAULT '',
                created_at TEXT,
                title TEXT NOT NULL DEFAULT '',
                body TEXT NOT NULL DEFAULT '',
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                raw_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS post_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id TEXT NOT NULL,
                run_id INTEGER,
                fetched_at TEXT NOT NULL,
                score REAL,
                comment_count INTEGER,
                source_endpoint TEXT NOT NULL DEFAULT '',
                source_sort TEXT NOT NULL DEFAULT '',
                source_submolt TEXT NOT NULL DEFAULT '',
                raw_hash TEXT NOT NULL DEFAULT '',
                UNIQUE(post_id, fetched_at, source_endpoint, source_sort, source_submolt),
                FOREIGN KEY(run_id) REFERENCES crawl_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS comments (
                comment_id TEXT PRIMARY KEY,
                platform TEXT NOT NULL,
                post_id TEXT NOT NULL,
                author_key TEXT NOT NULL,
                author_name TEXT NOT NULL,
                parent_comment_id TEXT,
                parent_author_key TEXT,
                created_at TEXT,
                body TEXT NOT NULL DEFAULT '',
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                raw_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS comment_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                comment_id TEXT NOT NULL,
                run_id INTEGER,
                fetched_at TEXT NOT NULL,
                score REAL,
                source_sort TEXT NOT NULL DEFAULT '',
                raw_hash TEXT NOT NULL DEFAULT '',
                UNIQUE(comment_id, fetched_at, source_sort),
                FOREIGN KEY(run_id) REFERENCES crawl_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS edges (
                edge_id TEXT PRIMARY KEY,
                platform TEXT NOT NULL,
                source_key TEXT NOT NULL,
                target_key TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                context_id TEXT NOT NULL DEFAULT '',
                comment_id TEXT NOT NULL DEFAULT '',
                submolt TEXT NOT NULL DEFAULT '',
                occurred_at TEXT,
                observed_at TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                raw_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS coverage (
                post_id TEXT PRIMARY KEY,
                expected_comment_count INTEGER,
                observed_comment_count INTEGER NOT NULL DEFAULT 0,
                coverage_ratio REAL,
                capped_flag INTEGER NOT NULL DEFAULT 0,
                last_checked_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reddit_threads (
                dataset TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                label TEXT NOT NULL DEFAULT '',
                node_count INTEGER NOT NULL,
                edge_count INTEGER NOT NULL,
                density REAL,
                imported_at TEXT NOT NULL,
                PRIMARY KEY(dataset, thread_id)
            );

            CREATE TABLE IF NOT EXISTS reddit_thread_edges (
                dataset TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                source_key TEXT NOT NULL,
                target_key TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                PRIMARY KEY(dataset, thread_id, source_key, target_key)
            );

            CREATE TABLE IF NOT EXISTS reddit_adjacency_snapshots (
                dataset TEXT NOT NULL,
                network_type TEXT NOT NULL,
                subreddit TEXT NOT NULL,
                snapshot_index INTEGER NOT NULL,
                node_count INTEGER NOT NULL,
                edge_entry_count INTEGER NOT NULL,
                unique_edge_count INTEGER NOT NULL,
                self_loop_count INTEGER NOT NULL,
                density REAL,
                imported_at TEXT NOT NULL,
                PRIMARY KEY(dataset, network_type, subreddit, snapshot_index)
            );

            CREATE INDEX IF NOT EXISTS idx_posts_author ON posts(author_key);
            CREATE INDEX IF NOT EXISTS idx_posts_submolt ON posts(submolt);
            CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(post_id);
            CREATE INDEX IF NOT EXISTS idx_edges_platform_time ON edges(platform, occurred_at);
            CREATE INDEX IF NOT EXISTS idx_edges_platform_type ON edges(platform, edge_type);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_key);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_key);
            CREATE INDEX IF NOT EXISTS idx_edges_comment ON edges(comment_id);
            CREATE INDEX IF NOT EXISTS idx_crawl_errors_run ON crawl_errors(run_id, occurred_at);
            CREATE INDEX IF NOT EXISTS idx_crawl_progress_status ON crawl_progress(source, target, status);
            CREATE INDEX IF NOT EXISTS idx_reddit_edges_dataset ON reddit_thread_edges(dataset);
            CREATE INDEX IF NOT EXISTS idx_reddit_adjacency_dataset
                ON reddit_adjacency_snapshots(dataset, network_type, subreddit);
            """
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

    def begin_run(self, source: str, target: str, config: dict[str, Any] | None = None) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO crawl_runs(source, target, started_at, status, config_json)
            VALUES (?, ?, ?, 'running', ?)
            """,
            (source, target, utc_now(), json_dumps(config or {})),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def end_run(self, run_id: int, status: str = "completed", notes: str = "") -> None:
        self.conn.execute(
            "UPDATE crawl_runs SET ended_at=?, status=?, notes=? WHERE run_id=?",
            (utc_now(), status, notes, run_id),
        )
        self.conn.commit()

    def save_raw_response(
        self,
        *,
        run_id: int | None,
        source: str,
        endpoint: str,
        query: dict[str, Any],
        status_code: int,
        headers: dict[str, Any],
        body_text: str,
        raw_dir: Path,
    ) -> str:
        response_hash = stable_hash(
            {
                "source": source,
                "endpoint": endpoint,
                "query": query,
                "status_code": status_code,
                "body": body_text,
            }
        )
        shard = response_hash[:2]
        body_path = raw_dir / source / shard / f"{response_hash}.json"
        ensure_dir(body_path.parent)
        if not body_path.exists():
            body_path.write_text(body_text, encoding="utf-8")
        self.conn.execute(
            """
            INSERT OR IGNORE INTO raw_responses(
                response_hash, run_id, source, endpoint, query_json, fetched_at,
                status_code, headers_json, body_path, body_bytes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                response_hash,
                run_id,
                source,
                endpoint,
                json_dumps(query),
                utc_now(),
                status_code,
                json_dumps(headers),
                str(body_path),
                len(body_text.encode("utf-8")),
            ),
        )
        self.conn.commit()
        return response_hash

    def save_probe(
        self,
        *,
        run_id: int | None,
        endpoint: str,
        query: dict[str, Any],
        status_code: int,
        ok: bool,
        item_count: int,
        keys: list[str],
        sample: Any,
        error: str = "",
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO endpoint_probes(
                run_id, endpoint, query_json, status_code, ok, item_count,
                keys_json, sample_json, error, fetched_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                endpoint,
                json_dumps(query),
                status_code,
                1 if ok else 0,
                item_count,
                json_dumps(keys),
                json_dumps(sample),
                error,
                utc_now(),
            ),
        )
        self.conn.commit()

    def record_error(
        self,
        *,
        run_id: int | None,
        source: str,
        scope: str,
        error: str,
        endpoint: str = "",
        query: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO crawl_errors(
                run_id, source, scope, endpoint, query_json, context_json, error, occurred_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                source,
                scope,
                endpoint,
                json_dumps(query or {}),
                json_dumps(context or {}),
                error,
                utc_now(),
            ),
        )
        self.conn.commit()

    def upsert_progress(
        self,
        *,
        progress_key: str,
        run_id: int | None,
        source: str,
        target: str,
        scope: str,
        subject_key: str = "",
        state: dict[str, Any] | None = None,
        status: str = "running",
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO crawl_progress(
                progress_key, run_id, source, target, scope, subject_key,
                state_json, status, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(progress_key) DO UPDATE SET
                run_id=excluded.run_id,
                source=excluded.source,
                target=excluded.target,
                scope=excluded.scope,
                subject_key=excluded.subject_key,
                state_json=excluded.state_json,
                status=excluded.status,
                updated_at=excluded.updated_at
            """,
            (
                progress_key,
                run_id,
                source,
                target,
                scope,
                subject_key,
                json_dumps(state or {}),
                status,
                utc_now(),
            ),
        )
        self.conn.commit()

    def upsert_agent(self, agent_key: str, platform: str, display_name: str, raw: Any | None = None) -> None:
        if not agent_key:
            return
        now = utc_now()
        self.conn.execute(
            """
            INSERT INTO agents(agent_key, platform, display_name, first_seen_at, last_seen_at, raw_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_key) DO UPDATE SET
                display_name=excluded.display_name,
                last_seen_at=excluded.last_seen_at,
                raw_json=excluded.raw_json
            """,
            (agent_key, platform, display_name or agent_key, now, now, json_dumps(raw or {})),
        )

    def upsert_submolt(self, row: dict[str, Any]) -> None:
        now = utc_now()
        key = str(row.get("id") or row.get("name") or row.get("display_name") or "")
        if not key:
            return
        self.conn.execute(
            """
            INSERT INTO submolts(
                submolt_key, name, display_name, description, subscribers,
                platform_post_count, first_seen_at, last_seen_at, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(submolt_key) DO UPDATE SET
                name=excluded.name,
                display_name=excluded.display_name,
                description=excluded.description,
                subscribers=excluded.subscribers,
                platform_post_count=excluded.platform_post_count,
                last_seen_at=excluded.last_seen_at,
                raw_json=excluded.raw_json
            """,
            (
                key,
                str(row.get("name") or key),
                str(row.get("display_name") or row.get("displayName") or row.get("name") or ""),
                str(row.get("description") or ""),
                _nullable_int(row.get("subscribers") or row.get("subscriber_count") or row.get("subscriberCount")),
                _nullable_int(row.get("post_count") or row.get("postCount")),
                now,
                now,
                json_dumps(row),
            ),
        )

    def upsert_post(
        self,
        post: dict[str, Any],
        *,
        run_id: int | None,
        fetched_at: str,
        source_endpoint: str,
        source_sort: str = "",
        source_submolt: str = "",
        raw_hash: str = "",
    ) -> str | None:
        post_id = str(
            post.get("id")
            or post.get("post_id")
            or post.get("postId")
            or post.get("uuid")
            or ""
        )
        if not post_id:
            return None
        author_key, author_name, author_raw = extract_author(post)
        submolt = normalize_submolt(
            post.get("submolt")
            or post.get("submolt_name")
            or post.get("submoltName")
            or source_submolt
            or ""
        )
        created_at = normalize_timestamp(
            post.get("created_at")
            or post.get("createdAt")
            or post.get("created_at_ts")
            or post.get("createdAtTs")
        )
        title = str(post.get("title") or "")
        body = str(post.get("body") or post.get("content") or post.get("selftext") or "")
        now = utc_now()
        self.upsert_agent(author_key, "moltbook", author_name, author_raw)
        self.conn.execute(
            """
            INSERT INTO posts(
                post_id, platform, author_key, author_name, submolt, created_at,
                title, body, first_seen_at, last_seen_at, raw_json
            )
            VALUES (?, 'moltbook', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(post_id) DO UPDATE SET
                author_key=excluded.author_key,
                author_name=excluded.author_name,
                submolt=excluded.submolt,
                created_at=excluded.created_at,
                title=excluded.title,
                body=excluded.body,
                last_seen_at=excluded.last_seen_at,
                raw_json=excluded.raw_json
            """,
            (
                post_id,
                author_key,
                author_name,
                submolt,
                created_at,
                title,
                body,
                now,
                now,
                json_dumps(post),
            ),
        )
        self.conn.execute(
            """
            INSERT OR IGNORE INTO post_snapshots(
                post_id, run_id, fetched_at, score, comment_count,
                source_endpoint, source_sort, source_submolt, raw_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                post_id,
                run_id,
                fetched_at,
                _nullable_float(post.get("score") or post.get("karma") or post.get("upvotes")),
                _nullable_int(post.get("comment_count") or post.get("commentCount") or post.get("commentsCount")),
                source_endpoint,
                source_sort,
                source_submolt,
                raw_hash,
            ),
        )
        return post_id

    def upsert_comment(
        self,
        comment: dict[str, Any],
        *,
        post_id: str,
        post_author_key: str,
        post_submolt: str,
        run_id: int | None,
        fetched_at: str,
        source_sort: str,
        parent_author_by_comment_id: dict[str, str] | None = None,
        raw_hash: str = "",
    ) -> str | None:
        comment_id = str(
            comment.get("id")
            or comment.get("comment_id")
            or comment.get("commentId")
            or comment.get("uuid")
            or ""
        )
        if not comment_id:
            return None
        author_key, author_name, author_raw = extract_author(comment)
        parent_comment_id = _normalize_parent_comment_id(
            _first_nonblank(
                comment.get("parent_comment_id"),
                comment.get("parentCommentId"),
                comment.get("parent_id"),
                comment.get("parentId"),
                comment.get("replyToCommentId"),
                comment.get("reply_to_comment_id"),
                comment.get("replyToId"),
                comment.get("reply_to_id"),
                comment.get("in_reply_to_id"),
                comment.get("inReplyToId"),
                comment.get("parent"),
                comment.get("reply_to"),
            )
        )
        if parent_comment_id == post_id:
            parent_comment_id = None
        parent_author_key = _extract_parent_author_key(comment)
        if parent_comment_id and parent_author_by_comment_id:
            parent_author_key = parent_author_key or parent_author_by_comment_id.get(parent_comment_id)
        if not parent_author_key:
            parent_author_key = post_author_key
        created_at = normalize_timestamp(
            comment.get("created_at")
            or comment.get("createdAt")
            or comment.get("created_at_ts")
            or comment.get("createdAtTs")
        )
        body = str(comment.get("body") or comment.get("content") or comment.get("text") or "")
        now = utc_now()
        self.upsert_agent(author_key, "moltbook", author_name, author_raw)
        self.conn.execute(
            """
            INSERT INTO comments(
                comment_id, platform, post_id, author_key, author_name,
                parent_comment_id, parent_author_key, created_at, body,
                first_seen_at, last_seen_at, raw_json
            )
            VALUES (?, 'moltbook', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(comment_id) DO UPDATE SET
                post_id=excluded.post_id,
                author_key=excluded.author_key,
                author_name=excluded.author_name,
                parent_comment_id=excluded.parent_comment_id,
                parent_author_key=excluded.parent_author_key,
                created_at=excluded.created_at,
                body=excluded.body,
                last_seen_at=excluded.last_seen_at,
                raw_json=excluded.raw_json
            """,
            (
                comment_id,
                post_id,
                author_key,
                author_name,
                parent_comment_id,
                parent_author_key,
                created_at,
                body,
                now,
                now,
                json_dumps(comment),
            ),
        )
        self.conn.execute(
            """
            INSERT OR IGNORE INTO comment_snapshots(
                comment_id, run_id, fetched_at, score, source_sort, raw_hash
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                comment_id,
                run_id,
                fetched_at,
                _nullable_float(comment.get("score") or comment.get("karma") or comment.get("upvotes")),
                source_sort,
                raw_hash,
            ),
        )
        edge_type = "reply_to_comment" if parent_comment_id else "comment_on_post"
        edge_id = stable_hash(
            {
                "platform": "moltbook",
                "comment_id": comment_id,
                "source": author_key,
                "target": parent_author_key,
                "type": edge_type,
            }
        )
        self.conn.execute(
            """
            INSERT OR REPLACE INTO edges(
                edge_id, platform, source_key, target_key, edge_type, context_id,
                comment_id, submolt, occurred_at, observed_at, weight, raw_json
            )
            VALUES (?, 'moltbook', ?, ?, ?, ?, ?, ?, ?, ?, 1.0, ?)
            """,
            (
                edge_id,
                author_key,
                parent_author_key,
                edge_type,
                post_id,
                comment_id,
                post_submolt,
                created_at,
                fetched_at,
                json_dumps(comment),
            ),
        )
        return comment_id

    def update_coverage(self, post_id: str) -> None:
        row = self.conn.execute(
            """
            SELECT
                (SELECT comment_count FROM post_snapshots
                 WHERE post_id = ? AND comment_count IS NOT NULL
                 ORDER BY fetched_at DESC LIMIT 1) AS expected,
                (SELECT COUNT(*) FROM comments WHERE post_id = ?) AS observed
            """,
            (post_id, post_id),
        ).fetchone()
        expected = row["expected"] if row else None
        observed = int(row["observed"] or 0) if row else 0
        ratio = None
        capped = 0
        if expected and expected > 0:
            ratio = observed / expected
            capped = 1 if observed < expected else 0
        self.conn.execute(
            """
            INSERT INTO coverage(
                post_id, expected_comment_count, observed_comment_count,
                coverage_ratio, capped_flag, last_checked_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(post_id) DO UPDATE SET
                expected_comment_count=excluded.expected_comment_count,
                observed_comment_count=excluded.observed_comment_count,
                coverage_ratio=excluded.coverage_ratio,
                capped_flag=excluded.capped_flag,
                last_checked_at=excluded.last_checked_at
            """,
            (post_id, expected, observed, ratio, capped, utc_now()),
        )

    def recent_posts_for_hydration(self, limit: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT p.post_id, p.author_key, p.submolt,
                   COALESCE(ps.comment_count, 0) AS comment_count,
                   COALESCE(c.observed_comment_count, 0) AS observed_comment_count
            FROM posts p
            LEFT JOIN post_snapshots ps ON ps.snapshot_id = (
                SELECT snapshot_id FROM post_snapshots
                WHERE post_id = p.post_id
                ORDER BY fetched_at DESC LIMIT 1
            )
            LEFT JOIN coverage c ON c.post_id = p.post_id
            ORDER BY p.created_at DESC, p.last_seen_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    def insert_reddit_thread(
        self,
        dataset: str,
        thread_id: str,
        label: str,
        edges: Iterable[tuple[str, str]],
    ) -> None:
        edge_list = list(edges)
        nodes = set()
        normalized_edges: dict[tuple[str, str], float] = {}
        for source, target in edge_list:
            if source == "" or target == "":
                continue
            left, right = sorted((str(source), str(target)))
            nodes.add(left)
            nodes.add(right)
            normalized_edges[(left, right)] = normalized_edges.get((left, right), 0.0) + 1.0
        n = len(nodes)
        m = len(normalized_edges)
        density = (2 * m / (n * (n - 1))) if n > 1 else 0.0
        now = utc_now()
        self.conn.execute(
            """
            INSERT OR REPLACE INTO reddit_threads(
                dataset, thread_id, label, node_count, edge_count, density, imported_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (dataset, thread_id, label, n, m, density, now),
        )
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO reddit_thread_edges(
                dataset, thread_id, source_key, target_key, weight
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                (dataset, thread_id, source, target, weight)
                for (source, target), weight in normalized_edges.items()
            ),
        )

    def insert_reddit_adjacency_snapshot(
        self,
        *,
        dataset: str,
        network_type: str,
        subreddit: str,
        snapshot_index: int,
        adjacency: dict[str, list[str]],
        store_edges: bool = True,
    ) -> dict[str, Any]:
        nodes: set[str] = set()
        edge_weights: dict[tuple[str, str], float] = {}
        edge_entries = 0
        self_loops = 0
        for source, targets in adjacency.items():
            source_text = str(source)
            if not source_text:
                continue
            nodes.add(source_text)
            if not isinstance(targets, list):
                continue
            for target in targets:
                target_text = str(target)
                if not target_text:
                    continue
                nodes.add(target_text)
                edge_entries += 1
                if source_text == target_text:
                    self_loops += 1
                key = (source_text, target_text)
                edge_weights[key] = edge_weights.get(key, 0.0) + 1.0

        n = len(nodes)
        non_self_unique_edges = sum(1 for source, target in edge_weights if source != target)
        density = (non_self_unique_edges / (n * (n - 1))) if n > 1 else 0.0
        now = utc_now()
        self.conn.execute(
            """
            INSERT OR REPLACE INTO reddit_adjacency_snapshots(
                dataset, network_type, subreddit, snapshot_index, node_count,
                edge_entry_count, unique_edge_count, self_loop_count, density, imported_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset,
                network_type,
                subreddit,
                snapshot_index,
                n,
                edge_entries,
                len(edge_weights),
                self_loops,
                density,
                now,
            ),
        )
        if not store_edges:
            return {
                "dataset": dataset,
                "network_type": network_type,
                "subreddit": subreddit,
                "snapshot_index": snapshot_index,
                "node_count": n,
                "edge_entry_count": edge_entries,
                "unique_edge_count": len(edge_weights),
                "self_loop_count": self_loops,
            }

        edge_type = f"reddit_{network_type}"
        context_id = f"{dataset}:{network_type}:{subreddit}:{snapshot_index}"
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO edges(
                edge_id, platform, source_key, target_key, edge_type, context_id,
                comment_id, submolt, occurred_at, observed_at, weight, raw_json
            )
            VALUES (?, 'reddit', ?, ?, ?, ?, '', ?, NULL, ?, ?, ?)
            """,
            (
                (
                    stable_hash(
                        {
                            "platform": "reddit",
                            "dataset": dataset,
                            "network_type": network_type,
                            "subreddit": subreddit,
                            "snapshot_index": snapshot_index,
                            "source": source,
                            "target": target,
                        }
                    ),
                    f"reddit:{source}",
                    f"reddit:{target}",
                    edge_type,
                    context_id,
                    subreddit,
                    now,
                    weight,
                    json_dumps(
                        {
                            "dataset": dataset,
                            "network_type": network_type,
                            "subreddit": subreddit,
                            "snapshot_index": snapshot_index,
                            "source": source,
                            "target": target,
                        }
                    ),
                )
                for (source, target), weight in edge_weights.items()
            ),
        )
        return {
            "dataset": dataset,
            "network_type": network_type,
            "subreddit": subreddit,
            "snapshot_index": snapshot_index,
            "node_count": n,
            "edge_entry_count": edge_entries,
            "unique_edge_count": len(edge_weights),
            "self_loop_count": self_loops,
        }

    def commit(self) -> None:
        self.conn.commit()

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()


def extract_author(obj: dict[str, Any]) -> tuple[str, str, Any]:
    raw = (
        obj.get("author")
        or obj.get("created_by")
        or obj.get("createdBy")
        or obj.get("user")
        or obj.get("agent")
        or {}
    )
    if isinstance(raw, dict):
        key = str(
            raw.get("id")
            or raw.get("name")
            or raw.get("username")
            or raw.get("display_name")
            or raw.get("displayName")
            or ""
        )
        name = str(raw.get("name") or raw.get("username") or raw.get("display_name") or raw.get("displayName") or key)
        if key:
            return key, name, raw
    author_value = obj.get("author") or obj.get("author_name") or obj.get("authorName") or obj.get("username")
    if author_value:
        return str(author_value), str(author_value), raw
    return "unknown", "unknown", raw


def normalize_submolt(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, dict):
        return str(
            value.get("name")
            or value.get("display_name")
            or value.get("displayName")
            or value.get("id")
            or ""
        )
    return str(value)


def _nullable_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _nullable_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_nonblank(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _normalize_parent_comment_id(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        value = _first_nonblank(
            value.get("id"),
            value.get("comment_id"),
            value.get("commentId"),
            value.get("uuid"),
        )
    if value in (None, ""):
        return None
    return str(value)


def _extract_parent_author_key(comment: dict[str, Any]) -> str | None:
    raw = _first_nonblank(
        comment.get("parent_author"),
        comment.get("parentAuthor"),
        comment.get("parent_user"),
        comment.get("parentUser"),
        comment.get("reply_to_author"),
        comment.get("replyToAuthor"),
        comment.get("parent_author_id"),
        comment.get("parentAuthorId"),
        comment.get("reply_to_author_id"),
        comment.get("replyToAuthorId"),
    )
    if raw in (None, ""):
        return None
    if isinstance(raw, dict):
        key, _name, _raw = extract_author({"author": raw})
        return key if key != "unknown" else None
    return str(raw)
