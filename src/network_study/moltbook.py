from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .config import MoltbookConfig
from .storage import normalize_submolt
from .storage import Store
from .util import as_list, get_path, json_dumps, utc_now


@dataclass
class ApiResponse:
    url: str
    path: str
    query: dict[str, Any]
    status_code: int
    headers: dict[str, Any]
    body: Any
    text: str


class MoltbookClient:
    def __init__(self, config: MoltbookConfig):
        self.config = config

    def get(self, path: str, query: dict[str, Any] | None = None) -> ApiResponse:
        return self.request("GET", path, query or {})

    def request(self, method: str, path: str, query: dict[str, Any]) -> ApiResponse:
        api_key = self.config.resolved_api_key()
        if not api_key:
            raise RuntimeError(
                f"No Moltbook API key found. Set ${self.config.api_key_env} or add api_key to config."
            )

        clean_query = {k: v for k, v in query.items() if v not in (None, "")}
        url = self._make_url(path, clean_query)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "moltbook-reddit-network-study/0.1",
        }
        max_retries = self.config.safety.max_retries

        for attempt in range(1, max_retries + 2):
            self._sleep_between_requests()
            request = urllib.request.Request(url, headers=headers, method=method)
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.config.safety.timeout_seconds,
                ) as response:
                    raw = response.read().decode("utf-8", errors="replace")
                    parsed = self._parse_body(raw)
                    response_headers = dict(response.headers.items())
                    self._cooldown_if_needed(response_headers)
                    return ApiResponse(
                        url=url,
                        path=path,
                        query=clean_query,
                        status_code=int(response.status),
                        headers=response_headers,
                        body=parsed,
                        text=raw,
                    )
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode("utf-8", errors="replace")
                status = int(exc.code)
                response_headers = dict(exc.headers.items()) if exc.headers else {}
                retryable = status in (408, 429) or 500 <= status < 600
                if not retryable or attempt > max_retries:
                    raise RuntimeError(f"Moltbook request failed: HTTP {status} {url}\n{raw}") from exc
                if status == 429:
                    self._cooldown_if_needed(response_headers, force=True)
                else:
                    self._backoff(attempt)
            except urllib.error.URLError as exc:
                if attempt > max_retries:
                    raise RuntimeError(f"Moltbook request failed: {url}\n{exc}") from exc
                self._backoff(attempt)

        raise RuntimeError(f"Moltbook request failed after retries: {url}")

    def _make_url(self, path: str, query: dict[str, Any]) -> str:
        base = self.config.base_url.rstrip("/")
        clean_path = "/" + path.lstrip("/")
        query_text = urllib.parse.urlencode(query, doseq=True)
        return f"{base}{clean_path}" + (f"?{query_text}" if query_text else "")

    def _parse_body(self, raw: str) -> Any:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}

    def _sleep_between_requests(self) -> None:
        delay_ms = self.config.safety.min_delay_ms
        delay_ms += random.randint(0, max(0, self.config.safety.jitter_ms))
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

    def _backoff(self, attempt: int) -> None:
        seconds = min(120.0, self.config.safety.backoff_base_seconds ** min(attempt, 6))
        seconds += random.random() * max(0, self.config.safety.jitter_ms / 1000.0)
        time.sleep(seconds)

    def _cooldown_if_needed(self, headers: dict[str, Any], force: bool = False) -> None:
        if not self.config.safety.honor_rate_limit_headers:
            return
        remaining = _header_int(headers, "X-RateLimit-Remaining")
        should_pause = force
        if remaining is not None:
            should_pause = should_pause or remaining <= self.config.safety.cool_down_when_remaining_at_or_below
        if not should_pause:
            return
        reset_seconds = _header_float(headers, "X-RateLimit-Reset")
        if reset_seconds is None:
            reset_seconds = self.config.safety.backoff_base_seconds
        time.sleep(max(0.0, reset_seconds))


class MoltbookCrawler:
    def __init__(self, client: MoltbookClient, store: Store, raw_dir):
        self.client = client
        self.store = store
        self.raw_dir = raw_dir

    def probe(self, run_id: int | None = None) -> dict[str, Any]:
        report: dict[str, Any] = {"checked_at": utc_now(), "probes": []}
        submolts = self._probe_one(run_id, "/submolts", {"limit": 50}, item_key="submolts")
        report["probes"].append(submolts)
        posts = self._probe_one(run_id, "/posts", {"sort": "new", "limit": 5, "offset": 0}, item_key="posts")
        report["probes"].append(posts)
        sample_posts = _items_from_body(posts.get("body"), "posts")
        first_post_id = _extract_post_id(sample_posts[0]) if sample_posts else ""
        first_submolt = _extract_submolt(sample_posts[0]) if sample_posts else ""

        for sort in ["new", "hot", "top"]:
            report["probes"].append(
                self._probe_one(run_id, "/posts", {"sort": sort, "limit": 5, "offset": 0}, item_key="posts")
            )

        if first_submolt:
            for param_name in ["submolt", "submoltName", "community", "subreddit"]:
                report["probes"].append(
                    self._probe_one(
                        run_id,
                        "/posts",
                        {"sort": "new", "limit": 5, "offset": 0, param_name: first_submolt},
                        item_key="posts",
                    )
                )
            for path in [f"/submolts/{first_submolt}/posts", f"/submolt/{first_submolt}/posts"]:
                report["probes"].append(
                    self._probe_one(run_id, path, {"sort": "new", "limit": 5, "offset": 0}, item_key="posts")
                )

        if first_post_id:
            report["sample_post_id"] = first_post_id
            report["probes"].append(self._probe_one(run_id, f"/posts/{first_post_id}", {}, item_key=""))
            for sort in ["new", "top", "old", "hot"]:
                report["probes"].append(
                    self._probe_one(
                        run_id,
                        f"/posts/{first_post_id}/comments",
                        {"sort": sort, "limit": 5, "offset": 0},
                        item_key="comments",
                    )
                )
                report["probes"].append(
                    self._probe_one(
                        run_id,
                        f"/posts/{first_post_id}/comments",
                        {"sort": sort, "limit": 5, "offset": 5},
                        item_key="comments",
                    )
                )
        return report

    def crawl_submolts(self, run_id: int | None = None, limit: int = 100) -> int:
        query = {"limit": limit}
        try:
            response = self._request_and_archive(run_id, "/submolts", query)
        except Exception as exc:
            self._record_error(
                run_id,
                scope="submolts",
                endpoint="/submolts",
                query=query,
                context={"limit": limit},
                error=exc,
            )
            if not self.client.config.safety.continue_on_error:
                raise
            print(f"[moltbook] warning: failed to refresh submolts; continuing: {exc}", flush=True)
            return 0
        items = _items_from_body(response.body, "submolts")
        for item in items:
            if isinstance(item, dict):
                self.store.upsert_submolt(item)
        self.store.commit()
        return len(items)

    def crawl_posts(
        self,
        *,
        run_id: int | None = None,
        sort: str = "new",
        limit: int = 50,
        max_pages: int = 20,
        submolt: str = "",
        submolt_query_param: str = "submolt",
        hydrate_comments: bool = True,
        comment_sorts: list[str] | None = None,
        comment_limit: int = 100,
        comment_max_pages: int = 20,
        deadline_monotonic: float | None = None,
    ) -> dict[str, int]:
        query = {"sort": sort, "limit": limit, "offset": 0}
        if submolt:
            query[submolt_query_param] = submolt
        post_count = 0
        comment_count = 0
        seen_post_ids: set[str] = set()

        for page_index in range(max_pages):
            if _deadline_reached(deadline_monotonic):
                print("[moltbook] active time budget reached; stopping this post page loop", flush=True)
                break
            self.store.upsert_progress(
                progress_key=f"posts:{submolt or 'global'}:{sort}",
                run_id=run_id,
                source="moltbook",
                target="crawl-posts",
                scope="post-page",
                subject_key=submolt or "global",
                state={"sort": sort, "page_index": page_index, "query": query},
            )
            print(
                f"[moltbook] posts sort={sort} submolt={submolt or 'global'} "
                f"page={page_index + 1}/{max_pages} offset={query.get('offset', 0)}",
                flush=True,
            )
            try:
                response = self._request_and_archive(run_id, "/posts", query)
            except Exception as exc:
                self._record_error(
                    run_id,
                    scope="post-page",
                    endpoint="/posts",
                    query=query,
                    context={"sort": sort, "submolt": submolt, "page_index": page_index},
                    error=exc,
                )
                if not self.client.config.safety.continue_on_error:
                    raise
                print(f"[moltbook] warning: failed post page; skipping page loop: {exc}", flush=True)
                break
            items = _items_from_body(response.body, "posts")
            if not items and isinstance(response.body, list):
                items = response.body
            if not items:
                print("[moltbook] no posts returned; stopping this post page loop", flush=True)
                break
            raw_hash = self.store.save_raw_response(
                run_id=run_id,
                source="moltbook",
                endpoint=response.path,
                query=response.query,
                status_code=response.status_code,
                headers=response.headers,
                body_text=response.text,
                raw_dir=self.raw_dir,
            )
            for item in items:
                if _deadline_reached(deadline_monotonic):
                    print("[moltbook] active time budget reached; stopping post processing", flush=True)
                    break
                if not isinstance(item, dict):
                    continue
                post_id = self.store.upsert_post(
                    item,
                    run_id=run_id,
                    fetched_at=utc_now(),
                    source_endpoint="/posts",
                    source_sort=sort,
                    source_submolt=submolt,
                    raw_hash=raw_hash,
                )
                if post_id and post_id not in seen_post_ids:
                    post_count += 1
                    seen_post_ids.add(post_id)
                    if hydrate_comments:
                        author_key = _extract_author_key(item)
                        submolt_name = _extract_submolt(item) or submolt
                        print(
                            f"[moltbook] hydrating comments post={post_id} "
                            f"observed_posts={post_count}",
                            flush=True,
                        )
                        comment_count += self.hydrate_comments(
                            post_id=post_id,
                            post_author_key=author_key,
                            post_submolt=submolt_name,
                            run_id=run_id,
                            sorts=comment_sorts or ["new", "top"],
                            limit=comment_limit,
                            max_pages=comment_max_pages,
                            deadline_monotonic=deadline_monotonic,
                        )
                    self.store.update_coverage(post_id)
            self.store.commit()
            if isinstance(response.body, dict) and response.body.get("has_more") is False:
                print("[moltbook] API returned has_more=false; stopping this post page loop", flush=True)
                break
            next_offset = response.body.get("next_offset") if isinstance(response.body, dict) else None
            if next_offset is not None:
                query["offset"] = next_offset
            else:
                query["offset"] = int(query.get("offset", 0)) + int(query.get("limit", limit))
        self.store.upsert_progress(
            progress_key=f"posts:{submolt or 'global'}:{sort}",
            run_id=run_id,
            source="moltbook",
            target="crawl-posts",
            scope="post-page",
            subject_key=submolt or "global",
            state={"sort": sort, "last_query": query, "posts": post_count, "comments": comment_count},
            status="completed",
        )
        return {"posts": post_count, "comments": comment_count}

    def hydrate_recent_posts(
        self,
        *,
        run_id: int | None,
        limit: int,
        sorts: list[str],
        comment_limit: int,
        comment_max_pages: int,
    ) -> int:
        total = 0
        for row in self.store.recent_posts_for_hydration(limit):
            total += self.hydrate_comments(
                post_id=row["post_id"],
                post_author_key=row["author_key"],
                post_submolt=row["submolt"],
                run_id=run_id,
                sorts=sorts,
                limit=comment_limit,
                max_pages=comment_max_pages,
            )
            self.store.update_coverage(row["post_id"])
        self.store.commit()
        return total

    def hydrate_comments(
        self,
        *,
        post_id: str,
        post_author_key: str,
        post_submolt: str,
        run_id: int | None,
        sorts: list[str],
        limit: int,
        max_pages: int,
        deadline_monotonic: float | None = None,
    ) -> int:
        total_new = 0
        comment_author_by_id: dict[str, str] = {}
        seen_comment_ids: set[str] = set()

        for sort in sorts:
            if _deadline_reached(deadline_monotonic):
                break
            query = {"sort": sort, "limit": limit, "offset": 0}
            for page_index in range(max_pages):
                if _deadline_reached(deadline_monotonic):
                    print("[moltbook] active time budget reached; stopping comment hydration", flush=True)
                    break
                self.store.upsert_progress(
                    progress_key=f"comments:{post_id}:{sort}",
                    run_id=run_id,
                    source="moltbook",
                    target="hydrate-comments",
                    scope="comment-page",
                    subject_key=post_id,
                    state={"post_id": post_id, "sort": sort, "page_index": page_index, "query": query},
                )
                print(
                    f"[moltbook] comments post={post_id} sort={sort} "
                    f"page={page_index + 1}/{max_pages} offset={query.get('offset', 0)}",
                    flush=True,
                )
                endpoint = f"/posts/{post_id}/comments"
                try:
                    response = self._request_and_archive(run_id, endpoint, query)
                except Exception as exc:
                    self._record_error(
                        run_id,
                        scope="comment-page",
                        endpoint=endpoint,
                        query=query,
                        context={"post_id": post_id, "sort": sort, "page_index": page_index},
                        error=exc,
                    )
                    if not self.client.config.safety.continue_on_error:
                        raise
                    print(f"[moltbook] warning: failed comment page; continuing: {exc}", flush=True)
                    break
                items = _items_from_body(response.body, "comments")
                if not items and isinstance(response.body, list):
                    items = response.body
                items = _flatten_comments(items)
                raw_hash = self.store.save_raw_response(
                    run_id=run_id,
                    source="moltbook",
                    endpoint=response.path,
                    query=response.query,
                    status_code=response.status_code,
                    headers=response.headers,
                    body_text=response.text,
                    raw_dir=self.raw_dir,
                )
                if not items:
                    break
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    cid = _extract_comment_id(item)
                    if cid:
                        comment_author_by_id[cid] = _extract_author_key(item)
                new_this_page = 0
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    cid = self.store.upsert_comment(
                        item,
                        post_id=post_id,
                        post_author_key=post_author_key,
                        post_submolt=post_submolt,
                        run_id=run_id,
                        fetched_at=utc_now(),
                        source_sort=sort,
                        parent_author_by_comment_id=comment_author_by_id,
                        raw_hash=raw_hash,
                    )
                    if cid and cid not in seen_comment_ids:
                        seen_comment_ids.add(cid)
                        total_new += 1
                        new_this_page += 1
                self.store.commit()
                if isinstance(response.body, dict) and response.body.get("has_more") is False:
                    break
                if new_this_page == 0 and int(query.get("offset", 0)) > 0:
                    break
                next_offset = response.body.get("next_offset") if isinstance(response.body, dict) else None
                if next_offset is not None:
                    query["offset"] = next_offset
                else:
                    query["offset"] = int(query.get("offset", 0)) + int(query.get("limit", limit))
            self.store.upsert_progress(
                progress_key=f"comments:{post_id}:{sort}",
                run_id=run_id,
                source="moltbook",
                target="hydrate-comments",
                scope="comment-page",
                subject_key=post_id,
                state={"post_id": post_id, "sort": sort, "last_query": query, "new_comments": total_new},
                status="completed",
            )
        return total_new

    def probe_parent_structure(
        self,
        *,
        run_id: int | None,
        post_limit: int = 25,
        comment_limit: int = 100,
        sorts: list[str] | None = None,
    ) -> dict[str, Any]:
        sorts = sorts or ["new", "top", "old"]
        posts = self._parent_probe_posts(post_limit)
        report: dict[str, Any] = {
            "checked_at": utc_now(),
            "post_limit": post_limit,
            "comment_limit": comment_limit,
            "sorts": sorts,
            "posts_checked": 0,
            "comment_batches_checked": 0,
            "comments_seen": 0,
            "raw_parent_field_hits": 0,
            "raw_parent_author_hits": 0,
            "raw_nested_reply_hits": 0,
            "flattened_reply_edges_possible": 0,
            "field_counts": {},
            "posts": [],
        }
        field_counts: Counter[str] = Counter()
        for post in posts:
            post_id = str(post.get("post_id") or "")
            if not post_id:
                continue
            post_record: dict[str, Any] = {
                "post_id": post_id,
                "submolt": post.get("submolt", ""),
                "comment_count": post.get("comment_count"),
                "detail_keys": [],
                "comment_sorts": [],
            }
            try:
                detail = self._request_and_archive(run_id, f"/posts/{post_id}", {})
                if isinstance(detail.body, dict):
                    post_record["detail_keys"] = sorted(detail.body.keys())
            except Exception as exc:
                self._record_error(
                    run_id,
                    scope="parent-probe-detail",
                    endpoint=f"/posts/{post_id}",
                    query={},
                    context={"post_id": post_id},
                    error=exc,
                )
                post_record["detail_error"] = str(exc)

            for sort in sorts:
                query = {"sort": sort, "limit": comment_limit, "offset": 0}
                sort_record: dict[str, Any] = {
                    "sort": sort,
                    "raw_comments": 0,
                    "flattened_comments": 0,
                    "raw_parent_field_hits": 0,
                    "raw_parent_author_hits": 0,
                    "raw_nested_reply_hits": 0,
                    "flattened_parent_field_hits": 0,
                    "keys": [],
                    "parent_field_examples": [],
                }
                try:
                    response = self._request_and_archive(run_id, f"/posts/{post_id}/comments", query)
                except Exception as exc:
                    self._record_error(
                        run_id,
                        scope="parent-probe-comments",
                        endpoint=f"/posts/{post_id}/comments",
                        query=query,
                        context={"post_id": post_id, "sort": sort},
                        error=exc,
                    )
                    sort_record["error"] = str(exc)
                    post_record["comment_sorts"].append(sort_record)
                    continue
                items = _items_from_body(response.body, "comments")
                if not items and isinstance(response.body, list):
                    items = response.body
                flat_items = _flatten_comments(items)
                raw_keys: set[str] = set()
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    raw_keys.update(item.keys())
                    field_counts.update(item.keys())
                    if _comment_parent_value(item) not in (None, ""):
                        sort_record["raw_parent_field_hits"] += 1
                    if _comment_parent_author_value(item) not in (None, ""):
                        sort_record["raw_parent_author_hits"] += 1
                    replies = item.get("replies")
                    if isinstance(replies, list) and replies:
                        sort_record["raw_nested_reply_hits"] += len(replies)
                for item in flat_items:
                    if not isinstance(item, dict):
                        continue
                    if _comment_parent_value(item) not in (None, ""):
                        sort_record["flattened_parent_field_hits"] += 1
                        if len(sort_record["parent_field_examples"]) < 3:
                            sort_record["parent_field_examples"].append(
                                {
                                    "id": _extract_comment_id(item),
                                    "author": _extract_author_key(item),
                                    "parent": _comment_parent_value(item),
                                    "parent_author": _comment_parent_author_value(item),
                                }
                            )
                sort_record["raw_comments"] = len(items)
                sort_record["flattened_comments"] = len(flat_items)
                sort_record["keys"] = sorted(raw_keys)
                report["comment_batches_checked"] += 1
                report["comments_seen"] += len(flat_items)
                report["raw_parent_field_hits"] += int(sort_record["raw_parent_field_hits"])
                report["raw_parent_author_hits"] += int(sort_record["raw_parent_author_hits"])
                report["raw_nested_reply_hits"] += int(sort_record["raw_nested_reply_hits"])
                report["flattened_reply_edges_possible"] += int(sort_record["flattened_parent_field_hits"])
                post_record["comment_sorts"].append(sort_record)
            report["posts_checked"] += 1
            report["posts"].append(post_record)
        report["field_counts"] = dict(field_counts.most_common())
        return report

    def _parent_probe_posts(self, post_limit: int) -> list[dict[str, Any]]:
        rows = self.store.query(
            """
            SELECT
                p.post_id,
                p.author_key,
                p.submolt,
                MAX(COALESCE(ps.comment_count, 0)) AS comment_count,
                MAX(ps.fetched_at) AS last_seen
            FROM posts p
            LEFT JOIN post_snapshots ps ON ps.post_id = p.post_id
            GROUP BY p.post_id, p.author_key, p.submolt
            HAVING comment_count > 0
            ORDER BY last_seen DESC
            LIMIT ?
            """,
            (post_limit,),
        )
        if rows:
            return [dict(row) for row in rows]

        response = self._request_and_archive(None, "/posts", {"sort": "new", "limit": post_limit, "offset": 0})
        items = _items_from_body(response.body, "posts")
        posts = []
        for item in items:
            if not isinstance(item, dict):
                continue
            posts.append(
                {
                    "post_id": _extract_post_id(item),
                    "author_key": _extract_author_key(item),
                    "submolt": _extract_submolt(item),
                    "comment_count": item.get("comment_count") or item.get("commentCount"),
                }
            )
        return posts

    def _probe_one(
        self,
        run_id: int | None,
        path: str,
        query: dict[str, Any],
        *,
        item_key: str,
    ) -> dict[str, Any]:
        try:
            response = self._request_and_archive(run_id, path, query)
            items = _items_from_body(response.body, item_key) if item_key else as_list(response.body)
            sample = items[0] if items else response.body
            keys = sorted(sample.keys()) if isinstance(sample, dict) else []
            record = {
                "endpoint": path,
                "query": query,
                "status_code": response.status_code,
                "ok": 200 <= response.status_code < 300,
                "item_count": len(items),
                "keys": keys,
                "sample": sample if isinstance(sample, dict) else {},
                "body": response.body,
            }
            self.store.save_probe(
                run_id=run_id,
                endpoint=path,
                query=query,
                status_code=response.status_code,
                ok=record["ok"],
                item_count=record["item_count"],
                keys=keys,
                sample=record["sample"],
            )
            return record
        except Exception as exc:
            self.store.save_probe(
                run_id=run_id,
                endpoint=path,
                query=query,
                status_code=0,
                ok=False,
                item_count=0,
                keys=[],
                sample={},
                error=str(exc),
            )
            return {
                "endpoint": path,
                "query": query,
                "status_code": 0,
                "ok": False,
                "item_count": 0,
                "keys": [],
                "sample": {},
                "error": str(exc),
            }

    def _request_and_archive(self, run_id: int | None, path: str, query: dict[str, Any]) -> ApiResponse:
        response = self.client.get(path, query)
        self.store.save_raw_response(
            run_id=run_id,
            source="moltbook",
            endpoint=path,
            query=query,
            status_code=response.status_code,
            headers=response.headers,
            body_text=response.text or json_dumps(response.body or {}),
            raw_dir=self.raw_dir,
        )
        return response

    def _record_error(
        self,
        run_id: int | None,
        *,
        scope: str,
        endpoint: str,
        query: dict[str, Any],
        context: dict[str, Any],
        error: Exception,
    ) -> None:
        self.store.record_error(
            run_id=run_id,
            source="moltbook",
            scope=scope,
            endpoint=endpoint,
            query=query,
            context=context,
            error=str(error),
        )


def _items_from_body(body: Any, item_key: str) -> list[Any]:
    if body is None:
        return []
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        if item_key and item_key in body:
            return as_list(body[item_key])
        for key in ["items", "data", "results", "posts", "comments", "submolts"]:
            if key in body and isinstance(body[key], list):
                return body[key]
    return []


def _deadline_reached(deadline_monotonic: float | None) -> bool:
    return deadline_monotonic is not None and time.monotonic() >= deadline_monotonic


def _flatten_comments(items: list[Any]) -> list[Any]:
    flat: list[Any] = []

    def visit(item: Any, parent_id: str | None = None, parent_author: str | None = None) -> None:
        if not isinstance(item, dict):
            return
        current = dict(item)
        if parent_id and not _comment_parent_value(current):
            current["parent_comment_id"] = parent_id
        if parent_author and not _comment_parent_author_value(current):
            current["parent_author_id"] = parent_author
        flat.append(current)
        current_id = _extract_comment_id(current)
        current_author = _extract_author_key(current)
        for key in ["replies", "children", "comments"]:
            replies = item.get(key)
            if isinstance(replies, list):
                for reply in replies:
                    visit(reply, current_id or parent_id, current_author or parent_author)

    for item in items:
        visit(item)
    return flat


def _comment_parent_value(comment: dict[str, Any]) -> Any:
    for key in [
        "parent_comment_id",
        "parentCommentId",
        "parent_id",
        "parentId",
        "replyToCommentId",
        "reply_to_comment_id",
        "replyToId",
        "reply_to_id",
        "in_reply_to_id",
        "inReplyToId",
    ]:
        if comment.get(key) not in (None, ""):
            return comment.get(key)
    return None


def _comment_parent_author_value(comment: dict[str, Any]) -> Any:
    for key in [
        "parent_author",
        "parentAuthor",
        "parent_user",
        "parentUser",
        "reply_to_author",
        "replyToAuthor",
        "parent_author_id",
        "parentAuthorId",
        "reply_to_author_id",
        "replyToAuthorId",
    ]:
        if comment.get(key) not in (None, ""):
            return comment.get(key)
    return None


def _extract_post_id(post: dict[str, Any]) -> str:
    return str(post.get("id") or post.get("post_id") or post.get("postId") or "")


def _extract_comment_id(comment: dict[str, Any]) -> str:
    return str(comment.get("id") or comment.get("comment_id") or comment.get("commentId") or "")


def _extract_submolt(post: dict[str, Any]) -> str:
    return normalize_submolt(post.get("submolt") or post.get("submolt_name") or post.get("submoltName") or "")


def _extract_author_key(obj: dict[str, Any]) -> str:
    author = get_path(obj, ["author", "created_by", "createdBy", "user", "agent"], {})
    if isinstance(author, dict):
        return str(author.get("id") or author.get("name") or author.get("username") or "unknown")
    if author:
        return str(author)
    return str(obj.get("author") or obj.get("username") or "unknown")


def _header_int(headers: dict[str, Any], name: str) -> int | None:
    value = _header_value(headers, name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _header_float(headers: dict[str, Any], name: str) -> float | None:
    value = _header_value(headers, name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _header_value(headers: dict[str, Any], name: str) -> str | None:
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return str(value)
    return None
