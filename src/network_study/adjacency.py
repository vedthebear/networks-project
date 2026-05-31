from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .storage import Store
from .util import ensure_dir, slugify, write_json


def export_moltbook_snap_adjacency(
    store: Store,
    out_dir: Path,
    *,
    network: str = "all",
    bins: int = 11,
    start: str = "",
    end: str = "",
    node_label: str = "key",
    exclude_capped: bool = False,
    include_attention: bool = False,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    networks = _selected_networks(network, include_attention)
    labels = _agent_labels(store, node_label)
    summary = {
        "out_dir": str(out_dir),
        "bins": bins,
        "start": start,
        "end": end,
        "node_label": node_label,
        "exclude_capped": exclude_capped,
        "networks": {},
    }
    for name in networks:
        if name == "chain":
            events = _moltbook_chain_events(store, start=start, end=end, exclude_capped=exclude_capped)
        else:
            events = _moltbook_direct_events(
                store,
                network=name,
                start=start,
                end=end,
                exclude_capped=exclude_capped,
            )
        target_dir = ensure_dir(out_dir / _network_dir(name))
        summary["networks"][name] = _write_snap_adjacency_files(
            events,
            target_dir,
            bins=bins,
            labels=labels,
            start=start,
            end=end,
        )
    write_json(out_dir / "export_summary.json", summary, pretty=True)
    return summary


def _selected_networks(network: str, include_attention: bool) -> list[str]:
    if network == "all":
        selected = ["direct", "chain"]
    else:
        selected = [network]
    if include_attention and "attention" not in selected:
        selected.append("attention")
    valid = {"direct", "chain", "reply", "attention"}
    unknown = [item for item in selected if item not in valid]
    if unknown:
        raise ValueError(f"Unknown adjacency network(s): {unknown}")
    return selected


def _network_dir(network: str) -> str:
    if network == "direct":
        return "reply_networks"
    if network == "chain":
        return "chain_networks"
    if network == "reply":
        return "strict_reply_networks"
    return "attention_networks"


def _agent_labels(store: Store, node_label: str) -> dict[str, str]:
    if node_label == "key":
        return {}
    if node_label != "display":
        raise ValueError("--node-label must be either key or display")
    labels = {}
    for row in store.query("SELECT agent_key, display_name FROM agents"):
        key = str(row["agent_key"])
        labels[key] = str(row["display_name"] or key)
    return labels


def _label(value: str, labels: dict[str, str]) -> str:
    return labels.get(value, value)


def _moltbook_direct_events(
    store: Store,
    *,
    network: str,
    start: str,
    end: str,
    exclude_capped: bool,
) -> list[dict[str, Any]]:
    edge_types = {
        "direct": ["reply_to_comment", "comment_on_post"],
        "reply": ["reply_to_comment"],
        "attention": ["comment_on_post"],
    }[network]
    where = ["e.platform = 'moltbook'"]
    params: list[Any] = []
    placeholders = ",".join("?" for _ in edge_types)
    where.append(f"e.edge_type IN ({placeholders})")
    params.extend(edge_types)
    if start:
        where.append("COALESCE(e.occurred_at, e.observed_at) >= ?")
        params.append(start)
    if end:
        where.append("COALESCE(e.occurred_at, e.observed_at) < ?")
        params.append(end)
    if exclude_capped:
        where.append(
            """
            EXISTS (
                SELECT 1 FROM coverage c
                WHERE c.post_id = e.context_id AND c.capped_flag = 0
            )
            """
        )
    rows = store.query(
        f"""
        SELECT
            e.source_key AS source,
            e.target_key AS target,
            COALESCE(e.submolt, '') AS submolt,
            COALESCE(e.occurred_at, e.observed_at) AS event_time
        FROM edges e
        WHERE {' AND '.join(where)}
        """,
        tuple(params),
    )
    return [dict(row) for row in rows]


def _moltbook_chain_events(
    store: Store,
    *,
    start: str,
    end: str,
    exclude_capped: bool,
) -> list[dict[str, Any]]:
    where = ["c.platform = 'moltbook'"]
    params: list[Any] = []
    if start:
        where.append("COALESCE(e.occurred_at, e.observed_at, c.created_at, c.first_seen_at) >= ?")
        params.append(start)
    if end:
        where.append("COALESCE(e.occurred_at, e.observed_at, c.created_at, c.first_seen_at) < ?")
        params.append(end)
    if exclude_capped:
        where.append(
            """
            EXISTS (
                SELECT 1 FROM coverage cov
                WHERE cov.post_id = c.post_id AND cov.capped_flag = 0
            )
            """
        )
    rows = store.query(
        f"""
        SELECT
            c.comment_id,
            c.author_key,
            c.parent_comment_id,
            c.parent_author_key,
            c.post_id,
            p.author_key AS post_author_key,
            COALESCE(p.submolt, e.submolt, '') AS submolt,
            COALESCE(e.occurred_at, e.observed_at, c.created_at, c.first_seen_at) AS event_time
        FROM comments c
        LEFT JOIN posts p ON p.post_id = c.post_id
        LEFT JOIN edges e ON e.comment_id = c.comment_id AND e.platform = 'moltbook'
        WHERE {' AND '.join(where)}
        """,
        tuple(params),
    )
    by_id = {str(row["comment_id"]): dict(row) for row in rows if row["comment_id"]}
    events: list[dict[str, Any]] = []
    for row in by_id.values():
        source = str(row["author_key"] or "")
        if not source:
            continue
        for target in _ancestor_authors(row, by_id):
            if not target:
                continue
            events.append(
                {
                    "source": source,
                    "target": target,
                    "submolt": str(row["submolt"] or ""),
                    "event_time": row["event_time"],
                }
            )
    return events


def _ancestor_authors(row: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> list[str]:
    targets: list[str] = []
    seen_comments: set[str] = set()
    parent_id = str(row.get("parent_comment_id") or "")
    if not parent_id:
        target = str(row.get("parent_author_key") or row.get("post_author_key") or "")
        return [target] if target else []
    while parent_id and parent_id not in seen_comments:
        seen_comments.add(parent_id)
        parent = by_id.get(parent_id)
        if parent is None:
            fallback = str(row.get("parent_author_key") or row.get("post_author_key") or "")
            if fallback:
                targets.append(fallback)
            break
        parent_author = str(parent.get("author_key") or "")
        if parent_author:
            targets.append(parent_author)
        parent_id = str(parent.get("parent_comment_id") or "")
        if not parent_id:
            post_author = str(parent.get("post_author_key") or row.get("post_author_key") or "")
            if post_author:
                targets.append(post_author)
    return list(dict.fromkeys(targets))


def _write_snap_adjacency_files(
    events: list[dict[str, Any]],
    out_dir: Path,
    *,
    bins: int,
    labels: dict[str, str],
    start: str,
    end: str,
) -> dict[str, Any]:
    bins = max(1, bins)
    parsed_times = [_parse_time(row.get("event_time")) for row in events]
    parsed_times = [dt for dt in parsed_times if dt is not None]
    start_dt = _parse_time(start) if start else (min(parsed_times) if parsed_times else None)
    end_dt = _parse_time(end) if end else (max(parsed_times) if parsed_times else None)
    grouped: dict[str, list[dict[str, set[str]]]] = defaultdict(lambda: [defaultdict(set) for _ in range(bins)])
    edge_entries = 0
    for row in events:
        source = str(row.get("source") or "")
        target = str(row.get("target") or "")
        if not source or not target:
            continue
        idx = _bin_index(_parse_time(row.get("event_time")), start_dt, end_dt, bins)
        submolt = str(row.get("submolt") or "global")
        grouped[submolt][idx][_label(source, labels)].add(_label(target, labels))
        edge_entries += 1
    files_written = 0
    nonempty_files = 0
    for submolt, snapshots in sorted(grouped.items()):
        payload = [
            {source: sorted(targets) for source, targets in sorted(snapshot.items())}
            for snapshot in snapshots
        ]
        if any(snapshot for snapshot in payload):
            nonempty_files += 1
        write_json(out_dir / f"{slugify(submolt, 'global')}.json", payload, pretty=False)
        files_written += 1
    return {
        "files_written": files_written,
        "nonempty_files": nonempty_files,
        "edge_entries": edge_entries,
        "start": start_dt.isoformat() if start_dt else "",
        "end": end_dt.isoformat() if end_dt else "",
    }


def _parse_time(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _bin_index(dt: datetime | None, start: datetime | None, end: datetime | None, bins: int) -> int:
    if dt is None or start is None or end is None or end <= start:
        return 0
    fraction = (dt - start).total_seconds() / (end - start).total_seconds()
    return min(bins - 1, max(0, int(fraction * bins)))
