from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

from .storage import Store
from .util import ensure_dir


def resolve_dataset_root(path: Path, work_dir: Path) -> Path:
    if path.is_dir():
        return path
    if path.suffix.lower() != ".zip":
        return path
    destination = work_dir / path.stem
    ensure_dir(destination)
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(destination)
    return destination


def import_reddit_threads(
    store: Store,
    dataset_path: Path,
    *,
    dataset_name: str = "reddit_threads",
    limit: int | None = None,
    batch_size: int = 1000,
) -> dict[str, Any]:
    edge_file = _find_file(dataset_path, "reddit_edges.json")
    target_file = _find_file(dataset_path, "reddit_target.csv")
    if edge_file is None:
        raise FileNotFoundError(f"Could not find reddit_edges.json under {dataset_path}")

    labels = _read_thread_labels(target_file) if target_file else {}
    imported = 0
    with edge_file.open("r", encoding="utf-8") as fh:
        edge_map = json.load(fh)

    for thread_id, edges in edge_map.items():
        if limit is not None and imported >= limit:
            break
        parsed_edges = []
        for edge in edges:
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                parsed_edges.append((str(edge[0]), str(edge[1])))
        store.insert_reddit_thread(
            dataset=dataset_name,
            thread_id=str(thread_id),
            label=str(labels.get(str(thread_id), "")),
            edges=parsed_edges,
        )
        imported += 1
        if imported % batch_size == 0:
            store.commit()
    store.commit()
    return {
        "dataset": dataset_name,
        "threads_imported": imported,
        "edge_file": str(edge_file),
        "label_file": str(target_file) if target_file else "",
    }


def import_temporal_reply_edges(
    store: Store,
    path: Path,
    *,
    dataset_name: str = "reddit_temporal_reply",
    source_col: str = "source",
    target_col: str = "target",
    time_col: str = "timestamp",
    delimiter: str = ",",
    limit: int | None = None,
) -> dict[str, Any]:
    """Import a generic temporal Reddit user-reply CSV as edge rows.

    The Cornell temporal Reddit data and many derived exports can be normalized
    into source,target,timestamp columns before using this importer.
    """
    imported = 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        required = {source_col, target_col}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
        for row in reader:
            if limit is not None and imported >= limit:
                break
            source = str(row.get(source_col, "")).strip()
            target = str(row.get(target_col, "")).strip()
            if not source or not target:
                continue
            timestamp = str(row.get(time_col, "")).strip() if time_col in row else ""
            edge_id = f"reddit:{dataset_name}:{imported}:{source}:{target}:{timestamp}"
            store.conn.execute(
                """
                INSERT OR REPLACE INTO edges(
                    edge_id, platform, source_key, target_key, edge_type, context_id,
                    comment_id, submolt, occurred_at, observed_at, weight, raw_json
                )
                VALUES (?, 'reddit', ?, ?, 'reply', ?, '', ?, ?, ?, 1.0, ?)
                """,
                (
                    edge_id,
                    f"reddit:{source}",
                    f"reddit:{target}",
                    str(row.get("context_id", row.get("subreddit", ""))),
                    str(row.get("subreddit", "")),
                    timestamp or None,
                    timestamp or "",
                    json.dumps(row, ensure_ascii=False),
                ),
            )
            imported += 1
            if imported % 10_000 == 0:
                store.commit()
    store.commit()
    return {"dataset": dataset_name, "edges_imported": imported, "path": str(path)}


def import_reddit_adjacency_snapshots(
    store: Store,
    *,
    reply_dir: Path | None = None,
    chain_dir: Path | None = None,
    dataset_name: str = "snap_reddit",
    limit_files: int | None = None,
    batch_size: int = 25,
    store_edges: bool = True,
) -> dict[str, Any]:
    """Import SNAP Reddit reply/chain adjacency JSON snapshots.

    Each file is expected to contain a list of adjacency maps:
    [{"source_user": ["target_user", ...]}, ...]. The importer keeps these
    graphs directed and writes one snapshot row plus directed edge rows per
    subreddit/snapshot/network type.
    """
    inputs: list[tuple[str, Path]] = []
    if reply_dir is not None:
        inputs.append(("reply", reply_dir))
    if chain_dir is not None:
        inputs.append(("chain", chain_dir))
    if not inputs:
        raise ValueError("Provide at least one of reply_dir or chain_dir.")

    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "network_types": {},
        "snapshots_imported": 0,
        "files_imported": 0,
        "unique_edges_imported": 0,
        "edge_rows_stored": bool(store_edges),
    }
    for network_type, directory in inputs:
        files = sorted(Path(directory).glob("*.json"))
        if limit_files is not None:
            files = files[:limit_files]
        type_summary = {
            "path": str(directory),
            "files": 0,
            "snapshots": 0,
            "unique_edges": 0,
            "edge_entries": 0,
        }
        for file_index, path in enumerate(files, start=1):
            with path.open("r", encoding="utf-8") as fh:
                snapshots = json.load(fh)
            if not isinstance(snapshots, list):
                raise ValueError(f"Expected list root in {path}")
            subreddit = path.stem
            type_summary["files"] += 1
            summary["files_imported"] += 1
            for snapshot_index, adjacency in enumerate(snapshots):
                if not isinstance(adjacency, dict):
                    continue
                normalized = {
                    str(source): [str(target) for target in targets]
                    for source, targets in adjacency.items()
                    if isinstance(targets, list)
                }
                result = store.insert_reddit_adjacency_snapshot(
                    dataset=dataset_name,
                    network_type=network_type,
                    subreddit=subreddit,
                    snapshot_index=snapshot_index,
                    adjacency=normalized,
                    store_edges=store_edges,
                )
                type_summary["snapshots"] += 1
                type_summary["unique_edges"] += int(result["unique_edge_count"])
                type_summary["edge_entries"] += int(result["edge_entry_count"])
                summary["snapshots_imported"] += 1
                summary["unique_edges_imported"] += int(result["unique_edge_count"])
            if file_index % batch_size == 0:
                store.commit()
        store.commit()
        summary["network_types"][network_type] = type_summary
    return summary


def reddit_thread_metric_rows(store: Store, dataset: str) -> list[dict[str, Any]]:
    rows = store.query(
        """
        SELECT dataset, thread_id, label, node_count, edge_count, density
        FROM reddit_threads
        WHERE dataset = ?
        """,
        (dataset,),
    )
    return [dict(row) for row in rows]


def iter_reddit_thread_edges(store: Store, dataset: str, limit_threads: int | None = None) -> Iterable[tuple[str, str, str]]:
    params: tuple[Any, ...] = (dataset,)
    limit_sql = ""
    if limit_threads is not None:
        limit_sql = """
        AND thread_id IN (
            SELECT thread_id FROM reddit_threads
            WHERE dataset = ?
            ORDER BY thread_id
            LIMIT ?
        )
        """
        params = (dataset, dataset, limit_threads)
    rows = store.query(
        f"""
        SELECT thread_id, source_key, target_key
        FROM reddit_thread_edges
        WHERE dataset = ?
        {limit_sql}
        """,
        params,
    )
    for row in rows:
        yield str(row["thread_id"]), str(row["source_key"]), str(row["target_key"])


def _find_file(root: Path, name: str) -> Path | None:
    if root.is_file() and root.name == name:
        return root
    if root.is_dir():
        for candidate in root.rglob(name):
            return candidate
    return None


def _read_thread_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        for idx, row in enumerate(reader):
            if not row:
                continue
            if len(row) == 1:
                labels[str(idx)] = row[0]
            else:
                labels[str(row[0])] = row[-1]
        if header and len(header) == 1 and not labels:
            labels["0"] = header[0]
    return labels
