from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Iterable

from .storage import Store
from .util import gini, top_share


@dataclass
class Edge:
    source: str
    target: str
    weight: float = 1.0
    context: str = ""


class WeightedDiGraph:
    def __init__(self, edges: Iterable[Edge]):
        self.edge_weights: dict[tuple[str, str], float] = {}
        self.out_adj: dict[str, set[str]] = defaultdict(set)
        self.in_adj: dict[str, set[str]] = defaultdict(set)
        self.nodes: set[str] = set()
        for edge in edges:
            if not edge.source or not edge.target:
                continue
            self.nodes.add(edge.source)
            self.nodes.add(edge.target)
            self.out_adj[edge.source].add(edge.target)
            self.in_adj[edge.target].add(edge.source)
            key = (edge.source, edge.target)
            self.edge_weights[key] = self.edge_weights.get(key, 0.0) + float(edge.weight or 1.0)

    @property
    def edge_count(self) -> int:
        return len(self.edge_weights)

    @property
    def total_weight(self) -> float:
        return sum(self.edge_weights.values())

    @property
    def self_loop_count(self) -> int:
        return sum(1 for source, target in self.edge_weights if source == target)

    def in_strengths(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for (_source, target), weight in self.edge_weights.items():
            counts[target] += weight
        return counts

    def out_strengths(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for (source, _target), weight in self.edge_weights.items():
            counts[source] += weight
        return counts

    def undirected_adj(self) -> dict[str, set[str]]:
        adj: dict[str, set[str]] = defaultdict(set)
        for source, target in self.edge_weights:
            adj[source].add(target)
            adj[target].add(source)
        return adj


def load_moltbook_graph(
    store: Store,
    start: str = "",
    end: str = "",
    *,
    edge_type: str = "",
    min_coverage_ratio: float | None = None,
    exclude_capped: bool = False,
) -> WeightedDiGraph:
    where = ["platform = 'moltbook'"]
    params: list[str] = []
    if start:
        where.append("COALESCE(occurred_at, observed_at) >= ?")
        params.append(start)
    if end:
        where.append("COALESCE(occurred_at, observed_at) < ?")
        params.append(end)
    if edge_type:
        where.append("edge_type = ?")
        params.append(edge_type)
    if min_coverage_ratio is not None or exclude_capped:
        coverage_where = ["c.post_id = edges.context_id"]
        if min_coverage_ratio is not None:
            coverage_where.append("c.coverage_ratio >= ?")
            params.append(str(min_coverage_ratio))
        if exclude_capped:
            coverage_where.append("c.capped_flag = 0")
        where.append(
            f"""
            EXISTS (
                SELECT 1 FROM coverage c
                WHERE {' AND '.join(coverage_where)}
            )
            """
        )
    rows = store.query(
        f"""
        SELECT source_key, target_key, weight, context_id
        FROM edges
        WHERE {' AND '.join(where)}
        """,
        tuple(params),
    )
    return WeightedDiGraph(
        Edge(str(row["source_key"]), str(row["target_key"]), float(row["weight"] or 1.0), str(row["context_id"]))
        for row in rows
    )


def load_reddit_temporal_graph(store: Store, start: str = "", end: str = "") -> WeightedDiGraph:
    where = ["platform = 'reddit'", "edge_type = 'reply'"]
    params: list[str] = []
    if start:
        where.append("COALESCE(occurred_at, observed_at) >= ?")
        params.append(start)
    if end:
        where.append("COALESCE(occurred_at, observed_at) < ?")
        params.append(end)
    rows = store.query(
        f"""
        SELECT source_key, target_key, weight, context_id
        FROM edges
        WHERE {' AND '.join(where)}
        """,
        tuple(params),
    )
    return WeightedDiGraph(
        Edge(str(row["source_key"]), str(row["target_key"]), float(row["weight"] or 1.0), str(row["context_id"]))
        for row in rows
    )


def load_reddit_adjacency_graph(
    store: Store,
    *,
    dataset: str = "snap_reddit",
    network_type: str,
    subreddit: str = "",
    snapshot_index: int | None = None,
) -> WeightedDiGraph:
    where = ["platform = 'reddit'", "edge_type = ?"]
    params: list[object] = [f"reddit_{network_type}"]
    context_prefix = f"{dataset}:{network_type}:"
    where.append("context_id LIKE ?")
    params.append(f"{context_prefix}%")
    if subreddit:
        where.append("submolt = ?")
        params.append(subreddit)
    if snapshot_index is not None:
        where.append("context_id LIKE ?")
        params.append(f"{context_prefix}%:{snapshot_index}")
    rows = store.query(
        f"""
        SELECT source_key, target_key, weight, context_id
        FROM edges
        WHERE {' AND '.join(where)}
        """,
        tuple(params),
    )
    return WeightedDiGraph(
        Edge(str(row["source_key"]), str(row["target_key"]), float(row["weight"] or 1.0), str(row["context_id"]))
        for row in rows
    )


def summarize_graph(graph: WeightedDiGraph, *, path_sample_limit: int = 1200) -> dict[str, object]:
    n = len(graph.nodes)
    m = graph.edge_count
    density = m / (n * (n - 1)) if n > 1 else 0.0
    in_strength = graph.in_strengths()
    out_strength = graph.out_strengths()
    in_values = [in_strength.get(node, 0.0) for node in graph.nodes]
    out_values = [out_strength.get(node, 0.0) for node in graph.nodes]
    weak = weak_components(graph.nodes, graph.undirected_adj())
    strong = strongly_connected_components(graph.nodes, graph.out_adj, graph.in_adj)
    lcc = weak[0] if weak else []
    path_stats = component_path_metrics(lcc, graph.undirected_adj(), sample_limit=path_sample_limit)
    clustering = clustering_and_core(lcc, graph.undirected_adj())
    pr = pagerank(graph)

    return {
        "nodes": n,
        "directed_edges": m,
        "self_loop_edges": graph.self_loop_count,
        "total_weight": graph.total_weight,
        "directed_density": density,
        "weak_component_count": len(weak),
        "largest_weak_component_size": len(lcc),
        "largest_weak_component_share": len(lcc) / n if n else 0.0,
        "largest_strong_component_size": len(strong[0]) if strong else 0,
        "largest_strong_component_share": len(strong[0]) / n if strong else 0.0,
        "reciprocity": reciprocity(graph),
        "weighted_reciprocity": weighted_reciprocity(graph),
        "avg_shortest_path_lcc": path_stats["avg_shortest_path"],
        "diameter_lcc": path_stats["diameter"],
        "path_sampled": path_stats["sampled"],
        "avg_local_clustering_lcc": clustering["avg_local_clustering"],
        "global_transitivity_lcc": clustering["global_transitivity"],
        "max_core_number_lcc": clustering["max_core_number"],
        "max_core_size_lcc": clustering["max_core_size"],
        "weighted_in_gini": gini(in_values),
        "weighted_out_gini": gini(out_values),
        "top_1pct_weighted_in_share": top_share(in_values, 0.01),
        "top_5pct_weighted_in_share": top_share(in_values, 0.05),
        "top_10pct_weighted_in_share": top_share(in_values, 0.10),
        "top_1pct_weighted_out_share": top_share(out_values, 0.01),
        "top_5pct_weighted_out_share": top_share(out_values, 0.05),
        "top_10pct_weighted_out_share": top_share(out_values, 0.10),
        "top_in_strength": top_counter(in_strength, 20),
        "top_out_strength": top_counter(out_strength, 20),
        "top_pagerank": top_counter(Counter(pr), 20),
    }


def summarize_reddit_threads(store: Store, dataset: str) -> dict[str, object]:
    rows = store.query(
        """
        SELECT node_count, edge_count, density, label
        FROM reddit_threads
        WHERE dataset = ?
        """,
        (dataset,),
    )
    nodes = [int(row["node_count"]) for row in rows]
    edges = [int(row["edge_count"]) for row in rows]
    densities = [float(row["density"] or 0.0) for row in rows]
    labels = Counter(str(row["label"]) for row in rows)
    return {
        "dataset": dataset,
        "thread_count": len(rows),
        "total_nodes_across_threads": sum(nodes),
        "total_edges_across_threads": sum(edges),
        "avg_nodes_per_thread": _mean(nodes),
        "median_nodes_per_thread": _median(nodes),
        "avg_edges_per_thread": _mean(edges),
        "median_edges_per_thread": _median(edges),
        "avg_density": _mean(densities),
        "median_density": _median(densities),
        "min_nodes": min(nodes) if nodes else 0,
        "max_nodes": max(nodes) if nodes else 0,
        "min_edges": min(edges) if edges else 0,
        "max_edges": max(edges) if edges else 0,
        "labels": dict(labels),
    }


def summarize_reddit_adjacency_snapshots(store: Store, dataset: str = "snap_reddit") -> dict[str, object]:
    rows = store.query(
        """
        SELECT network_type, node_count, edge_entry_count, unique_edge_count, self_loop_count, density
        FROM reddit_adjacency_snapshots
        WHERE dataset = ?
        """,
        (dataset,),
    )
    by_type: dict[str, dict[str, object]] = {}
    for network_type in sorted({str(row["network_type"]) for row in rows}):
        subset = [row for row in rows if str(row["network_type"]) == network_type]
        nodes = [int(row["node_count"]) for row in subset]
        entries = [int(row["edge_entry_count"]) for row in subset]
        unique_edges = [int(row["unique_edge_count"]) for row in subset]
        self_loops = [int(row["self_loop_count"]) for row in subset]
        densities = [float(row["density"] or 0.0) for row in subset]
        by_type[network_type] = {
            "snapshot_count": len(subset),
            "total_edge_entries": sum(entries),
            "total_unique_edges": sum(unique_edges),
            "total_self_loops": sum(self_loops),
            "avg_nodes": _mean(nodes),
            "median_nodes": _median(nodes),
            "avg_unique_edges": _mean(unique_edges),
            "median_unique_edges": _median(unique_edges),
            "avg_density": _mean(densities),
            "median_density": _median(densities),
        }
    return {"dataset": dataset, "network_types": by_type}


def summarize_reddit_adjacency_graph_if_small(
    store: Store,
    *,
    dataset: str = "snap_reddit",
    network_type: str,
    max_edge_rows: int = 100_000,
) -> dict[str, object]:
    edge_type = f"reddit_{network_type}"
    context_prefix = f"{dataset}:{network_type}:"
    row = store.query(
        """
        SELECT COUNT(*) AS edges
        FROM edges
        WHERE platform = 'reddit' AND edge_type = ? AND context_id LIKE ?
        """,
        (edge_type, f"{context_prefix}%"),
    )[0]
    edge_rows = int(row["edges"] or 0)
    if edge_rows == 0:
        snapshot_row = store.query(
            """
            SELECT COUNT(*) AS snapshots
            FROM reddit_adjacency_snapshots
            WHERE dataset = ? AND network_type = ?
            """,
            (dataset, network_type),
        )[0]
        if int(snapshot_row["snapshots"] or 0) > 0:
            return {
                "skipped": True,
                "reason": "snapshot summaries were imported without edge rows",
                "edge_rows": 0,
                "nodes": None,
                "directed_edges": None,
            }
    if edge_rows > max_edge_rows:
        return {
            "skipped": True,
            "reason": f"edge row count {edge_rows} exceeds max_edge_rows {max_edge_rows}",
            "edge_rows": edge_rows,
            "nodes": None,
            "directed_edges": None,
        }
    summary = summarize_graph(load_reddit_adjacency_graph(store, dataset=dataset, network_type=network_type))
    summary["skipped"] = False
    summary["edge_rows"] = edge_rows
    return summary


def weak_components(nodes: Iterable[str], adj: dict[str, set[str]]) -> list[list[str]]:
    seen: set[str] = set()
    components: list[list[str]] = []
    for node in nodes:
        if node in seen:
            continue
        comp = []
        q = deque([node])
        seen.add(node)
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nbr in adj.get(cur, set()):
                if nbr not in seen:
                    seen.add(nbr)
                    q.append(nbr)
        components.append(comp)
    return sorted(components, key=len, reverse=True)


def strongly_connected_components(
    nodes: Iterable[str],
    out_adj: dict[str, set[str]],
    in_adj: dict[str, set[str]],
) -> list[list[str]]:
    seen: set[str] = set()
    order: list[str] = []
    for node in nodes:
        if node in seen:
            continue
        stack: list[tuple[str, bool]] = [(node, False)]
        while stack:
            cur, expanded = stack.pop()
            if expanded:
                order.append(cur)
                continue
            if cur in seen:
                continue
            seen.add(cur)
            stack.append((cur, True))
            for nbr in out_adj.get(cur, set()):
                if nbr not in seen:
                    stack.append((nbr, False))

    seen.clear()
    components: list[list[str]] = []
    for node in reversed(order):
        if node in seen:
            continue
        comp = []
        stack = [node]
        seen.add(node)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nbr in in_adj.get(cur, set()):
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        components.append(comp)
    return sorted(components, key=len, reverse=True)


def component_path_metrics(
    component_nodes: list[str],
    adj: dict[str, set[str]],
    *,
    sample_limit: int,
) -> dict[str, object]:
    if not component_nodes:
        return {"avg_shortest_path": None, "diameter": None, "sampled": False}
    nodes = component_nodes
    sampled = False
    if len(nodes) > sample_limit:
        step = max(1, len(nodes) // sample_limit)
        nodes = nodes[::step][:sample_limit]
        sampled = True
    allowed = set(component_nodes)
    total = 0
    pairs = 0
    diameter = 0
    for source in nodes:
        dist = {source: 0}
        q = deque([source])
        while q:
            cur = q.popleft()
            for nbr in adj.get(cur, set()):
                if nbr in allowed and nbr not in dist:
                    dist[nbr] = dist[cur] + 1
                    q.append(nbr)
        total += sum(dist.values())
        pairs += len(dist) - 1
        if dist:
            diameter = max(diameter, max(dist.values()))
    return {
        "avg_shortest_path": total / pairs if pairs else None,
        "diameter": diameter,
        "sampled": sampled,
    }


def clustering_and_core(component_nodes: list[str], adj: dict[str, set[str]]) -> dict[str, object]:
    allowed = set(component_nodes)
    local_values: list[float] = []
    closed_triplets = 0
    triplets = 0
    for node in component_nodes:
        neighbors = list(adj.get(node, set()) & allowed)
        k = len(neighbors)
        if k < 2:
            continue
        links = 0
        for idx, left in enumerate(neighbors):
            left_neighbors = adj.get(left, set())
            for right in neighbors[idx + 1 :]:
                if right in left_neighbors:
                    links += 1
        local_values.append((2 * links) / (k * (k - 1)))
        closed_triplets += links
        triplets += k * (k - 1) // 2

    core = core_numbers(component_nodes, adj)
    max_core = max(core.values()) if core else 0
    return {
        "avg_local_clustering": _mean(local_values),
        "global_transitivity": (closed_triplets / triplets) if triplets else None,
        "max_core_number": max_core,
        "max_core_size": sum(1 for value in core.values() if value == max_core),
    }


def core_numbers(nodes: list[str], adj: dict[str, set[str]]) -> dict[str, int]:
    allowed = set(nodes)
    degree = {node: len(adj.get(node, set()) & allowed) for node in nodes}
    heap: list[tuple[int, str]] = []
    for node, deg in degree.items():
        heappush(heap, (deg, node))
    removed: set[str] = set()
    core: dict[str, int] = {}
    while heap:
        deg, node = heappop(heap)
        if node in removed or deg != degree[node]:
            continue
        removed.add(node)
        core[node] = deg
        for nbr in adj.get(node, set()):
            if nbr in allowed and nbr not in removed:
                degree[nbr] -= 1
                heappush(heap, (degree[nbr], nbr))
    return core


def reciprocity(graph: WeightedDiGraph) -> float:
    if graph.edge_count == 0:
        return 0.0
    reciprocal_directed = 0
    for source, target in graph.edge_weights:
        if (target, source) in graph.edge_weights:
            reciprocal_directed += 1
    return reciprocal_directed / graph.edge_count


def weighted_reciprocity(graph: WeightedDiGraph) -> float:
    total = graph.total_weight
    if total == 0:
        return 0.0
    reciprocal = 0.0
    for (source, target), weight in graph.edge_weights.items():
        opposite = graph.edge_weights.get((target, source), 0.0)
        reciprocal += min(weight, opposite)
    return reciprocal / total


def pagerank(graph: WeightedDiGraph, damping: float = 0.85, iterations: int = 50) -> dict[str, float]:
    nodes = list(graph.nodes)
    n = len(nodes)
    if n == 0:
        return {}
    rank = {node: 1.0 / n for node in nodes}
    out_weight = graph.out_strengths()
    incoming: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (source, target), weight in graph.edge_weights.items():
        incoming[target].append((source, weight))
    base = (1.0 - damping) / n
    for _ in range(iterations):
        dangling = sum(rank[node] for node in nodes if out_weight.get(node, 0.0) == 0.0)
        new_rank = {node: base + damping * dangling / n for node in nodes}
        for target in nodes:
            for source, weight in incoming.get(target, []):
                total = out_weight.get(source, 0.0)
                if total:
                    new_rank[target] += damping * rank[source] * (weight / total)
        rank = new_rank
    return rank


def top_counter(counter: Counter[str] | dict[str, float], n: int) -> list[dict[str, object]]:
    return [
        {"node": key, "value": value}
        for key, value in sorted(counter.items(), key=lambda item: item[1], reverse=True)[:n]
    ]


def _mean(values: list[float] | list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float] | list[int]) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return (vals[mid - 1] + vals[mid]) / 2.0
