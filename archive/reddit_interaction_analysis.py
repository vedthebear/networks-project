"""
Reddit Interaction Network — Setup & Exploration
================================================

End-to-end script that:
  1. Downloads & extracts the Stanford SNAP Reddit Interaction Network tarballs
     (chain-based and reply-based monthly user interaction networks for 2014)
  2. Loads all per-subreddit JSON files into structured Python objects
  3. Builds NetworkX graphs from the reply networks:
       - one aggregate weighted DiGraph per subreddit (cached to disk)
       - a weighted undirected subreddit-similarity graph (shared-user overlap)
       - a combined multi-subreddit DiGraph for the 5 most active subreddits
  4. Computes per-subreddit graph statistics
  5. Computes statistics on the subreddit-similarity network
  6. Produces a per-month temporal breakdown for the top 5 subreddits

Dataset: https://snap.stanford.edu/data/web-RedditNetworks.html

Outputs:
  data/reddit_interactions/
    reddit_chain_networks.tar.gz   (downloaded)
    reddit_reply_networks.tar.gz   (downloaded)
    chain/*.json                   (one file per subreddit)
    reply/*.json                   (one file per subreddit)
    graphs/<subreddit>.pkl         (aggregate weighted DiGraph per subreddit)
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tarfile
from collections import Counter
from itertools import combinations
from pathlib import Path


# ---------------------------------------------------------------------------
# Dependency bootstrap
# ---------------------------------------------------------------------------
def _ensure(pkg: str, import_name: str | None = None) -> None:
    name = import_name or pkg
    try:
        __import__(name)
    except ImportError:
        print(f"[setup] Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])


for _pkg, _imp in [("pandas", "pandas"), ("networkx", "networkx"),
                   ("requests", "requests"), ("tqdm", "tqdm")]:
    _ensure(_pkg, _imp)

import pandas as pd           # noqa: E402
import networkx as nx         # noqa: E402
import requests               # noqa: E402
from tqdm import tqdm         # noqa: E402


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "reddit_interactions"
CHAIN_DIR = DATA_DIR / "chain"
REPLY_DIR = DATA_DIR / "reply"
GRAPHS_DIR = DATA_DIR / "graphs"
for _d in (DATA_DIR, CHAIN_DIR, REPLY_DIR, GRAPHS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

URLS = {
    "chain": "https://snap.stanford.edu/data/reddit_chain_networks.tar.gz",
    "reply": "https://snap.stanford.edu/data/reddit_reply_networks.tar.gz",
}

EXTRACT_DIRS = {
    "chain": CHAIN_DIR,
    "reply": REPLY_DIR,
}


# =============================================================================
# SECTION 1 — Download & extract
# =============================================================================
def download_file(url: str, dest: Path) -> None:
    """Stream-download `url` to `dest`; skip if already present."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[1] Already have {dest.name} ({dest.stat().st_size/1e6:.1f} MB) — skipping download.")
        return
    print(f"[1] Downloading {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) or None
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    print(f"[1] Saved to {dest}")


def extract_tarball(archive: Path, target_dir: Path) -> None:
    """Extract `archive` into `target_dir`; skip if target already has JSON files."""
    existing = list(target_dir.glob("*.json"))
    if existing:
        print(f"[1] {target_dir.name}/ already contains {len(existing)} JSON files — skipping extraction.")
        return
    print(f"[1] Extracting {archive.name} → {target_dir} ...")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for m in tqdm(members, desc=f"extract {archive.name}", unit="file"):
            # Flatten: if the archive contains a nested folder, pull just the JSON
            # files up into target_dir. Skip directory entries.
            if m.isdir():
                continue
            if not m.name.lower().endswith(".json"):
                continue
            m.name = Path(m.name).name  # strip any internal directory prefix
            tar.extract(m, target_dir)
    n_extracted = len(list(target_dir.glob("*.json")))
    print(f"[1] Extracted {n_extracted} JSON files to {target_dir}")


print("\n" + "=" * 72)
print("SECTION 1 — Download & extract")
print("=" * 72)

archive_paths = {}
for key, url in URLS.items():
    dest = DATA_DIR / Path(url).name
    download_file(url, dest)
    archive_paths[key] = dest

for key, archive in archive_paths.items():
    extract_tarball(archive, EXTRACT_DIRS[key])


# =============================================================================
# SECTION 2 — Load & parse
# =============================================================================
def _clean_monthly_net(net: dict | None) -> dict[str, list[str]]:
    """Lowercase usernames, drop self-loops, drop empty target lists."""
    if not net:
        return {}
    cleaned: dict[str, list[str]] = {}
    for u, targets in net.items():
        if not targets:
            continue
        u_l = u.lower()
        out = [t.lower() for t in targets if t and t.lower() != u_l]
        if out:
            cleaned[u_l] = out
    return cleaned


def load_subreddit_jsons(directory: Path) -> dict[str, list[dict]]:
    """Load every *.json in `directory` as a list of monthly nets keyed by subreddit."""
    files = sorted(directory.glob("*.json"))
    print(f"[2] Reading {len(files)} JSON files from {directory.name}/ ...")
    out: dict[str, list[dict]] = {}
    empty_months = 0
    for f in tqdm(files, desc=f"load {directory.name}", unit="file"):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            print(f"    [warn] could not load {f.name}: {e}")
            continue
        if not isinstance(raw, list):
            print(f"    [warn] {f.name}: expected list of monthly nets, got {type(raw).__name__}")
            continue
        months: list[dict] = []
        for i, m in enumerate(raw):
            cleaned = _clean_monthly_net(m)
            if not cleaned:
                empty_months += 1
            months.append(cleaned)
        out[f.stem] = months
    if empty_months:
        print(f"    [warn] {empty_months} subreddit-month networks were empty/null.")
    return out


print("\n" + "=" * 72)
print("SECTION 2 — Load & parse")
print("=" * 72)

reply_data = load_subreddit_jsons(REPLY_DIR)
chain_data = load_subreddit_jsons(CHAIN_DIR)

# Report totals
all_users_reply: set[str] = set()
for months in reply_data.values():
    for m in months:
        all_users_reply.update(m.keys())
        for tgts in m.values():
            all_users_reply.update(tgts)

all_users_chain: set[str] = set()
for months in chain_data.values():
    for m in months:
        all_users_chain.update(m.keys())
        for tgts in m.values():
            all_users_chain.update(tgts)

print(f"\n[2] reply: {len(reply_data)} subreddits, {len(all_users_reply):,} unique users")
print(f"[2] chain: {len(chain_data)} subreddits, {len(all_users_chain):,} unique users")


# =============================================================================
# SECTION 3 — Build NetworkX graphs (from reply networks)
# =============================================================================
print("\n" + "=" * 72)
print("SECTION 3 — NetworkX graphs (reply)")
print("=" * 72)


def _info(name: str, g: nx.Graph) -> None:
    print(f"  {name:40s}  nodes={g.number_of_nodes():>7d}  edges={g.number_of_edges():>8d}")


# nx.write_gpickle / nx.read_gpickle were removed in NetworkX 3.0.
# Honor the spirit of the brief: try the NetworkX-native API first, fall back to
# plain pickle so the script works on both 2.x and 3.x.
def _gpickle_write(g: nx.Graph, path: Path) -> None:
    fn = getattr(nx, "write_gpickle", None)
    if fn is not None:
        fn(g, path)
    else:
        with open(path, "wb") as fh:
            pickle.dump(g, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _gpickle_read(path: Path) -> nx.Graph:
    fn = getattr(nx, "read_gpickle", None)
    if fn is not None:
        return fn(path)
    with open(path, "rb") as fh:
        return pickle.load(fh)


# --- 3.1 Aggregate weighted DiGraph per subreddit -------------------------
def build_aggregate_graph(months: list[dict]) -> nx.DiGraph:
    """Collapse 11 monthly reply dicts into a single weighted DiGraph."""
    counter: Counter = Counter()
    for m in months:
        for u, tgts in m.items():
            for v in tgts:
                counter[(u, v)] += 1
    g = nx.DiGraph()
    g.add_weighted_edges_from((u, v, w) for (u, v), w in counter.items())
    return g


print("[3.1] Building / loading aggregate DiGraph per subreddit ...")
existing_pkls = list(GRAPHS_DIR.glob("*.pkl"))
subreddit_graphs: dict[str, nx.DiGraph] = {}

if existing_pkls:
    print(f"[3.1] Found {len(existing_pkls)} cached graphs in {GRAPHS_DIR.name}/ — loading from disk.")
    for p in tqdm(existing_pkls, desc="load .pkl", unit="file"):
        subreddit_graphs[p.stem] = _gpickle_read(p)
else:
    print(f"[3.1] No cache found — building fresh from {len(reply_data)} subreddits.")
    for name, months in tqdm(reply_data.items(), desc="aggregate", unit="sub"):
        g = build_aggregate_graph(months)
        subreddit_graphs[name] = g
        _gpickle_write(g, GRAPHS_DIR / f"{name}.pkl")
    print(f"[3.1] Wrote {len(subreddit_graphs)} graphs to {GRAPHS_DIR}")

total_nodes_311 = sum(g.number_of_nodes() for g in subreddit_graphs.values())
total_edges_311 = sum(g.number_of_edges() for g in subreddit_graphs.values())
print(f"[3.1] Total across subreddits: nodes={total_nodes_311:,}  edges={total_edges_311:,}")


# --- 3.2 Subreddit similarity graph (shared-user overlap) -----------------
print("[3.2] Building subreddit-similarity Graph (shared users >= 10) ...")
user_sets: dict[str, set[str]] = {name: set(g.nodes()) for name, g in subreddit_graphs.items()}
similarity = nx.Graph()
similarity.add_nodes_from(user_sets.keys())

MIN_SHARED = 10
sub_names = list(user_sets.keys())
for a, b in tqdm(list(combinations(sub_names, 2)), desc="pairs", unit="pair"):
    shared = len(user_sets[a] & user_sets[b])
    if shared >= MIN_SHARED:
        similarity.add_edge(a, b, weight=shared)
_info("subreddit similarity (undirected)", similarity)


# --- 3.3 Combined multi-subreddit DiGraph (top 5 by edges) ----------------
print("[3.3] Building combined MultiDiGraph from top-5 subreddits by edge count ...")
top5 = sorted(subreddit_graphs.items(), key=lambda kv: kv[1].number_of_edges(), reverse=True)[:5]
top5_names = [name for name, _ in top5]
print(f"[3.3] Top 5 subreddits: {top5_names}")

combined = nx.MultiDiGraph()
for name, g in top5:
    for u, v, data in g.edges(data=True):
        combined.add_edge(u, v, subreddit=name, weight=data.get("weight", 1))
_info("combined top-5 multi-subreddit", combined)


# =============================================================================
# SECTION 4 — Per-subreddit statistics
# =============================================================================
print("\n" + "=" * 72)
print("SECTION 4 — Per-subreddit statistics")
print("=" * 72)


def reciprocity_fraction(g: nx.DiGraph) -> float:
    n_edges = g.number_of_edges()
    if n_edges == 0:
        return 0.0
    reciprocal = sum(1 for u, v in g.edges() if g.has_edge(v, u))
    return reciprocal / n_edges


def top_by_weighted_degree(g: nx.DiGraph, direction: str) -> str:
    iter_fn = g.out_degree if direction == "out" else g.in_degree
    best_user, best_w = "", -1
    for u, w in iter_fn(weight="weight"):
        if w > best_w:
            best_user, best_w = u, w
    return best_user


print("[4] Computing per-subreddit stats ...")
rows = []
for name, g in tqdm(subreddit_graphs.items(), desc="stats", unit="sub"):
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    if n_nodes == 0:
        continue
    density = nx.density(g)
    reciprocity = reciprocity_fraction(g)
    top_out = top_by_weighted_degree(g, "out") if n_edges else ""
    top_in = top_by_weighted_degree(g, "in") if n_edges else ""
    wcc = list(nx.weakly_connected_components(g))
    n_wcc = len(wcc)
    largest_wcc = max((len(c) for c in wcc), default=0)
    rows.append({
        "subreddit": name,
        "nodes": n_nodes,
        "edges": n_edges,
        "density": density,
        "reciprocity": reciprocity,
        "top_out_user": top_out,
        "top_in_user": top_in,
        "n_wcc": n_wcc,
        "largest_wcc_frac": largest_wcc / n_nodes,
    })

stats_df = pd.DataFrame(rows).sort_values("edges", ascending=False).reset_index(drop=True)

print("\n[4] Summary table (all subreddits, sorted by edges desc):")
with pd.option_context("display.max_rows", 200, "display.width", 160):
    print(stats_df.to_string(index=False,
                             formatters={"density": "{:.3e}".format,
                                         "reciprocity": "{:.3f}".format,
                                         "largest_wcc_frac": "{:.3f}".format}))

print("\n[4] Detailed stats for top 5 most active subreddits:")
for _, row in stats_df.head(5).iterrows():
    print(f"\n  --- {row['subreddit']} ---")
    print(f"  nodes              : {row['nodes']:,}")
    print(f"  edges              : {row['edges']:,}")
    print(f"  density            : {row['density']:.3e}")
    print(f"  reciprocity        : {row['reciprocity']:.3f}")
    print(f"  top out-degree user: {row['top_out_user']}")
    print(f"  top in-degree user : {row['top_in_user']}")
    print(f"  # WCCs             : {row['n_wcc']}")
    print(f"  largest WCC frac   : {row['largest_wcc_frac']:.3f}")


# =============================================================================
# SECTION 5 — Subreddit similarity network statistics
# =============================================================================
print("\n" + "=" * 72)
print("SECTION 5 — Subreddit similarity network")
print("=" * 72)

S = similarity
print(f"[5] nodes={S.number_of_nodes()}  edges={S.number_of_edges()}")
print(f"[5] density: {nx.density(S):.3e}")

cc = list(nx.connected_components(S))
largest_cc = max((len(c) for c in cc), default=0)
print(f"[5] # connected components: {len(cc)}")
print(f"[5] largest CC size       : {largest_cc} "
      f"({(largest_cc / S.number_of_nodes() * 100 if S.number_of_nodes() else 0):.2f}% of nodes)")

weighted_deg = pd.Series(dict(S.degree(weight="weight")), name="weighted_degree")
weighted_deg = weighted_deg.sort_values(ascending=False).head(10)
print("\n[5] Top 10 subreddits by weighted degree (most user overlap):")
print(weighted_deg.to_string())


# =============================================================================
# SECTION 6 — Temporal analysis (top 5 subreddits, per-month)
# =============================================================================
print("\n" + "=" * 72)
print("SECTION 6 — Temporal analysis (top 5 subreddits, per-month)")
print("=" * 72)


def monthly_stats(month_net: dict[str, list[str]]) -> tuple[int, int, float]:
    """Return (active_users, edges, reciprocity_fraction) for one month."""
    if not month_net:
        return 0, 0, 0.0
    edge_set: set[tuple[str, str]] = set()
    users: set[str] = set()
    for u, tgts in month_net.items():
        users.add(u)
        for v in tgts:
            users.add(v)
            edge_set.add((u, v))
    n_edges = len(edge_set)
    if n_edges == 0:
        return len(users), 0, 0.0
    recip = sum(1 for (u, v) in edge_set if (v, u) in edge_set)
    return len(users), n_edges, recip / n_edges


temporal_rows = []
for name in top5_names:
    months = reply_data.get(name, [])
    for i, m in enumerate(months, start=1):
        au, ne, rf = monthly_stats(m)
        temporal_rows.append({
            "subreddit": name,
            "month": i,
            "active_users": au,
            "edges": ne,
            "reciprocity": rf,
        })

temporal_df = pd.DataFrame(temporal_rows)

for name in top5_names:
    sub = temporal_df[temporal_df["subreddit"] == name].drop(columns=["subreddit"])
    print(f"\n[6] {name} — month-by-month:")
    print(sub.to_string(index=False,
                        formatters={"reciprocity": "{:.3f}".format}))


# =============================================================================
# Final summary
# =============================================================================
total_edges_all = sum(g.number_of_edges() for g in subreddit_graphs.values())

print("\nDone.")
print(f"  subreddits processed : {len(subreddit_graphs)}")
print(f"  unique users seen    : {len(all_users_reply):,}")
print(f"  total aggregate edges: {total_edges_all:,}")
