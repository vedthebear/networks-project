"""
Reddit Hyperlink Network — Setup & Exploration
==============================================

End-to-end script that:
  1. Downloads the Stanford SNAP Reddit Hyperlink Network TSVs
  2. Loads and parses them into pandas DataFrames (expanding POST_PROPERTIES)
  3. Builds four directed NetworkX graphs from the body data
  4. Computes basic graph statistics on the weighted collapsed graph

Dataset: https://snap.stanford.edu/data/soc-RedditHyperlinks.html
"""

from __future__ import annotations

import os
import subprocess
import sys
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
DATA_DIR = SCRIPT_DIR / "data" / "reddit_hyperlinks"
DATA_DIR.mkdir(parents=True, exist_ok=True)

URLS = {
    "body":  "https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv",
    "title": "https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv",
}


# =============================================================================
# SECTION 1 — Download the data
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


print("\n" + "=" * 72)
print("SECTION 1 — Download")
print("=" * 72)
paths = {}
for key, url in URLS.items():
    dest = DATA_DIR / Path(url).name
    download_file(url, dest)
    paths[key] = dest


# =============================================================================
# SECTION 2 — Load and parse the data
# =============================================================================
N_PROPS = 65


def load_hyperlink_tsv(path: Path) -> pd.DataFrame:
    """Load a Reddit Hyperlink TSV and expand POST_PROPERTIES into prop_0…prop_64."""
    print(f"[2] Reading {path.name} ...")
    df = pd.read_csv(path, sep="\t")

    # Parse TIMESTAMP; coerce malformed entries to NaT and report.
    ts = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    n_bad_ts = ts.isna().sum()
    if n_bad_ts:
        print(f"    [warn] {n_bad_ts} TIMESTAMP values failed to parse (set to NaT).")
    df["TIMESTAMP"] = ts

    # The TSV header uses PROPERTIES; the user brief calls it POST_PROPERTIES.
    # Normalize to POST_PROPERTIES.
    if "POST_PROPERTIES" not in df.columns and "PROPERTIES" in df.columns:
        df = df.rename(columns={"PROPERTIES": "POST_PROPERTIES"})

    # Split POST_PROPERTIES (comma-separated 65 floats) into prop_0..prop_64.
    prop_cols = [f"prop_{i}" for i in range(N_PROPS)]
    split = df["POST_PROPERTIES"].astype(str).str.split(",", expand=True)

    # Handle edge case: row with unexpected number of fields.
    actual_cols = split.shape[1]
    if actual_cols != N_PROPS:
        print(f"    [warn] POST_PROPERTIES yielded {actual_cols} fields "
              f"(expected {N_PROPS}); reindexing with NaN fill.")
        split = split.reindex(columns=range(N_PROPS))

    # Per-row length mismatch (some rows short/long even if max is 65).
    row_lengths = df["POST_PROPERTIES"].astype(str).str.count(",") + 1
    n_mismatch = int((row_lengths != N_PROPS).sum())
    if n_mismatch:
        print(f"    [warn] {n_mismatch} rows had != {N_PROPS} comma-separated values.")

    split.columns = prop_cols
    split = split.apply(pd.to_numeric, errors="coerce")

    df = pd.concat([df.drop(columns=["POST_PROPERTIES"]), split], axis=1)
    return df


print("\n" + "=" * 72)
print("SECTION 2 — Load & parse")
print("=" * 72)

body_df = load_hyperlink_tsv(paths["body"])
title_df = load_hyperlink_tsv(paths["title"])

print(f"\n[body]  shape: {body_df.shape}")
print(body_df.head())
print(f"\n[title] shape: {title_df.shape}")
print(title_df.head())


# =============================================================================
# SECTION 3 — Build NetworkX graphs from the body DataFrame
# =============================================================================
print("\n" + "=" * 72)
print("SECTION 3 — NetworkX graphs (body)")
print("=" * 72)


def _info(name: str, g: nx.Graph) -> None:
    print(f"  {name:40s}  nodes={g.number_of_nodes():>7d}  edges={g.number_of_edges():>8d}")


# --- 3.1 Raw multigraph: every hyperlink preserved as its own edge --------
print("[3.1] Building raw MultiDiGraph ...")
raw_multi = nx.MultiDiGraph()
edges_iter = zip(
    body_df["SOURCE_SUBREDDIT"].values,
    body_df["TARGET_SUBREDDIT"].values,
    body_df["TIMESTAMP"].values,
    body_df["LINK_SENTIMENT"].values,
)
raw_multi.add_edges_from(
    (s, t, {"timestamp": ts, "sentiment": int(sent)}) for s, t, ts, sent in edges_iter
)
_info("raw multigraph", raw_multi)


# --- 3.2 Weighted collapsed graph: one edge per (source,target) -----------
def build_weighted(df: pd.DataFrame) -> nx.DiGraph:
    """Collapse (source, target) pairs into a single weighted edge."""
    agg = (df.groupby(["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"])
             .size()
             .reset_index(name="weight"))
    g = nx.DiGraph()
    g.add_weighted_edges_from(
        zip(agg["SOURCE_SUBREDDIT"], agg["TARGET_SUBREDDIT"], agg["weight"])
    )
    return g


print("[3.2] Building weighted collapsed DiGraph ...")
weighted = build_weighted(body_df)
_info("weighted collapsed", weighted)


# --- 3.3 Positive-only weighted subgraph ----------------------------------
print("[3.3] Building positive-only weighted DiGraph ...")
pos_weighted = build_weighted(body_df[body_df["LINK_SENTIMENT"] == 1])
_info("positive-only weighted", pos_weighted)


# --- 3.4 Negative-only weighted subgraph ----------------------------------
print("[3.4] Building negative-only weighted DiGraph ...")
neg_weighted = build_weighted(body_df[body_df["LINK_SENTIMENT"] == -1])
_info("negative-only weighted", neg_weighted)


# =============================================================================
# SECTION 4 — Graph statistics on the weighted collapsed graph
# =============================================================================
print("\n" + "=" * 72)
print("SECTION 4 — Statistics on weighted collapsed graph")
print("=" * 72)

G = weighted
n_nodes = G.number_of_nodes()


def top_n_series(pairs, name: str, n: int = 20) -> pd.Series:
    s = pd.Series(dict(pairs), name=name).sort_values(ascending=False).head(n)
    return s


# --- 4.1 Top 20 in-degree (most cited) ------------------------------------
in_deg_top = top_n_series(G.in_degree(weight="weight"), "weighted_in_degree")
print("\n[4.1] Top 20 subreddits by weighted in-degree (most cited):")
print(in_deg_top.to_string())


# --- 4.2 Top 20 out-degree (most citing) ----------------------------------
out_deg_top = top_n_series(G.out_degree(weight="weight"), "weighted_out_degree")
print("\n[4.2] Top 20 subreddits by weighted out-degree (most citing):")
print(out_deg_top.to_string())


# --- 4.3 Top 20 PageRank (weighted) ---------------------------------------
print("\n[4.3] Computing PageRank (weighted) ...")
pr = nx.pagerank(G, weight="weight")
pr_top = pd.Series(pr, name="pagerank").sort_values(ascending=False).head(20)
print("Top 20 subreddits by weighted PageRank:")
print(pr_top.to_string())


# --- 4.4 Top 20 betweenness centrality (approximated) ---------------------
k_sample = min(500, n_nodes)
print(f"\n[4.4] Computing approximate betweenness centrality "
      f"(weighted, k={k_sample} pivot nodes, seed=42). This may take a few minutes ...")
bc = nx.betweenness_centrality(G, k=k_sample, weight="weight", seed=42)
bc_top = pd.Series(bc, name="betweenness_approx").sort_values(ascending=False).head(20)
print("Top 20 subreddits by approximate weighted betweenness centrality:")
print(bc_top.to_string())
print("  (Note: sampled approximation — full computation is infeasible on this graph.)")


# --- 4.5 Density and weakly connected components -------------------------
density = nx.density(G)
n_wcc = nx.number_weakly_connected_components(G)
largest_wcc = max(nx.weakly_connected_components(G), key=len)
print("\n[4.5] Overall structural summary:")
print(f"  density                        : {density:.3e}")
print(f"  # weakly connected components  : {n_wcc}")
print(f"  size of largest WCC            : {len(largest_wcc)} "
      f"({len(largest_wcc)/n_nodes*100:.2f}% of all nodes)")

print("\nDone.")
