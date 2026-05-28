"""
Reddit Hyperlink Network — Shared Graph Builder
================================================

Reusable functions for loading the SNAP Reddit hyperlink TSV and constructing
the canonical graph objects used across all RQ analysis files:

    body_df       — per-hyperlink DataFrame (raw, with parsed properties)
    weighted      — DiGraph, one edge per (source, target), weight = count
    raw_multi     — MultiDiGraph, one edge per hyperlink (timestamps preserved)
    pos_weighted  — weighted subgraph, positive-sentiment edges only
    neg_weighted  — weighted subgraph, negative-sentiment edges only

All graphs are cached to reddit/data/processed/graphs/*.pkl so subsequent
calls are fast. Pass force_rebuild=True to invalidate the cache.

Usage:
    from graph_builder import load_graphs
    graphs = load_graphs()
    G = graphs["weighted"]
"""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = SCRIPT_DIR / "data" / "reddit_hyperlinks"
CACHE_DIR = SCRIPT_DIR / "data" / "processed" / "graphs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BODY_TSV = RAW_DATA_DIR / "soc-redditHyperlinks-body.tsv"

N_PROPS = 65


# ---------------------------------------------------------------------------
# DataFrame loader
# ---------------------------------------------------------------------------
def load_body_df(force_rebuild: bool = False) -> pd.DataFrame:
    """Load the body hyperlink TSV and expand POST_PROPERTIES into 65 columns.

    Cached as reddit/data/processed/graphs/body_df.pkl.
    """
    cache = CACHE_DIR / "body_df.pkl"
    if cache.exists() and not force_rebuild:
        return pd.read_pickle(cache)

    if not BODY_TSV.exists():
        raise FileNotFoundError(
            f"Raw data missing: {BODY_TSV}. Run reddit_hyperlink_analysis.py once "
            f"to download the SNAP TSV files."
        )

    df = pd.read_csv(BODY_TSV, sep="\t")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")

    if "POST_PROPERTIES" not in df.columns and "PROPERTIES" in df.columns:
        df = df.rename(columns={"PROPERTIES": "POST_PROPERTIES"})

    prop_cols = [f"prop_{i}" for i in range(N_PROPS)]
    split = df["POST_PROPERTIES"].astype(str).str.split(",", expand=True)
    if split.shape[1] != N_PROPS:
        split = split.reindex(columns=range(N_PROPS))
    split.columns = prop_cols
    split = split.apply(pd.to_numeric, errors="coerce")

    df = pd.concat([df.drop(columns=["POST_PROPERTIES"]), split], axis=1)
    df.to_pickle(cache)
    return df


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------
def build_weighted(df: pd.DataFrame) -> nx.DiGraph:
    """Collapse (source, target) pairs into one weighted edge each."""
    agg = (df.groupby(["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"])
             .size()
             .reset_index(name="weight"))
    g = nx.DiGraph()
    g.add_weighted_edges_from(
        zip(agg["SOURCE_SUBREDDIT"], agg["TARGET_SUBREDDIT"], agg["weight"])
    )
    return g


def build_raw_multi(df: pd.DataFrame) -> nx.MultiDiGraph:
    """One edge per individual hyperlink; timestamps and sentiment preserved."""
    g = nx.MultiDiGraph()
    edges = zip(
        df["SOURCE_SUBREDDIT"].values,
        df["TARGET_SUBREDDIT"].values,
        df["TIMESTAMP"].values,
        df["LINK_SENTIMENT"].values,
    )
    g.add_edges_from(
        (s, t, {"timestamp": ts, "sentiment": int(sent)}) for s, t, ts, sent in edges
    )
    return g


def build_sentiment_subgraphs(df: pd.DataFrame) -> tuple[nx.DiGraph, nx.DiGraph]:
    """Return (positive-only, negative-only) weighted DiGraphs."""
    pos = build_weighted(df[df["LINK_SENTIMENT"] == 1])
    neg = build_weighted(df[df["LINK_SENTIMENT"] == -1])
    return pos, neg


# ---------------------------------------------------------------------------
# Cached loader for everything at once
# ---------------------------------------------------------------------------
def load_graphs(force_rebuild: bool = False) -> dict:
    """Return all canonical graph objects, building and caching as needed.

    Keys: body_df, weighted, raw_multi, pos_weighted, neg_weighted.
    """
    cache = CACHE_DIR / "all_graphs.pkl"
    if cache.exists() and not force_rebuild:
        with cache.open("rb") as f:
            return pickle.load(f)

    body_df = load_body_df(force_rebuild=force_rebuild)
    weighted = build_weighted(body_df)
    raw_multi = build_raw_multi(body_df)
    pos_weighted, neg_weighted = build_sentiment_subgraphs(body_df)

    result = {
        "body_df": body_df,
        "weighted": weighted,
        "raw_multi": raw_multi,
        "pos_weighted": pos_weighted,
        "neg_weighted": neg_weighted,
    }
    with cache.open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return result


if __name__ == "__main__":
    g = load_graphs()
    print(f"body_df:      {g['body_df'].shape}")
    print(f"weighted:     {g['weighted'].number_of_nodes():,} nodes, "
          f"{g['weighted'].number_of_edges():,} edges")
    print(f"raw_multi:    {g['raw_multi'].number_of_nodes():,} nodes, "
          f"{g['raw_multi'].number_of_edges():,} edges")
    print(f"pos_weighted: {g['pos_weighted'].number_of_nodes():,} nodes, "
          f"{g['pos_weighted'].number_of_edges():,} edges")
    print(f"neg_weighted: {g['neg_weighted'].number_of_nodes():,} nodes, "
          f"{g['neg_weighted'].number_of_edges():,} edges")
