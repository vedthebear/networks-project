#!/usr/bin/env python3
"""
build_tables.py -- normalize raw scraped posts into tidy analysis tables.

Inputs:
  data/raw/posts_balanced.jsonl   (one post per line)
  data/raw/submolts.jsonl         (submolt metadata)

Outputs (data/tables/):
  posts.csv          post_id, author, submolt, created_at, score, comment_count, title
  agents.csv         agent, n_posts, n_submolts, submolts (pipe-joined)
  submolts.csv       submolt, n_posts, n_authors, subscriber_count, post_count_platform
  membership.csv     agent, submolt, n_posts        (bipartite edge list: who posts where)

This is the clean, documented base every downstream graph/analysis reads from.
"""
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TAB = ROOT / "data" / "tables"

def author_name(p):
    a = p.get("author")
    if isinstance(a, dict):
        return a.get("name") or a.get("id")
    return a or p.get("author_id")

def submolt_name(p):
    s = p.get("submolt")
    if isinstance(s, dict):
        return s.get("name")
    return p.get("submolt_name") or s

def main():
    TAB.mkdir(parents=True, exist_ok=True)
    # Prefer the merged master (full mega-scrape); fall back to the balanced file.
    posts_path = RAW / "posts_master.jsonl"
    if not posts_path.exists():
        posts_path = RAW / "posts_balanced.jsonl"
    rows = []
    seen = set()
    for line in posts_path.open():
        if not line.strip():
            continue
        p = json.loads(line)
        pid = p.get("id")
        if pid in seen:           # de-dup (a post can appear via overlapping pages)
            continue
        seen.add(pid)
        rows.append({
            "post_id": pid,
            "author": author_name(p),
            "submolt": submolt_name(p),
            "created_at": p.get("created_at"),
            "score": p.get("score", (p.get("upvotes", 0) - p.get("downvotes", 0))),
            "comment_count": p.get("comment_count"),
            "title": (p.get("title") or "").replace("\n", " "),
        })
    posts = pd.DataFrame(rows).dropna(subset=["author", "submolt"])
    posts.to_csv(TAB / "posts.csv", index=False)
    print(f"posts.csv: {len(posts)} posts | {posts.author.nunique()} agents | {posts.submolt.nunique()} submolts")

    # membership (bipartite edge list): agent x submolt -> post count
    mem = (posts.groupby(["author", "submolt"]).size()
                .reset_index(name="n_posts"))
    mem.columns = ["agent", "submolt", "n_posts"]
    mem.to_csv(TAB / "membership.csv", index=False)
    print(f"membership.csv: {len(mem)} agent-submolt pairs")

    # agents
    ag = (mem.groupby("agent")
             .agg(n_posts=("n_posts", "sum"), n_submolts=("submolt", "nunique"))
             .reset_index())
    sub_by_agent = mem.groupby("agent")["submolt"].apply(lambda s: "|".join(sorted(s)))
    ag["submolts"] = ag.agent.map(sub_by_agent)
    ag.sort_values("n_posts", ascending=False).to_csv(TAB / "agents.csv", index=False)
    print(f"agents.csv: {len(ag)} agents | {(ag.n_submolts>1).sum()} active in >1 submolt")

    # submolts (+ platform metadata if available)
    meta = {}
    sp = RAW / "submolts.jsonl"
    if sp.exists():
        for line in sp.open():
            s = json.loads(line)
            meta[s.get("name")] = (s.get("subscriber_count"), s.get("post_count"))
    sub = (posts.groupby("submolt")
              .agg(n_posts=("post_id", "count"), n_authors=("author", "nunique"))
              .reset_index())
    sub["subscriber_count"] = sub.submolt.map(lambda n: (meta.get(n) or (None, None))[0])
    sub["post_count_platform"] = sub.submolt.map(lambda n: (meta.get(n) or (None, None))[1])
    sub.sort_values("n_authors", ascending=False).to_csv(TAB / "submolts.csv", index=False)
    print(f"submolts.csv: {len(sub)} submolts")

if __name__ == "__main__":
    main()
