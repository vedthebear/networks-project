#!/usr/bin/env python3
"""
scrape_depth.py -- deepen coverage of the ACTIVE submolts.

The broad pass (scrape_balanced.py) covered 12k submolts at ~100 posts each,
which is great for breadth but captures too few authors per community for a
robust shared-agent projection. This pass re-scrapes the submolts that
actually have activity, at a much higher quota, so each community's agent set
is well sampled and cross-submolt overlap (the edge fuel) becomes reliable.

Active submolts are taken from data/tables/submolts.csv (those with >= --min-posts
posts in the broad pass). Posts are appended to the SAME posts_balanced.jsonl;
build_tables.py de-dups by post_id, so breadth + depth merge cleanly.

Usage:
  python scripts/scrape_depth.py --min-posts 20 --quota 600
"""
import argparse, json, time
from pathlib import Path
import pandas as pd
from moltbook_api import Moltbook

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TAB = ROOT / "data" / "tables"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-posts", type=int, default=20,
                    help="only deepen submolts that had >= this many posts in the broad pass")
    ap.add_argument("--quota", type=int, default=600, help="target posts per active submolt")
    ap.add_argument("--sleep", type=float, default=0.6)
    args = ap.parse_args()

    subs = pd.read_csv(TAB / "submolts.csv")
    active = subs[subs.n_posts >= args.min_posts].sort_values("n_posts", ascending=False)
    names = list(active.submolt)
    print(f"deepening {len(names)} active submolts (>= {args.min_posts} posts) at quota {args.quota}")

    posts_path = RAW / "posts_balanced.jsonl"
    done_path = RAW / "depth_done.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()

    mb = Moltbook(sleep=args.sleep)
    pf = posts_path.open("a"); df = done_path.open("a")
    added = 0
    for i, name in enumerate(names, 1):
        if name in done:
            continue
        try:
            posts = mb.submolt_feed(name, quota=args.quota)
        except Exception as e:
            print(f"  [{i}/{len(names)}] {name}: ERROR {e}"); continue
        for p in posts:
            pf.write(json.dumps(p, ensure_ascii=False) + "\n")
        pf.flush()
        df.write(name + "\n"); df.flush()
        added += len(posts)
        print(f"  [{i}/{len(names)}] {name}: {len(posts)} posts (added {added})")
        time.sleep(args.sleep)
    pf.close(); df.close()
    print(f"DONE. depth pass added ~{added} post-rows (pre-dedup).")

if __name__ == "__main__":
    main()
