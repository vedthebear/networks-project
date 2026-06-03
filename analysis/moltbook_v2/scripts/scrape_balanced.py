#!/usr/bin/env python3
"""
scrape_balanced.py -- collect a community-balanced post sample from Moltbook.

Strategy (chosen for a fair community-level comparison):
  1. Enumerate the top-N submolts by popularity.
  2. From EACH submolt, pull a fixed quota of posts (default 200).
This prevents a few giant communities (e.g. 'general') from dominating the
shared-agent graph, so every community is represented on comparable footing.

Checkpointing:
  - Posts are appended to data/raw/posts_balanced.jsonl as they arrive.
  - data/raw/done_submolts.txt records finished submolts so re-runs resume.

Usage:
  python moltbook_v2/scripts/scrape_balanced.py --submolts 800 --quota 200
"""
import argparse, json, time
from pathlib import Path
from moltbook_api import Moltbook

ROOT = Path(__file__).resolve().parents[1]            # moltbook_v2/
RAW = ROOT / "data" / "raw"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submolts", type=int, default=800, help="top-N submolts to cover")
    ap.add_argument("--quota", type=int, default=200, help="posts per submolt")
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    posts_path = RAW / "posts_balanced.jsonl"
    done_path = RAW / "done_submolts.txt"
    subs_path = RAW / "submolts.jsonl"

    done = set(done_path.read_text().split()) if done_path.exists() else set()
    mb = Moltbook(sleep=args.sleep)

    print(f"enumerating top {args.submolts} submolts...")
    submolts = mb.list_submolts(max_n=args.submolts)
    # persist the submolt metadata (subscriber_count, post_count, etc.)
    with subs_path.open("w") as f:
        for s in submolts:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"got {len(submolts)} submolts; {len(done)} already scraped")

    pf = posts_path.open("a")
    df = done_path.open("a")
    total_posts = sum(1 for _ in posts_path.open()) if posts_path.exists() else 0
    for i, s in enumerate(submolts, 1):
        name = s.get("name")
        if not name or name in done:
            continue
        try:
            posts = mb.submolt_feed(name, quota=args.quota)
        except Exception as e:
            print(f"  [{i}/{len(submolts)}] {name}: ERROR {e}")
            continue
        for p in posts:
            pf.write(json.dumps(p, ensure_ascii=False) + "\n")
        pf.flush()
        df.write(name + "\n"); df.flush()
        total_posts += len(posts)
        if i % 25 == 0 or len(posts) > 0:
            print(f"  [{i}/{len(submolts)}] {name}: +{len(posts)} posts (total {total_posts})")
        time.sleep(args.sleep)
    pf.close(); df.close()
    print(f"DONE. total posts on disk: {total_posts}")

if __name__ == "__main__":
    main()
