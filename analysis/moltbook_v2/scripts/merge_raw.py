#!/usr/bin/env python3
"""
merge_raw.py -- merge every raw posts JSONL into one deduped master file.

Streams through all raw sources (the mega/ workers + the earlier balanced
scrape), dedups by post id, and writes data/raw/posts_master.jsonl. Memory
stays bounded: we only hold the set of seen ids, not the posts.
"""
import json, glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

def main():
    sources = sorted(glob.glob(str(RAW / "mega" / "*.jsonl")))
    extra = RAW / "posts_balanced.jsonl"
    if extra.exists():
        sources.append(str(extra))
    print(f"merging {len(sources)} source files:")
    for s in sources:
        print("  ", Path(s).name)

    out = RAW / "posts_master.jsonl"
    seen = set()
    kept = total = 0
    with out.open("w") as w:
        for src in sources:
            for line in open(src):
                if not line.strip():
                    continue
                total += 1
                try:
                    pid = json.loads(line).get("id")
                except Exception:
                    continue
                if pid and pid not in seen:
                    seen.add(pid)
                    w.write(line if line.endswith("\n") else line + "\n")
                    kept += 1
            print(f"  after {Path(src).name}: {kept:,} unique / {total:,} read")
    print(f"\nMASTER: {kept:,} unique posts -> {out}")

if __name__ == "__main__":
    main()
