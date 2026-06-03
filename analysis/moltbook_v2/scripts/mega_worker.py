#!/usr/bin/env python3
"""
mega_worker.py -- one worker process in the overnight max-volume scrape.

Modes (pick one via --mode):
  global-new     : cursor-paginate /posts?sort=new  as deep as the API serves
                   (the recent firehose; auto-covers every submolt).
  global-top     : cursor-paginate /posts?sort=top  back to platform genesis
                   (all-time, score-ranked).
  submolt-shard  : for a shard of known submolts, scrape each under ALL sort
                   orders (new/top/hot/old/controversial) with page pagination.
                   Each sort surfaces a mostly-different post set, maximizing
                   per-community yield.

Every worker:
  - reads the API key from the workspace .env / credentials.json,
  - appends posts (one JSON per line) to its own file in data/raw/mega/,
  - checkpoints (cursor file or done-submolt list) so it resumes after a stop,
  - dedups within its own stream by post id,
  - relies on the rate-limit budget being shared across ~5 workers; keep --sleep
    so the COMBINED rate stays under 600 req/min (5 workers x 0.6s ~= 500/min).
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from moltbook_api import Moltbook

ROOT = Path(__file__).resolve().parents[1]
MEGA = ROOT / "data" / "raw" / "mega"
SORTS = ["new", "top", "hot", "old", "controversial"]

def load_seen(path):
    seen = set()
    if path.exists():
        for line in path.open():
            try: seen.add(json.loads(line)["id"])
            except Exception: pass
    return seen

def global_stream(mb, sort, sleep):
    out = MEGA / f"global_{sort}.jsonl"
    ckpt = MEGA / f"global_{sort}.cursor"
    seen = load_seen(out)
    cursor = ckpt.read_text().strip() if ckpt.exists() else None
    f = out.open("a")
    n = len(seen); empty_pages = 0
    print(f"[global-{sort}] resuming at {n} posts", flush=True)
    while True:
        path = f"/posts?sort={sort}&limit=100" + (f"&cursor={cursor}" if cursor else "")
        try:
            d = mb.get(path)
        except Exception as e:
            print(f"[global-{sort}] fatal get error {e}; sleeping 30", flush=True); time.sleep(30); continue
        batch = d.get("posts", [])
        fresh = [p for p in batch if p.get("id") not in seen]
        for p in fresh:
            seen.add(p["id"]); f.write(json.dumps(p, ensure_ascii=False) + "\n")
        f.flush()
        n += len(fresh)
        cursor = d.get("next_cursor")
        if cursor: ckpt.write_text(cursor)
        if not fresh:
            empty_pages += 1
        else:
            empty_pages = 0
        if n % 2000 < 100:
            print(f"[global-{sort}] {n} posts", flush=True)
        if not d.get("has_more") or not cursor or empty_pages >= 3:
            print(f"[global-{sort}] DONE at {n} posts (has_more={d.get('has_more')})", flush=True)
            break
        time.sleep(sleep)
    f.close()

def submolt_shard(mb, shard, nshards, sleep):
    import pandas as pd
    subs = list(pd.read_csv(ROOT / "data" / "tables" / "submolts.csv").submolt)
    mine = [s for i, s in enumerate(subs) if i % nshards == shard]
    out = MEGA / f"submolt_shard{shard}.jsonl"
    done_path = MEGA / f"submolt_shard{shard}.done"
    done = set(done_path.read_text().split("\n")) if done_path.exists() else set()
    seen = load_seen(out)
    f = out.open("a"); df = done_path.open("a")
    print(f"[shard{shard}] {len(mine)} submolts, {len(done)} done, {len(seen)} posts", flush=True)
    for name in mine:
        for sort in SORTS:
            key = f"{name}|{sort}"
            if key in done:
                continue
            page = 1
            while True:
                try:
                    d = mb.get(f"/submolts/{name}/feed?limit=100&sort={sort}&page={page}")
                except Exception:
                    break
                batch = d.get("posts", [])
                fresh = [p for p in batch if p.get("id") not in seen]
                for p in fresh:
                    seen.add(p["id"]); p.setdefault("submolt_name", name)
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
                f.flush()
                if not batch or not fresh:
                    break
                page += 1
                time.sleep(sleep)
            df.write(key + "\n"); df.flush()
        print(f"[shard{shard}] {name}: total posts now {len(seen)}", flush=True)
    f.close(); df.close()
    print(f"[shard{shard}] DONE, {len(seen)} posts", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["global-new", "global-top", "submolt-shard"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.6)
    args = ap.parse_args()
    MEGA.mkdir(parents=True, exist_ok=True)
    mb = Moltbook(sleep=args.sleep)
    if args.mode == "global-new":
        global_stream(mb, "new", args.sleep)
    elif args.mode == "global-top":
        global_stream(mb, "top", args.sleep)
    else:
        submolt_shard(mb, args.shard, args.nshards, args.sleep)

if __name__ == "__main__":
    main()
