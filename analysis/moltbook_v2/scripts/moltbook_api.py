#!/usr/bin/env python3
"""
moltbook_api.py -- thin, resilient client for the Moltbook public API.

Endpoints discovered empirically (2026-06-01):
  GET /submolts?sort=popular&page=N&limit=100   -> list communities (paginate via page=)
  GET /submolts/{name}/feed?limit=100&page=N     -> posts in one submolt (paginate via page=)
  GET /posts?sort=new&limit=100&cursor=...        -> global feed (cursor pagination)
  GET /agents/me                                  -> auth check

Notes:
  - submolt_name / submolt_id query filters on /posts are silently IGNORED by the
    server, so per-submolt collection MUST use /submolts/{name}/feed.
  - /submolts/{name}/feed posts carry submolt_name, author{}, content, created_at.
  - Read access works without the agent being "claimed".
"""
import json, sys, time, urllib.request, urllib.error
from pathlib import Path

BASE = "https://www.moltbook.com/api/v1"

def load_key():
    # project root is two levels up from this file (networks-project/), but the key
    # lives in the workspace .env one level above the repo; check several spots.
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / ".env",          # Desktop/Math_168/.env
        here.parents[2] / ".env",          # networks-project/.env
        Path.home() / ".config/moltbook/credentials.json",
    ]
    for c in candidates:
        if not c.exists():
            continue
        if c.suffix == ".json":
            return json.loads(c.read_text())["api_key"]
        for line in c.read_text().splitlines():
            if line.startswith("MOLTBOOK_API_KEY="):
                return line.split("=", 1)[1].strip()
    sys.exit("No MOLTBOOK_API_KEY found (.env or ~/.config/moltbook/credentials.json)")

class Moltbook:
    def __init__(self, key=None, sleep=0.5):
        self.key = key or load_key()
        self.sleep = sleep

    def get(self, path, retries=6):
        url = BASE + path
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.key}"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    return json.load(r)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                code = getattr(e, "code", None)
                if code == 429:               # rate limited -> back off harder
                    wait = min(5 * attempt, 60)
                else:
                    wait = min(2 ** attempt, 30)
                print(f"  [retry {attempt}/{retries}] {code or e} sleep {wait}s", file=sys.stderr)
                time.sleep(wait)
        raise RuntimeError(f"GET {url} failed after {retries} retries")

    def list_submolts(self, max_n=1000, sort="popular"):
        """Enumerate up to max_n submolts via page-based pagination."""
        out, page, seen = [], 1, set()
        while len(out) < max_n:
            d = self.get(f"/submolts?sort={sort}&limit=100&page={page}")
            items = d.get("submolts", [])
            fresh = [s for s in items if s.get("name") not in seen]
            for s in fresh:
                seen.add(s.get("name"))
            out += fresh
            if not items or not fresh:
                break
            page += 1
            time.sleep(self.sleep)
        return out[:max_n]

    def submolt_feed(self, name, quota=200):
        """Pull up to `quota` posts from one submolt via page-based pagination."""
        out, page, seen = [], 1, set()
        while len(out) < quota:
            try:
                d = self.get(f"/submolts/{name}/feed?limit=100&page={page}")
            except RuntimeError:
                break
            posts = d.get("posts", [])
            fresh = [p for p in posts if p.get("id") not in seen]
            for p in fresh:
                seen.add(p.get("id"))
                p.setdefault("submolt_name", name)   # ensure community label present
            out += fresh
            if not posts or not fresh:
                break
            page += 1
            time.sleep(self.sleep)
        return out[:quota]
