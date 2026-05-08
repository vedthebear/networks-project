"""
Moltbook data collector for Math 168 networks project.

Pulls agent profiles, posts, and comments from the Moltbook API and
saves them to data/ as JSON + CSV for downstream network analysis
(networkx, pandas, etc.).

Auth: set MOLTBOOK_API_KEY in .env (the API key Moltbook gives you on
agent registration, prefixed `moltbook_sk_` or `moltdev_`).

Endpoint paths and rate limits are based on Moltbook's public API
docs (https://www.moltbook.com/developers and community write-ups).
If the API has shifted, adjust the constants in the ENDPOINTS block.
"""

import csv
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("MOLTBOOK_API_KEY")
BASE_URL = os.getenv("MOLTBOOK_BASE_URL", "https://www.moltbook.com/api/v1")

# Moltbook's documented limit is 100 req/min on read endpoints. Sleeping
# 0.7s between calls keeps us comfortably under that.
REQUEST_DELAY_SEC = 0.7

# Retry settings for transient failures (5xx and network errors)
# Tune via environment if desired: MOLTBOOK_MAX_RETRIES, MOLTBOOK_BACKOFF_FACTOR
MAX_RETRIES = int(os.getenv("MOLTBOOK_MAX_RETRIES", "3"))
BACKOFF_FACTOR = float(os.getenv("MOLTBOOK_BACKOFF_FACTOR", "1.5"))
# Where to save collected data.
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Endpoint paths in one place so you can patch them if docs differ.
ENDPOINTS = {
    "me":         "/agents/me",
    "agent":      "/agents/{agent_id}",          # GET single agent profile
    "posts":      "/posts",                      # GET global feed (?sort=&limit=&after=)
    "submolts":   "/submolts",                   # GET list of submolts (communities)
    "submolt":    "/submolts/{name}/feed",       # GET feed for a specific submolt
    "comments":   "/posts/{post_id}/comments",   # GET comments on a post
}

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
    "User-Agent": "math168-networks-project/0.1",
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

class MoltbookError(Exception):
    pass


def _get(path, params=None):
    """GET with rate limiting + basic error handling. Returns parsed JSON."""
    if not API_KEY:
        raise MoltbookError("MOLTBOOK_API_KEY is not set. Add it to .env.")
    url = f"{BASE_URL}{path}"

    attempt = 0
    while attempt < MAX_RETRIES:
        attempt += 1
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        except requests.RequestException as e:
            # Network-level error (DNS, timeout, connection, etc.)
            if attempt >= MAX_RETRIES:
                raise MoltbookError(f"GET {url} -> exception after {attempt} attempts: {e}")
            wait = BACKOFF_FACTOR * (2 ** (attempt - 1))
            print(f"  request error ({e}); retrying in {wait:.1f}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait)
            continue

        # keep us under Moltbook's documented rate limits
        time.sleep(REQUEST_DELAY_SEC)

        # 429 = rate limited. Honor Retry-After if present, then retry.
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", "30"))
            print(f"  rate-limited; sleeping {wait}s")
            time.sleep(wait)
            continue

        # 5xx server errors: retry a few times with exponential backoff
        if 500 <= resp.status_code < 600:
            if attempt < MAX_RETRIES:
                wait = BACKOFF_FACTOR * (2 ** (attempt - 1))
                print(f"  server error {resp.status_code}; retrying in {wait:.1f}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            # final attempt failed; include headers/body to aid debugging
            raise MoltbookError(
                f"GET {url} -> {resp.status_code}: {resp.text[:1000]} Headers: {dict(resp.headers)}"
            )

        if not resp.ok:
            # include headers and a larger slice of the body for diagnostics
            raise MoltbookError(
                f"GET {url} -> {resp.status_code}: {resp.text[:1000]} Headers: {dict(resp.headers)}"
            )

        return resp.json()


def check_auth():
    """Smoke test: verify the API key works."""
    me = _get(ENDPOINTS["me"])
    print(f"Authed as agent: {me.get('name', '?')} (id={me.get('id', '?')})")
    return me


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------

def fetch_posts(submolt=None, sort="new", max_posts=None):
    """
    Page through the posts feed (global or one submolt).

    If max_posts is None, run pagination until exhausted (every page).
    If max_posts is an int, stop once we have at least that many.

    Moltbook's feed pagination uses `after=<post_id>` cursor style
    (Reddit-flavored). If the response uses a different key, adjust
    `cursor_key` below.
    """
    cursor_key = "after"
    page_size = 100
    posts = []
    cursor = None

    if submolt:
        path = ENDPOINTS["submolt"].format(name=submolt)
        label = f"submolt={submolt}"
    else:
        path = ENDPOINTS["posts"]
        label = "global feed"

    print(f"Fetching posts ({label}, sort={sort}, max={max_posts or 'no cap'}) ...")
    while True:
        if max_posts is not None and len(posts) >= max_posts:
            break

        params = {"sort": sort, "limit": page_size}
        if cursor:
            params[cursor_key] = cursor

        data = _get(path, params=params)

        # Response shape can be either {"posts": [...], "after": "..."} or
        # a bare list. Handle both.
        if isinstance(data, dict):
            batch = data.get("posts") or data.get("data") or []
            cursor = data.get("after") or data.get("next_cursor")
        else:
            batch = data
            cursor = None

        if not batch:
            break

        posts.extend(batch)
        print(f"  +{len(batch)} (total {len(posts)})")

        if not cursor:
            break

    if max_posts is not None:
        return posts[:max_posts]
    return posts


def fetch_agent(agent_id):
    """Get a single agent's profile."""
    return _get(ENDPOINTS["agent"].format(agent_id=agent_id))


def discover_submolts(top_n=20):
    """
    Return up to top_n submolt names, sorted by activity (subscribers / posts).

    Tries the /submolts listing endpoint first; if that's not exposed, falls
    back to sampling the global feed and counting which submolts show up.
    """
    try:
        data = _get(ENDPOINTS["submolts"], params={"sort": "popular", "limit": top_n})
        if isinstance(data, dict):
            items = data.get("submolts") or data.get("data") or []
        else:
            items = data
        names = [s.get("name") for s in items if s.get("name")]
        if names:
            return names[:top_n]
    except MoltbookError as e:
        print(f"  /submolts listing not available ({e}); falling back to feed sample.")

    # Fallback: scan a slice of the global feed.
    sample = fetch_posts(max_posts=500)
    counts = {}
    for p in sample:
        s = p.get("submolt") or (p.get("submolt_obj") or {}).get("name")
        if s:
            counts[s] = counts.get(s, 0) + 1
    return [name for name, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]]


def fetch_comments(post_id):
    """Get all comments on a post (a flat list; thread structure is in
    the parent_id field of each comment)."""
    data = _get(ENDPOINTS["comments"].format(post_id=post_id))
    if isinstance(data, dict):
        return data.get("comments") or data.get("data") or []
    return data


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------

def save_json(name, obj):
    path = DATA_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  wrote {path} ({len(obj) if hasattr(obj, '__len__') else '?'} records)")


def save_csv(name, rows, fieldnames):
    """Write a list of dicts to data/<name>.csv with a fixed schema."""
    path = DATA_DIR / f"{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  wrote {path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Network edge construction
# ---------------------------------------------------------------------------
#
# These functions take the raw posts/comments collected above and emit
# edge-list CSVs ready for networkx / igraph.
#
# Three networks are built:
#   1. agent_submolt_edges.csv  -- bipartite: agent <-> submolt, weight = activity
#   2. submolt_overlap_edges.csv -- projected: submolt -- submolt, weight = shared agents (Jaccard)
#   3. agent_reply_edges.csv    -- directed:  replier -> author, scoped per submolt
#
# Field-name assumptions are documented inline; tweak `_post_author`,
# `_post_submolt`, etc. if the API's actual JSON differs.

def _post_author(post):
    return post.get("author_id") or (post.get("author") or {}).get("id")


def _post_submolt(post):
    return post.get("submolt") or (post.get("submolt_obj") or {}).get("name")


def _comment_author(c):
    return c.get("author_id") or (c.get("author") or {}).get("id")


def build_agent_submolt_edges(posts, comments):
    """Bipartite agent <-> submolt with weight = #posts + #comments."""
    posts_by_id = {p.get("id"): p for p in posts}
    weights = {}  # (agent_id, submolt) -> {"posts": n, "comments": n}

    for p in posts:
        a = _post_author(p)
        s = _post_submolt(p)
        if not (a and s):
            continue
        key = (a, s)
        weights.setdefault(key, {"posts": 0, "comments": 0})
        weights[key]["posts"] += 1

    for c in comments:
        a = _comment_author(c)
        parent_post = posts_by_id.get(c.get("post_id"))
        s = _post_submolt(parent_post) if parent_post else None
        if not (a and s):
            continue
        key = (a, s)
        weights.setdefault(key, {"posts": 0, "comments": 0})
        weights[key]["comments"] += 1

    rows = [
        {
            "agent_id": a, "submolt": s,
            "posts": v["posts"], "comments": v["comments"],
            "weight": v["posts"] + v["comments"],
        }
        for (a, s), v in weights.items()
    ]
    save_csv("agent_submolt_edges", rows,
             fieldnames=["agent_id", "submolt", "posts", "comments", "weight"])
    return rows


def build_submolt_overlap_edges(agent_submolt_rows):
    """
    Project the bipartite graph onto submolts: edge weight = #shared agents,
    plus Jaccard similarity for fair comparison with Reddit.
    """
    members = {}  # submolt -> set(agent_id)
    for r in agent_submolt_rows:
        members.setdefault(r["submolt"], set()).add(r["agent_id"])

    submolts = sorted(members)
    rows = []
    for i, a in enumerate(submolts):
        for b in submolts[i + 1:]:
            shared = members[a] & members[b]
            if not shared:
                continue
            union = members[a] | members[b]
            rows.append({
                "submolt_a": a, "submolt_b": b,
                "shared_agents": len(shared),
                "size_a": len(members[a]), "size_b": len(members[b]),
                "jaccard": len(shared) / len(union) if union else 0.0,
            })
    save_csv("submolt_overlap_edges", rows,
             fieldnames=["submolt_a", "submolt_b", "shared_agents",
                         "size_a", "size_b", "jaccard"])
    return rows


def build_agent_reply_edges(posts, comments):
    """
    Directed reply graph: replier -> recipient, with one row per submolt.
    Recipient is:
      - the comment's parent comment author, if parent_id matches another comment;
      - else the post author.
    Use this per-submolt for within-community analysis (centrality, communities).
    """
    posts_by_id = {p.get("id"): p for p in posts}
    comments_by_id = {c.get("id"): c for c in comments if c.get("id")}
    weights = {}  # (replier, recipient, submolt) -> count

    for c in comments:
        replier = _comment_author(c)
        if not replier:
            continue
        post = posts_by_id.get(c.get("post_id"))
        if not post:
            continue
        submolt = _post_submolt(post)

        parent_id = c.get("parent_id")
        parent_comment = comments_by_id.get(parent_id) if parent_id else None
        if parent_comment:
            recipient = _comment_author(parent_comment)
        else:
            recipient = _post_author(post)

        if not recipient or recipient == replier:
            continue  # skip self-loops

        key = (replier, recipient, submolt)
        weights[key] = weights.get(key, 0) + 1

    rows = [
        {"replier": r, "recipient": rc, "submolt": s, "weight": w}
        for (r, rc, s), w in weights.items()
    ]
    save_csv("agent_reply_edges", rows,
             fieldnames=["replier", "recipient", "submolt", "weight"])
    return rows


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(submolts=None, max_posts_per_feed=300, fetch_comments_for_posts=True,
        build_edges=True, auto_discover_top_n=0):
    """
    End-to-end collection.

    submolts: list of submolt names to crawl, or None for the global feed.
    auto_discover_top_n: if > 0 and submolts is None, discover the top N
        most active submolts and crawl each (good for community-to-community
        comparisons, since the global feed is biased toward popular posts).
    build_edges: also emit network edge-list CSVs after fetching.
    """
    check_auth()

    if submolts is None and auto_discover_top_n > 0:
        submolts = discover_submolts(top_n=auto_discover_top_n)
        print(f"Discovered submolts: {submolts}")

    # 1. Posts ---------------------------------------------------------------
    all_posts = []
    if submolts:
        for s in submolts:
            batch = fetch_posts(submolt=s, max_posts=max_posts_per_feed)
            for p in batch:
                p.setdefault("submolt", s)
            all_posts.extend(batch)
            # checkpoint after each submolt so a crash mid-crawl is survivable
            save_json("posts", all_posts)
    else:
        all_posts = fetch_posts(max_posts=max_posts_per_feed)

    # de-dupe by post id
    by_id = {p.get("id"): p for p in all_posts if p.get("id")}
    all_posts = list(by_id.values())
    save_json("posts", all_posts)
    save_csv(
        "posts",
        all_posts,
        fieldnames=[
            "id", "submolt", "title", "content", "author_id", "author_name",
            "created_at", "score", "upvotes", "downvotes", "num_comments",
        ],
    )

    # 2. Agents (collected from post authors) -------------------------------
    author_ids = {
        p.get("author_id") or (p.get("author") or {}).get("id")
        for p in all_posts
    }
    author_ids.discard(None)
    print(f"\nFetching {len(author_ids)} agent profiles ...")

    agents = []
    for aid in sorted(author_ids):
        try:
            agents.append(fetch_agent(aid))
        except MoltbookError as e:
            print(f"  skip agent {aid}: {e}")

    save_json("agents", agents)
    save_csv(
        "agents",
        agents,
        fieldnames=[
            "id", "name", "description", "owner", "created_at",
            "post_count", "comment_count", "karma", "model",
        ],
    )

    # 3. Comments (for interaction graph) -----------------------------------
    all_comments = []
    if fetch_comments_for_posts:
        print(f"\nFetching comments for {len(all_posts)} posts ...")
        for i, p in enumerate(all_posts):
            pid = p.get("id")
            if not pid:
                continue
            try:
                cs = fetch_comments(pid)
                for c in cs:
                    c["post_id"] = pid
                all_comments.extend(cs)
            except MoltbookError as e:
                print(f"  skip comments for post {pid}: {e}")
            # checkpoint every 500 posts so a long crawl is survivable
            if (i + 1) % 500 == 0:
                save_json("comments", all_comments)
                print(f"  checkpoint: {i + 1}/{len(all_posts)} posts processed, "
                      f"{len(all_comments)} comments collected")

        save_json("comments", all_comments)
        save_csv(
            "comments",
            all_comments,
            fieldnames=[
                "id", "post_id", "parent_id", "author_id", "author_name",
                "content", "created_at", "score",
            ],
        )

    # 4. Network edge lists --------------------------------------------------
    if build_edges:
        print("\nBuilding network edge lists ...")
        agent_submolt_rows = build_agent_submolt_edges(all_posts, all_comments)
        build_submolt_overlap_edges(agent_submolt_rows)
        if all_comments:
            build_agent_reply_edges(all_posts, all_comments)
        else:
            print("  (skipping reply graph: no comments collected)")

    print("\nDone. Files are in data/.")


if __name__ == "__main__":
    # Full crawl: top 50 submolts, no per-feed cap (every available post),
    # and every comment on every post. Posts and comments are checkpointed
    # to disk during the crawl, so a crash will not lose previous progress.
    #
    # Expect this to take a while (hours, depending on platform activity).
    # Reduce `auto_discover_top_n` or set `max_posts_per_feed` to test.
    run(
        submolts=None,
        auto_discover_top_n=50,
        max_posts_per_feed=None,       # no cap; pagination runs until exhausted
        fetch_comments_for_posts=True,
        build_edges=True,
    )
