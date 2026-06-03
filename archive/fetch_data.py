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
import random
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
# 1.0s between calls keeps us comfortably under that.
REQUEST_DELAY_SEC = 1.0

# Retry settings for transient failures (5xx and network errors).
# On a 500 the server is stressed; we wait longer and add jitter so
# a burst of retries doesn't all hit again at the same moment.
#   attempt 1 → wait ~5s,  attempt 2 → ~10s,  attempt 3 → ~20s ...
# Tune via environment: MOLTBOOK_MAX_RETRIES, MOLTBOOK_BACKOFF_FACTOR
MAX_RETRIES    = int(os.getenv("MOLTBOOK_MAX_RETRIES",    "5"))
BACKOFF_FACTOR = float(os.getenv("MOLTBOOK_BACKOFF_FACTOR", "5.0"))
BACKOFF_JITTER = 2.0   # add up to this many extra random seconds to each wait
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
            wait = BACKOFF_FACTOR * (2 ** (attempt - 1)) + random.uniform(0, BACKOFF_JITTER)
            print(f"  network error ({e}); retrying in {wait:.1f}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait)
            continue

        # keep us under Moltbook's documented rate limits
        time.sleep(REQUEST_DELAY_SEC)

        # 429 = rate limited. Honor Retry-After if present, then retry.
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", "60"))
            print(f"  rate-limited (429); sleeping {wait}s")
            time.sleep(wait)
            continue

        # 5xx server errors: back off and retry with jitter so a cluster of
        # in-flight requests doesn't all hammer the server at the same moment.
        if 500 <= resp.status_code < 600:
            if attempt < MAX_RETRIES:
                wait = BACKOFF_FACTOR * (2 ** (attempt - 1)) + random.uniform(0, BACKOFF_JITTER)
                print(f"  server error {resp.status_code}; retrying in {wait:.1f}s "
                      f"(attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            raise MoltbookError(
                f"GET {url} -> {resp.status_code} after {attempt} attempts: "
                f"{resp.text[:500]}"
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

def fetch_posts(submolt=None, sort="new", max_posts=300):
    """
    Page through the posts feed (global or one submolt).

    If max_posts is None, run pagination until exhausted (every page).
    If max_posts is an int, stop once we have at least that many.

    Tries cursor-based pagination first (after=<post_id>). If the API
    does not return a cursor, falls back to offset-based pagination
    (offset=N) so the second and third pages are still collected.
    """
    page_size = 100
    posts = []
    cursor = None
    offset = 0

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
            params["after"] = cursor
        elif offset:
            params["offset"] = offset

        data = _get(path, params=params)

        # Response shape can be {"posts": [...], "after": "..."} or a bare list.
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

        # If the API returned a cursor, use it next iteration.
        # If not but we got a full page, try offset pagination for the next page.
        if cursor:
            offset = 0           # cursor takes priority; reset offset
        elif len(batch) == page_size:
            offset = len(posts)  # full page with no cursor — try offset next
        else:
            break                # partial page and no cursor means we're done

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
    """Get all comments on a post, paginating through all pages."""
    path = ENDPOINTS["comments"].format(post_id=post_id)
    comments = []
    cursor = None

    while True:
        params = {"limit": 100}
        if cursor:
            params["after"] = cursor

        data = _get(path, params=params)

        if isinstance(data, dict):
            batch = data.get("comments") or data.get("data") or []
            cursor = data.get("after") or data.get("next_cursor")
        else:
            batch = data
            cursor = None

        if not batch:
            break

        comments.extend(batch)

        if not cursor:
            break

    return comments


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
# Incremental helpers shared by posts, agents, and comments
# ---------------------------------------------------------------------------

# Flush cadence (in number of items processed)
POSTS_SAVE_EVERY   = 1      # save after every submolt (already fast)
AGENT_SAVE_EVERY   = 50     # save after every 50 agent profiles
COMMENT_SAVE_EVERY = 25     # save after every 25 posts worth of comments

# Progress + data files
_POSTS_PROGRESS    = DATA_DIR / ".posts_progress.json"
_AGENTS_PROGRESS   = DATA_DIR / ".agents_progress.json"
_COMMENTS_PROGRESS = DATA_DIR / ".comments_progress.json"

POSTS_FILE    = DATA_DIR / "posts.json"
AGENTS_FILE   = DATA_DIR / "agents.json"
COMMENTS_FILE = DATA_DIR / "comments.json"

POSTS_CSV_FIELDS = [
    "id", "submolt", "title", "content", "author_id", "author_name",
    "created_at", "score", "upvotes", "downvotes", "num_comments",
]
AGENTS_CSV_FIELDS = [
    "id", "name", "description", "owner", "created_at",
    "post_count", "comment_count", "karma", "model",
]
COMMENTS_CSV_FIELDS = [
    "id", "post_id", "parent_id", "author_id", "author_name",
    "content", "created_at", "score",
]


def _read_progress(path):
    """Return set of already-done item keys from a progress file."""
    if path.exists():
        try:
            return set(json.load(open(path, encoding="utf-8")).get("done", []))
        except (json.JSONDecodeError, KeyError):
            pass
    return set()


def _write_progress(path, done):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"done": list(done)}, f)


def _load_json(path):
    """Load a JSON list from disk, returning [] on missing/corrupt file."""
    if path.exists():
        try:
            return json.load(open(path, encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  Warning: {path} was corrupt, starting fresh.")
    return []


def _flush_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _flush_csv(name, rows, fieldnames):
    path = DATA_DIR / f"{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def fetch_posts_incremental(submolts, max_posts_per_feed):
    """
    Fetch posts for each submolt with incremental saving and resume support.

    Progress is tracked per submolt name in .posts_progress.json.
    Already-completed submolts are skipped entirely on restart.
    posts.json and posts.csv are updated after every successful submolt.
    Submolts that fail after all retries are skipped for this run and
    retried automatically on the next run (they are NOT marked done).
    """
    done_submolts = _read_progress(_POSTS_PROGRESS)
    all_posts = _load_json(POSTS_FILE)
    by_id = {p["id"]: p for p in all_posts if p.get("id")}

    todo = [s for s in submolts if s not in done_submolts]
    print(f"\nFetching posts: {len(todo)} submolts remaining "
          f"({len(done_submolts)} already done) ...")

    failed = []
    for s in todo:
        try:
            batch = fetch_posts(submolt=s, max_posts=max_posts_per_feed)
        except MoltbookError as e:
            print(f"  !! submolt '{s}' skipped after all retries: {e}")
            failed.append(s)
            continue   # not marked done — will be retried next run

        added = 0
        for p in batch:
            p.setdefault("submolt", s)
            if p.get("id") and p["id"] not in by_id:
                by_id[p["id"]] = p
                added += 1

        done_submolts.add(s)
        all_posts = list(by_id.values())
        _flush_json(POSTS_FILE, all_posts)
        _write_progress(_POSTS_PROGRESS, done_submolts)
        print(f"  submolt '{s}': +{added} new posts (total {len(all_posts)})")

    _flush_csv("posts", all_posts, POSTS_CSV_FIELDS)
    print(f"  wrote {POSTS_FILE} ({len(all_posts)} posts)")
    if failed:
        print(f"  !! {len(failed)} submolts skipped due to server errors "
              f"(will retry next run): {failed}")
    return all_posts


def fetch_agents_incremental(all_posts):
    """
    Fetch agent profiles for every unique post author with incremental saving
    and resume support.

    Progress is tracked per agent ID in .agents_progress.json.
    Already-fetched agents are skipped on restart.
    agents.json is flushed every AGENT_SAVE_EVERY new profiles.
    """
    done_ids  = _read_progress(_AGENTS_PROGRESS)
    agents    = _load_json(AGENTS_FILE)
    by_id     = {a["id"]: a for a in agents if a.get("id")}

    author_ids = {
        p.get("author_id") or (p.get("author") or {}).get("id")
        for p in all_posts
    }
    author_ids.discard(None)

    todo = sorted(author_ids - done_ids)
    print(f"\nFetching agent profiles: {len(todo)} remaining "
          f"({len(done_ids)} already done) ...")

    pending = 0
    failed = []
    for i, aid in enumerate(todo):
        try:
            agent = fetch_agent(aid)
            if agent.get("id"):
                by_id[agent["id"]] = agent
            else:
                by_id[aid] = agent
            done_ids.add(aid)   # only marked done on success
        except MoltbookError as e:
            print(f"  !! agent {aid} skipped after all retries: {e}")
            failed.append(aid)  # not marked done — retried next run

        pending += 1
        if pending >= AGENT_SAVE_EVERY or i == len(todo) - 1:
            agents = list(by_id.values())
            _flush_json(AGENTS_FILE, agents)
            _write_progress(_AGENTS_PROGRESS, done_ids)
            pending = 0
            print(f"  {i + 1}/{len(todo)} agents processed "
                  f"({len(agents)} fetched, {len(failed)} failed so far)")

    agents = list(by_id.values())
    _flush_csv("agents", agents, AGENTS_CSV_FIELDS)
    print(f"  wrote {AGENTS_FILE} ({len(agents)} agents)")
    if failed:
        print(f"  !! {len(failed)} agents skipped due to server errors "
              f"(will retry next run)")
    return agents


def fetch_comments_incremental(all_posts):
    """
    Fetch comments for every post with incremental saving and resume support.

    On each run:
      - Reads .comments_progress.json to find which post IDs are already done.
      - Loads comments.json for comments already collected.
      - Skips posts that are already done.
      - Saves comments.json + .comments_progress.json every COMMENT_SAVE_EVERY posts.
      - Writes comments.csv at the very end.

    If the run is interrupted and restarted, it picks up exactly where it left off.
    """
    done_ids    = _read_progress(_COMMENTS_PROGRESS)
    all_comments = _load_json(COMMENTS_FILE)

    if done_ids:
        print(f"  Resuming: {len(done_ids)} posts already done, "
              f"{len(all_comments)} comments already collected.")

    posts_todo = [p for p in all_posts if p.get("id") and p["id"] not in done_ids]
    print(f"\nFetching comments for {len(posts_todo)} posts "
          f"({len(done_ids)} already done, {len(all_posts)} total) ...")

    pending = 0
    failed = []

    for i, p in enumerate(posts_todo):
        pid = p["id"]
        try:
            cs = fetch_comments(pid)
            for c in cs:
                c["post_id"] = pid
            all_comments.extend(cs)
            done_ids.add(pid)   # only marked done on success
        except MoltbookError as e:
            print(f"  !! post {pid} skipped after all retries: {e}")
            failed.append(pid)  # not marked done — retried next run

        pending += 1
        if pending >= COMMENT_SAVE_EVERY or i == len(posts_todo) - 1:
            _flush_json(COMMENTS_FILE, all_comments)
            _write_progress(_COMMENTS_PROGRESS, done_ids)
            pending = 0
            print(f"  saved — {i + 1}/{len(posts_todo)} posts processed, "
                  f"{len(all_comments)} comments total "
                  f"({len(failed)} failed so far)")

    _flush_csv("comments", all_comments, COMMENTS_CSV_FIELDS)
    print(f"  wrote {COMMENTS_FILE} ({len(all_comments)} comments)")
    if failed:
        print(f"  !! {len(failed)} posts skipped due to server errors "
              f"(will retry next run)")
    return all_comments


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(submolts=None, max_posts_per_feed=300, fetch_comments_for_posts=True,
        build_edges=True, auto_discover_top_n=0):
    """
    End-to-end collection with full incremental save + resume for all three
    data types (posts, agents, comments).

    submolts: list of submolt names to crawl, or None for the global feed.
    auto_discover_top_n: if > 0 and submolts is None, discover the top N
        most active submolts and crawl each.
    build_edges: emit network edge-list CSVs after fetching.

    On restart, already-completed submolts / agents / comment-posts are
    detected from their progress files and skipped automatically.
    """
    check_auth()

    if submolts is None and auto_discover_top_n > 0:
        submolts = discover_submolts(top_n=auto_discover_top_n)
        print(f"Discovered submolts: {submolts}")

    # 1. Posts ---------------------------------------------------------------
    if submolts:
        all_posts = fetch_posts_incremental(submolts, max_posts_per_feed)
    else:
        # Global feed: no per-submolt tracking needed, just load if exists
        if POSTS_FILE.exists():
            print("\nLoading existing posts.json ...")
            all_posts = _load_json(POSTS_FILE)
            print(f"  {len(all_posts)} posts loaded")
        else:
            all_posts = fetch_posts(max_posts=max_posts_per_feed)
            _flush_json(POSTS_FILE, all_posts)
            _flush_csv("posts", all_posts, POSTS_CSV_FIELDS)

    # 2. Agents --------------------------------------------------------------
    agents = fetch_agents_incremental(all_posts)

    # 3. Comments ------------------------------------------------------------
    all_comments = []
    if fetch_comments_for_posts:
        all_comments = fetch_comments_incremental(all_posts)

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
    run(
        submolts=None,
        auto_discover_top_n=200,
        max_posts_per_feed=300,
        fetch_comments_for_posts=True,
        build_edges=True,
    )
