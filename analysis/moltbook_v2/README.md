# Moltbook v2 — community-level data pipeline

This directory builds a **community-level network for Moltbook** (an AI-agent
social platform) so it can be compared against the SNAP Reddit subreddit
hyperlink network. It supersedes the `develop-moltbook-v1` exploratory scripts.

## Why this exists (the short version)

We want to ask: **do AI agents form communities the way humans do?** Reddit's
SNAP dataset is a *subreddit → subreddit* graph (communities are nodes). To
compare fairly, Moltbook must also be a *community → community* graph.

We empirically tested three ways to draw edges between Moltbook communities
(submolts), on live API data:

| Edge definition | Result | Verdict |
|---|---|---|
| **Hyperlinks** (`m/submolt` refs in post bodies) | 0% of posts | dead — agents don't cross-link communities |
| **Hashtag diffusion** (shared hashtags across submolts) | ~29 nodes even at 20k posts | too sparse — hashtags are community-private |
| **Shared agents** (same agents post in both submolts) | 264 nodes, 6k edges, 99% connected | ✅ used here |

The shared-agent approach is the standard **bipartite projection** (Newman ch. 6):
agents ↔ submolts, projected onto submolts. Two submolts are linked when the
same agents post in both; edge weight = Jaccard overlap of their agent sets.

That hyperlinks and hashtags *fail* is itself a finding: Moltbook agents form
**siloed communities** that humans would otherwise bridge.

## Pipeline

```
scrape_balanced.py   →  data/raw/posts_balanced.jsonl   (balanced per-submolt scrape)
build_tables.py      →  data/tables/{posts,agents,submolts,membership}.csv
build_shared_agent_graph.py → graphs/{shared_agent_edges,shared_agent_nodes,shared_agent_stats}.*
```

### 1. Scrape (community-balanced)
`scrape_balanced.py` enumerates the top-N submolts by popularity and pulls a
fixed **quota** of posts from *each* (via `/submolts/{name}/feed?page=`). Equal
quotas stop giant submolts like `general` from dominating the graph. The scrape
is checkpointed (`done_submolts.txt`) so it resumes after interruption.

```bash
python scripts/scrape_balanced.py --submolts 1000 --quota 250
```

### 2. Normalize into tidy tables
`build_tables.py` de-dups posts and emits four tables:
- `posts.csv` — post_id, author, submolt, created_at, score, comment_count, title
- `membership.csv` — agent, submolt, n_posts  (the bipartite edge list)
- `agents.csv` — per-agent activity + which submolts they post in
- `submolts.csv` — per-submolt post/author counts + platform subscriber counts

### 3. Build the shared-agent graph
`build_shared_agent_graph.py` projects `membership.csv` onto submolts. Use
`--min-shared 2` (default) so a single one-off crossover doesn't create an edge.

```bash
python scripts/build_shared_agent_graph.py --min-shared 2
```

## API notes (discovered empirically 2026-06-01)

- Read access works with an unclaimed agent key (`MOLTBOOK_API_KEY` in `.env`).
- Platform scale: ~31,937 submolts, ~3.16M posts.
- `submolt_name` / `submolt_id` filters on `/posts` are **silently ignored** —
  per-submolt collection must use `/submolts/{name}/feed`.
- Submolt lists and per-submolt feeds paginate via `page=` (not cursor/offset);
  the global `/posts` feed paginates via `cursor=`.

## Reproducing
```bash
uv venv && source .venv/bin/activate
uv pip install pandas networkx scipy matplotlib
python scripts/scrape_balanced.py --submolts 1000 --quota 250
python scripts/build_tables.py
python scripts/build_shared_agent_graph.py
```
Raw and derived data are gitignored; rerun the scripts to regenerate them.
