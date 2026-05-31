# Moltbook Reddit Network Study

This is a standalone research scraper and analysis system for comparing Moltbook agent behavior with Reddit user-network data. It is designed to fix the biggest limitation in the first Moltbook dataset: the old data was essentially a one-time snapshot. This project stores repeated crawls in a database so the final paper can analyze temporal change, coverage, centrality stability, community structure, thread shape, and cross-community movement.

## What It Builds

- A SQLite database in `data/study.sqlite`.
- Raw Moltbook API responses under `raw/moltbook/`.
- Normalized tables for agents, submolts, posts, comments, snapshots, coverage, and edges.
- Reddit import tables for thread graph datasets and generic user-reply edge streams.
- Markdown and JSON comparison reports under `reports/`.

For sharing this project with a group, see `docs/GITHUB_SHARING.md` and `docs/DATA_MANIFEST.md`.

## First Setup

Use an environment variable for the Moltbook key:

```powershell
$env:MOLTBOOK_API_KEY = "your-key"
```

Initialize the database:

```powershell
.\run.ps1 --config config.example.json init
```

Probe Moltbook endpoints and schemas:

```powershell
.\run.ps1 --config config.example.json probe-moltbook
```

The probe writes `reports/moltbook_endpoint_probe.json`. Read this first. It tells you which endpoints, filters, comment sorts, limits, offsets, and parent-comment fields actually work.

## Moltbook Crawls

Run one deep crawl pass:

```powershell
.\run.ps1 --config config.example.json crawl-moltbook --sorts new hot top --limit 50 --max-pages 20 --comment-sorts new top old --comment-limit 100 --comment-max-pages 20
```

Crawl all known submolts separately:

```powershell
.\run.ps1 --config config.example.json crawl-moltbook --submolts --sorts new hot top --limit 50 --max-pages 10
```

Rehydrate recent posts to improve comment coverage:

```powershell
.\run.ps1 --config config.example.json hydrate-recent --limit 500 --comment-sorts new top old --comment-limit 100 --comment-max-pages 30
```

Run repeated snapshots every 30 minutes:

```powershell
.\run.ps1 --config config.example.json run-study --iterations 48 --sleep-minutes 30 --crawl-submolts
```

## Reddit Imports

Import your local Reddit Threads dataset:

```powershell
.\run.ps1 --config config.example.json import-reddit-threads --path "C:\Users\blufi\OneDrive\Documents\New project\reddit_threads_unpacked\reddit_threads"
```

For quick testing, import only part of it:

```powershell
.\run.ps1 --config config.example.json import-reddit-threads --path "C:\Users\blufi\OneDrive\Documents\New project\reddit_threads_unpacked\reddit_threads" --limit 1000
```

Import a generic temporal Reddit reply CSV after normalizing it to source/target/timestamp columns:

```powershell
.\run.ps1 --config config.example.json import-reddit-temporal --path reddit_reply_edges.csv --source-col source --target-col target --time-col timestamp
```

Import SNAP Reddit `reply_networks` and `chain_networks` folders. For the full SNAP folders, prefer `--summary-only` first; it imports every snapshot's node/edge/density metrics without expanding hundreds of millions of edge rows into SQLite:

```powershell
.\run.ps1 --config config.example.json --db data\study_12h.sqlite import-reddit-adjacency --reply-dir "C:\Users\blufi\Downloads\reddit_reply_networks\reply_networks" --chain-dir "C:\Users\blufi\Downloads\reddit_chain_networks (1)\chain_networks" --dataset snap_reddit --summary-only
```

## Analysis

Generate comparison reports:

```powershell
.\run.ps1 --config config.example.json analyze
```

Outputs:

- `reports/network_analysis_summary.json`
- `reports/network_analysis_report.md`
- `reports/moltbook_top_agents.csv`

## Network Definitions

The Moltbook actor graph uses:

```text
comment author -> parent comment author
```

If no parent comment ID is available, it falls back to:

```text
comment author -> post author
```

That fallback is important. It preserves the available evidence but also reveals whether Moltbook is being captured as true threaded discussion or as a star-like attention network around post authors.

Reddit Threads are imported as per-thread undirected user reply graphs. Generic temporal Reddit reply CSVs are imported as directed actor graphs.

## What This Fixes

- Repeated snapshots replace one-time scraping.
- Raw responses make every normalized row auditable.
- Post snapshots track score and comment-count growth over time.
- Comment hydration uses multiple sorts and offset pagination attempts.
- Coverage checks compare observed comments to platform comment counts.
- Submolt crawls reduce bias from global feed ranking.
- Reddit importers normalize outside datasets into the same analysis pipeline.

## Recommended Research Protocol

1. Run `probe-moltbook`.
2. Adjust `submolt_query_param` in `config.example.json` if the probe discovers a better submolt filter.
3. Run `run-study` for at least 7 days, ideally 30 days.
4. Rehydrate recent posts daily.
5. Import Reddit Threads for per-thread comparison.
6. Import SNAP/Cornell temporal reply data for actor-network comparison if storage allows.
7. Run `analyze` on matched windows and matched sample sizes.

## Good Reddit Sources

- SNAP Reddit interaction networks: monthly subreddit user interaction networks, best for actor-network comparison.
- Cornell temporal Reddit reply network: long-running timestamped user reply edges, best for temporal dynamics.
- SNAP Reddit Threads: already local in this workspace, best for thread-shape comparison.
- SNAP Reddit Hyperlink Network: subreddit-to-subreddit network, useful if you later build submolt-to-submolt links.

## Database Notes

The most important tables are:

- `posts`, `post_snapshots`
- `comments`, `comment_snapshots`
- `edges`
- `coverage`
- `endpoint_probes`
- `reddit_threads`, `reddit_thread_edges`

The `coverage` table is the warning light. If many rows have `capped_flag = 1`, the crawler is still not observing all comments for those posts.
