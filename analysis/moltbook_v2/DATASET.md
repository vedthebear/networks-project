# Moltbook v2 dataset manifest

Fresh max-volume scrape via own API key (`netscicartographer`), 2026-06-01.
All raw/derived data is gitignored; regenerate with the scripts below.

## How it was collected (max-volume scrape)
Five parallel workers on one API key (~500 req/min, under the 600/min ceiling),
run detached overnight via `launch_mega.sh`:
- **global-new** — `/posts?sort=new` cursor firehose (deep recent backlog).
- **global-top** — `/posts?sort=top` cursor, all-time back to genesis.
- **3× submolt shards** — each known submolt scraped under all 5 sort orders
  (new/top/hot/old/controversial); each sort surfaces a mostly-different set.

`merge_raw.py` then dedups all streams by post id → `posts_master.jsonl`
(1,337,035 rows read → **1,089,907 unique posts**, ~18% cross-stream overlap).

## Final tables (`data/tables/`)
| file | rows | description |
|---|---:|---|
| posts.csv | **1,089,902** | post_id, author, submolt, created_at, score, comment_count, title |
| membership.csv | 59,631 | agent × submolt bipartite edge list (n_posts) |
| agents.csv | **27,342** | per-agent activity; **11,017 post in >1 submolt** |
| submolts.csv | **2,654** | per-submolt post/author counts + platform subscriber counts |

**Date span: 2026-01-28 → 2026-06-01 (124 days — the platform's full life).**
Monthly volume shows a clear lifecycle: Jan 5k · Feb 90k · Mar 231k ·
**Apr 408k (peak)** · May 348k · Jun 8k — explosive growth, an April peak,
then decline. (Agent platform rises and falls in months; Reddit grew for years.)

Activity profile: 2,654 submolts have posts; **533 have ≥5 authors**; 343 have ≥10.

## Graphs (`graphs/`)
Shared-agent projection (submolts linked when ≥`min_shared` agents post in both;
weight = `shared` raw count and `jaccard` overlap). `*_master` keeps all edges
(threshold later); `*_full` and `*_core` use min_shared=2.

| artifact | nodes | edges | largest comp | notes |
|---|---:|---:|---|---|
| `*_full` (all submolts) | 2,654 | 29,149 | 33% | fragmented tail of tiny submolts |
| `*_core` (≥5 authors) | **533** | **25,445** | **98.9%** | the analyzable community network |

The active core grew from 60 → **533 nodes** with the full scrape — large and
connected enough for robust configuration-model null testing.

## Reproduce
```bash
source .venv/bin/activate
bash scripts/launch_mega.sh          # overnight; stop with pkill -f mega_worker.py
python scripts/merge_raw.py          # -> data/raw/posts_master.jsonl
python scripts/build_tables.py
python scripts/build_shared_agent_graph.py --min-shared 1 --tag master
python scripts/build_shared_agent_graph.py --min-shared 2 --tag full
python scripts/build_shared_agent_graph.py --min-shared 2 --min-authors 5 --tag core
python scripts/dataset_report.py
```

## Open modeling choices (for analysis)
- **Author threshold** for "active" (5 vs 10) — graph size vs density.
- **min_shared** edge cutoff (1 = includes single-agent/spam edges; ≥2 robust).
- **Hub/spam agents**: a few agents post across hundreds of submolts (projection
  clique-inflation); weight by `shared` and/or threshold to control.
