# Moltbook v2 dataset manifest

Fresh scrape via own API key (`netscicartographer`), collected 2026-06-01.
Two-pass design: a **breadth** pass across the platform, then a **depth** pass
on the active communities. All raw/derived data is gitignored; regenerate with
the scripts below.

## How it was collected
1. **Breadth pass** (`scrape_balanced.py --submolts 12000 --quota 100`)
   Enumerated the top 12,000 submolts by popularity and pulled ~100 posts from
   each. Result: 12,000 submolts touched, 641 with any posts (the rest are
   near-empty tail communities), ~6.9k posts.
2. **Depth pass** (`scrape_depth.py --min-posts 5 --quota 800`)
   Re-scraped the 148 active submolts much deeper so each community's agent set
   is well sampled (overlap is the edge fuel for the projection).

## Final tables (`data/tables/`)
| file | rows | description |
|---|---|---|
| posts.csv | 19,947 | post_id, author, submolt, created_at, score, comment_count, title |
| membership.csv | 2,478 | agent × submolt bipartite edge list (n_posts) |
| agents.csv | 864 | per-agent activity; 376 post in >1 submolt |
| submolts.csv | 641 | per-submolt post/author counts + platform subscriber counts |
| dataset_report.json | — | health metrics (active-submolt counts, graph sizes) |

Activity profile: **641** submolts have posts; **60** have ≥5 distinct authors;
**29** have ≥10. The active core is what carries the community graph.

## Graphs (`graphs/`)
Shared-agent projection (submolts linked when ≥`min_shared` agents post in both;
weight = Jaccard overlap of agent sets).

| artifact | nodes | edges | largest comp | clustering (unwtd) | notes |
|---|---|---|---|---|---|
| `*_full` (all submolts, min_shared=2) | 641 | 863 | 22% | low | fragmented — dragged down by ~580 tiny submolts |
| `*_core` (≥5 authors, min_shared=2) | **60** | **493** | **95%** | **0.58** | the analyzable community network |

`min_shared=1` makes even the full graph dense (~128k edges, 94% connected) but
each edge is a single shared agent — a hairball driven by a few hyper-active
agents. `min_shared=2` on the active core is the recommended object.

## Open modeling choices (for analysis)
- **Author threshold** for "active" (5 vs 10) — trades graph size vs density.
- **min_shared** edge threshold (1 vs 2) — trades connectivity vs robustness.
- **Hub agents**: a few agents post across many submolts; decide whether to cap
  or down-weight their contribution to edges.

## Reproduce
```bash
source .venv/bin/activate
python scripts/scrape_balanced.py --submolts 12000 --quota 100
python scripts/scrape_depth.py --min-posts 5 --quota 800
python scripts/build_tables.py
python scripts/build_shared_agent_graph.py --min-shared 2 --tag full
python scripts/build_shared_agent_graph.py --min-shared 2 --min-authors 5 --tag core
python scripts/dataset_report.py
```
