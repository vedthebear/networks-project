# Concentrated but not Conversational: AI-Agent Social Networks vs Human Ones at Two Scales

**Math 168 (Networks) — UCLA, 2026**
Riley Leong · August Curtis · Ved Vedere · Samuel Kelly · Linlong Wang

---

## Thesis

Agent social networks concentrate attention and activity far more extremely than human ones, and are far less reciprocal — agents barely converse. Yet where structure does form, it organizes exactly like humans do: central communities act as bridges, not cliques. This pattern holds at two independent scales simultaneously: communities and individuals.

---

## Overview

This project compares **Moltbook** (an AI-agent social network) against **Reddit** (a human social network) to ask a fundamental question: do AI agents, interacting autonomously at scale, reproduce the structural properties of human online social behavior?

We study this at **two levels**:

- **Community level** — how submolts (Moltbook) and subreddits (Reddit) link to each other. Edges represent shared membership: two communities are connected when the same agents or users participate in both. This is structurally analogous to the Reddit subreddit hyperlink network (Kumar et al., 2018), where a post in subreddit A that links to subreddit B creates a directed reference edge.

- **Individual level** — how individual agents (Moltbook) and users (Reddit) interact directly via replies. Edges represent conversational turns: a reply to someone's post or comment creates a directed edge from replier to author.

Running the same structural tests on both levels and both platforms lets us separate what is universal in online social structure from what is specific to human behavior.

---

## The Three Findings

### Finding 1 — Extreme Concentration of Attention (both scales)

Activity on Moltbook is far more concentrated than on Reddit. The top 1% of agents receive 31% of all inbound replies; the in-degree Gini coefficient is 0.85. At the community level, the top 1% of submolts account for the majority of posts. Reddit's distributions are substantially flatter.

Critically, this concentration is in **activity and attention**, not in connectivity. Degree distributions for both Moltbook and Reddit are better described by lognormal or exponential fits than by a power law — neither is truly scale-free by the Clauset (2009) maximum-likelihood + likelihood-ratio test standard. The heavy tail of Moltbook comes from a winner-take-all attention economy, not a preferential-attachment degree structure.

### Finding 2 — Agents Barely Converse (individual level)

The Moltbook agent reply graph has a reciprocity of ~0.23 in the reply-to-comment subgraph, with global transitivity of 0.059. Agents post at each other far more than they talk with each other. On Reddit, user-user reciprocity is substantially higher — human conversations are genuinely bidirectional in a way agent interactions are not.

This suggests that autonomous agents default to broadcasting rather than exchanging. Even when the platform supports threaded replies, agents rarely respond to responses. The result is a network that looks conversational in structure (directed reply edges) but is functionally one-directional in practice.

### Finding 3 — Core-Periphery Structure Converges (community level)

Despite the differences in concentration and reciprocity, the *structural role* of central communities is identical across platforms. On both Moltbook and Reddit, **high-k-core communities have lower local clustering** — central nodes are bridges, not cliques. The raw Spearman correlations differ (Reddit within-core ρ ≈ −0.15; Moltbook within-core ρ ≈ −0.76), but this is explained by density differences between the platforms.

The key comparison is null-relative: each platform's real within-core ρ bends approximately **0.7 below its own degree-preserving configuration-model null**. Both platforms bend the same amount, in the same direction. Agents and humans build the same kind of community structure — dense periphery, sparse-but-connected core — even though agents do it with far more concentrated activity and far less conversation.

---

## Platforms

### Moltbook
Moltbook is a social network populated entirely by autonomous AI agents. Agents are assigned to communities (called **submolts**), post content, and reply to each other — all without direct human intervention. This makes Moltbook a rare fully-observable social system: complete post history, complete membership, no sampling. We scraped its entire lifespan (January 28 to June 1, 2026 — 124 days) for the community-level analysis.

**Platform lifecycle:** Monthly post volumes show explosive growth followed by a decline consistent with an attention-economy collapse — Jan 5k, Feb 90k, Mar 231k, Apr 408k (peak), May 348k, Jun 8k.

### Reddit
Reddit is one of the largest human social networks, organized into topic-specific communities (subreddits). We use two independent SNAP datasets for the two levels of analysis:

- **Reddit Hyperlink Network** (Kumar et al., 2018): 35,776 subreddits, hyperlinks extracted from post bodies. Community-level reference, matched structurally to Moltbook's shared-agent projection.
- **Reddit Interaction Network** (SNAP, 2014): directed user-to-user reply edges across subreddits. Individual-level reference, matched structurally to the Moltbook agent reply graph.

The choice of which Reddit dataset to use at each level was principled: hyperlinks are the natural substrate for community-level reference; reply edges are the natural substrate for individual-level conversation.

---

## Datasets

### Moltbook v2 — Community Level
Full-platform scrape via five parallel workers, collected June 2026.

| File | Rows | Description |
|---|---:|---|
| `posts.csv` | **1,089,902** | post_id, author, submolt, created_at, score, comment_count, title |
| `agents.csv` | 27,342 | per-agent activity (posts, submolts active in) |
| `submolts.csv` | 2,654 | per-submolt post and author counts |
| `membership.csv` | 59,631 | agent × submolt bipartite edge list (n_posts per pair) |

The **shared-agent graph** is a projection of the membership table: two submolts are linked when ≥ `min_shared` agents post in both. We use the **core graph** (submolts with ≥ 5 authors, edges with ≥ 2 shared agents) as the primary analysis object — 533 nodes, 25,445 edges, 98.9% in the largest component.

### Moltbook v1 — Individual Level
Smaller scrape collected over a 12-hour window via the Moltbook API. Used for agent-to-agent reply analysis only.

| Metric | Value |
|---|---|
| Agents (nodes) | 1,476 |
| Reply edges | 4,837 |
| LCC share | 94.7% |
| Comment coverage | ~17% (estimated from expected vs observed counts) |

The limited coverage is acknowledged in the paper's Limitations. The v2 dataset does not include comment threads, so individual-level metrics come exclusively from this scrape.

### Reddit — Community Level (SNAP Hyperlink)
- **35,776 subreddits** (nodes after undirected projection)
- **124,330 edges** (undirected, subreddit A–B if any hyperlink exists in either direction)
- Full post-body hyperlink records; date range 2014–2017
- Auto-downloaded by `run_all.py` (~304 MB)

### Reddit — Individual Level (SNAP Interaction)
- Directed user-to-user reply edges, aggregated across subreddits
- 2014 monthly interaction networks
- Used for reciprocity and concentration comparison against Moltbook agent reply graph

---

## Methods

### Shared-Agent Projection (Community Graph)
For each pair of submolts (A, B), count the number of agents who posted in both. Draw an undirected edge when this count exceeds `min_shared`. The edge weight is the raw shared-agent count; Jaccard overlap is also computed for normalized comparison. This projection is structurally analogous to Reddit's hyperlink graph: both capture cross-community reference, just through different mechanisms (shared membership vs explicit links).

### Configuration-Model Null (Shared Inferential Tool)
All structural tests use a **degree-preserving configuration-model null**: rewire the real graph 500 times while preserving each node's exact degree sequence. This ensures that any observed structural signal (clustering, core-periphery correlation, reciprocity) is measured *beyond what degree alone explains*. Significance reported as permutation p-values with one-tailed tests matching the hypothesized direction.

### k-Core Decomposition + Core-Periphery Test
Each node is assigned a k-core number (the largest k such that the node belongs to a subgraph where every node has degree ≥ k). We compute Spearman ρ(k-core, clustering) two ways:
- **Full graph**: includes peripheral leaf nodes, which mechanically inflate ρ toward positive (leaves have degree 1 and high clustering ratio)
- **Within-core** (k ≥ 2): removes the leaf fringe; the remaining correlation measures whether *central* communities are bridges (negative ρ) or cliques (positive ρ)

We report both the raw within-core ρ and the null-relative value (real ρ − null-mean ρ), which is the fair cross-platform comparison.

### Power-Law + Concentration Analysis
Degree sequences and activity distributions are fit using Clauset, Shalizi & Newman (2009) maximum-likelihood estimation with KS-minimized lower cutoff k_min. Each fit is compared against lognormal and exponential alternatives via likelihood-ratio tests — "looks like a straight line on a log-log plot" is insufficient; LR test significance determines whether scale-free can be claimed. Gini coefficients and top-k attention shares are computed projection-free (directly from post/reply count tables) as independent concentration measures.

### IC/SIR Diffusion Simulation (Reddit SI)
On the Reddit hyperlink network, we simulate Independent Cascade (IC) and SIR epidemic spreading to characterize whether Reddit diffusion is broadcast-dominant (high-degree nodes drive spread) or viral-dominant (structural bridging nodes drive spread). Results are compared against the same configuration-model null to determine how much is degree-explained vs structure-explained.

### Reciprocity Analysis
Global reciprocity = fraction of directed edges (A→B) for which the reverse (B→A) also exists. Weighted reciprocity accounts for edge weights (reply counts). We also compute the reciprocity ratio min(w_AB, w_BA)/max(w_AB, w_BA) per reciprocated pair to measure one-sidedness of acknowledged exchanges.

---

## Reproduction

### Prerequisites

Python 3.8+. Install all dependencies:

```bash
pip install -r requirements.txt
```

### Moltbook Data

The v2 dataset is gitignored (1.09M posts, ~300 MB). Obtain the four CSV files from the authors and place them at:

```
data/data/tables/posts.csv
data/data/tables/agents.csv
data/data/tables/submolts.csv
data/data/tables/membership.csv
```

The Reddit hyperlink data (~304 MB) is downloaded **automatically** in Step 1.

### Run Everything

```bash
python run_all.py
```

`run_all.py` runs all five steps in order, checks for prerequisites before each step, and prints where every output lands. It will skip Moltbook-dependent steps gracefully if the v2 tables are not present.

**What each step does:**

| Step | What runs | Output |
|---|---|---|
| 1 | Download Reddit hyperlink TSV (~304 MB, auto-skipped if present) | `data/reddit/` |
| 2 | Extract degree sequences; power-law fit + Gini + FIG 1 CCDF | `analysis/individual/results/` |
| 3 | Moltbook + Reddit core-periphery null test; FIG 2 comparison arrow plot | `analysis/community/results/` + `figures/` |
| 4 | Moltbook community cohesion vs null (SI) | `analysis/community/results/` + `figures/` |
| 5 | Reddit IC/SIR diffusion, core-periphery, reciprocity (Sam's SI pipeline) | `analysis/reddit_diffusion/figures/` |

**Expected runtime:** Steps 3–5 each run a 300-500 graph null ensemble and will take 5–20 minutes on a standard laptop. Steps 1–2 are fast once the TSV is downloaded.

---

## File Structure

```
networks-project/
│
├── run_all.py                        # Single entry point — runs full analysis
├── fetch_data.py                     # Moltbook API crawler (incremental, for v1 data)
├── requirements.txt
│
├── analysis/
│   ├── individual/                   # R1 + R2: concentration & reciprocity
│   │   ├── fetch_reddit.py           # Downloads SNAP Reddit hyperlink TSV
│   │   ├── extract_degrees.py        # Builds degree sequence CSVs
│   │   └── analyze_degrees.py        # Power-law fit, Gini, CCDF figures
│   │
│   ├── community/                    # R3 + SI: core-periphery & cohesion
│   │   ├── moltbook_coreperiphery.py # k-core + clustering + null (Moltbook)
│   │   ├── reddit_coreperiphery.py   # same test on Reddit hyperlink graph
│   │   ├── make_comparison.py        # FIG 2: cross-platform arrow plot
│   │   └── moltbook_cohesion.py      # SI: clustering vs null model
│   │
│   └── reddit_diffusion/             # SI: Sam's Reddit analysis pipeline
│       ├── graph_builder.py          # Loads + caches Reddit hyperlink graph
│       ├── rq1_diffusion_mode.py     # IC/SIR diffusion simulation (Method 1)
│       ├── rq1_null_model.py         # Diffusion null model (Method 2)
│       ├── rq4_core_periphery.py     # Reddit core-periphery structure
│       └── rq5_reciprocity.py        # Reddit reciprocity + structural prediction
│
├── data/
│   ├── data/tables/                  # Moltbook v2 (gitignored — contact authors)
│   │   ├── posts.csv                 # 1,089,902 posts
│   │   ├── agents.csv                # 27,342 agents
│   │   ├── submolts.csv              # 2,654 submolts
│   │   └── membership.csv            # 59,631 agent×submolt pairs
│   ├── graphs/
│   │   ├── shared_agent_edges_core.csv   # community graph edges (≥5 authors, ≥2 shared)
│   │   ├── shared_agent_nodes_core.csv   # community graph nodes + degree
│   │   └── shared_agent_stats_core.json  # graph summary statistics
│   └── reddit/                       # auto-downloaded by run_all.py
│       └── soc-redditHyperlinks-body.tsv
│
├── figures/                          # pre-computed outputs from earlier analysis runs
│
└── archive/                          # superseded scripts from earlier iterations
```

---

## Key Numbers at a Glance

| Metric | Moltbook | Reddit |
|---|---|---|
| Platform type | AI agents | Humans |
| Communities (analysis graph) | 533 submolts (core) | 35,776 subreddits |
| Edges (community graph) | 25,445 | 124,330 |
| Top 1% agent attention share | **31%** | lower |
| In-degree Gini (agents) | **0.85** | lower |
| Degree distribution | not scale-free (lognormal) | not scale-free |
| Reply reciprocity (individual) | **~0.23** | higher |
| Global transitivity (individual) | 0.059 | — |
| Within-core ρ (k-core vs clustering) | −0.76 (raw) | −0.15 (raw) |
| Within-core ρ vs null (null-relative) | **−0.68 below null** | **−0.72 below null** |
| Core-periphery verdict | central = bridges | central = bridges |

The null-relative within-core ρ is the key comparable metric: both platforms bend ~0.7 below their own degree-null, meaning the same structural role for high-k-core communities emerges independently on both platforms.

---

## Authors & Contributions

| Author | Contribution |
|---|---|
| August Curtis | Moltbook Data Collection, Individual-level agent graph analysis; reciprocity and attention metrics |
| Linlong Wang | Individual-level analysis; concentration measures |
| Samuel Kelly | Reddit hyperlink community graph; IC/SIR diffusion (RQ1); core-periphery (RQ4); reciprocity (RQ5) |
| Ved Vedere | Moltbook community graph; core-periphery comparison (RQ3); cohesion analysis (RQ1 Moltbook side) |
| Riley Leong | Moltbook data collection (v1 + v2 scraping pipeline); individual-level concentration and adoption analysis |

---

## References

- Kumar et al. (2018). Community Interaction and Conflict on the Web. *WWW 2018*. [SNAP Reddit Hyperlink dataset]
- Barabási & Albert (1999). Emergence of scaling in random networks. *Science*.
- Mislove et al. (2007). Measurement and analysis of online social networks. *IMC*.
- Clauset, Shalizi & Newman (2009). Power-law distributions in empirical data. *SIAM Review*.
- Broido & Clauset (2019). Scale-free networks are rare. *Nature Communications*.
- Newman (2018). *Networks*, 2nd ed. Oxford University Press.
- Kempe, Kleinberg & Tardos (2003). Maximizing the spread of influence. *KDD*.
- Hagberg et al. (2008). NetworkX: network analysis in Python. *SciPy Conference*.
