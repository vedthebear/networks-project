# Who Drives Information Flow on Moltbook?

A network analysis of topic diffusion on Moltbook — an AI-agent social platform — comparing how information spreads at two levels: submolt-to-submolt (analogous to Reddit hyperlink analysis) and agent-to-agent (reply graph).

Built for Math 168 (Networks), UCLA.

---

## What This Project Does

Moltbook is organized into **submolts** (communities), populated entirely by autonomous AI agents. We treat **hashtags as hyperlinks**: when a hashtag first appears in submolt A and later shows up in submolt B, we draw a directed edge A → B, exactly the way the Reddit hyperlink network treats cross-community links.

From this we build two graphs:

| Graph | Nodes | Edges | Method |
|---|---|---|---|
| Submolt diffusion graph | 213 submolts | 7,985 directed edges | First-adoption of 2,201 hashtags |
| Shared-agent graph (core) | 533 submolts | 25,445 edges | Submolts linked when ≥2 agents post in both |

We then run centrality analysis (PageRank, betweenness, HITS hub/authority) on both graphs to identify which submolts drive information flow.

---

## Dataset

The v2 dataset is a full-platform scrape of Moltbook, covering its entire existence:

| File | Rows | Description |
|---|---:|---|
| `posts.csv` | **1,089,902** | post_id, author, submolt, created_at, score, title |
| `agents.csv` | 27,342 | per-agent activity summary |
| `submolts.csv` | 2,654 | per-submolt post/author counts |
| `membership.csv` | 59,631 | agent × submolt bipartite edge list |

**Date range: 2026-01-28 → 2026-06-01 (124 days — full platform lifetime)**

Monthly post volume: Jan 5k · Feb 90k · Mar 231k · **Apr 408k (peak)** · May 348k · Jun 8k

The data lives in `data/data/tables/` and is gitignored (too large). See [Reproducing the Dataset](#reproducing-the-dataset) to regenerate it.

---

## Reproducing the Analysis

### Prerequisites

Python 3.8+ required.

```bash
pip install -r requirements.txt
```

### Step 1 — Get the Data

**Option A: Use the pre-built v2 dataset (recommended)**

The dataset was collected via a max-volume API scrape. If you have access to the raw files, place them at:

```
data/data/tables/posts.csv
data/data/tables/agents.csv
data/data/tables/submolts.csv
data/data/tables/membership.csv
```

**Option B: Collect fresh data from the Moltbook API**

Set your API key in `.env`:

```bash
echo "MOLTBOOK_API_KEY=your_key_here" > .env
```

Then run the crawler (incremental — safe to stop and resume):

```bash
python fetch_data.py
```

This writes to `data/posts.json`, `data/agents.json`, `data/comments.json` with checkpointing. The v1 JSON format is also accepted by both analysis scripts as a fallback.

### Step 2 — Run Topic Diffusion Analysis

Builds the submolt-to-submolt hashtag diffusion graph and computes centrality:

```bash
python analyze_topic_diffusion.py
```

Outputs to `figures/`:
- `submolt_influence_ranking.png` — top submolts by PageRank
- `topic_diffusion_network.png` — network diagram of the diffusion graph
- `hashtag_adoption_curves.png` — spread curves for top hashtags
- `submolt_centrality.csv` — full per-submolt metrics table
- `submolt_diffusion_edges.csv` — full edge list with weights
- `topic_first_touch.csv` — per-hashtag first-adoption record
- `diffusion_summary.txt` — text summary of key results

### Step 3 — Run Adoption Influence Analysis

Analyzes whether hub agents or broker agents are the first to adopt new topics:

```bash
python analyze_adoption_influence.py
```

Outputs to `figures/`:
- `adoption_centrality_by_topic.csv` — per-topic early vs late adopter centrality
- `adoption_hub_vs_broker.png` — centrality distributions by adoption quartile
- `adoption_timing_vs_centrality.png` — scatter of adoption order vs centrality

> **Note:** This script requires a reply graph (`data/agent_reply_edges.csv`) for agent-level centrality. Without it, it falls back to reporting hashtag adoption statistics only. The v2 dataset does not include comment threads, so building the reply graph requires v1 API data from `fetch_data.py`.

---

## File Structure

```
networks-project/
├── fetch_data.py                     # Moltbook API crawler (incremental, resumable)
├── analyze_topic_diffusion.py        # Submolt hashtag diffusion analysis
├── analyze_adoption_influence.py     # Agent hub vs broker adoption analysis
├── requirements.txt
├── .env                              # API key (gitignored)
│
├── data/
│   ├── data/tables/                  # v2 dataset (gitignored, 1.09M posts)
│   │   ├── posts.csv
│   │   ├── agents.csv
│   │   ├── submolts.csv
│   │   └── membership.csv
│   ├── graphs/
│   │   ├── shared_agent_edges_core.csv   # submolt graph (≥5 authors, ≥2 shared agents)
│   │   ├── shared_agent_nodes_core.csv
│   │   └── shared_agent_stats_core.json
│   └── archive/                      # old v1 fetched data (superseded)
│
├── figures/                          # all analysis outputs (charts + CSVs)
│
└── archive/                          # old scripts (Reddit reference, early iterations)
```

---

## Key Results

From the full 1.09M-post dataset:

- **2,201 hashtags** tracked across **213 submolts**, producing **7,985 directed diffusion edges**
- **`creativeprojects`** is the dominant topic originator (HITS hub = 0.204, out-degree 163) — seeds topics into the broader network but has low PageRank itself
- **`general`** and **`mbc-20`** are dual-role: high hub score AND high PageRank — both originate and receive topics
- **`general`** has the highest betweenness centrality (0.146), making it the key bridge between topic communities
- Out-weight vs in-weight Spearman r = 0.54 — the same submolts dominate both sending and receiving topics

---

## Dependencies

| Package | Use |
|---|---|
| `pandas` | Data loading and vectorized processing |
| `networkx` | Graph construction and centrality algorithms |
| `scipy` | Spearman correlation |
| `matplotlib` | Visualizations |
| `requests` | Moltbook API calls |
| `python-dotenv` | `.env` file loading |
| `numpy` | Numerical operations |
