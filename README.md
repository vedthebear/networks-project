# Moltbook Analysis: Who Drives Information Flow?

**Research Question:** Are network HUBS (high-PageRank agents) the first to adopt new topics, or do BROKERS (high-betweenness agents) introduce topics earlier and drive initial spread?

## Background

Moltbook is a decentralized social network platform where autonomous agents post, comment, and interact. This project analyzes information adoption patterns through the lens of network centrality.

**Hypothesis:** In decentralized networks, information typically spreads through peripheral BROKERS (agents with high betweenness—cross-community connectors) rather than central HUBS (agents with high PageRank—within-community influencers). We test this by comparing the centrality profiles of early vs. late topic adopters.

---

## Quick Start

### 0. Install Dependencies
```bash
pip install -r requirements.txt
```

### 1. Collect Data
```bash
# Set up your .env file with Moltbook API credentials
echo "MOLTBOOK_API_KEY=your_key_here" > .env

# Fetch posts, comments, agents, and build network edge lists
python fetch_data.py
```

This generates:
- `data/posts.json` — all posts
- `data/comments.json` — all comments  
- `data/agents.json` — agent profiles

### 2. Build Reply Graph
```bash
# Extract author information and build reply network edges
python build_reply_graph.py
```

This generates:
- `data/agent_reply_edges.csv` — reply graph edges (directed, weighted)

### 3. Analyze Adoption & Influence
```bash
python analyze_adoption_influence.py
```

Outputs (in `figures/`):
- `adoption_centrality_by_topic.csv` — per-topic statistics
- `adoption_hub_vs_broker.png` — violin plots of centrality by adoption quartile
- `adoption_timing_vs_centrality.png` — scatter plots of adoption order vs centrality
- `adoption_influence_summary.txt` — narrative summary

---

## Analysis Pipeline

### Step 1: Network Construction
Load the reply graph from `agent_reply_edges.csv`. Edges are directed (replier → recipient) with weights = number of replies.

### Step 2: Centrality Computation
Compute two node centrality measures:
- **PageRank** (hub score) — agents who receive many replies / replies to important agents
- **Betweenness centrality** (broker score) — agents who bridge otherwise-disconnected parts of the network

### Step 3: Topic Extraction
Extract the top-K topics from post/comment text:
1. Prefer hashtags (e.g., `#topic`)
2. Fall back to high-frequency keywords (5+ letters, non-stopwords)
3. Keep topics with ≥20 mentions

### Step 4: Adoption Tracking
For each topic and agent, record the **first timestamp** they mention it.

### Step 5: Stratification & Comparison
For each topic:
- Rank agents by adoption order (earliest first)
- Compare centrality of early adopters (Q1, ~25%) vs late adopters (Q4, ~75%)
- Test correlation: adoption rank vs PageRank / betweenness

### Step 6: Visualization & Summary
Generate violin plots (centrality distributions by adoption quartile) and scatter plots (adoption rank vs centrality), plus a narrative summary.

---

## Key Metrics

**Per Topic:**
- `early_pagerank_mean` / `late_pagerank_mean` — hub score comparison
- `early_betweenness_mean` / `late_betweenness_mean` — broker score comparison
- `pagerank_adoption_corr` — Spearman rank correlation between adoption order and PageRank
- `betweenness_adoption_corr` — Spearman rank correlation between adoption order and betweenness
- `early_adopters_are_hubs` — Boolean: do early adopters have higher PageRank than late?
- `early_adopters_are_brokers` — Boolean: do early adopters have higher betweenness than late?

**Interpretation:**
- If early adopters have **higher PageRank**: HUB-driven adoption (popular agents lead)
- If early adopters have **higher betweenness**: BROKER-driven adoption (connectors lead)
- If correlations are **weak/insignificant**: adoption is **not** driven by network position

---

## Expected Findings

We hypothesize:
1. Early topic adopters will have **lower PageRank** on average (not the central hubs)
2. Early adopters will have **higher betweenness** on average (brokers connect communities)
3. Adoption order will **negatively correlate** with PageRank
4. Adoption order will **positively correlate** with betweenness
5. Topics vary: some are hub-driven, others broker-driven

---

## File Structure

```
networks-project/
├── fetch_data.py                      # Data collection (Moltbook API → JSON + CSVs)
├── analyze_adoption_influence.py      # Main analysis: adoption × centrality
├── README.md                          # This file
├── data/
│   ├── posts.json / posts.csv
│   ├── comments.json / comments.csv
│   ├── agents.json / agents.csv
│   └── agent_reply_edges.csv          # Reply graph (primary input for analysis)
└── figures/                           # Outputs
    ├── adoption_centrality_by_topic.csv
    ├── adoption_hub_vs_broker.png
    ├── adoption_timing_vs_centrality.png
    └── adoption_influence_summary.txt
```

---

## Dependencies

- Python 3.8+
- networkx
- pandas
- numpy
- scipy
- matplotlib
- requests (for API)
- python-dotenv (for .env)

Install:
```bash
pip install networkx pandas numpy scipy matplotlib requests python-dotenv
```

---

## Paper Outline (3 pages)

1. **Introduction** — Decentralized platforms, information diffusion, hub vs. broker hypothesis
2. **Method** — Network construction, centrality measures, adoption tracking, stratification
3. **Results** — Per-topic findings, violin plots, correlation analysis, patterns across topics
4. **Discussion** — Implications: brokers as tastemakers? Platform design insights
5. **Conclusion** — Summary + future work (e.g., causal inference, agent behavior)

---

## Contact

Built for Math 168 (Networks) course project.
