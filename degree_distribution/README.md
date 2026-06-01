# Scale-Free or Not? Degree Distributions of Agent vs. Human Community Networks

**Research question (RQ2 for the paper).**
Human community networks (Reddit) are famously *heavy-tailed*: a handful of
subreddits are wildly more connected than the rest, and the degree distribution
follows a power law (this is what "scale-free" means). Do **AI-agent** community
networks (Moltbook) look the same — or is the "ghost-town" concentration we see
on Moltbook (a few giant submolts, thousands of near-empty ones) a *different*,
distinctly agentic signature?

We answer this by comparing the **degree distributions** of both platforms'
community networks: log-log plots, formal power-law fits, and a comparison of the
fitted exponents.

---

## TL;DR for teammates (the 30-second version)

- A network's **degree distribution** = "how many connections does each node have,
  and how common is each connection-count?" Plot it and you can see whether a few
  hubs dominate (heavy tail) or everyone is roughly similar (light tail).
- **Reddit** subreddits: classic heavy-tailed / scale-free shape (expected).
- **Moltbook** submolts: *[FILL IN after running — e.g., "even more extreme / not a
  clean power law / similar exponent"]*.
- The headline figure is `results/degree_ccdf_comparison.png`.
- Full numbers (fitted exponent α, goodness-of-fit, distribution comparisons) are in
  `results/powerlaw_summary.csv`.

---

## Why this is a good question for the paper

1. **It uses a core course concept directly.** Power laws and scale-free networks are
   Lecture Notes §3.3–§3.3.1, and degree distributions are §1.5 / §3.2. We use the
   exact detection method the notes describe (see *Methods*).
2. **One killer figure.** The whole result is visible in a single side-by-side log-log
   plot — ideal for a 4-page paper.
3. **It reuses data we already have** (the SNAP Reddit hyperlink graph and the Moltbook
   shared-agent community graph) and ties to our running finding that agent communities
   are extremely concentrated.

---

## What "degree" means in each network (important — they are built differently)

| | **Reddit (human)** | **Moltbook (agent)** |
|---|---|---|
| Node | subreddit | submolt (a community) |
| Edge | subreddit A links to B (a hyperlink in a post/comment) | A and B share posting agents |
| Degree of a node | # of other subreddits it links to / is linked from | # of other submolts it shares agents with |
| Source | SNAP Reddit Hyperlink corpus | our live Moltbook scrape → bipartite projection |

Both are **community-to-community networks**, so "degree distribution of the community
network" is a fair thing to compare. But the *edges mean different things* (a reference
vs. shared membership), so we focus on the **shape** of the distribution (heavy-tailed or
not) and the **fitted exponent**, not on raw degree values. See *Caveats*.

---

## Methods (and how they map to the lecture notes)

### 1. Build the degree sequences
- **Reddit:** load the hyperlink edge list, build the **undirected projection** (an edge
  exists if a link runs in *either* direction — same projection our RQ4 core-periphery
  work uses), and read off each subreddit's degree. (~35.8k nodes.)
- **Moltbook:** the shared-agent community graph is already built
  (`moltbook_v2/graphs/`). We read each submolt's degree from the node table. We analyze
  two variants for robustness:
  - **core** — submolts with ≥5 distinct authors, edge if ≥2 shared agents (533 nodes,
    the clean analyzable graph);
  - **full** — all submolts (2,654 nodes, includes the dead tail).

### 2. Visualize on a log-log scale — using the CCDF, not a raw histogram
The notes (§3.3.1) warn that **raw log-log histograms are unreliable**: with small bins
you get noise in the tail, with big bins you lose detail. The recommended fix is the
**complementary cumulative distribution function (CCDF)**:

> P(k) = fraction of nodes with degree ≥ k.

If the degree distribution has a power-law tail `p(k) ∝ k^(−α)`, then the CCDF is also a
power law, with exponent `(α−1)`, and shows up as a straight line on a log-log plot. We
plot the CCDF for every distribution.

### 3. Fit the power law *properly* (not by eyeballing a line)
The notes also warn (§3.3.1) that you **can't** just fit a straight line to the log-log
CCDF, because successive points are correlated. So instead of a line fit we use the
standard **maximum-likelihood method of Clauset, Shalizi & Newman (2009)**, via the
`powerlaw` Python package:
- estimate the lower cutoff `k_min` (the point beyond which the tail is power-law) by
  minimizing the Kolmogorov–Smirnov (KS) distance between data and fit;
- estimate the exponent **α** by maximum likelihood on the tail `k ≥ k_min`;
- report a **goodness-of-fit** and, crucially, **compare the power law against
  alternative heavy-tailed shapes** (lognormal, exponential) with a likelihood-ratio
  test. "It looks straight-ish on a log-log plot" is *not* evidence of a power law on its
  own; the comparison test is what lets us actually claim scale-free or not.

### 4. Also measure raw concentration (the "ghost town" signal)
Because the shared-agent projection can mechanically inflate degrees (one hyperactive
agent posting in K submolts links all K together), we *also* report concentration
measures that don't depend on the projection at all:
- **posts per submolt** and **authors per submolt** (2,654 submolts),
- **posts per agent** (27,342 agents),
- the **Gini coefficient** and **top-k share** (e.g., "the top 1% of submolts hold X% of
  all posts").

These make the concentration finding robust to how the graph was built.

---

## How to reproduce

```bash
# from repo root, with the project venv active
cd degree_distribution

# 1. Reddit hyperlink data (~304 MB; skipped if already present)
python scripts/fetch_reddit.py

# 2. Extract degree sequences + concentration tables for both platforms
python scripts/extract_degrees.py

# 3. Fit power laws, run comparison tests, make figures
python scripts/analyze.py
```

Outputs:
- `results/degree_ccdf_comparison.png` — the main figure (Reddit vs. Moltbook).
- `results/powerlaw_summary.csv` — fitted α, k_min, goodness-of-fit, and
  power-law-vs-alternative comparisons for every distribution.
- `results/concentration.csv` — Gini and top-k shares.
- `data/` (git-ignored) — raw Reddit TSV and intermediate degree-sequence CSVs.

---

## Directory layout

```
degree_distribution/
├── README.md            ← this file
├── scripts/
│   ├── fetch_reddit.py      download the SNAP Reddit hyperlink TSV
│   ├── extract_degrees.py   build degree sequences + concentration tables
│   └── analyze.py           power-law fits + comparison figure
├── data/    (git-ignored)   raw + intermediate data
└── results/ (committed)     final figure + summary tables
```

---

## Caveats (state these honestly in the paper)

- **Different edge meanings.** Reddit edges are *references*; Moltbook edges are *shared
  membership*. We compare distribution *shape* and *exponents*, not raw degrees.
- **Projection inflation.** Shared-agent projections turn one multi-community agent into
  a clique, which can fatten the degree tail. This is exactly why we also report the
  projection-free concentration measures (posts/authors per submolt).
- **Size gap.** Reddit has ~35.8k community-nodes; the Moltbook core graph has ~533.
  Power-law fits are less certain on smaller samples; we report fit diagnostics honestly
  and use the full 2,654-node graph as a robustness check.

---

## Results

*(filled in after running `analyze.py` — see `results/` for the figure and tables.)*
