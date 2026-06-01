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
- **Reddit** subreddits: classic heavy-tailed shape — degree spans from 1 to 2,336,
  the top 10% of subreddits hold 72% of all the connections. A long, straight-ish
  tail on the log-log plot.
- **Moltbook** submolts: **not** scale-free in the same way. The community network's
  degree is *bounded* and drops off sharply (a "characteristic scale"), and the
  active core is actually *more even* than Reddit (Gini 0.50 vs 0.76).
- **The distinctly agentic signature is in *activity*, not *connectivity*.** Where
  agents *post* is brutally concentrated: the top 1% of submolts hold **81%** of all
  posts (Gini 0.975) — a far more extreme winner-take-all than anything on the Reddit
  side. That's the "ghost town" — a few hyperactive communities, thousands near-dead.
- **One-line answer to the RQ:** *No — agent community networks are not scale-free
  like human ones. Their hallmark is extreme concentration of activity, not a
  heavy-tailed web of community-to-community links.*
- Headline figure: `results/degree_ccdf_comparison.png`. Numbers: `results/powerlaw_summary.csv`
  and `results/concentration.csv`.

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

### The headline figure (`results/degree_ccdf_comparison.png`)
On a log-log CCDF plot, **Reddit (blue)** is a long, gently-sloping tail stretching
across four orders of magnitude (degree 1 → 2,336). **Moltbook (orange)** is
compressed into a much narrower range and then **bends sharply downward** — the
signature of a distribution with a *characteristic scale* rather than a scale-free
power law.

### Power-law fits (`results/powerlaw_summary.csv`)
Using the Clauset–Shalizi–Newman maximum-likelihood method (`R` = likelihood-ratio;
`R>0` favors a power law, `R<0` favors the alternative; small `p` = significant):

| Distribution | n | α | k_min | vs. exponential | vs. lognormal | verdict |
|---|---|---|---|---|---|---|
| Reddit subreddit degree | 35,776 | 1.80 | 1 | R=+33.5 (p≈0) | R=−10.8 (p≈0) | heavy-tailed; lognormal competitive |
| Moltbook submolt degree (core) | 527 | 2.95 | 111 | R=−8.3 (p≈0) | R=−4.9 (p≈0) | **no clear power-law tail** |
| Moltbook submolt degree (full) | 868 | 2.80 | 109 | R=−7.3 | R=−4.7 | **no clear power-law tail** |
| Moltbook posts per submolt | 2,654 | 1.71 | 281 | R=+2.6 (p=.009) | R=−0.8 (ns) | heavy-tailed; power law plausible |
| Moltbook authors per submolt | 2,654 | 1.62 | 4 | R=+5.6 | R=−1.9 (p=.06) | heavy-tailed |
| Moltbook posts per agent | 27,342 | 1.59 | 2 | R=+33.5 | R=−8.2 | heavy-tailed |

**Reading this honestly:** Reddit's community degree is strongly heavy-tailed —
a power law fits the tail far better than an exponential. But, as is true for *most*
real "scale-free" networks (Broido & Clauset 2019, *Scale-free networks are rare*),
a lognormal fits at least as well, so we say "heavy-tailed" rather than claiming a
pure power law. The key contrast is that the **Moltbook community-graph degree fails
the test in the other direction**: an exponential beats the power law (R<0), i.e. it
has a *thin* tail with a characteristic scale — the opposite of scale-free.

### Concentration / the "ghost town" (`results/concentration.csv`)

| Distribution | Gini | top 1% share | top 10% share |
|---|---|---|---|
| Reddit subreddit degree | 0.76 | 34% | 72% |
| Moltbook submolt degree (core) | 0.50 | 4% | 29% |
| **Moltbook posts per submolt** | **0.975** | **81%** | **97%** |
| Moltbook authors per submolt | 0.92 | 62% | 92% |
| Moltbook posts per agent | 0.91 | 54% | 90% |

This is the punchline. The Moltbook *connection* network is more even than Reddit's
(Gini 0.50 < 0.76), but Moltbook *activity* is wildly more concentrated: **81% of all
posts come from the top 1% of submolts.** The agentic signature isn't a heavy-tailed
web of hubs — it's a winner-take-all distribution of *where the activity happens*.

### What this means for the paper
A clean, slightly counter-intuitive result that's easy to defend: **agent community
networks are not scale-free the way human ones are.** Humans build a heavy-tailed
web of cross-community connections; agents instead pile nearly all activity into a
few communities while the connection structure among active communities stays
relatively flat. Pairs naturally with our other finding that agents don't bridge
communities (no hyperlinks / cross-community hashtags).
