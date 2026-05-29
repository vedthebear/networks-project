# Reddit Subreddit Diffusion — Research Plan

## Research Questions & Motivation

This project is **a set of questions we are deriving answers for, not a thesis we are defending.** We do not assert an outcome up front. The final conclusion is whatever the evidence supports — and where the evidence overturns an initial expectation, that overturning is itself a legitimate result, reported plainly.

The investigation is loosely organized around one through-line we are *interested in testing* (not assuming): **does a subreddit's structural position in the hyperlink network determine the role it plays in information flow, beyond what its raw activity level (degree) would predict?** Each research question contributes evidence for or against this, and the synthesis is written from the results.

The original framing of this project predicted **broadcast-dominant diffusion** — that information spreads through a few structurally-determined gateway hubs rather than peer-to-peer chains. We treat that prediction as the open question **RQ1 set out to answer**, not as a settled claim. As it turns out (see RQ1 below and `docs/rq1/`), the evidence **disproves** it: diffusion is viral in shape, and that shape is fully explained by the degree distribution. That is a valid finding. It also redirects the rest of the project toward the structural features where a degree-preserving null model *can* be beaten — local clustering hierarchy (RQ4), reciprocity (RQ5), and sentiment-stratified roles (RQ3).

Every quantitative claim is tested against a null model (what we'd expect by chance given the degree sequence), so that findings are inferences, not just observed patterns.

---

## Data Infrastructure

### Primary Dataset: Stanford SNAP Reddit Hyperlink Corpus

The core dataset is a table (`body_df`) where each row represents one hyperlink that appeared in a Reddit post or comment. Key columns:

| Column | Description |
|---|---|
| `SOURCE_SUBREDDIT` | The subreddit that posted the link |
| `TARGET_SUBREDDIT` | The subreddit being linked to |
| `TIMESTAMP` | When the post was made |
| `LINK_SENTIMENT` | Whether the post was positive or negative toward the target |
| + 63 more | Various post-level properties |

From this table, the existing code builds:
- **Weighted DiGraph (`weighted`)** — one directed edge per (source, target) pair, weight = number of hyperlinks. Primary graph for structural analysis.
- **Raw multigraph (`raw_multi`)** — one edge per individual hyperlink (preserves timestamps; used for temporal analysis).
- **Positive/negative subgraphs** — weighted graph filtered by sentiment.

### Analysis File Layout

Each research question lives in its own module: `rq1_diffusion_mode.py`, `rq1_null_model.py`, `rq3_roles.py`, `rq4_core_periphery.py`, `rq5_reciprocity.py`. Shared graph construction is in `graph_builder.py`. (See `CLAUDE.md` for the full file tree.)

### Dependencies (in `requirements.txt`)
- `scikit-learn` — logistic regression (RQ5); `seaborn`, `tqdm` — plotting/progress.

---

## Null Model Methodology

A core methodological principle running through this entire analysis is that **a statistic computed on a single network is not a finding — it only becomes a finding when compared to a baseline**. Without a baseline, we cannot distinguish "Reddit is structured this way" from "any network of this size and density would look like this."

The standard baseline in network science is the **configuration model** — a family of random graphs that preserve the degree sequence of the real network while randomizing everything else. The degree sequence is the list of how many connections each node has. Preserving it means the random graphs have the same size and the same distribution of highly-connected vs. poorly-connected nodes as Reddit's actual network. What they don't preserve is *which* nodes are connected to *which* — that is randomized.

This matters because many structural properties (like high clustering or clear sender/receiver asymmetry) can emerge just from having a few extremely high-degree nodes, regardless of any meaningful organization in the network. By comparing our real findings to configuration model baselines, we test: is what we observe a product of Reddit's actual social structure, or would any network with the same degree distribution look the same?

### How the Null Model is Generated

For a directed network like the Reddit hyperlink graph, the directed configuration model works as follows:

1. Record every node's in-degree and out-degree from the real network
2. Generate a new random graph where each node is assigned exactly that many incoming and outgoing "stubs" (half-edges)
3. Randomly match outgoing stubs to incoming stubs across the full network
4. The result is a random directed graph with the same in- and out-degree sequence as the original, but with edges placed randomly

In Python, `nx.directed_configuration_model(in_degree_sequence, out_degree_sequence)` handles this directly. The result may have self-loops and multi-edges (a node connecting to itself, or two nodes having multiple edges between them), which are artifacts of the random matching. These are typically removed or filtered out before analysis.

### Generating a Distribution, Not a Single Random Graph

One random graph is not a useful baseline — it could be an unusual draw. We generate an **ensemble** of 500–1000 configuration model graphs and compute our metric of interest on each one. This produces a null distribution: a histogram of what values the metric takes in random networks with Reddit's degree sequence.

We then compare the real network's observed metric to this distribution using a **permutation test**:
- If the real value falls far in the tail of the null distribution (e.g., above the 95th percentile), we conclude the finding is statistically non-random
- We report a p-value: the fraction of null graphs whose metric exceeded the real network's value

### Where Null Model Testing Applies in This Analysis

The null model runs through three of the five research questions:

- **RQ1 (SIR simulation):** Is Reddit's diffusion shape (broadcast vs. viral) different from what a random network with the same degree sequence would produce? *(Result: no — the viral shape is degree-explained.)*
- **RQ3 (role asymmetry):** Is the variance in sender/receiver roles across subreddits significantly higher than in random networks — meaning role assignment is non-random?
- **RQ4 (core-periphery correlation):** Is the anti-correlation between k-core number and clustering coefficient significantly stronger in Reddit's network than in random networks of the same degree distribution?

In each case, the null model testing is what elevates the analysis from descriptive to inferential — from "we observe X" to "X is a statistically meaningful property of this specific network."

---

## Research Question 1: Where Does Reddit Fall on the Broadcast-to-Viral Spectrum?

> **✅ COMPLETE — Finding:** Diffusion is **viral** (median width at hop 1 = 1, median depth = 10–12 in active cascades), the opposite of the broadcast prediction. Method 2's configuration-model null shows this viral shape is **not statistically distinguishable from a degree-matched random graph** (all p > 0.27, none near significance). **Conclusion: the viral shape is real but fully explained by the degree distribution — it is not a non-random property of Reddit's wiring.** Full write-up in `docs/rq1/README.md`.

### The Question
When information begins in one subreddit and spreads via cross-community hyperlinks, does it tend to spread directly outward from a single hub (broadcast), or does it pass through a chain of intermediate communities (viral)? Is the pattern more structured than we'd expect in a randomly wired network?

### Why This Anchors the Paper
This is the thesis-testing question. All other RQs either characterize the mechanism behind this answer or test whether the pattern is structurally determined.

### Methods

#### SIR Simulation
SIR (Susceptible-Infected-Recovered) is a diffusion model borrowed from epidemiology. Each subreddit starts susceptible (hasn't seen the information). At each time step, infected nodes spread to neighbors with probability β. After spreading, nodes move to recovered.

Running many simulations from many starting nodes produces a distribution of outcomes. Two signals distinguish broadcast from viral:
- **Cascade depth** (maximum hops from seed to furthest reached node) — longer = more viral
- **Cascade width** (breadth at each level of the tree) — wider and shallower = more broadcast

Parameters β and γ are varied in sensitivity analyses to ensure findings aren't parameter-dependent.

#### Configuration Model Null Comparison
See the Null Model Methodology section above for the full explanation. For RQ1 specifically: we generate 500 configuration model graphs, run the same SIR simulation on each, and build a null distribution of cascade depth and width values. The real network's values are compared to this distribution via permutation test. **Result:** Reddit's cascade depth and width fell squarely inside the null distribution at every β (all p > 0.27), so the observed viral shape is an artifact of the degree distribution, not a non-random property of Reddit's wiring. See `docs/rq1/README.md` for the full Method 2 write-up.

### Data Needed
- `weighted` DiGraph (already built)
- **Build:** SIR simulation function, cascade depth/width measurement, configuration model loop, permutation test

---

## Research Question 2: ~~Which Subreddits Function as Structural Gatekeepers?~~ — RETIRED

**RQ2 has been retired.** It was always contingent on RQ1: a broadcast result would justify identifying gatekeeper *hubs*, while a viral result would have it reframed toward identifying *relay nodes* in long diffusion chains.

RQ1 Method 2 invalidates **both** framings. The configuration-model null showed that Reddit's cascade structure is **fully explained by its degree distribution** — a degree-matched random graph produces the same narrow, deep cascades. If the aggregate diffusion structure is degree-generic, then any node-level *diffusion-position* analysis (gatekeeper or relay) has no statistical footing: the positions nodes occupy in cascades are not a non-random property of Reddit's wiring, so "which node is the critical relay/hub" cannot be distinguished from chance.

Retiring RQ2 is the honest consequence of the RQ1 finding, not a gap. The project re-anchors on the structural features a degree-preserving null *cannot* reproduce — local clustering hierarchy (RQ4), reciprocity (RQ5), and sentiment-stratified role structure (RQ3).

**Note on the k-core dependency:** RQ5's logistic regression originally drew k-core numbers from RQ2. K-core is a pure graph statistic (`nx.core_number(UG)`) computed independently of RQ2; it is now produced in **RQ4** and reused by RQ5.

---

## Research Question 3: Are Sender and Receiver Roles Structurally Determined?

### The Question
Do subreddits occupy persistent asymmetric roles as net senders or net receivers of cross-community links? Is this asymmetry more pronounced than a random network would produce?

### Aggregation Caveat
The hyperlink dataset is aggregate flow — it records total links between subreddit pairs, not individual events. We can characterize structural net senders but cannot claim "origin" in the content-diffusion sense. This distinction is stated clearly in the paper.

### Methods

#### Fan-Out Ratio (Origin Score)
`origin_score = total_outgoing_weight / (total_outgoing + total_incoming_weight)`

Score near 1.0 = structural sender. Near 0.0 = structural receiver. Near 0.5 = balanced. Simple and directly interpretable.

#### HITS Algorithm
HITS assigns two scores per node:
- **Hub score:** High if the node links to many well-linked-to destinations — captures "good sender" behavior
- **Authority score:** High if the node is linked to by many active senders — captures "good receiver" behavior

Computed iteratively via the adjacency matrix until convergence. Complements fan-out ratio by accounting for *who* you link to, not just volume. `nx.hits(weighted, max_iter=300)`.

#### Null Model Test for Role Asymmetry — with a pre-registered caveat
See the Null Model Methodology section. For RQ3: we compute origin scores for every node in 500+ configuration model graphs and record the variance of those scores in each random graph. This builds a null distribution of "how spread-out are sender/receiver roles in a random network with Reddit's degree sequence?"

If Reddit's real origin score variance is significantly higher than the null, role assignment is a structural property beyond degree. **However, we pre-register the expectation that this test may come back null — for the same reason RQ1 did.** The origin score, `out_weight / (out_weight + in_weight)`, is largely a function of the in/out-degree (and strength) marginals that the configuration model *preserves*. So basic role asymmetry may be degree-explained, just as cascade shape was. We report this test honestly either way: a null result here is itself a finding, consistent with RQ1, that role *magnitude* is degree-driven.

#### Sentiment-Stratified Role Consistency (primary contribution)
This is the angle the degree-preserving null **cannot** trivialize, because that null ignores sentiment entirely. Using the `pos_weighted` and `neg_weighted` subgraphs (already built by `graph_builder.py`), we compute each node's origin score separately within positive-sentiment links and within negative-sentiment links, then test whether a subreddit's sender/receiver role **persists or flips with sentiment**:

- Do communities that are net *senders* of positive links remain senders when linking negatively, or do roles reorganize by sentiment?
- Is role-by-sentiment consistency itself structured (e.g., correlated with k-core or clustering)?

Because the degree null carries no sentiment information, any systematic role-by-sentiment structure is a genuine, non-degree feature of Reddit — making this the part of RQ3 most likely to yield an inferential finding. We quantify consistency (e.g., correlation between positive-subgraph and negative-subgraph origin scores) and, where a structural statistic is computed, compare against the configuration-model null per project convention.

#### Structural Properties by Role
Bin into sender (>0.7), balanced (0.3–0.7), receiver (<0.3). Compare k-core, clustering coefficient, and PageRank distributions across bins to test whether role is structurally determined.

### Data Needed
- `weighted` DiGraph (built); `pos_weighted` and `neg_weighted` subgraphs (built)
- **Build:** Fan-out ratio per node, HITS, null model variance test (pre-registered caveat), sentiment-stratified role consistency, role binning and comparison

---

## Research Question 4: Does Global Position Predict Local Behavior?

### The Question
Do subreddits in the network's dense core connect across many diverse communities (low local clustering), while periphery subreddits connect within tight local clusters (high clustering)? Is this core-periphery anti-correlation more pronounced than chance?

### Conceptual Background
In many real networks, there is a strong anti-correlation between global embeddedness (k-core number) and local clustering (clustering coefficient). If this holds here, it means broadcast diffusion is concentrated in the core while viral-style local spreading occurs in the periphery — a finding with real implications for how information moves through the platform.

### Methods

#### Local Clustering Coefficient
For node X: what fraction of all possible edges between X's neighbors actually exist? High = X lives in a tight clique. Low = X bridges unconnected groups. `nx.clustering(UG, weight="weight")`.

#### K-Core vs. Clustering Correlation (Spearman)
Scatter k-core number vs. clustering coefficient for all nodes. Spearman rank correlation (robust to skewed distributions, appropriate here). A significant negative correlation confirms the core-periphery anti-correlation hypothesis.

Spearman is used instead of Pearson because neither variable is normally distributed and the relationship may be monotonic but non-linear.

#### Rich-Club Coefficient
Tests whether the highest-degree nodes preferentially connect to each other more than to lower-degree nodes. If Reddit's top subreddits form a rich club, it suggests a self-reinforcing core that captures and recirculates information — amplifying broadcast-style diffusion. `nx.rich_club_coefficient(UG, normalized=False)` restricted to degree ≥ 5.

#### Sentiment by Core Shell
Using `body_df`, compute mean outgoing link sentiment per k-core shell. Do core subreddits link more positively or negatively than periphery subreddits? Connects network structure to the character of information flow.

### Data Needed
- `weighted` DiGraph (built), `body_df` (built), undirected projection
- **Build:** Clustering computation, Spearman test, rich-club analysis, sentiment-by-shell aggregation

---

## Research Question 5: Do Structural Features Predict Reciprocal Linking, and Is There Evidence of Temporal Influence?

### The Question
When subreddit A links to B, does B subsequently link back — and is this predictable from structural features of A and B? When reciprocal linking develops, does one direction tend to precede the other in time, consistent with directional influence?

### Why This Completes the Arc
The previous four questions characterize the static diffusion structure. This question tests whether that structure has dynamic consequences: does structural position shape how inter-community relationships form over time?

### Methods

#### Reciprocity Analysis
A directed edge A→B is reciprocated if B→A also exists. Global reciprocity = fraction of all directed edges with a mutual counterpart. Per-node reciprocity = fraction of a subreddit's outgoing links that are reciprocated. `nx.reciprocity(weighted)`.

#### Logistic Regression on Reciprocity
Logistic regression predicts binary outcomes (reciprocated: yes/no) from predictor features. Unlike linear regression, it outputs probabilities between 0 and 1.

Build an edge-level table where each row is one directed edge. Outcome: `is_reciprocated`. Predictors:

| Feature | What it captures |
|---|---|
| Edge weight | Volume of the relationship |
| Out-degree of source | Sender activity level |
| In-degree of target | Receiver popularity |
| K-core number of both nodes | Structural embeddedness |
| Origin score of both nodes | Sender/receiver role |
| Fraction of positive-sentiment links | Relationship character |

Output: coefficients showing direction and magnitude of each feature's effect, controlling for all others. Tests whether structural position predicts reciprocity independently of activity level.

#### Timestamp Gap Analysis
Infrastructure: load `body_df` → group by (SOURCE, TARGET) → take earliest timestamp per directed pair → for each (A,B) pair where both directions exist, compute gap = first(B→A) - first(A→B).

Positive gap = A→B came first. Negative gap = B→A came first. Distribution shape tells us whether reciprocal relationships have a clear initiator or emerge simultaneously.

**Limitation:** This tells us about the *start* of a relationship — which direction first appeared — but not whether one direction continued to drive the other over time. Framed in the paper as "temporal co-occurrence consistent with directional influence," not causation.

### Data Needed
- `weighted` DiGraph (built), `raw_multi` with timestamps (built), `body_df` (built)
- **Build:** Per-edge reciprocity labels, logistic regression pipeline, timestamp gap computation
- **New deps:** `scikit-learn`

---

## How the Questions Connect

```
RQ1: Asks where Reddit falls on the broadcast–viral spectrum, and whether the pattern is non-random.
     FINDING: viral in shape, but the shape is fully degree-explained (null model not beaten).
  ↓
  [RESOLVED FORK: a degree-explained diffusion structure means node-level diffusion-position
   analysis has no statistical footing → RQ2 (gatekeeper/relay) RETIRED.]
  ↓
  RQ1 redirects the project: look for structure where a degree-preserving null CAN be beaten.
  ↓
RQ3: Are sender/receiver roles structural? Tests role-magnitude vs. degree null (may be degree-explained,
     pre-registered), and — the part the degree null cannot trivialize — sentiment-stratified role consistency.
  ↓
RQ4: Does global position predict local behavior? K-core ↔ clustering anti-correlation and rich-club —
     features the configuration model does NOT preserve, so the strongest candidates for a non-random finding.
     (Produces k-core numbers reused by RQ5.)
  ↓
RQ5: Do structural features predict reciprocal linking over time? Logistic regression on reciprocity
     (using RQ4 k-core + RQ3 origin scores) + timestamp-gap analysis.
```

The synthesis is written from these results. The through-line under test — does structural position determine role/behavior beyond degree? — is supported only where the evidence beats the null; RQ1 shows it does *not* for diffusion shape, and RQ3–RQ5 test where it might.
