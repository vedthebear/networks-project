# Reddit Subreddit Diffusion — Research Plan

## Central Thesis

Reddit's subreddit-level information network exhibits **broadcast-dominant diffusion**: information spreads primarily through a small set of structurally determined gateway communities rather than through peer-to-peer viral chains. This broadcast structure is non-random — it is organized by the topology of the hyperlink network itself, meaning a subreddit's structural position predicts the role it plays in information flow.

This thesis has two intertwined claims:
1. **Broadcast over viral** — the diffusion mode is identifiable and skewed toward hub-driven spread
2. **Structural determinism** — that skew is explained by network position, not by subreddit size or topic

Both claims are tested against null models (what we'd expect by chance) so that findings are hypotheses confirmed, not just patterns observed.

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

### New File to Create

All research-question-specific analysis goes in: **`reddit_network_analysis.py`**

### New Dependencies to Add to `requirements.txt`
- `scikit-learn` — logistic regression (RQ5)

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

- **RQ1 (SIR simulation):** Is Reddit's diffusion more broadcast-dominant than a random network with the same degree sequence would produce?
- **RQ3 (role asymmetry):** Is the variance in sender/receiver roles across subreddits significantly higher than in random networks — meaning role assignment is non-random?
- **RQ4 (core-periphery correlation):** Is the anti-correlation between k-core number and clustering coefficient significantly stronger in Reddit's network than in random networks of the same degree distribution?

In each case, the null model testing is what elevates the analysis from descriptive to inferential — from "we observe X" to "X is a statistically meaningful property of this specific network."

---

## Research Question 1: Where Does Reddit Fall on the Broadcast-to-Viral Spectrum?

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
See the Null Model Methodology section above for the full explanation. For RQ1 specifically: we generate 500–1000 configuration model graphs, run the same SIR simulation on each, and build a null distribution of cascade depth and width values. The real network's values are compared to this distribution via permutation test. If Reddit's diffusion is significantly shallower and wider than the null, that is statistical evidence for broadcast-dominant structure — not merely an artifact of the network's size or degree distribution.

### Data Needed
- `weighted` DiGraph (already built)
- **Build:** SIR simulation function, cascade depth/width measurement, configuration model loop, permutation test

---

## Research Question 2: Which Subreddits Function as Structural Gatekeepers?

### Conditional Dependency on RQ1
**This question is only meaningful if RQ1 establishes broadcast-dominant diffusion.** If RQ1 instead finds viral diffusion — information spreading through long peer-to-peer chains rather than through central hubs — the gatekeeper framing breaks down. In a viral network, no single node is the bottleneck; spread is distributed across many intermediate nodes. In that case, RQ2 would need to reframe toward identifying the most active relay nodes in long diffusion chains rather than the hubs controlling outward broadcast.

This is a deliberate analytical fork: **run RQ1 first, observe the result, then decide whether RQ2 proceeds as written or adapts.**

### The Question
Assuming broadcast-dominant diffusion: which subreddits sit at the critical junctions that control information flow across the network? How do we distinguish true structural bottlenecks from communities that are simply large and active?

### Methods

#### Betweenness Centrality (Extend Existing)
For every pair of nodes (A, B), betweenness centrality counts what fraction of all shortest paths pass through node X. Already computed (sampled, k=500). Extension: add **edge betweenness** — which specific A→B links are the critical conduits, not just which nodes.

#### K-Core Decomposition
Assigns each node a shell number k: it belongs to the largest subgraph where every node has at least k neighbors within that subgraph. Think of it as peeling an onion — strip away low-degree nodes iteratively until only the densely interconnected core remains. Nodes at the **boundary between a high-core and low-core region** are natural structural bridges: embedded enough to receive from the dense interior, connected enough to pass information outward to the periphery. This complements betweenness by capturing embeddedness rather than path centrality. `nx.core_number(UG)`. O(E). Also computed in RQ3 and RQ4 so there is no additional cost.

*Note: Articulation points (nodes whose removal disconnects the graph entirely) and max-flow/min-cut analysis were considered but cut. Articulation points produce a binary label that adds little beyond what betweenness and k-core already capture, and require losing directionality via undirected projection. Max-flow/min-cut is theoretically stronger but computationally expensive to run across many community pairs and adds implementation complexity disproportionate to the marginal insight over betweenness.*

#### Composite Gatekeeper Score
Normalize betweenness rank and k-core boundary score to [0,1] and average into a single interpretable ranking. Nodes that are both high-betweenness and on the high-to-low k-core boundary are the strongest gatekeeper candidates.

### Data Needed
- `weighted` DiGraph (built), undirected projection (one line)
- **Build:** Edge betweenness extension, k-core decomposition, composite score

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

#### Null Model Test for Role Asymmetry
See the Null Model Methodology section. For RQ3: we compute origin scores for every node in 500+ configuration model graphs and record the variance of those scores in each random graph. This builds a null distribution of "how spread-out are sender/receiver roles in a random network with Reddit's degree sequence?"

If Reddit's real origin score variance is significantly higher than the null — meaning subreddits are more asymmetric than chance predicts — it tells us that role assignment is a structural property of the network, not a consequence of degree variation alone. Intuitively: even if you know which subreddits are highly active (degree), you wouldn't predict from that alone that some are exclusively senders while others are exclusively receivers. Structural position, beyond just activity level, is determining role.

#### Structural Properties by Role
Bin into sender (>0.7), balanced (0.3–0.7), receiver (<0.3). Compare k-core, clustering coefficient, and PageRank distributions across bins to test whether role is structurally determined.

### Data Needed
- `weighted` DiGraph (built)
- **Build:** Fan-out ratio per node, HITS, null model variance test, role binning and comparison

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

## How the Questions Build Toward the Thesis

```
RQ1: Establishes the diffusion mode (broadcast vs. viral) and confirms it is non-random
  ↓
  [DECISION POINT: if broadcast → proceed to RQ2 as written]
  [DECISION POINT: if viral → reframe RQ2 toward relay node identification in long chains]
  ↓
RQ2: Identifies which communities drive the broadcast structure — the gatekeeper mechanism
  ↓
RQ3: Shows sender/receiver roles are structurally determined — explains why gatekeepers emerge
  ↓
RQ4: Shows global structural position predicts local behavior — generalizes structural determinism
  ↓
RQ5: Tests whether the static structure has dynamic consequences for how relationships form over time
```
