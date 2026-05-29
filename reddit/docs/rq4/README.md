# RQ4 — Core-Periphery Structure

> **Final result (one line):** Reddit's hyperlink network is **far more locally clustered than a degree-matched random graph** (mean clustering 0.181 vs. null 0.034, p < 0.002), and it carries a **modest but real rich club** at low-to-mid degree. The k-core ↔ clustering relationship has a two-layer structure: a *mechanical* positive correlation across the whole graph (driven by the huge mass of degree-1 leaves), but a **negative** correlation *within the connected core* — i.e., among genuinely connected communities, the more globally embedded ones are the *less* locally clustered (the classic bridging pattern). All four results beat the configuration-model null.

---

## Research Question

Does a community's **global** structural position predict its **local** connection behavior? Concretely: are communities deep in the network's dense core the ones that **bridge** across many otherwise-disconnected groups (low local clustering), while peripheral communities sit inside **tight cliques** (high clustering)? And is whatever pattern we find stronger than a random network with Reddit's degree distribution would produce?

This is the deliberate contrast to RQ1. RQ1 found that Reddit's diffusion shape was *fully* explained by its degree distribution — the configuration-model null reproduced it exactly. RQ4 targets the features that null **cannot** reproduce: random rewiring preserves degree but **destroys triangles**, so clustering and core-periphery organization are exactly where Reddit can plausibly beat chance.

---

## Methods

All metrics are computed on the **undirected projection** of the `weighted` graph (`load_graphs()["weighted"]`).

- **Projection rule:** an undirected edge exists iff a hyperlink runs in *either* direction (A→B or B→A); reciprocal weights are summed. Self-loops are dropped. Result: 35,776 nodes, 124,330 undirected edges (down from 137,821 directed — the difference is reciprocal pairs collapsing).

### Metrics

| Metric | What it measures | Implementation |
|---|---|---|
| **K-core number** | Global embeddedness — recursive depth in the dense core | `nx.core_number(UG)` (unweighted) |
| **Local clustering coefficient** | Local cohesion / bridging — fraction of a node's neighbor-pairs that are themselves connected | `nx.clustering(UG)` (unweighted, primary); `nx.clustering(UG, weight="weight")` (weighted robustness) |
| **Spearman ρ** | Rank correlation between k-core and clustering — the central test | `scipy.stats.spearmanr` |
| **Rich-club coefficient φ(k)** | Whether the highest-degree nodes preferentially connect to each other | `nx.rich_club_coefficient(UG, normalized=False)`, normalized against the null |

### Weighting choice
The **primary** analysis is unweighted/topological — the cleanest core-periphery test, standard in the literature, and directly comparable to Moltbook. **Weighted clustering** is reported as a robustness check, using RQ1's 95th-percentile weight cap (`normalize_edge_weights`, cap = 6.0, 4.28% of edges affected) so the project is internally consistent. K-core and rich-club are inherently unweighted.

### Null model (500 configuration-model graphs)
We reuse **only the null-graph construction** from RQ1 (`build_null_graph` → `nx.directed_configuration_model`, preserving each node's in/out-degree, removing self-loops/multi-edges). No RQ1 simulation output is reused. Each null graph is projected to undirected **identically** to the real graph, and the same four metrics are computed. One-tailed permutation tests:

- **Spearman:** tail chosen by the *observed* sign (we do not pre-assume the hypothesis direction). `p = P(null_ρ ≥ real_ρ)` since the observed ρ is positive; a two-sided `p = P(|null_ρ| ≥ |real_ρ|)` is also reported.
- **Clustering:** right tail, `p = P(null_clustering ≥ real_clustering)`.
- **Rich-club:** real φ(k) compared to the null's 2.5–97.5 percentile band at each k.

Ensemble size 500, RNG seed 2024, parallelized across 7 workers (~9 minutes).

---

## Results

### 1. Per-node metric distributions (`rq4_node_metrics.csv`, 35,776 rows)

| Metric | mean | median | 75th pct | max |
|---|---|---|---|---|
| k-core | 3.60 | 1 | 3 | 46 |
| clustering (unweighted) | 0.181 | 0 | 0.244 | 1.0 |
| clustering (weighted) | 0.033 | 0 | 0.036 | 0.822 |
| degree | 6.95 | 1 | 4 | 2,336 |

**Commentary.** The network is a **small dense core wrapped in a vast sparse periphery**. **51.7%** of communities have degree 1 (a single connection) and **53.4%** have k-core number 1 — these are leaf subreddits hanging off the network by a thread. By definition a degree-1 node has clustering 0, so **62.7% of all nodes have clustering exactly 0**. At the other extreme, the innermost core (k-core = 46) contains 169 tightly interlocked communities. This heavy periphery mass is the single most important fact for interpreting the correlation below.

### 2. K-core ↔ clustering correlation — the two-layer story

| Subset | Spearman ρ | n | Interpretation |
|---|---|---|---|
| **Full graph** | **+0.776** | 35,776 | Positive — but see below |
| Connected core (k-core ≥ 2) | **−0.146** | 16,683 | Negative — bridging pattern |
| Degree ≥ 3 | −0.170 | 11,763 | Negative — bridging pattern |

**Commentary — this is the subtle, important part.** The full-graph correlation is strongly **positive** (+0.776), which at first looks like it *overturns* our predicted negative anti-correlation. But that positive value is **largely mechanical**: the 53% of nodes that are leaves all sit at (low k-core, clustering = 0), and that single dense cloud of points forces a positive rank correlation no matter how the core is organized. The configuration-model null confirms this — the null's own ρ is **+0.684** (std 0.0017), i.e. *most* of the positive correlation is reproduced by degree structure alone.

When we condition on actually-connected communities (k-core ≥ 2), the correlation **flips to negative** (−0.146), and the per-bucket trend is cleanly monotonic:

| k-core bucket | mean clustering | median | n |
|---|---|---|---|
| 2 | 0.516 | 0.667 | 5,895 |
| 3 | 0.427 | 0.333 | 2,836 |
| 4–5 | 0.350 | 0.333 | 2,935 |
| 6–10 | 0.272 | 0.231 | 2,427 |
| 11–20 | 0.222 | 0.194 | 1,452 |
| 21–46 | 0.184 | 0.149 | 1,138 |

So **within the core, the original hypothesis holds**: the more globally embedded a community is, the *less* locally clustered it becomes — exactly the bridging behavior we predicted. The most central communities connect across many otherwise-separate neighborhoods rather than sitting in a single closed clique. The lesson is that "the" correlation depends entirely on whether you include the leaf periphery, and an honest reading reports both layers.

**Does Reddit beat the null on ρ?** Yes, decisively. The full-graph real ρ (0.776) exceeds the null mean (0.684) by ~54 null standard deviations (p < 0.002, the resolution floor of a 500-graph ensemble). So even the aggregate correlation is *significantly more positive* than degree alone predicts — Reddit's connected core adds genuine clustering on top of the mechanical baseline. The weighted variant agrees (ρ = 0.782, same p).

### 3. Mean clustering vs. null — the strongest, cleanest result

| | Reddit | Null mean | Null std | p | Reddit / Null |
|---|---|---|---|---|---|
| Clustering (unweighted) | 0.1809 | 0.0339 | 0.00063 | < 0.002 | **5.3×** |
| Clustering (weighted) | 0.0332 | 0.0041 | 0.00013 | < 0.002 | **8.2×** |

**Commentary.** This is the headline structural finding. Reddit's communities are **5.3× more locally clustered** than a degree-matched random graph (8.2× under weighting). Random rewiring scatters edges and rarely closes triangles, so the null's clustering is low (0.034); Reddit's is far higher because communities genuinely cluster by topic and interest. Unlike RQ1 — where Reddit was indistinguishable from the null — here the gap is enormous and unambiguous. This is precisely the "structure beyond degree" that the configuration model cannot manufacture, and it is the clearest evidence in the project so far that Reddit's wiring is non-random.

### 4. Rich-club coefficient

- Reddit's φ(k) exceeds the null's 95% band at **370 of 718** degree thresholds.
- Normalized φ(k)/φ_null(k) ranges from **0.91 to 1.27** — modestly above 1 across low-to-mid degree.
- At the **very highest** degree thresholds (k ≈ 714–718) the ratio dips just **below** 1 (≈ 0.98), i.e. the extreme-elite is *not* denser than chance.

**Commentary.** There is a **real but modest rich club**: across most of the low-to-mid degree range, well-connected communities link to each other somewhat more than degree alone would force (up to 27% above null). This complements the clustering result — the core isn't just locally cohesive node-by-node, its well-connected members are also preferentially interlinked. But the effect is gentle, not a dominant oligarchy, and it actually **disappears at the very top**: the handful of highest-degree hubs are no more interconnected than random. So Reddit's cohesion lives in the **broad mid-core**, not in a tight super-elite — consistent with the within-core negative ρ finding, where the most extreme nodes are the bridges, not the clique-members.

---

## Figures (`figures/rq4/`)

- **`kcore_vs_clustering_scatter.png`** — per-node scatter with the median-clustering-per-k-core trend overlaid. The trend line shows the within-core *decline* in clustering with k-core; the dense band at clustering 0 along the bottom is the leaf periphery that drives the positive aggregate ρ. Read the two together to see the two-layer story.
- **`spearman_null_distribution.png`** — the null distribution of ρ (tightly centered at 0.684) with Reddit's 0.776 marked far to the right. Visually, Reddit is well outside the null spread.
- **`clustering_real_vs_null.png`** — Reddit's mean clustering (0.181) sits dramatically to the right of the null distribution (centered 0.034). The single most decisive figure in RQ4.
- **`richclub_curve_vs_null.png`** — φ(k) for Reddit (red) against the null mean and 95% band over degree. Reddit rides just above the band through low-to-mid k and converges to / dips below it at the extreme tail.

---

## Synthesis

Taken together, the four results paint one coherent picture of Reddit's core-periphery architecture:

1. **A dense, genuinely clustered core exists** — clustering is 5× the null, decisively non-random.
2. **The core is a broad mid-tier, not a tight elite** — the rich club is modest and vanishes at the extreme top.
3. **Within that core, global centrality buys bridging, not cliquishness** — conditional on being connected, higher k-core means lower clustering (ρ = −0.15), the classic finding that the most embedded communities span many neighborhoods.
4. **The aggregate positive correlation is mostly a periphery artifact** — half the network is degree-1 leaves, and they mechanically dominate the whole-graph ρ.

This is the inferential payoff RQ1 lacked. Where diffusion shape was degree-explained, **clustering and core-periphery organization are not** — they are real structural properties of how Reddit's communities wire themselves, exactly the features a degree-preserving null is blind to.

---

## Caveats and Limitations

1. **The aggregate Spearman ρ is direction-ambiguous.** Reported on the full graph it is positive and largely mechanical; reported on the connected core it is negative. Neither is "wrong" — they answer different questions. We report both and lead analysis with the conditional (k-core ≥ 2) version for the substantive claim.
2. **Mechanical degree contribution.** A large share of the positive aggregate ρ (and a small share of clustering) is forced by the degree distribution. The null comparison is what isolates the genuine signal; we never interpret a raw statistic without it.
3. **Config-model degree near-match.** Multi-edge collapse and self-loop removal shave a few edges, so a null graph's degree sequence is a near- (not exact-) match to the real one. Standard configuration-model practice; identical artifact to RQ1.
4. **Weighted-clustering distortion.** The Onnela weighted clustering depends on the weight scale; we cap weights at the 95th percentile (matching RQ1) to limit heavy-tail distortion. The weighted result agrees directionally with the unweighted one, so the conclusion is not weighting-dependent.
5. **Undirected projection discards direction.** K-core, clustering, and rich-club are undirected concepts; collapsing A→B and B→A loses directional information. This is appropriate for these metrics but means RQ4 says nothing about directional flow (that is RQ3/RQ5 territory).

---

## How to Re-Run

```bash
cd /path/to/networks-project
python reddit/rq4_core_periphery.py
```

The script caches the null ensemble (`rq4_null_distribution.csv` + `rq4_richclub_curve.csv`); if both exist it skips the 500-graph sweep and re-runs only the permutation tests and figures. Delete those CSVs to force a full re-run.

**Dependencies:** `networkx`, `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `tqdm` (all in `requirements.txt`).

**Expected runtime:** ~9 minutes for a full sweep on 7 workers; seconds if the null cache exists.

**Outputs:**
- `data/processed/metrics/rq4_node_metrics.csv` — per-node k-core, clustering (uw + w), degree. **The `kcore` column is the dependency RQ5 consumes.**
- `data/processed/metrics/rq4_global_stats.csv` — real vs. null for Spearman ρ and mean clustering, with permutation p-values.
- `data/processed/metrics/rq4_richclub_curve.csv` — φ(k) real, null mean, null band, normalized ratio.
- `data/processed/metrics/rq4_null_distribution.csv` — per-null-graph ρ and mean clustering (the null distributions).
- `figures/rq4/*.png` — 4 figures.
