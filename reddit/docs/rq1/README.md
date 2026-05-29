# RQ1 — Broadcast vs. Viral Diffusion (Method 1: SIR Simulation + Method 2: Null Model)

> **Final result:** Diffusion is **viral** in shape, but that shape is **fully explained by the degree distribution** (not statistically distinguishable from a configuration-model null). Original broadcast thesis disproved; RQ2 retired as a consequence. See *Final RQ1 Conclusion* below.

---

## Document Map
- **Method 1 — SIR Simulation:** establishes the observed diffusion shape (viral).
- **Method 2 — Null Model Comparison:** tests whether that shape is non-random (it is not).

## Research Question

When information spreads through Reddit's subreddit-to-subreddit hyperlink network, does it travel in a **broadcast** pattern (a central hub community simultaneously reaching many others in one step) or a **viral** pattern (information chaining through a sequence of intermediate communities, one hop at a time)?

---

## Methods

### Model: Independent Cascade (IC)

We use the **Independent Cascade model**, the field-standard for information diffusion on networks (Kempe, Kleinberg & Tardos 2003). This is a discrete-time SIR model with a fixed one-step infectious period (γ = 1).

At each time step, every infected node simultaneously attempts to infect each of its susceptible out-neighbors. The probability of successful spread along edge A→B is:

```
P(spread) = 1 − (1 − β)^w'_AB
```

where `w'_AB` is the capped edge weight (see normalization below) and β is the spread probability parameter. This formula treats each of the `w` directed hyperlinks between A and B as an independent transmission attempt.

After each time step, infected nodes recover and cannot spread again. Updates are synchronous — all attempts happen in the same step before any recoveries.

### Edge Weight Normalization

Raw edge weights (hyperlink counts between subreddit pairs) have a heavy tail. Without normalization, a few high-weight edges would become near-deterministic at moderate β (e.g., w=50, β=0.1 → p≈0.99). We cap weights at the **95th percentile** before running the simulation.

- **Cap value:** 6.0 hyperlinks
- **Fraction of edges affected:** 4.28%
- **Provenance:** `data/processed/metrics/rq1_edge_weight_normalization.csv`

### Parameter Sweep

| Parameter | Value |
|---|---|
| β values | 0.02, 0.05, 0.10, 0.15, 0.20, 0.30 |
| Seeds sampled | 500 (random, seed=42) |
| Runs per seed | 20 Monte Carlo realizations |
| Total simulations | 60,000 |
| Depth cap | 45 (safety bound; 0 runs hit it) |

β sweeps from well below to above the network's epidemic threshold (β_c ≈ 0.26 for this graph's mean out-degree of 3.85). Running all six values verifies that findings hold across the β range, not just at one parameter choice.

### Broadcast Scores

Two complementary measures per β, computed on **active cascades only** (total_reached > 1):

- **Ratio score** = median(width_1) / median(depth) — High = wide-and-shallow = broadcast
- **Reach score** = median(width_1 / total_reached per run) — High = most reach achieved at hop 1 = broadcast

We require both scores to agree directionally before drawing a conclusion.

---

## Key Findings

### Cascade Activation Rate

Most seeds produce no spread at all. 21.4% of seeds in the sampled set have out-degree 0 (they receive links but never send them), and 76% have out-degree ≤ 2. As a result:

| β | % of runs producing any spread |
|---|---|
| 0.02 | 6.8% |
| 0.05 | 13.8% |
| 0.10 | 21.8% |
| 0.15 | 28.7% |
| 0.20 | 34.7% |
| 0.30 | 43.3% |

The all-run median depth and width are 0 across all β values — dominated by zero-spread seeds. All meaningful statistics below are computed on **active cascades only** (total_reached > 1).

### Active Cascade Shape

Among cascades that spread at all:

| β | Median depth | Median width₁ | Mean depth | Mean width₁ | Mean total reach |
|---|---|---|---|---|---|
| 0.02 | 2 | 1 | 7.4 | 1.9 | 282 |
| 0.05 | 12 | 1 | 9.0 | 2.4 | 1,513 |
| 0.10 | 11 | 1 | 9.0 | 2.9 | 3,355 |
| 0.15 | 10 | 1 | 9.1 | 3.2 | 4,966 |
| 0.20 | 10 | 1 | 9.1 | 3.5 | 6,336 |
| 0.30 | 10 | 1 | 9.3 | 3.9 | 8,734 |

### Broadcast Score Summary

| β | Ratio score | Reach score |
|---|---|---|
| 0.02 | 0.50 | 0.333 |
| 0.05 | 0.083 | 0.001 |
| 0.10 | 0.091 | 0.0004 |
| 0.15 | 0.10 | 0.0003 |
| 0.20 | 0.10 | 0.0001 |
| 0.30 | 0.10 | 0.0001 |

### Interpretation

**The data shows viral-dominant diffusion, not broadcast.** Among active cascades:

- **Median width₁ = 1** across all β — even at the highest spread rate, the typical infected subreddit directly reaches only 1 new community in the first step
- **Median depth = 10–12** — cascades propagate through ~10 relay hops
- **Reach score ≈ 0.0001–0.0003** — virtually none of the cascade's total reach is achieved at the first hop; it accumulates over many steps
- **Ratio score ≈ 0.08–0.10** — shallow-to-depth ratios strongly below 1, consistent with deep-chain viral patterns

Information does not spread explosively from central hub communities. Instead, it moves through extended chains of relay communities, each passing it along to one or two others.

### Implication for RQ2 (Decision Gate)

Per the research plan, RQ1 determines the framing for RQ2. **Because RQ1 finds viral-dominant diffusion, RQ2 must be reframed away from gatekeeper hubs and toward identifying the relay nodes that occupy the key positions in diffusion chains** — the communities that consistently appear as middle-links in long spreading sequences. See `STATUS.md` for the updated RQ2 framing.

---

## Caveats and Limitations

1. **Aggregate-flow data.** The hyperlink dataset is a static aggregate — it records total links over the full observation window, not individual spreading events. The SIR model on this graph simulates a structurally-informed spreading process, not actual observed diffusion.

2. **β as a modeling assumption.** β is not estimated from the data; it is an analysis parameter. The finding (viral dominance) holds consistently across all six β values, which validates robustness, but the absolute cascade sizes at any single β should not be over-interpreted.

3. **Seed sampling.** Seeds are drawn uniformly at random, which oversamples low-degree leaf nodes. This depresses activation rates but does not bias the shape of active cascades, since those are analyzed conditional on spreading.

4. **Null model (now complete).** Method 1 establishes the observed diffusion pattern; **Method 2** (below) tests whether it is more viral than expected from the degree distribution alone. The answer is no — the viral shape is degree-explained.

5. **Weight normalization.** Capping at the 95th percentile affects 4.28% of edges. Results are not materially sensitive to this choice — the viral signature appears at all β values regardless.

---

# Method 2 — Configuration-Model Null Comparison

Method 1 established *that* Reddit's cascades are viral (narrow and deep). Method 2 asks the inferential question: **is this viral shape a non-random property of Reddit's specific wiring, or is it just what any network with Reddit's degree distribution would produce?**

## Why a Null Model Is Required

A statistic measured on a single network is a description, not a finding. 76% of Reddit's nodes have out-degree ≤ 2, so narrow cascades could be an automatic consequence of the degree distribution alone — nothing to do with how Reddit is actually organized. To separate "Reddit's structure causes viral diffusion" from "any graph with these degrees diffuses virally," we compare Reddit to an ensemble of random graphs that share its degree sequence but scramble everything else.

## The Null Ensemble

We generate **500 directed configuration-model graphs**. Each one:

- **Preserves** Reddit's exact in-degree and out-degree sequence (every node keeps its number of incoming and outgoing connections).
- **Randomizes** which node connects to which (stub matching).
- **Removes** self-loops and multi-edges (artifacts of random stub matching).
- **Assigns edge weights by resampling from Reddit's empirical (95th-percentile-capped) weight distribution.**

The weight resampling is the key design choice. Because Method 1's IC spread formula `P = 1 − (1−β)^w` depends directly on edge weights, an unweighted null would not be comparable. By giving null edges weights drawn from the same distribution Method 1 used, we run the **identical** spread process on both — an apples-to-apples comparison. This preserves the *marginal* weight distribution while breaking any weight–topology correlation (e.g., Reddit's hubs having systematically heavy edges). That correlation is itself a structural feature we are correctly *not* preserving in the null.

## Simulation Budget

| Parameter | Method 2 value | (Method 1 value) |
|---|---|---|
| Null graphs | 500 | — |
| β values | 0.02–0.30 (same six) | same |
| Seeds per graph | 100 | (500) |
| Runs per seed | 10 | (20) |
| Total simulations | 3,000,000 | (60,000) |

Seeds/runs are reduced per graph because we run 500 graphs; the per-graph medians remain stable estimates. RNG seed = 1234 (distinct from Method 1's 42); each worker uses an independent prime-offset stream for reproducibility.

## Permutation Test

For each β we compare Reddit's real median to the null distribution of medians, one-tailed in the viral direction:

```
p_depth  = fraction of null graphs with median_depth  >= Reddit's median_depth   (viral = deeper)
p_width1 = fraction of null graphs with median_width1 <= Reddit's median_width1  (viral = narrower)
```

With 6 β × 2 metrics = **12 simultaneous tests**, we apply Bonferroni correction: significance threshold = 0.05 / 12 ≈ **0.0042**.

## Results

| β | Reddit depth | Null depth (mean ± sd) | p_depth | Reddit width₁ | Null width₁ (mean ± sd) | p_width1 |
|---|---|---|---|---|---|---|
| 0.02 | 2 | 1.82 ± 0.61 | 0.718 | 1 | 1.00 ± 0.00 | 1.000 |
| 0.05 | 12 | 12.46 ± 1.91 | 0.938 | 1 | 1.00 ± 0.00 | 1.000 |
| 0.10 | 11 | 11.05 ± 0.22 | 1.000 | 1 | 1.01 ± 0.08 | 0.994 |
| 0.15 | 10 | 10.32 ± 0.46 | 1.000 | 1 | 1.00 ± 0.06 | 0.996 |
| 0.20 | 10 | 10.00 ± 0.00 | 1.000 | 1 | 1.01 ± 0.11 | 0.988 |
| 0.30 | 10 | 9.28 ± 0.45 | 0.274 | 1 | 1.03 ± 0.15 | 0.974 |

**No test is significant — not at any β, not on either metric, not even at the uncorrected 0.05 threshold.** Null graphs reproduce Reddit's cascade depth almost exactly (the real value sits squarely inside the null distribution at every β) and reproduce width₁ = 1 essentially perfectly.

Figures (in `figures/rq1/`):
- `null_depth_distribution_by_beta.png` — Reddit's depth (red line) falls inside the null histogram at every β.
- `null_width1_distribution_by_beta.png` — null width₁ is ≈ 1 everywhere, identical to Reddit.
- `permutation_test_summary.png` — all p-values far above both thresholds.
- `real_vs_null_scatter.png` — Reddit's (depth, width₁) point sits in the middle of the null cloud.

## Final RQ1 Conclusion

**Reddit's diffusion is viral in shape (Method 1), but that shape is fully explained by the network's degree distribution (Method 2).** The viral signature is *not* a product of Reddit's specific wiring — any directed graph with the same in/out-degree sequence produces the same narrow, deep cascades. The original broadcast hypothesis is disproved; the viral alternative, while descriptively accurate, is not a structurally non-random property of Reddit.

This is a valid and useful result. It tells us precisely **where not to look** for Reddit's distinctive structure: diffusion-position analyses (which node sits where in a cascade) cannot be statistically grounded, because the cascades themselves are degree-generic.

### Implication — RQ2 Retired

RQ2 was contingent on RQ1. Under a broadcast result it would have identified gatekeeper hubs; under a viral result it was to be reframed toward relay-node identification. **Method 2 invalidates both framings**: if the aggregate diffusion structure is degree-explained, then a node-level diffusion-position analysis (gatekeeper *or* relay) has no statistical footing. RQ2 is therefore retired. The project re-anchors on structural features the degree-preserving null *cannot* reproduce — local clustering hierarchy (RQ4), reciprocity (RQ5), and sentiment-stratified role structure (RQ3). See `research.md` and `STATUS.md`.

## How to Re-Run Method 2

```bash
cd /path/to/networks-project
python reddit/rq1_null_model.py
```

The script detects an existing `rq1_null_ensemble.csv` and skips the 3M-simulation sweep, re-running only the permutation test and figures. Delete the CSV to force a full re-run (~45 min on 7 workers). Outputs: `rq1_null_ensemble.csv` (3,000 rows), `rq1_permutation_test.csv` (6 rows), 4 figures.

---

## How to Re-Run (Method 1)

```bash
cd /path/to/networks-project
python reddit/rq1_diffusion_mode.py
```

The script detects the existing `rq1_cascade_runs.csv` and skips the 60,000-simulation sweep, re-running only aggregation and figures (~30 seconds). To force a full re-run, delete the CSV first.

**Dependencies:** `pandas`, `networkx`, `numpy`, `matplotlib`, `seaborn`, `tqdm` (all in `requirements.txt`)

**Expected runtime:** ~90 seconds for a full sweep; ~30 seconds if CSV cache exists.

**Outputs:**
- `reddit/data/processed/metrics/rq1_cascade_runs.csv` — 60,000 simulation rows
- `reddit/data/processed/metrics/rq1_broadcast_score.csv` — per-β summary
- `reddit/data/processed/metrics/rq1_survival_curves.csv` — S(d) per β
- `reddit/data/processed/metrics/rq1_edge_weight_normalization.csv` — weight cap provenance
- `reddit/figures/rq1/*.png` — 8 figures
