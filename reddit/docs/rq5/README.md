# RQ5 — Reciprocity & Temporal Influence

> **Final result (one line):** Reddit's cross-community linking is **far more reciprocal than degree alone can explain** (global reciprocity **0.196 vs. null 0.016**, ~12× the null, p < 0.002), and *which* edges get reciprocated is **strongly and predictably structural** (logistic regression AUC **0.88**, pseudo-R² **0.34**). Reciprocation is driven by **role complementarity** — a link is most likely returned when a receiver-type community links to a sender-type one — and, controlling for degree, **more globally embedded (high-k-core) communities are *less* likely to be reciprocated**, the same bridging signature RQ4 found. Mutual ties mostly build slowly (median lag **111 days**), with a co-emergence spike (10.5% returned within a day).

---

## Research Question

When community A links to community B, does B link back — and can we predict which links get reciprocated from the structural position of A and B? When a mutual relationship does form, which direction comes first in time? This is the capstone: RQ1–RQ4 characterized the *static* structure; RQ5 tests whether that structure has *dynamic consequences* for how inter-community relationships form.

This closes the project's through-line — *does structural position shape behavior beyond what degree predicts?* RQ1 said **no** for diffusion shape (degree-explained). RQ4 said **yes** for clustering. RQ5 adds a second decisive **yes**: reciprocity is non-random, and structural features predict it even after degree is controlled for.

**No NLP/sentiment features are used** — every predictor is comparison-valid (runnable on Moltbook).

---

## Methods

All metrics use the `weighted` directed graph (`load_graphs()["weighted"]`): 35,776 nodes, 137,821 directed edges. K-core per node is consumed from RQ4 (`rq4_node_metrics.csv`).

### Component 1 — Reciprocity (2 metrics)
- **Global reciprocity:** `nx.overall_reciprocity` — fraction of directed edges whose reverse also exists (binary, weight-ignoring). Tested against a **500-graph directed configuration-model null** (reusing RQ1's `build_null_graph`), right-tailed permutation test `p = P(null ≥ real)`. This is the one network-level structural statistic in RQ5, so it gets the null comparison.
- **Reciprocity ratio** (weight one-sidedness): for each reciprocated unordered pair, `min(w_AB, w_BA) / max(w_AB, w_BA)`. 1.0 = perfectly balanced flow; near 0 = nearly one-way. Reported as a distribution.

### Component 2 — Logistic regression on `is_reciprocated`
One row per directed edge; outcome `is_reciprocated = 1` iff the reverse edge exists. Seven predictors: `log(edge weight)`, `log(out-degree of source)`, `log(in-degree of target)`, `k-core(source)`, `k-core(target)`, `origin(source)`, `origin(target)`. **Origin score** = `out_strength / (out_strength + in_strength)` (1 = pure sender, 0 = pure receiver) computed directly here (RQ3 parked). Weight and degrees are log-transformed (heavy-tailed); all predictors are standardized, then fit with **`statsmodels` Logit** (intercept added) so each coefficient carries a p-value.

> **Why no config-model null on the regression.** The null-model convention targets single structural statistics. The regression instead **controls for degree by including it as a predictor**, which is the direct way to ask "beyond degree?" — its built-in Wald inference is that test. Component 1 still carries the config-model null for the global statistic.

### Component 3 — Timestamp gap
From `body_df`, the earliest link time per directed pair (`groupby([SOURCE,TARGET]).TIMESTAMP.min()`). For each reciprocated pair: `lag_days = |first(A→B) − first(B→A)|`. An **initiator cross-tab** asks whether the earlier-linking endpoint tends to be the higher-k-core / higher-origin node. **Temporal precedence only — never causation.**

---

## Results

### 1. Global reciprocity decisively beats the null
| | Reddit | Null mean | Null std | p | Reddit / Null |
|---|---|---|---|---|---|
| Global reciprocity | **0.196** | 0.0164 | 0.0004 | < 0.002 | **~12×** |

**Commentary.** Almost **20% of directed links are mutual**, versus **1.6%** in a degree-matched random graph — the real value sits ~450 null standard deviations out, the most extreme null separation in the entire project. Random rewiring almost never places both A→B and B→A, so reciprocity is *structurally manufactured* by Reddit's communities, not a degree artifact. This is the same kind of "structure beyond degree" RQ4 found for clustering, and the opposite of RQ1's degree-explained diffusion.

### 2. Mutual links are fairly balanced (reciprocity ratio)
13,491 reciprocated pairs; ratio **median 0.57, mean 0.64**. When two communities do link mutually, the flow is more balanced than one-sided — though the distribution is bimodal: a large spike at exactly **1.0** (perfectly balanced, largely weight-1↔weight-1 pairs) alongside a broad spread of partially-asymmetric pairs. So "reciprocated" is not a weak technicality here — returned links carry comparable volume to the originals.

### 3. Structural features strongly predict reciprocation (logistic regression)
137,821 edges modeled; base rate 0.196; **pseudo-R² = 0.339, AUC = 0.880** (strong fit). All seven predictors significant. Standardized coefficients (log-odds per 1 SD):

| Predictor | Coef | Odds ratio | p | Reading |
|---|---|---|---|---|
| **origin score of target** | **+1.14** | **3.14** | <1e-300 | linking to a **sender-type** target → much more likely returned |
| log(edge weight) | +1.00 | 2.71 | <1e-300 | heavier relationships are far more likely reciprocated |
| log(out-degree of source) | +0.37 | 1.45 | 3e-47 | more active senders get returned more |
| log(in-degree of target) | +0.05 | 1.06 | 0.025 | popular targets slightly more (weak) |
| **k-core of target** | **−0.11** | 0.90 | 2e-06 | more embedded target → slightly *less* likely |
| **k-core of source** | **−0.46** | 0.63 | 5e-74 | more embedded source → *less* likely (beyond degree) |
| **origin score of source** | **−1.27** | 0.28 | <1e-300 | **sender-type** sources are rarely returned |

**Commentary — two stories.**
- **Role complementarity drives reciprocation.** The two origin-score coefficients are the largest and point in opposite directions: an edge is most likely reciprocated when a **receiver-type** community (low origin) links to a **sender-type** community (high origin). Intuitive — a target that sends many links is mechanically likely to send one back, while a source that mostly broadcasts rarely gets a reply. Reciprocity lives between *complementary* roles, not between two broadcasters.
- **Embedding works against reciprocity, beyond degree.** Both k-core coefficients are **negative and significant after controlling for degree and origin.** The deeper a community sits in the dense core, the *less* its links are reciprocated — exactly RQ4's within-core finding that the most embedded communities **bridge** across many neighborhoods rather than lock into tight mutual ties. Because degree is already in the model, this is a genuine *beyond-degree* structural effect, not an activity artifact.

### 4. Reciprocation is mostly slow, with a co-emergence spike (timestamp gap)
13,491 reciprocated pairs with timestamps; **median lag 111 days**; **10.5%** returned within a single day. The lag distribution is broad and right-skewed on a log axis: a sharp spike at <1 day (communities discovering each other together, e.g. around a shared event) plus a large hump peaking around 3–12 months. Most mutual relationships are **built gradually**, not reciprocated immediately.

**Initiator cross-tab** (among pairs that differ on the feature):
| Feature | Initiator is the higher-valued node |
|---|---|
| origin score | **58.1%** — sender-type communities tend to initiate |
| k-core | **46.0%** — core communities slightly more often *respond* than initiate |

Both are mild but directionally consistent with the regression: sender-types start relationships; embedded core communities are more often on the receiving/responding end. *(Temporal precedence, consistent with directional influence — not causation.)*

---

## Figures (`figures/rq5/`)
- **`reciprocity_overview.png`** — (left) Reddit's 0.196 marked far to the right of the null distribution (centered 0.016); (right) the reciprocity-ratio distribution with its median-0.57 line and the spike at 1.0.
- **`logit_coefficients_forest.png`** — standardized coefficients with 95% CIs, sorted; green = positive, orange = negative, significance stars. The two origin-score bars and the two negative k-core bars are the story at a glance.
- **`timestamp_gap_histogram.png`** — lag distribution (log-x): the <1-day co-emergence spike and the multi-month hump, median 111 days.

---

## Synthesis
RQ5 delivers the project's second clean "structure beyond degree" result and ties the static findings to dynamics:
1. **Reciprocity itself is non-random** — 12× the null, the project's strongest null separation.
2. **It is predictable and structural** — AUC 0.88 from position alone; role complementarity (sender↔receiver) is the dominant driver.
3. **Embedding suppresses reciprocity, beyond degree** — high-k-core communities bridge rather than reciprocate, echoing RQ4 and confirmed conditional on degree.
4. **Mutual ties build slowly** — median 111-day lag, with sender-types initiating.

Read with RQ1 and RQ4: diffusion *shape* is degree-generic, but *clustering* (RQ4) and *reciprocity* (RQ5) are real, degree-independent properties of how Reddit's communities wire themselves over time.

---

## Caveats and Limitations
1. **Aggregate flow, not content events.** Edges are total link counts between communities, not individual diffusion events; reciprocity is structural mutual linking, not "B re-shared A's content."
2. **In-sample AUC.** The 0.88 AUC / 0.34 pseudo-R² describe fit on the full edge set, not a held-out estimate; they index association strength, not out-of-sample prediction.
3. **Collinearity.** K-core correlates with degree, and origin with the degree marginals; coefficients are *partial* effects sharing variance. Signs and significance are robust, but exact magnitudes should not be over-read. The negative k-core coefficient is notable *because* it survives conditioning on degree.
4. **Config-model degree near-match.** Self-loop/multi-edge removal shaves a few edges, so null degree sequences are near- (not exact-) matches — the standard artifact, identical to RQ1/RQ4.
5. **Ratio spike at 1.0.** Weight-1↔weight-1 mutual pairs have ratio exactly 1.0 by construction, inflating that bin; the median (0.57) is the more representative summary.
6. **Temporal precedence ≠ causation.** The gap uses only the first link in each direction; it says nothing about ongoing dynamics and is never framed causally.

---

## How to Re-Run
```bash
cd /path/to/networks-project
python reddit/rq5_reciprocity.py
```
Caches the reciprocity null (`rq5_null_reciprocity.csv`); if present, the 500-graph sweep is skipped and only the permutation test, regression, gaps, and figures re-run. Delete that CSV to force a full re-run.

**Dependencies:** `networkx`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`, `tqdm` (all in `requirements.txt`). **Requires** `data/processed/metrics/rq4_node_metrics.csv` — run `rq4_core_periphery.py` first.

**Expected runtime:** ~3 minutes (null sweep on 7 workers); regression and gap analysis are near-instant.

**Outputs:**
- `data/processed/metrics/rq5_node_reciprocity.csv` — per-node out-degree, reciprocity, origin score, k-core.
- `rq5_null_reciprocity.csv` — 500 null global-reciprocity values.
- `rq5_edge_features.csv` — per-edge predictor table + `is_reciprocated` (137,821 rows).
- `rq5_logit_coefficients.csv` — coefficient, std err, CI, odds ratio, p-value.
- `rq5_timestamp_gaps.csv` — per reciprocated pair: first-link times, lag, initiator + its features.
- `rq5_global_stats.csv` — headline summary (global reciprocity, ratio median, pseudo-R², AUC, median lag).
- `figures/rq5/*.png` — 3 figures.
