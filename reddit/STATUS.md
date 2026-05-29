# Project Status — Networks Project (Reddit Side)

**Last Updated:** 2026-05-28  
**Current Focus:** RQ1 fully complete (both methods); RQ2 retired; RQ4 is next  
**Next Action:** Plan and implement RQ4 — k-core vs. clustering (Spearman), rich-club, sentiment-by-shell. Produces the k-core CSV that RQ5 needs.

---

## Overall Progress

| RQ | Description | Status |
|---|---|---|
| Setup | File structure, graph_builder.py, requirements.txt | ✅ Complete |
| RQ1 | Broadcast vs. viral diffusion mode | ✅ Complete — viral, but degree-explained (null not beaten) |
| ~~RQ2~~ | ~~Relay/gatekeeper node identification~~ | ❌ Retired — RQ1 showed diffusion is degree-explained (see Decision Points) |
| ~~RQ3~~ | ~~Sender/receiver role asymmetry~~ | ⏸ Parked → Future Interests (sentiment angle not comparison-valid; rest likely degree-explained) |
| RQ4 | Global position vs. local clustering | ⏳ Not Started (**next**) |
| RQ5 | Reciprocity prediction *(depends on RQ4; computes origin score directly)* | ⏳ Not Started |

---

## Decision Points

- [x] **RQ1 → RQ2 gate (RESOLVED — RQ2 retired):** RQ1 Method 1 found viral-dominant diffusion (median width₁ = 1, median depth = 10–12). Method 2's configuration-model null (500 graphs) then showed this viral shape is **not statistically distinguishable from a degree-matched random graph** (all p > 0.27, none near significance). Because the diffusion structure is fully degree-explained, a node-level diffusion-position analysis (gatekeeper *or* relay) has no statistical footing. **RQ2 is retired.** The project re-anchors on RQ3 (sentiment-stratified roles), RQ4 (clustering hierarchy), RQ5 (reciprocity) — features the degree null cannot reproduce. See `docs/rq1/README.md` and `research.md`.
- [ ] **RQ4 → RQ5 gate:** RQ5 logistic regression requires k-core numbers (now from **RQ4**). Origin scores are computed directly inside RQ5 (RQ3 parked). Do not begin RQ5 until the RQ4 k-core CSV is confirmed present.
- [x] **NLP comparison boundary:** Moltbook has no sentiment/NLP data, so no comparative claim may use it. RQ3's sentiment-stratified plan is parked; RQ4's "sentiment by core shell" is **removed**. RQ5's "positive-sentiment fraction" predictor is still flagged for review (to be discussed before RQ5).

---

## RQ Detail

### Setup
- [x] Consolidate all Reddit files into `reddit/` mother folder
- [x] Move CLAUDE.md, research.md, STATUS.md into `reddit/`
- [x] Move `reddit_hyperlink_analysis.py` into `reddit/`
- [x] Move raw data into `reddit/data/reddit_hyperlinks/`
- [x] Delete `reddit_interaction_analysis.py` (user interactions data not used)
- [x] Create `data/processed/graphs/` and `data/processed/metrics/`
- [x] Create `figures/rq1/` through `figures/rq5/`
- [x] Create `docs/rq1/` through `docs/rq5/`
- [x] Write `graph_builder.py`
- [x] Add `scikit-learn`, `seaborn`, `tqdm` to `requirements.txt`

### RQ1 — Broadcast vs. Viral
- [x] Independent Cascade SIR simulation (`sir_run`, `measure_all_cascades`)
- [x] Weight normalization (95th-percentile cap = 6.0; 4.28% of edges affected)
- [x] 60,000 simulations: 6 β × 500 seeds × 20 runs (seed=42)
- [x] Both broadcast scores (ratio and reach) on active cascades
- [x] Survival curves S(d) per β
- [x] 8 publication-quality figures → `figures/rq1/`
- [x] Per-cascade CSV → `data/processed/metrics/rq1_cascade_runs.csv`
- [x] Broadcast score summary → `data/processed/metrics/rq1_broadcast_score.csv`
- [x] Documentation (Method 1) → `docs/rq1/README.md`
- [x] **Method 2:** Config-model null (500 graphs) + permutation test → `rq1_null_model.py`
- [x] Null ensemble CSV (3,000 rows) → `data/processed/metrics/rq1_null_ensemble.csv`
- [x] Permutation-test CSV (6 rows) → `data/processed/metrics/rq1_permutation_test.csv`
- [x] 4 null-comparison figures → `figures/rq1/`
- [x] Method 2 documentation appended → `docs/rq1/README.md`
- [x] **Finding:** viral shape is degree-explained; original broadcast thesis disproved

### ~~RQ2 — Relay / Gatekeeper Node Identification~~ — RETIRED
RQ2 was contingent on RQ1. RQ1 Method 2 showed the diffusion structure is fully degree-explained, so any node-level diffusion-position analysis (gatekeeper or relay) lacks statistical footing. **Retired — see `research.md` and `docs/rq1/README.md`.** No code was written. The k-core input RQ5 needs is now produced by RQ4.

### ~~RQ3 — Sender/Receiver Roles~~ — PARKED → Future Interests
Parked because its only null-beatable angle (sentiment-stratified role consistency) depends on NLP/sentiment data Moltbook lacks, and the remaining role-magnitude test is expected to be degree-explained (like RQ1). See research.md "RQ3 — PARKED" banner. A sentiment-free revival would pivot to **directed role assortativity** (net-senders → net-receivers vs. config null). No code written.

### RQ4 — Core-Periphery
- [ ] Local clustering coefficient
- [ ] Spearman correlation (k-core vs. clustering)
- [ ] Rich-club coefficient
- [ ] K-core number per node → `data/processed/metrics/` (reused by RQ5)
- [ ] Figures → `figures/rq4/`
- [ ] Documentation → `docs/rq4/README.md`
- [x] ~~Sentiment by core shell~~ — REMOVED (NLP not comparison-valid; see Comparison Boundary in research.md)

### RQ5 — Reciprocity *(start only after RQ4 k-core CSV confirmed; origin score computed inside RQ5)*
- [ ] Global and per-node reciprocity
- [ ] Logistic regression on `is_reciprocated`
- [ ] Timestamp gap analysis
- [ ] Figures → `figures/rq5/`
- [ ] Documentation → `docs/rq5/README.md`

---

## Completed Work Log

| Date | Action | File(s) |
|---|---|---|
| 2026-05-20 | Created research plan | `research.md` |
| 2026-05-20 | Created project instructions | `CLAUDE.md` |
| 2026-05-20 | Created status document | `STATUS.md` |
| 2026-05-27 | Consolidated all Reddit files into `reddit/` mother folder; deleted interaction data | `reddit/` |
| 2026-05-28 | Built shared graph builder with pickle caching | `reddit/graph_builder.py` |
| 2026-05-28 | RQ1 Method 1: IC SIR simulation, 60k runs, 8 figures, 4 CSVs; finding: viral-dominant | `reddit/rq1_diffusion_mode.py`, `reddit/figures/rq1/`, `reddit/data/processed/metrics/rq1_*.csv` |
| 2026-05-28 | RQ1 Method 1 documentation | `reddit/docs/rq1/README.md` |
| 2026-05-28 | RQ1 Method 2: config-model null (500 graphs, 3M sims), permutation test, 4 figures; finding: viral shape is degree-explained (null not beaten, all p > 0.27) | `reddit/rq1_null_model.py`, `reddit/figures/rq1/`, `reddit/data/processed/metrics/rq1_null_ensemble.csv`, `rq1_permutation_test.csv` |
| 2026-05-28 | RQ1 Method 2 documentation appended | `reddit/docs/rq1/README.md` |
| 2026-05-28 | Retired RQ2; reframed research.md from a thesis to research-questions; rerouted RQ5 k-core dependency to RQ4 | `reddit/research.md`, `reddit/CLAUDE.md`, `reddit/STATUS.md` |
| 2026-05-28 | Added NLP comparison-boundary rule (no sentiment in cross-platform claims); parked RQ3 → Future Interests | `reddit/research.md`, `reddit/CLAUDE.md`, `reddit/STATUS.md` |
| 2026-05-28 | Removed "sentiment by core shell" from RQ4 (NLP not comparison-valid); RQ4 now fully structural | `reddit/research.md`, `reddit/STATUS.md` |

---

## Future Interests (Parked)

- **RQ3 — Sender/receiver roles.** Parked: sentiment-stratified angle isn't comparison-valid (no Moltbook sentiment data), and the role-magnitude test is likely degree-explained. Sentiment-free revival path: **directed role assortativity** (net-senders → net-receivers vs. configuration-model null) — comparison-valid and null-beatable.

---

## Blocked / Needs Attention

- **RQ5 sentiment predictor — pending discussion.** RQ5's "fraction of positive-sentiment links" predictor depends on NLP data and violates the comparison boundary. Decision deferred to the RQ5 discussion. *(RQ4's sentiment-by-shell piece is now resolved — removed.)*
- Otherwise nothing blocked. RQ4 (core-periphery) is the immediate next step — it also produces the k-core CSV RQ5 depends on.
