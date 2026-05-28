# Project Status — Networks Project (Reddit Side)

**Last Updated:** 2026-05-28  
**Current Focus:** RQ1 — Method 1 complete; Method 2 (null model) is next  
**Next Action:** Plan and implement RQ1 Method 2 — configuration-model null comparison + permutation test

---

## Overall Progress

| RQ | Description | Status |
|---|---|---|
| Setup | File structure, graph_builder.py, requirements.txt | ✅ Complete |
| RQ1 | Broadcast vs. viral diffusion mode | 🔄 In Progress (Method 1 done, Method 2 pending) |
| RQ2 | Relay node identification *(reframed from gatekeepers — see Decision Points)* | ⏳ Not Started |
| RQ3 | Sender/receiver role asymmetry | ⏳ Not Started |
| RQ4 | Global position vs. local clustering | ⏳ Not Started |
| RQ5 | Reciprocity prediction *(depends on RQ2, RQ3)* | ⏳ Not Started |

---

## Decision Points

- [x] **RQ1 → RQ2 gate:** RQ1 Method 1 SIR simulation is complete. **Result: viral-dominant diffusion.** Median cascade width at hop 1 = 1 across all β values; median depth = 10–12 in active cascades. Cascades propagate through relay chains, not broadcast hubs. **RQ2 is therefore reframed toward identifying relay nodes in long diffusion chains** (not gatekeeper hubs). Await Method 2 null-model p-value before finalising RQ2 implementation.
- [ ] **RQ2+3 → RQ5 gate:** RQ5 logistic regression requires k-core numbers (from RQ2) and origin scores (from RQ3). Do not begin RQ5 until both CSVs are confirmed present.

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
- [x] Documentation → `docs/rq1/README.md`
- [ ] **Method 2:** Configuration-model null comparison (500–1000 graphs) + permutation test + p-value

### RQ2 — Relay Node Identification *(reframed from Gatekeepers; start after RQ1 Method 2)*
- [ ] Identify nodes with high betweenness centrality in active cascade paths
- [ ] K-core decomposition — relay nodes expected in mid-core boundary (not just the innermost core)
- [ ] Composite relay score
- [ ] Save per-node metrics → `data/processed/metrics/rq2_relay_metrics.csv`
- [ ] Figures → `figures/rq2/`
- [ ] Documentation → `docs/rq2/README.md`

### RQ3 — Sender/Receiver Roles
- [ ] Fan-out ratio (origin score) per node
- [ ] HITS hub/authority scores
- [ ] Null model variance test
- [ ] Role binning and structural comparison
- [ ] Save per-node metrics → `data/processed/metrics/rq3_role_metrics.csv`
- [ ] Figures → `figures/rq3/`
- [ ] Documentation → `docs/rq3/README.md`

### RQ4 — Core-Periphery
- [ ] Local clustering coefficient
- [ ] Spearman correlation (k-core vs. clustering)
- [ ] Rich-club coefficient
- [ ] Sentiment by core shell
- [ ] Figures → `figures/rq4/`
- [ ] Documentation → `docs/rq4/README.md`

### RQ5 — Reciprocity *(start only after RQ2 + RQ3 metrics confirmed)*
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

---

## Blocked / Needs Attention

*Nothing currently blocked. RQ1 Method 2 is the immediate next step.*
