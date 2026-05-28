# Project Status — Networks Project (Reddit Side)

**Last Updated:** 2026-05-27  
**Current Focus:** Setup — file structure and graph builder  
**Next Action:** Write `reddit/graph_builder.py`

---

## Overall Progress

| RQ | Description | Status |
|---|---|---|
| Setup | File structure, graph_builder.py, requirements.txt | 🔄 In Progress |
| RQ1 | Broadcast vs. viral diffusion mode | ⏳ Not Started |
| RQ2 | Structural gatekeepers *(contingent on RQ1)* | ⏳ Not Started |
| RQ3 | Sender/receiver role asymmetry | ⏳ Not Started |
| RQ4 | Global position vs. local clustering | ⏳ Not Started |
| RQ5 | Reciprocity prediction *(depends on RQ2, RQ3)* | ⏳ Not Started |

---

## Decision Points

These are forks that require a result before downstream work can proceed:

- [ ] **RQ1 → RQ2 gate:** Run RQ1 SIR simulation first. If broadcast-dominant → proceed with RQ2 gatekeeper framing as written. If viral → reframe RQ2 toward relay node identification. Do not begin RQ2 until this is resolved.
- [ ] **RQ2+3 → RQ5 gate:** RQ5 logistic regression requires k-core numbers (from RQ2) and origin scores (from RQ3) to exist in `data/processed/metrics/`. Do not begin RQ5 until both CSVs are confirmed present.

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
- [ ] Write `graph_builder.py` — extract graph construction from `reddit_hyperlink_analysis.py`
- [ ] Add `scikit-learn` to `requirements.txt`

### RQ1 — Broadcast vs. Viral
- [ ] SIR simulation function
- [ ] Cascade depth/width measurement
- [ ] Configuration model null comparison (500–1000 graphs)
- [ ] Permutation test + p-value
- [ ] Figures → `figures/rq1/`
- [ ] Documentation → `docs/rq1/README.md`

### RQ2 — Gatekeepers *(start only after RQ1 result confirmed)*
- [ ] Edge betweenness centrality
- [ ] K-core decomposition
- [ ] Composite gatekeeper score
- [ ] Save per-node metrics → `data/processed/metrics/rq2_gatekeeper_metrics.csv`
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

---

## Blocked / Needs Attention

*Nothing currently blocked.*
