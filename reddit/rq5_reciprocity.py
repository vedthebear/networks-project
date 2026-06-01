"""
RQ5 — Reciprocity & Temporal Influence
==============================================================================

Closes the project arc. RQ1 found diffusion shape is degree-explained; RQ4
found clustering/core-periphery structure BEATS the degree-preserving null.
RQ5 asks: do structural features predict whether inter-community links get
RECIPROCATED, and which direction tends to come first in time?

Three components (NLP/sentiment fully excluded — comparison-boundary compliant):

  Component 1 — Reciprocity (2 metrics)
    * Global reciprocity  nx.overall_reciprocity  vs. a 500-graph directed
      configuration-model null (right-tailed permutation test).
    * Reciprocity ratio  min(w_AB,w_BA)/max(w_AB,w_BA)  per reciprocated pair
      — the weight "one-sidedness" distribution.

  Component 2 — Logistic regression on is_reciprocated (all directed edges)
    * Predictors: log edge weight, log out-deg(src), log in-deg(tgt),
      k-core(src), k-core(tgt), origin(src), origin(tgt).
    * statsmodels Logit on standardized features → coefficient, sign, p-value.
    * No config-model null here: the regression controls for degree by
      INCLUDING it as a predictor; its own inference is the "beyond-degree" test.

  Component 3 — Timestamp gap
    * lag_days = |first(A→B) − first(B→A)| per reciprocated pair.
    * Initiator cross-tab: is the earlier direction more often the higher
      k-core / higher origin-score node? (temporal precedence, NOT causation).

Reuses load_graphs (graph_builder) and build_null_graph (rq1_null_model).
Consumes rq4_node_metrics.csv (per-node k-core) — run RQ4 first.

Usage:
    python reddit/rq5_reciprocity.py
"""

from __future__ import annotations

import multiprocessing
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from graph_builder import load_graphs
from rq1_null_model import build_null_graph


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
METRICS_DIR = SCRIPT_DIR / "data" / "processed" / "metrics"
FIGURES_DIR = SCRIPT_DIR / "figures" / "rq5"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

KCORE_CSV = METRICS_DIR / "rq4_node_metrics.csv"   # RQ5 dependency (from RQ4)

N_NULL    = 500
RNG_SEED  = 2025                                    # distinct from RQ1 (42/1234), RQ4 (2024)
N_WORKERS = max(1, multiprocessing.cpu_count() - 1)

PREDICTORS = [
    "log_weight", "log_out_deg_src", "log_in_deg_tgt",
    "kcore_src", "kcore_tgt", "origin_src", "origin_tgt",
]
PRETTY = {
    "log_weight":      "log(edge weight)",
    "log_out_deg_src": "log(out-degree of source)",
    "log_in_deg_tgt":  "log(in-degree of target)",
    "kcore_src":       "k-core of source",
    "kcore_tgt":       "k-core of target",
    "origin_src":      "origin score of source",
    "origin_tgt":      "origin score of target",
}

# ---------------------------------------------------------------------------
# Plot style (per reddit/CLAUDE.md visualization standards)
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "DejaVu Sans",
})


def _save(fig, name: str) -> None:
    fig.savefig(FIGURES_DIR / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Component 1 — Reciprocity metrics
# ---------------------------------------------------------------------------
def reciprocity_ratios(G: nx.DiGraph) -> np.ndarray:
    """min(w_AB, w_BA) / max(w_AB, w_BA) for each reciprocated unordered pair.

    1.0 = balanced flow both ways; near 0 = nearly one-sided. Counts each
    {A,B} pair once.
    """
    ratios = []
    seen = set()
    for u, v, d in G.edges(data=True):
        if (v, u) in seen or (u, v) in seen:
            continue
        if G.has_edge(v, u):
            w_uv = d.get("weight", 1.0)
            w_vu = G[v][u].get("weight", 1.0)
            hi = max(w_uv, w_vu)
            ratios.append(min(w_uv, w_vu) / hi if hi > 0 else np.nan)
            seen.add((u, v))
    return np.asarray(ratios, dtype=float)


def node_reciprocity(G: nx.DiGraph) -> dict[str, float]:
    """Fraction of each node's outgoing links that are reciprocated."""
    out = {}
    for u in G.nodes():
        succ = list(G.successors(u))
        if not succ:
            out[u] = np.nan
            continue
        out[u] = sum(1 for v in succ if G.has_edge(v, u)) / len(succ)
    return out


# ----- Component 1 null ensemble (global reciprocity vs. config model) ------
def _recip_worker(args: tuple) -> float:
    """Build one directed config-model null graph, return its overall reciprocity.

    Top-level for multiprocessing pickling. Reuses RQ1's build_null_graph
    (weights are assigned but irrelevant — reciprocity is purely topological).
    """
    (null_idx, in_seq, out_seq, real_weights, base_seed) = args
    rng = random.Random(base_seed + null_idx * 7919)   # prime offset for independence
    G_null = build_null_graph(in_seq, out_seq, real_weights, rng)
    return float(nx.overall_reciprocity(G_null))


def build_reciprocity_null(in_seq, out_seq, real_weights,
                           n_null, rng_seed, n_workers) -> np.ndarray:
    args_list = [(i, in_seq, out_seq, real_weights, rng_seed) for i in range(n_null)]
    vals = []
    if n_workers > 1:
        with multiprocessing.Pool(processes=n_workers) as pool:
            for r in tqdm(pool.imap_unordered(_recip_worker, args_list),
                          total=n_null, desc=f"Reciprocity null ({n_workers} workers)"):
                vals.append(r)
    else:
        for args in tqdm(args_list, desc="Reciprocity null (sequential)"):
            vals.append(_recip_worker(args))
    return np.asarray(vals, dtype=float)


# ---------------------------------------------------------------------------
# Component 2 — origin scores, edge table, logistic regression
# ---------------------------------------------------------------------------
def origin_scores(G: nx.DiGraph) -> dict[str, float]:
    """origin = out_strength / (out_strength + in_strength). 1=sender, 0=receiver."""
    out_str = dict(G.out_degree(weight="weight"))
    in_str = dict(G.in_degree(weight="weight"))
    scores = {}
    for n in G.nodes():
        o, i = out_str.get(n, 0.0), in_str.get(n, 0.0)
        tot = o + i
        scores[n] = (o / tot) if tot > 0 else np.nan
    return scores


def build_edge_table(G: nx.DiGraph, kcore: dict, origin: dict) -> pd.DataFrame:
    """One row per directed edge with structural predictors + is_reciprocated."""
    out_deg = dict(G.out_degree())     # unweighted (# distinct targets)
    in_deg = dict(G.in_degree())       # unweighted (# distinct sources)
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append({
            "source":          u,
            "target":          v,
            "weight":          d.get("weight", 1.0),
            "out_deg_src":     out_deg[u],
            "in_deg_tgt":      in_deg[v],
            "kcore_src":       kcore.get(u, np.nan),
            "kcore_tgt":       kcore.get(v, np.nan),
            "origin_src":      origin.get(u, np.nan),
            "origin_tgt":      origin.get(v, np.nan),
            "is_reciprocated": int(G.has_edge(v, u)),
        })
    df = pd.DataFrame(rows)
    df["log_weight"]      = np.log1p(df["weight"])
    df["log_out_deg_src"] = np.log1p(df["out_deg_src"])
    df["log_in_deg_tgt"]  = np.log1p(df["in_deg_tgt"])
    return df


def fit_logit(edge_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Standardize predictors, fit statsmodels Logit, return (coef_table, fit_stats)."""
    data = edge_df.dropna(subset=PREDICTORS + ["is_reciprocated"]).copy()
    X_raw = data[PREDICTORS].to_numpy(dtype=float)
    y = data["is_reciprocated"].to_numpy(dtype=int)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)
    X_design = sm.add_constant(X_std)               # intercept term

    model = sm.Logit(y, X_design)
    res = model.fit(disp=False, maxiter=200)

    names = ["const"] + PREDICTORS
    conf = res.conf_int()
    coef_rows = []
    for i, name in enumerate(names):
        coef_rows.append({
            "predictor":   name,
            "pretty":      "(intercept)" if name == "const" else PRETTY[name],
            "coef":        float(res.params[i]),
            "std_err":     float(res.bse[i]),
            "ci_low":      float(conf[i][0]),
            "ci_high":     float(conf[i][1]),
            "odds_ratio":  float(np.exp(res.params[i])),
            "p_value":     float(res.pvalues[i]),
        })
    coef_table = pd.DataFrame(coef_rows)

    # in-sample AUC (descriptive fit quality; not a held-out estimate)
    auc = float(roc_auc_score(y, res.predict(X_design)))
    fit_stats = {
        "n_edges":       int(len(data)),
        "base_rate":     float(y.mean()),
        "pseudo_r2":     float(res.prsquared),
        "auc":           auc,
        "llf":           float(res.llf),
        "converged":     bool(res.mle_retvals.get("converged", True)),
    }
    return coef_table, fit_stats


# ---------------------------------------------------------------------------
# Component 3 — timestamp gap & initiator cross-tab
# ---------------------------------------------------------------------------
def timestamp_gaps(body_df: pd.DataFrame, G: nx.DiGraph,
                   kcore: dict, origin: dict) -> pd.DataFrame:
    """Per reciprocated unordered pair: first-link times, |lag| in days, and
    which endpoint initiated (earlier first-link) with its structural features.
    """
    first = (body_df.groupby(["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"])["TIMESTAMP"]
                    .min())
    first_dict = first.to_dict()

    rows = []
    seen = set()
    for u, v in G.edges():
        if (u, v) in seen or (v, u) in seen:
            continue
        if not G.has_edge(v, u):
            continue
        t_uv = first_dict.get((u, v))
        t_vu = first_dict.get((v, u))
        if pd.isna(t_uv) or pd.isna(t_vu):
            continue
        seen.add((u, v))

        lag_days = abs((t_vu - t_uv).total_seconds()) / 86400.0
        # initiator = endpoint whose outgoing link appeared first
        if t_uv <= t_vu:
            initiator, responder = u, v
        else:
            initiator, responder = v, u
        rows.append({
            "node_a":           u,
            "node_b":           v,
            "first_ab":         t_uv,
            "first_ba":         t_vu,
            "lag_days":         lag_days,
            "initiator":        initiator,
            "responder":        responder,
            "initiator_kcore":  kcore.get(initiator, np.nan),
            "responder_kcore":  kcore.get(responder, np.nan),
            "initiator_origin": origin.get(initiator, np.nan),
            "responder_origin": origin.get(responder, np.nan),
        })
    return pd.DataFrame(rows)


def initiator_crosstab(gaps: pd.DataFrame) -> pd.DataFrame:
    """Among pairs where the two endpoints DIFFER on a feature, how often is the
    initiator the higher-valued one? 0.5 = no tendency; >0.5 = initiator tends high.
    """
    rows = []
    for feat, lo, hi in [
        ("k-core",       "initiator_kcore",  "responder_kcore"),
        ("origin score", "initiator_origin", "responder_origin"),
    ]:
        sub = gaps.dropna(subset=[lo, hi])
        diff = sub[sub[lo] != sub[hi]]
        n = len(diff)
        init_higher = int((diff[lo] > diff[hi]).sum())
        rows.append({
            "feature":              feat,
            "n_pairs_differ":       n,
            "initiator_higher":     init_higher,
            "initiator_lower":      n - init_higher,
            "frac_initiator_higher": (init_higher / n) if n > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_reciprocity_overview(null_vals: np.ndarray, real_recip: float,
                              p_val: float, ratios: np.ndarray) -> None:
    palette = sns.color_palette("colorblind")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) global reciprocity vs. null
    ax = axes[0]
    ax.hist(null_vals, bins=30, color=palette[0], alpha=0.8, edgecolor="white",
            label="Null graphs")
    ax.axvline(real_recip, color="red", linewidth=2, linestyle="--",
               label=f"Reddit = {real_recip:.3f}")
    ax.set_xlabel("Global reciprocity")
    ax.set_ylabel("Count")
    ax.set_title(f"Global Reciprocity vs. Null   (p = {p_val:.3f})")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, color="#cccccc")

    # (b) reciprocity ratio distribution
    ax = axes[1]
    clean = ratios[~np.isnan(ratios)]
    ax.hist(clean, bins=30, color=palette[2], alpha=0.85, edgecolor="white")
    ax.axvline(float(np.median(clean)), color="red", linewidth=2, linestyle="--",
               label=f"median = {np.median(clean):.2f}")
    ax.set_xlabel("Reciprocity ratio  min(w)/max(w)")
    ax.set_ylabel("Reciprocated pairs")
    ax.set_title("How Balanced Are Mutual Links?")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, color="#cccccc")

    fig.suptitle("RQ5 Component 1 — Reciprocity", fontweight="bold", y=1.02)
    _save(fig, "reciprocity_overview.png")


def plot_logit_forest(coef_table: pd.DataFrame) -> None:
    palette = sns.color_palette("colorblind")
    tbl = coef_table[coef_table["predictor"] != "const"].copy()
    tbl = tbl.sort_values("coef")
    y = np.arange(len(tbl))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [palette[2] if c > 0 else palette[3] for c in tbl["coef"]]
    ax.errorbar(tbl["coef"], y,
                xerr=[tbl["coef"] - tbl["ci_low"], tbl["ci_high"] - tbl["coef"]],
                fmt="none", ecolor="#888888", elinewidth=1.5, capsize=3, zorder=1)
    ax.scatter(tbl["coef"], y, color=colors, s=70, zorder=2)
    ax.axvline(0, color="black", linewidth=1, linestyle="-")
    ax.set_yticks(y)
    ax.set_yticklabels(tbl["pretty"])
    ax.set_xlabel("Standardized logistic coefficient (log-odds per 1 SD)")
    ax.set_title("What Predicts Reciprocation?  (Logistic Regression)")
    # significance annotation
    for yi, (_, r) in zip(y, tbl.iterrows()):
        star = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 \
               else "*" if r["p_value"] < 0.05 else "n.s."
        ax.text(r["ci_high"] + 0.02, yi, star, va="center", fontsize=9)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4, color="#cccccc")
    _save(fig, "logit_coefficients_forest.png")


def plot_timestamp_gap(gaps: pd.DataFrame) -> None:
    palette = sns.color_palette("colorblind")
    lags = gaps["lag_days"].to_numpy()
    lags = lags[~np.isnan(lags)]

    fig, ax = plt.subplots(figsize=(10, 6))
    # log1p axis to handle the heavy tail; plot in days but log-scaled x
    bins = np.logspace(0, np.log10(max(lags.max(), 2)), 40)
    ax.hist(np.clip(lags, 1, None), bins=bins, color=palette[0],
            alpha=0.85, edgecolor="white")
    ax.set_xscale("log")
    ax.axvline(float(np.median(lags)), color="red", linewidth=2, linestyle="--",
               label=f"median = {np.median(lags):.0f} days")
    ax.set_xlabel("Lag between first link and reciprocal link (days, log scale)")
    ax.set_ylabel("Reciprocated pairs")
    ax.set_title("Time Until a Link Is Reciprocated")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
    _save(fig, "timestamp_gap_histogram.png")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("RQ5 — Reciprocity & Temporal Influence")
    print("=" * 72)

    # ---- [1/7] load graphs + RQ4 k-core dependency ----
    print("\n[1/7] Loading graphs and RQ4 k-core dependency...")
    if not KCORE_CSV.exists():
        raise FileNotFoundError(
            f"Missing {KCORE_CSV}. RQ5 depends on RQ4's per-node k-core. "
            f"Run `python reddit/rq4_core_periphery.py` first."
        )
    graphs = load_graphs()
    G = graphs["weighted"]
    body_df = graphs["body_df"]
    kcore_df = pd.read_csv(KCORE_CSV)
    kcore = dict(zip(kcore_df["node"], kcore_df["kcore"]))
    print(f"      weighted: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"      k-core loaded for {len(kcore):,} nodes")

    # ---- [2/7] Component 1: reciprocity metrics ----
    print("\n[2/7] Component 1 — reciprocity metrics...")
    real_recip = float(nx.overall_reciprocity(G))
    ratios = reciprocity_ratios(G)
    node_recip = node_reciprocity(G)
    origin = origin_scores(G)
    nr_vals = np.array([v for v in node_recip.values() if not np.isnan(v)])
    print(f"      global reciprocity      = {real_recip:.4f}")
    print(f"      reciprocated pairs      = {len(ratios):,}")
    print(f"      reciprocity ratio median= {np.nanmedian(ratios):.3f} "
          f"(mean {np.nanmean(ratios):.3f})")
    print(f"      per-node reciprocity    : mean {nr_vals.mean():.3f}, "
          f"median {np.median(nr_vals):.3f}")

    node_df = pd.DataFrame({
        "node":         list(G.nodes()),
        "out_degree":   [G.out_degree(n) for n in G.nodes()],
        "reciprocity":  [node_recip[n] for n in G.nodes()],
        "origin_score": [origin[n] for n in G.nodes()],
        "kcore":        [kcore.get(n, np.nan) for n in G.nodes()],
    })
    node_df.to_csv(METRICS_DIR / "rq5_node_reciprocity.csv", index=False)

    # ---- [3/7] Component 1 null ensemble (cached) ----
    null_csv = METRICS_DIR / "rq5_null_reciprocity.csv"
    if null_csv.exists():
        print(f"\n[3/7] Loading cached reciprocity null from {null_csv.name}...")
        null_vals = pd.read_csv(null_csv)["overall_reciprocity"].to_numpy()
    else:
        print(f"\n[3/7] Building reciprocity null: {N_NULL} config-model graphs "
              f"({N_WORKERS} workers)")
        in_seq  = [d for _, d in G.in_degree()]
        out_seq = [d for _, d in G.out_degree()]
        real_weights = [d.get("weight", 1.0) for _, _, d in G.edges(data=True)]
        null_vals = build_reciprocity_null(
            in_seq, out_seq, real_weights, N_NULL, RNG_SEED, N_WORKERS)
        pd.DataFrame({"overall_reciprocity": null_vals}).to_csv(null_csv, index=False)
        print(f"      saved {len(null_vals):,} null values → {null_csv.name}")

    p_recip = float((null_vals >= real_recip).mean())   # right-tailed
    print(f"      null reciprocity mean   = {null_vals.mean():.4f} "
          f"(std {null_vals.std():.4f})")
    print(f"      permutation p (real ≥ null) = {p_recip:.3f}")

    # ---- [4/7] Component 2: logistic regression ----
    print("\n[4/7] Component 2 — logistic regression on is_reciprocated...")
    edge_df = build_edge_table(G, kcore, origin)
    edge_df.to_csv(METRICS_DIR / "rq5_edge_features.csv", index=False)
    coef_table, fit_stats = fit_logit(edge_df)
    coef_table.to_csv(METRICS_DIR / "rq5_logit_coefficients.csv", index=False)
    print(f"      edges modeled = {fit_stats['n_edges']:,}  "
          f"base rate (reciprocated) = {fit_stats['base_rate']:.3f}")
    print(f"      pseudo-R² = {fit_stats['pseudo_r2']:.4f}   AUC = {fit_stats['auc']:.4f}")
    print(coef_table[["pretty", "coef", "odds_ratio", "p_value"]]
          .to_string(index=False))

    # ---- [5/7] Component 3: timestamp gaps ----
    print("\n[5/7] Component 3 — timestamp gap analysis...")
    gaps = timestamp_gaps(body_df, G, kcore, origin)
    gaps.to_csv(METRICS_DIR / "rq5_timestamp_gaps.csv", index=False)
    ct = initiator_crosstab(gaps)
    med_lag = float(gaps["lag_days"].median())
    same_day = float((gaps["lag_days"] < 1).mean())
    print(f"      reciprocated pairs with timestamps = {len(gaps):,}")
    print(f"      median lag = {med_lag:.1f} days; {same_day:.1%} within 1 day")
    print(ct.to_string(index=False))

    # ---- global summary CSV ----
    summary = pd.DataFrame([
        {"metric": "global_reciprocity",     "value": real_recip,
         "null_mean": float(null_vals.mean()), "p_value": p_recip},
        {"metric": "reciprocity_ratio_median", "value": float(np.nanmedian(ratios)),
         "null_mean": np.nan, "p_value": np.nan},
        {"metric": "logit_pseudo_r2",        "value": fit_stats["pseudo_r2"],
         "null_mean": np.nan, "p_value": np.nan},
        {"metric": "logit_auc",              "value": fit_stats["auc"],
         "null_mean": np.nan, "p_value": np.nan},
        {"metric": "timestamp_lag_median_days", "value": med_lag,
         "null_mean": np.nan, "p_value": np.nan},
    ])
    summary.to_csv(METRICS_DIR / "rq5_global_stats.csv", index=False)

    # ---- [6/7] figures ----
    print("\n[6/7] Generating figures...")
    plot_reciprocity_overview(null_vals, real_recip, p_recip, ratios)
    plot_logit_forest(coef_table)
    plot_timestamp_gap(gaps)
    print(f"      3 figures saved → {FIGURES_DIR}")

    # ---- [7/7] sanity checks ----
    print("\n[7/7] Sanity checks...")
    assert abs(edge_df["is_reciprocated"].mean() - real_recip) < 1e-9, \
        "edge-table reciprocity rate disagrees with overall_reciprocity"
    assert len(edge_df) == G.number_of_edges(), "edge-table row count mismatch"
    assert 0 <= p_recip <= 1, "p-value out of [0,1]"
    assert np.nanmin(ratios) >= 0 and np.nanmax(ratios) <= 1, "ratio out of [0,1]"
    assert coef_table["p_value"].between(0, 1).all(), "coef p-value out of [0,1]"
    print("      ✓ all sanity checks passed")

    # ---- plain-language conclusion ----
    print("\nConclusion:")
    beats = p_recip < 0.05
    print(f"  Global reciprocity {real_recip:.3f} vs null {null_vals.mean():.3f} "
          f"→ {'BEATS null (non-random mutual linking)' if beats else 'not distinguishable from null'}")
    sig = coef_table[(coef_table["predictor"] != "const") &
                     (coef_table["p_value"] < 0.05)].copy()
    sig = sig.reindex(sig["coef"].abs().sort_values(ascending=False).index)
    print(f"  Significant predictors (|coef|, sign):")
    for _, r in sig.iterrows():
        print(f"     {r['pretty']:<28} coef={r['coef']:+.3f}  OR={r['odds_ratio']:.2f}  p={r['p_value']:.1e}")
    print(f"  Reciprocity ratio median = {np.nanmedian(ratios):.2f} "
          f"({'mostly one-sided' if np.nanmedian(ratios) < 0.5 else 'fairly balanced'} mutual links)")
    print(f"  Median reciprocation lag = {med_lag:.0f} days; "
          f"{same_day:.0%} of reciprocal links appear within a day (co-emergence).")
    print("\nDone.")


if __name__ == "__main__":
    main()
