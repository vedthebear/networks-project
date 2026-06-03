#!/usr/bin/env python3
"""
make_comparison.py -- the cross-platform core-periphery comparison, from both
platforms' permutation-test results (computed by identical code).

Key insight: raw within-core rho is NOT comparable across platforms (Reddit
-0.15 vs Moltbook -0.76 differ mostly because of density/degree range). The
fair comparison is each platform's real rho vs its OWN degree-preserving null
-- the "beyond-degree bridging" = (real - null). Both platforms bend ~0.7
below their null, i.e. comparable core-periphery organization.
"""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R = HERE / "results"; FIGS = HERE / "figures"

mb = pd.read_csv(R / "permutation_tests.csv").set_index("statistic")
rd = pd.read_csv(R / "reddit_permutation_tests.csv").set_index("statistic")

rows = []
for stat in ["rho_core", "rho_full", "mean_clustering"]:
    rows.append({
        "statistic": stat,
        "moltbook_real": mb.loc[stat, "real"], "moltbook_null": mb.loc[stat, "null_mean"],
        "moltbook_beyond_null": mb.loc[stat, "real"] - mb.loc[stat, "null_mean"],
        "reddit_real": rd.loc[stat, "real"], "reddit_null": rd.loc[stat, "null_mean"],
        "reddit_beyond_null": rd.loc[stat, "real"] - rd.loc[stat, "null_mean"],
    })
comp = pd.DataFrame(rows)
comp.to_csv(R / "comparison.csv", index=False)
print(comp.to_string(index=False))

# Figure: within-core rho, real vs null, each platform. Arrow = beyond-degree bridging.
fig, ax = plt.subplots(figsize=(6.8, 4.2))
plats = [("Reddit\n(humans)", rd.loc["rho_core", "null_mean"], rd.loc["rho_core", "real"], "#2171b5"),
         ("Moltbook\n(agents)", mb.loc["rho_core", "null_mean"], mb.loc["rho_core", "real"], "#d62728")]
for i, (name, null_v, real_v, color) in enumerate(plats):
    ax.scatter([i], [null_v], color="#969696", s=90, zorder=3)
    ax.scatter([i], [real_v], color=color, s=110, marker="D", zorder=3)
    ax.annotate("", xy=(i, real_v), xytext=(i, null_v),
                arrowprops=dict(arrowstyle="->", color=color, lw=2))
    ax.text(i + 0.08, null_v, "degree-null", va="center", fontsize=8, color="#666")
    ax.text(i + 0.08, real_v, f"real = {real_v:+.2f}", va="center", fontsize=8, color=color)
    ax.text(i, min(real_v, null_v) - 0.12, f"bends {real_v-null_v:+.2f}\nbelow null",
            ha="center", va="top", fontsize=8, fontweight="bold")
ax.axhline(0, color="#bbb", lw=0.8, ls=":")
ax.set_xticks([0, 1]); ax.set_xticklabels([p[0] for p in plats])
ax.set_ylabel("within-core ρ (k-core vs clustering)")
ax.set_ylim(-1.0, 0.85)
ax.set_title("Central communities are bridges on BOTH platforms\n"
             "real ρ bends ~0.7 below its degree-null on each", fontsize=11)
fig.tight_layout(); fig.savefig(FIGS / "comparison_bridging.png", dpi=300)
print("\nfigure -> comparison_bridging.png")
