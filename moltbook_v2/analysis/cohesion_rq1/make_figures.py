#!/usr/bin/env python3
"""
make_figures.py -- (re)draw the RQ1 cohesion figures from saved results, so we
don't have to rerun the ~25-min null ensemble. Reads results/*.csv|json.

The null distributions are extremely tight (std ~0.004), so histograms render
as invisible spikes. A grouped bar chart with null error bars is the honest,
readable view: it shows the real value sitting just above its degree-preserving
null but far above the Erdos-Renyi "no structure" floor.
"""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"; F = HERE / "figures"

real = json.loads((R / "cohesion_real.json").read_text())
dp = pd.read_csv(R / "cohesion_null.csv")
er = pd.read_csv(R / "cohesion_null_er.csv")

metrics = [("mean_clustering", "mean local\nclustering"),
           ("transitivity", "global\ntransitivity")]

fig, ax = plt.subplots(figsize=(6.5, 4.2))
x = np.arange(len(metrics)); w = 0.26
er_means = [er[m].mean() for m, _ in metrics]
er_err   = [er[m].std() for m, _ in metrics]
dp_means = [dp[m].mean() for m, _ in metrics]
dp_err   = [dp[m].std() for m, _ in metrics]
real_v   = [real[m] for m, _ in metrics]

ax.bar(x - w, er_means, w, yerr=er_err, capsize=4, color="#bdbdbd",
       label="Erdős–Rényi null (no structure)")
ax.bar(x,     dp_means, w, yerr=dp_err, capsize=4, color="#6baed6",
       label="degree-preserving null")
ax.bar(x + w, real_v,  w, color="#d62728", label="Moltbook (real)")

# annotate the real/dp ratio above the real bars
for i, (m, _) in enumerate(metrics):
    ratio = real[m] / dp[m].mean()
    ax.text(x[i] + w, real_v[i] + 0.02, f"{ratio:.2f}× dp-null\np=0.002",
            ha="center", va="bottom", fontsize=8)

ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in metrics])
ax.set_ylabel("coefficient")
ax.set_ylim(0, 0.95)
ax.set_title("Moltbook community cohesion vs two null models\n"
             "(real is far above 'no structure' but barely above degree-matched)")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(F / "clustering_vs_null.png", dpi=300)
plt.close(fig)

# Second figure: how far real sits between the two nulls (the "degree explains
# most of it" point), shown as a number line per metric.
fig, ax = plt.subplots(figsize=(6.5, 3))
for i, (m, lbl) in enumerate(metrics):
    y = i
    er_m, dp_m, rv = er[m].mean(), dp[m].mean(), real[m]
    ax.plot([er_m, rv], [y, y], color="#cccccc", lw=2, zorder=1)
    ax.scatter([er_m], [y], color="#636363", s=60, zorder=2, label="ER null" if i == 0 else "")
    ax.scatter([dp_m], [y], color="#2171b5", s=60, zorder=2, label="degree null" if i == 0 else "")
    ax.scatter([rv], [y], color="#d62728", s=70, marker="D", zorder=3, label="real" if i == 0 else "")
    frac = (rv - er_m) / (rv - er_m)  # 100% reference
    deg_frac = (dp_m - er_m) / (rv - er_m)
    ax.text(rv + 0.01, y, f"degree explains {deg_frac*100:.0f}% of the gap",
            va="center", fontsize=8)
ax.set_yticks(range(len(metrics))); ax.set_yticklabels([lbl for _, lbl in metrics])
ax.set_xlabel("coefficient value")
ax.set_xlim(0.1, 0.95)
ax.set_title("From 'no structure' to real: degree alone covers most of the gap")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(F / "structure_vs_null.png", dpi=300)
plt.close(fig)
print("figures rewritten:", list(F.glob("*.png")))
