#!/usr/bin/env python3
"""
run_all.py — Full reproduction of all analysis for:
  "Concentrated but not Conversational: AI-Agent Social Networks vs Human Ones at Two Scales"
  Riley Leong, August Curtis, Ved Vedere, Samuel Kelly, Linlong Wang
  Math 168 (Networks), UCLA

Usage:
    python run_all.py

Requirements:
    pip install -r requirements.txt

Data requirements:
    Moltbook v2 dataset (gitignored — contact authors):
        data/data/tables/posts.csv
        data/data/tables/agents.csv
        data/data/tables/submolts.csv
        data/data/tables/membership.csv

    Reddit SNAP hyperlink network (~304 MB):
        Automatically downloaded to data/reddit/ in Step 1.

Output locations:
    analysis/individual/results/   R1 figures + concentration/power-law tables
    analysis/community/results/    R3 permutation test tables
    analysis/community/figures/    R3 + SI cohesion figures (incl. FIG 2)
    analysis/reddit_diffusion/figures/  SI Reddit diffusion + reciprocity figures
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(title: str, description: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    for line in textwrap.wrap(description, width - 4):
        print(f"  {line}")
    print("=" * width)


def run(script: Path, label: str) -> bool:
    """Run a script with the current Python interpreter. Returns True on success."""
    print(f"\n  >> {label}")
    print(f"     {script.relative_to(ROOT)}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(script.parent),
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"     done in {elapsed:.0f}s")
        return True
    else:
        print(f"     FAILED (exit {result.returncode}) after {elapsed:.0f}s")
        return False


def check_moltbook_data() -> bool:
    required = [
        ROOT / "data" / "data" / "tables" / "membership.csv",
        ROOT / "data" / "data" / "tables" / "submolts.csv",
        ROOT / "data" / "data" / "tables" / "agents.csv",
        ROOT / "data" / "data" / "tables" / "posts.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("\n  [!] Missing Moltbook v2 data files:")
        for p in missing:
            print(f"        {p.relative_to(ROOT)}")
        print("      Contact the authors or re-run the scraper to obtain these.")
        return False
    print("  [ok] Moltbook v2 data present")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("  Who Drives Information Flow? — Full Analysis Reproduction")
    print("  Math 168 (Networks), UCLA, 2026")

    # ------------------------------------------------------------------
    # Pre-flight check
    # ------------------------------------------------------------------
    banner(
        "Pre-flight: checking Moltbook data",
        "Verifying that the v2 dataset (1.09M posts, 27k agents, 2654 submolts) "
        "is present under data/data/tables/."
    )
    moltbook_ok = check_moltbook_data()

    # ------------------------------------------------------------------
    # Step 1 — Download Reddit data
    # ------------------------------------------------------------------
    banner(
        "Step 1 — Download Reddit hyperlink network  [all steps]",
        "Downloads the SNAP soc-redditHyperlinks-body.tsv (~304 MB) to "
        "data/reddit/. Skipped automatically if already present."
    )
    run(ROOT / "analysis" / "individual" / "fetch_reddit.py",
        "fetch SNAP Reddit hyperlink TSV")

    # ------------------------------------------------------------------
    # Step 2 — R1 / FIG 1: Degree concentration + CCDF
    # ------------------------------------------------------------------
    banner(
        "Step 2 — R1 / FIG 1: Degree concentration & power-law analysis",
        "Extracts degree sequences for Reddit subreddits and Moltbook submolts, "
        "fits power laws (Clauset 2009 MLE), computes Gini + top-k shares, and "
        "produces the headline CCDF comparison figure."
    )
    if moltbook_ok:
        run(ROOT / "analysis" / "individual" / "extract_degrees.py",
            "extract degree sequences (Reddit + Moltbook)")
        run(ROOT / "analysis" / "individual" / "analyze_degrees.py",
            "power-law fit + Gini + CCDF figure  ->  analysis/individual/results/")
    else:
        print("  [skip] Moltbook data missing — skipping Steps 2/3/4")

    # ------------------------------------------------------------------
    # Step 3 — R3 / FIG 2: Core-periphery comparison  (MAIN RESULT)
    # ------------------------------------------------------------------
    banner(
        "Step 3 — R3 / FIG 2: Core-periphery comparison  [MAIN RESULT]",
        "Runs the identical k-core + clustering + degree-preserving null-model "
        "test on both Moltbook and Reddit. Produces the key comparison figure "
        "(central communities = bridges on both platforms, each bending ~0.7 "
        "below its own degree-null).  Output: analysis/community/figures/comparison_bridging.png"
    )
    if moltbook_ok:
        run(ROOT / "analysis" / "community" / "moltbook_coreperiphery.py",
            "Moltbook core-periphery + null ensemble  ->  analysis/community/results/")
        run(ROOT / "analysis" / "community" / "reddit_coreperiphery.py",
            "Reddit core-periphery + null ensemble   ->  analysis/community/results/")
        run(ROOT / "analysis" / "community" / "make_comparison.py",
            "cross-platform comparison figure        ->  analysis/community/figures/")

    # ------------------------------------------------------------------
    # Step 4 — SI: Community cohesion (clustering vs null)
    # ------------------------------------------------------------------
    banner(
        "Step 4 — SI: Community cohesion (Moltbook clustering vs null)",
        "Permutation test: is Moltbook community clustering higher than a "
        "degree-preserving null? Mirrors Sam's Reddit RQ4 method. "
        "Output: analysis/community/figures/"
    )
    if moltbook_ok:
        run(ROOT / "analysis" / "community" / "moltbook_cohesion.py",
            "Moltbook cohesion null test  ->  analysis/community/results/ + figures/")

    # ------------------------------------------------------------------
    # Step 5 — SI: Reddit diffusion analysis (Sam's RQ1/RQ4/RQ5)
    # ------------------------------------------------------------------
    banner(
        "Step 5 — SI: Reddit diffusion + reciprocity (Sam's analysis)",
        "Runs the full Reddit hyperlink RQ pipeline: IC/SIR diffusion simulation "
        "(RQ1), core-periphery structure (RQ4), and reciprocity analysis (RQ5). "
        "These are the Reddit baselines cited in the SI. "
        "RQ4 must finish before RQ5 (RQ5 reads RQ4 node metrics). "
        "Output: analysis/reddit_diffusion/figures/"
    )
    # RQ1 must run before the null model (null model reads RQ1 broadcast scores)
    rq1_ok  = run(ROOT / "analysis" / "reddit_diffusion" / "rq1_diffusion_mode.py",
                  "RQ1 diffusion mode (Method 1: broadcast score)")
    if rq1_ok:
        run(ROOT / "analysis" / "reddit_diffusion" / "rq1_null_model.py",
            "RQ1 null model (Method 2: permutation test)")
    rq4_ok = run(ROOT / "analysis" / "reddit_diffusion" / "rq4_core_periphery.py",
                 "RQ4 core-periphery structure")
    if rq4_ok:
        run(ROOT / "analysis" / "reddit_diffusion" / "rq5_reciprocity.py",
            "RQ5 reciprocity + structural prediction")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    banner(
        "Done — output locations",
        ""
    )
    print("  Paper figures:")
    print("    FIG 1  analysis/individual/results/degree_ccdf_comparison.png")
    print("    FIG 2  analysis/community/figures/comparison_bridging.png")
    print()
    print("  SI figures:")
    print("    analysis/community/figures/          (cohesion, k-core vs clustering)")
    print("    analysis/reddit_diffusion/figures/   (RQ1/RQ4/RQ5)")
    print()
    print("  Key tables:")
    print("    analysis/individual/results/concentration.csv    (Gini, top-k shares)")
    print("    analysis/individual/results/powerlaw_summary.csv (alpha, KS, LR tests)")
    print("    analysis/community/results/permutation_tests.csv (Moltbook core-periphery)")
    print("    analysis/community/results/reddit_permutation_tests.csv")
    print("    analysis/community/results/comparison.csv        (cross-platform delta)")
    print()


if __name__ == "__main__":
    main()
