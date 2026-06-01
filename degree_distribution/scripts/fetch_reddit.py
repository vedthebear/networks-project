"""
fetch_reddit.py — download the SNAP Reddit hyperlink network.

We use the BODY hyperlink file (links that appear in the body of a post),
matching the graph our Reddit RQ1/RQ4/RQ5 analysis is built on. The file is
~304 MB and is skipped if already present.

Dataset: https://snap.stanford.edu/data/soc-RedditHyperlinks.html
Kumar, Hamilton, Leskovec, Jurafsky, "Community Interaction and Conflict on
the Web", WWW 2018.

Usage:
    python scripts/fetch_reddit.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "reddit"
DATA_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv"
DEST = DATA_DIR / "soc-redditHyperlinks-body.tsv"


def main() -> None:
    if DEST.exists() and DEST.stat().st_size > 1_000_000:
        print(f"[skip] already have {DEST.name} "
              f"({DEST.stat().st_size / 1e6:.0f} MB)")
        return
    print(f"[download] {URL}")
    print("          (~304 MB, streaming to disk)")

    def _progress(block_num, block_size, total_size):
        done = block_num * block_size
        if total_size > 0:
            pct = min(100.0, 100.0 * done / total_size)
            sys.stdout.write(f"\r          {pct:5.1f}%  ({done/1e6:6.0f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(URL, DEST, reporthook=_progress)
    print(f"\n[done] saved {DEST} ({DEST.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
