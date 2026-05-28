"""
Analyze Reddit user-user interaction network summary.

Input:
    reddit_all_user_network_summary.csv

Expected columns:
    subreddit, nodes, edges, density, reciprocity,
    top_out_user, top_in_user, n_wcc, largest_wcc_frac

Outputs:
    outputs/reddit_summary_statistics.csv
    outputs/reddit_top10_by_edges.csv
    outputs/reddit_top10_by_nodes.csv
    outputs/reddit_user_network_cleaned.csv
    outputs/figures/top10_edges.png
    outputs/figures/reciprocity_distribution.png
    outputs/figures/density_vs_nodes.png
    outputs/figures/nodes_vs_edges.png
    outputs/figures/largest_wcc_distribution.png
"""

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_default_input(base_dir: Path) -> Path:
    """Find the most likely Reddit summary CSV in the project."""
    candidates = [
        base_dir / "reddit_all_user_network_summary.csv",
        base_dir / "outputs" / "reddit_all_user_network_summary.csv",
        base_dir / "data" / "reddit_interactions" / "reddit_all_user_network_summary.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find reddit_all_user_network_summary.csv. "
        "Pass the path explicitly with --input."
    )


def load_and_clean(path: Path) -> pd.DataFrame:
    """Load the summary CSV and coerce metric columns to numeric."""
    df = pd.read_csv(path)

    required_cols = {
        "subreddit",
        "nodes",
        "edges",
        "density",
        "reciprocity",
        "largest_wcc_frac",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    numeric_cols = [
        "nodes",
        "edges",
        "density",
        "reciprocity",
        "largest_wcc_frac",
    ]

    if "n_wcc" in df.columns:
        numeric_cols.append("n_wcc")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["nodes", "edges", "density", "reciprocity", "largest_wcc_frac"])
    df = df[df["nodes"] > 0].copy()

    # Useful transformed fields for plotting and interpretation.
    df["log10_nodes"] = np.log10(df["nodes"])
    df["log10_edges"] = np.log10(df["edges"].clip(lower=1))
    df["is_large_subreddit"] = df["nodes"] >= df["nodes"].quantile(0.75)

    return df.sort_values("edges", ascending=False).reset_index(drop=True)


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Create compact summary statistics for poster/report use."""
    metrics = ["nodes", "edges", "density", "reciprocity", "largest_wcc_frac"]

    rows = []
    for metric in metrics:
        s = df[metric].dropna()
        rows.append(
            {
                "metric": metric,
                "count": int(s.count()),
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(),
                "min": s.min(),
                "q25": s.quantile(0.25),
                "q75": s.quantile(0.75),
                "max": s.max(),
            }
        )

    return pd.DataFrame(rows)


def print_key_findings(df: pd.DataFrame, stats: pd.DataFrame) -> None:
    """Print a concise interpretation to the console."""
    n_subreddits = len(df)

    median_nodes = df["nodes"].median()
    median_edges = df["edges"].median()
    median_density = df["density"].median()
    median_reciprocity = df["reciprocity"].median()
    median_wcc = df["largest_wcc_frac"].median()

    top_edges = df.iloc[0]
    top_nodes = df.sort_values("nodes", ascending=False).iloc[0]

    print("\n" + "=" * 72)
    print("REDDIT USER-USER INTERACTION NETWORK SUMMARY")
    print("=" * 72)

    print(f"Number of subreddit networks analyzed: {n_subreddits:,}")
    print(f"Median nodes per subreddit: {median_nodes:,.0f}")
    print(f"Median edges per subreddit: {median_edges:,.0f}")
    print(f"Median density: {median_density:.6f}")
    print(f"Median reciprocity: {median_reciprocity:.3f}")
    print(f"Median largest weakly connected component fraction: {median_wcc:.3f}")

    print("\nLargest network by edge count:")
    print(
        f"  r/{top_edges['subreddit']}: "
        f"{int(top_edges['nodes']):,} nodes, {int(top_edges['edges']):,} edges, "
        f"density={top_edges['density']:.6f}, reciprocity={top_edges['reciprocity']:.3f}"
    )

    print("\nLargest network by node count:")
    print(
        f"  r/{top_nodes['subreddit']}: "
        f"{int(top_nodes['nodes']):,} nodes, {int(top_nodes['edges']):,} edges, "
        f"density={top_nodes['density']:.6f}, reciprocity={top_nodes['reciprocity']:.3f}"
    )

    print("\nInterpretation notes:")
    print("- Reddit user-user reply networks are generally sparse: density is low across subreddits.")
    print("- Despite low density, most subreddits have a large weakly connected component.")
    print("- Reciprocity varies by subreddit, suggesting different levels of back-and-forth interaction.")
    print("- Large subreddits tend to have many interactions but very low density because possible user pairs grow rapidly.")


def save_tables(df: pd.DataFrame, stats: pd.DataFrame, output_dir: Path) -> None:
    """Save cleaned data and compact tables."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "reddit_user_network_cleaned.csv", index=False)
    stats.to_csv(output_dir / "reddit_summary_statistics.csv", index=False)

    top10_edges = df.sort_values("edges", ascending=False).head(10)
    top10_nodes = df.sort_values("nodes", ascending=False).head(10)

    top10_edges.to_csv(output_dir / "reddit_top10_by_edges.csv", index=False)
    top10_nodes.to_csv(output_dir / "reddit_top10_by_nodes.csv", index=False)


def plot_top10_edges(df: pd.DataFrame, fig_dir: Path) -> None:
    top10 = df.sort_values("edges", ascending=False).head(10).copy()
    top10 = top10.sort_values("edges", ascending=True)

    plt.figure(figsize=(9, 5))
    plt.barh(top10["subreddit"], top10["edges"]/1000000)
    plt.xlabel("Number of directed user-user edges (millions)")
    plt.ylabel("Subreddit")
    plt.title("Top 10 subreddits by user-user interaction edges")
    plt.tight_layout()
    plt.savefig(fig_dir / "top10_edges.png", dpi=300)
    plt.close()


def plot_reciprocity_distribution(df: pd.DataFrame, fig_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(df["reciprocity"].dropna(), bins=40)
    plt.xlabel("Reciprocity")
    plt.ylabel("Number of subreddits")
    plt.title("Distribution of reciprocity across subreddit networks")
    plt.tight_layout()
    plt.savefig(fig_dir / "reciprocity_distribution.png", dpi=300)
    plt.close()


def plot_density_vs_nodes(df: pd.DataFrame, fig_dir: Path) -> None:
    plot_df = df[(df["nodes"] > 0) & (df["density"] > 0)].copy()

    plt.figure(figsize=(8, 5))
    plt.scatter(plot_df["nodes"], plot_df["density"], alpha=0.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of users/nodes (log scale)")
    plt.ylabel("Density (log scale)")
    plt.title("Subreddit size vs. user-user network density")
    plt.tight_layout()
    plt.savefig(fig_dir / "density_vs_nodes.png", dpi=300)
    plt.close()


def plot_nodes_vs_edges(df: pd.DataFrame, fig_dir: Path) -> None:
    plot_df = df[(df["nodes"] > 0) & (df["edges"] > 0)].copy()

    plt.figure(figsize=(8, 5))
    plt.scatter(plot_df["nodes"], plot_df["edges"], alpha=0.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of users/nodes (log scale)")
    plt.ylabel("Number of edges (log scale)")
    plt.title("Subreddit network size: users vs. interactions")
    plt.tight_layout()
    plt.savefig(fig_dir / "nodes_vs_edges.png", dpi=300)
    plt.close()


def plot_largest_wcc_distribution(df: pd.DataFrame, fig_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(df["largest_wcc_frac"].dropna(), bins=40)
    plt.xlabel("Largest weakly connected component fraction")
    plt.ylabel("Number of subreddits")
    plt.title("Most subreddit networks contain a giant weakly connected component")
    plt.tight_layout()
    plt.savefig(fig_dir / "largest_wcc_distribution.png", dpi=300)
    plt.close()


def make_figures(df: pd.DataFrame, output_dir: Path) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_top10_edges(df, fig_dir)
    plot_reciprocity_distribution(df, fig_dir)
    plot_density_vs_nodes(df, fig_dir)
    plot_nodes_vs_edges(df, fig_dir)
    plot_largest_wcc_distribution(df, fig_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Reddit user-user interaction network summary."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to reddit_all_user_network_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for output tables and figures.",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    input_path = args.input if args.input is not None else find_default_input(base_dir)

    print(f"[1] Loading summary CSV from: {input_path}")
    df = load_and_clean(input_path)

    print(f"[2] Loaded {len(df):,} subreddit networks.")
    stats = summary_statistics(df)

    print("[3] Saving summary tables...")
    save_tables(df, stats, args.output_dir)

    print("[4] Creating figures...")
    make_figures(df, args.output_dir)

    print_key_findings(df, stats)

    print("\nSaved outputs to:")
    print(f"  {args.output_dir.resolve()}")
    print(f"  {(args.output_dir / 'figures').resolve()}")


if __name__ == "__main__":
    main()
