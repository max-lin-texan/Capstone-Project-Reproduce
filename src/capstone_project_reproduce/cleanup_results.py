"""Remove rank-specific artifacts that were not selected for analysis.

The full rank sweep is retained in ``fit_summary.csv`` and the model
selection/diagnostic tables. Factor tables and factor figures are reduced to
the selected rank for each result directory. Cosine heatmaps are kept when
their rank is selected by at least one cohort in the corresponding experiment,
because each heatmap requires factor files from every cohort at that rank.
"""

import argparse
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_DIR / "results"


def remove_unselected_factor_files(
    table: pd.DataFrame,
    results_root: Path = RESULTS_ROOT,
) -> int:
    """Keep selected rank artifacts and remove other rank-specific files."""
    removed = 0
    for directory_name, group in table.groupby("result_directory"):
        directory = results_root / directory_name
        selected_rank = int(group.loc[group["selected"], "rank"].iloc[0])
        for path in directory.glob("rank*_*.csv"):
            if not path.name.startswith(f"rank{selected_rank}_"):
                path.unlink()
                removed += 1
        for path in directory.glob("rank*_*.png"):
            if not path.name.startswith(f"rank{selected_rank}_"):
                path.unlink()
                removed += 1
    return removed


def trim_fit_summaries(
    table: pd.DataFrame,
    results_root: Path = RESULTS_ROOT,
) -> int:
    """Keep only the selected rank in each result directory's fit summary."""
    trimmed = 0
    for directory_name, group in table.groupby("result_directory"):
        directory = results_root / directory_name
        summary_path = directory / "fit_summary.csv"
        if not summary_path.exists():
            continue
        selected_rank = int(group.loc[group["selected"], "rank"].iloc[0])
        summary = pd.read_csv(summary_path)
        summary = summary[summary["rank"] == selected_rank]
        summary.to_csv(summary_path, index=False, float_format="%.6f")
        trimmed += 1
    return trimmed


def remove_unselected_cosine_files(
    table: pd.DataFrame,
    results_root: Path = RESULTS_ROOT,
) -> int:
    """Keep cosine ranks selected by at least one cohort in each experiment."""
    removed = 0
    group_columns = ["feature_set", "method", "aggregation"]
    for keys, group in table.groupby(group_columns):
        panel, method, aggregation = keys
        keep_ranks = set(group.loc[group["selected"], "rank"].astype(int))
        directory = results_root / "cosine_similarity" / panel / f"{method}_{aggregation}"
        for path in directory.glob("rank*_*.csv"):
            rank = int(path.name.split("_", 1)[0].removeprefix("rank"))
            if rank not in keep_ranks:
                path.unlink()
                removed += 1
        for path in directory.glob("rank*_*.png"):
            rank = int(path.name.split("_", 1)[0].removeprefix("rank"))
            if rank not in keep_ranks:
                path.unlink()
                removed += 1
    return removed


def cleanup(
    selection_path: Path,
    results_root: Path = RESULTS_ROOT,
) -> tuple[int, int]:
    """Clean factor artifacts and cosine heatmaps using a selection table."""
    table = pd.read_csv(selection_path)
    required = {"result_directory", "selected", "rank", "feature_set", "method", "aggregation"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    factor_removed = remove_unselected_factor_files(table, results_root)
    summaries_trimmed = trim_fit_summaries(table, results_root)
    cosine_removed = remove_unselected_cosine_files(table, results_root)
    return factor_removed, summaries_trimmed + cosine_removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=RESULTS_ROOT / "model_selection" / "model_selection.csv",
    )
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()
    factor_removed, other_changes = cleanup(args.selection, args.results_root)
    print(f"Removed {factor_removed} unselected factor files")
    print(f"Trimmed fit summaries and removed unselected cosine files: {other_changes}")


if __name__ == "__main__":
    main()