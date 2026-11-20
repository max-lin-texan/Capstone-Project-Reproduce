"""Select a simple, stable rank for each saved CPD result set.

CPD and NNCPD are evaluated separately because their fits are based on
different input scaling and constraints.  Within one result directory, the
smallest rank within ``fit_tolerance`` of the best fit is selected.  The
output also records fit improvements and random-start variability so the
selection can be reviewed rather than treated as an absolute truth.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .cosine_similarity import loadings, paired_scores


PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_DIR / "results"
MODEL_SELECTION_ROOT = RESULTS_ROOT / "model_selection"
OUTPUT_PATH = MODEL_SELECTION_ROOT / "model_selection.csv"
SECOND_LEVEL_OUTPUT_NAME = "model_selection_summary.csv"
DEFAULT_ELBOW_IMPROVEMENT = 0.02


def result_directories(results_root: Path) -> list[Path]:
    """Return result directories containing a fit summary."""
    return sorted(path.parent for path in results_root.glob("**/fit_summary.csv"))


def metadata_for(directory: Path, results_root: Path) -> dict[str, str]:
    """Extract method, aggregation, feature panel, and cohort from a path."""
    relative = directory.relative_to(results_root)
    parts = relative.parts
    if len(parts) < 3 or not parts[-1].startswith(("cpd_", "nncpd_")):
        raise ValueError(f"Unexpected result directory: {directory}")
    panel, cohort, result_name = parts[-3:]
    method, how = result_name.split("_", 1)
    return {
        "feature_set": panel,
        "cohort": cohort,
        "method": method,
        "aggregation": how,
    }


def add_selection_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    """Add rank-complexity and stability metrics to one fit summary."""
    required = {"rank", "best_fit", "fit_std"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    result = summary.copy()
    result = result.sort_values("rank").reset_index(drop=True)
    result["relative_error"] = 1.0 - result["best_fit"]
    result["fit_improvement"] = result["best_fit"].diff()
    result.loc[result.index[0], "fit_improvement"] = np.nan
    best_fit = result["best_fit"].max()
    result["fit_gap_to_best"] = best_fit - result["best_fit"]
    return result


def select_rank(
    metrics: pd.DataFrame,
    fit_tolerance: float,
    stability_tolerance: float,
) -> tuple[int, str]:
    """Select the lowest near-best rank and report any stability warning."""
    near_best = metrics[metrics["fit_gap_to_best"] <= fit_tolerance]
    if near_best.empty:
        raise ValueError("No rank satisfies fit_tolerance")

    stable = near_best[near_best["fit_std"] <= stability_tolerance]
    candidates = stable if not stable.empty else near_best
    selected_rank = int(candidates.iloc[0]["rank"])
    reason = "lowest rank within fit tolerance"
    if stable.empty:
        reason += "; no near-best rank met stability tolerance"
    return selected_rank, reason


def select_models(
    results_root: Path = RESULTS_ROOT,
    fit_tolerance: float = 0.01,
    stability_tolerance: float = 0.01,
) -> pd.DataFrame:
    """Read all summaries and return candidate rows with one selected rank per set."""
    selected_frames: list[pd.DataFrame] = []
    for directory in result_directories(results_root):
        metadata = metadata_for(directory, results_root)
        metrics = add_selection_metrics(pd.read_csv(directory / "fit_summary.csv"))
        selected_rank, reason = select_rank(
            metrics,
            fit_tolerance=fit_tolerance,
            stability_tolerance=stability_tolerance,
        )
        metrics["selected"] = metrics["rank"].eq(selected_rank)
        metrics["selection_reason"] = ""
        metrics.loc[metrics["selected"], "selection_reason"] = reason
        for key, value in metadata.items():
            metrics[key] = value
        metrics["result_directory"] = str(directory.relative_to(results_root))
        selected_frames.append(metrics)

    if not selected_frames:
        raise FileNotFoundError(f"No fit_summary.csv found under {results_root}")
    return pd.concat(selected_frames, ignore_index=True)


def build_diagnostics(
    table: pd.DataFrame,
    elbow_improvement: float = DEFAULT_ELBOW_IMPROVEMENT,
) -> pd.DataFrame:
    """Summarize rank growth, stability, and elbow candidates per result set."""
    group_columns = [
        "feature_set",
        "cohort",
        "method",
        "aggregation",
        "result_directory",
    ]
    rows: list[dict[str, object]] = []
    for _, group in table.groupby(group_columns, sort=True):
        ordered = group.sort_values("rank")
        selected = ordered[ordered["selected"]]
        selected_row = selected.iloc[0]
        last_row = ordered.iloc[-1]
        elbow_candidates = ordered[
            ordered["fit_improvement"].notna()
            & (ordered["fit_improvement"] <= elbow_improvement)
        ]
        elbow_row = elbow_candidates.iloc[0] if not elbow_candidates.empty else last_row
        rows.append(
            {
                **{column: selected_row[column] for column in group_columns},
                "selected_rank": int(selected_row["rank"]),
                "selected_fit": float(selected_row["best_fit"]),
                "selected_fit_std": float(selected_row["fit_std"]),
                "selected_fit_improvement": float(selected_row["fit_improvement"])
                if pd.notna(selected_row["fit_improvement"])
                else np.nan,
                "maximum_rank": int(last_row["rank"]),
                "maximum_rank_fit": float(last_row["best_fit"]),
                "last_fit_improvement": float(last_row["fit_improvement"])
                if pd.notna(last_row["fit_improvement"])
                else np.nan,
                "elbow_rank": int(elbow_row["rank"]),
                "elbow_fit": float(elbow_row["best_fit"]),
                "elbow_fit_improvement": float(elbow_row["fit_improvement"])
                if pd.notna(elbow_row["fit_improvement"])
                else np.nan,
                "elbow_improvement_threshold": elbow_improvement,
                "elbow_is_fallback": bool(elbow_candidates.empty),
                "rank_reduction_if_elbow": int(selected_row["rank"] - elbow_row["rank"]),
                "fit_loss_if_elbow": float(selected_row["best_fit"] - elbow_row["best_fit"]),
                "relative_fit_loss_if_elbow": float(
                    (selected_row["best_fit"] - elbow_row["best_fit"])
                    / selected_row["best_fit"]
                )
                if selected_row["best_fit"] != 0
                else np.nan,
                "selected_at_maximum_rank": bool(
                    selected_row["rank"] == last_row["rank"]
                ),
                "stable_at_selected_rank": bool(selected_row["fit_std"] <= 0.01),
            }
        )
    return pd.DataFrame(rows)


def select_primary_models(
    table: pd.DataFrame,
    feature_set: str = "feature_11",
    cohort: str = "overall",
    results_root: Path = RESULTS_ROOT,
) -> pd.DataFrame:
    """Select primary and sensitivity aggregations after rank selection.

    CPD and NNCPD are ranked separately because their fit values are not
    comparable. Within each method, the aggregation with the highest selected
    fit is primary; fit variability and rank break ties. The other aggregations
    remain sensitivity models.
    """
    required = {
        "feature_set",
        "cohort",
        "method",
        "aggregation",
        "result_directory",
        "rank",
        "best_fit",
        "fit_std",
        "selected",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    candidates = table[
        table["selected"]
        & table["feature_set"].eq(feature_set)
        & table["cohort"].eq(cohort)
    ].copy()
    if candidates.empty:
        raise ValueError(
            f"No selected models found for feature_set={feature_set!r}, "
            f"cohort={cohort!r}"
        )

    candidates["second_level_role"] = "sensitivity"
    candidates["second_level_selected"] = False
    candidates["second_level_reason"] = "alternative aggregation for sensitivity analysis"

    for method, group in candidates.groupby("method", sort=True):
        ordered = group.sort_values(
            ["best_fit", "fit_std", "rank"],
            ascending=[False, True, True],
        )
        primary_index = ordered.index[0]
        candidates.loc[primary_index, "second_level_role"] = "primary"
        candidates.loc[primary_index, "second_level_selected"] = True
        candidates.loc[primary_index, "second_level_reason"] = (
            f"highest selected fit within {method}; stability and rank used as tie-breakers"
        )

        primary = candidates.loc[primary_index]
        primary_path = results_root / primary["result_directory"]
        primary_feature, primary_time = loadings(primary_path, int(primary["rank"]))
        for index, row in group.iterrows():
            path = results_root / row["result_directory"]
            feature, time = loadings(path, int(row["rank"]))
            feature_score, time_score = paired_scores(
                primary_feature,
                primary_time,
                feature,
                time,
            )
            candidates.loc[index, "feature_cosine_to_primary"] = feature_score
            candidates.loc[index, "time_cosine_to_primary"] = time_score
            candidates.loc[index, "cosine_score_to_primary"] = float(
                np.sqrt(feature_score * time_score)
            )

    return candidates.sort_values(
        ["method", "second_level_role", "aggregation"]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-tolerance", type=float, default=0.01)
    parser.add_argument("--stability-tolerance", type=float, default=0.01)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--diagnostics-output", type=Path, default=None)
    parser.add_argument("--second-level-output", type=Path, default=None)
    parser.add_argument("--primary-feature-set", default="feature_11")
    parser.add_argument("--primary-cohort", default="overall")
    parser.add_argument("--elbow-improvement", type=float, default=DEFAULT_ELBOW_IMPROVEMENT)
    args = parser.parse_args()
    if args.fit_tolerance < 0 or args.stability_tolerance < 0 or args.elbow_improvement < 0:
        parser.error("tolerances must be non-negative")

    output = args.output or args.results_root / "model_selection" / "model_selection.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table = select_models(
        results_root=args.results_root,
        fit_tolerance=args.fit_tolerance,
        stability_tolerance=args.stability_tolerance,
    )
    table.to_csv(output, index=False, float_format="%.6f")
    diagnostics_output = args.diagnostics_output or output.with_name("model_diagnostics.csv")
    diagnostics = build_diagnostics(table, elbow_improvement=args.elbow_improvement)
    diagnostics.to_csv(diagnostics_output, index=False, float_format="%.6f")
    second_level_output = args.second_level_output or output.with_name(
        SECOND_LEVEL_OUTPUT_NAME
    )
    second_level = select_primary_models(
        table,
        feature_set=args.primary_feature_set,
        cohort=args.primary_cohort,
        results_root=args.results_root,
    )
    second_level.to_csv(second_level_output, index=False, float_format="%.6f")
    selected = table[table["selected"]]
    print(f"Saved {len(table):,} candidate rows to {output}")
    print(f"Saved {len(diagnostics):,} diagnostic rows to {diagnostics_output}")
    print(f"Selected {len(selected):,} models")
    print(f"Saved {len(second_level):,} second-level candidates to {second_level_output}")
    print(
        "Primary models: "
        f"{int(second_level['second_level_selected'].sum())}"
    )


if __name__ == "__main__":
    main()