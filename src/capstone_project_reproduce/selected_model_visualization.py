"""Visualize primary models selected by ``model_selection.py``.

The primary model definitions are read from ``model_selection_summary.csv``;
no method, aggregation, or rank is hard-coded here. Hospital ranks are read
from the first-level selection table for the same feature set, method, and
aggregation.
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .cosine_similarity import abs_cosine_matrix, loadings, paired_scores

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_DIR / "results"
SELECTION_NAME = "model_selection.csv"
SUMMARY_NAME = "model_selection_summary.csv"
MODEL_SELECTION_NAME = "model_selection"
OUTPUT_NAME = "selected_model_visualizations"


def selected_primary_models(summary_path: Path) -> pd.DataFrame:
    """Read primary models selected by the second-level selection."""
    summary = pd.read_csv(summary_path)
    required = {"second_level_selected", "feature_set", "cohort", "method", "aggregation", "rank", "result_directory"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Missing columns in {summary_path}: {sorted(missing)}")
    primary = summary[
        summary["second_level_selected"].astype(bool)
        & summary["cohort"].eq("overall")
    ].copy()
    if primary.empty:
        raise ValueError(f"No primary models found in {summary_path}")
    return primary


def selected_hospital_models(
    selection: pd.DataFrame,
    primary: pd.Series,
) -> pd.DataFrame:
    """Find first-level selected ranks for the primary setting across cohorts."""
    matches = selection[
        selection["selected"].astype(bool)
        & selection["feature_set"].eq(primary["feature_set"])
        & selection["method"].eq(primary["method"])
        & selection["aggregation"].eq(primary["aggregation"])
    ].copy()
    if matches.empty:
        raise ValueError(f"No cohort models found for {primary['result_directory']}")
    return matches.sort_values("cohort").drop_duplicates("cohort")


def component_alignment(
    reference_feature: np.ndarray,
    reference_time: np.ndarray,
    other_feature: np.ndarray,
    other_time: np.ndarray,
) -> np.ndarray:
    """Return other-component indices aligned to the reference components."""
    feature_similarity = abs_cosine_matrix(reference_feature, other_feature)
    time_similarity = abs_cosine_matrix(reference_time, other_time)
    rows, columns = linear_sum_assignment(
        feature_similarity * time_similarity,
        maximize=True,
    )
    aligned = np.full(reference_feature.shape[0], -1, dtype=int)
    aligned[rows] = columns
    return aligned


def load_model(directory: Path, rank: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load variable, time, and patient factors for one selected model."""
    variable = pd.read_csv(directory / f"rank{rank}_variable.csv")
    time = pd.read_csv(directory / f"rank{rank}_time.csv")
    patient = pd.read_csv(directory / f"rank{rank}_patient.csv")
    return variable, time, patient


def plot_loading_heatmap(
    frame: pd.DataFrame,
    index_column: str,
    path: Path,
    title: str,
    absolute: bool = False,
) -> None:
    """Save a feature/component or time/component loading heatmap."""
    values = frame.set_index(index_column)
    values = values.loc[:, [column for column in values.columns if column.startswith("comp")]]
    if absolute:
        values = values.abs()
    figure, axis = plt.subplots(figsize=(max(8, values.shape[1] * 0.55), 7))
    image = axis.imshow(values.to_numpy(float), aspect="auto", cmap="viridis")
    axis.set_xticks(range(values.shape[1]))
    axis.set_xticklabels(values.columns, rotation=45, ha="right")
    axis.set_yticks(range(values.shape[0]))
    axis.set_yticklabels(values.index)
    axis.set_title(title)
    axis.set_xlabel("Component")
    axis.set_ylabel(index_column.replace("_", " ").title())
    figure.colorbar(image, ax=axis, label="Absolute loading" if absolute else "Loading")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_patient_distributions(patient: pd.DataFrame, path: Path, title: str) -> None:
    """Save boxplots for patient component scores."""
    component_columns = [column for column in patient.columns if column.startswith("comp")]
    if not component_columns:
        raise ValueError("Patient factor table has no component columns")
    figure, axis = plt.subplots(figsize=(max(8, len(component_columns) * 0.55), 6))
    axis.boxplot([patient[column].dropna() for column in component_columns], showfliers=False)
    axis.set_xticks(range(1, len(component_columns) + 1))
    axis.set_xticklabels(component_columns, rotation=45, ha="right")
    axis.set_title(title)
    axis.set_xlabel("Component")
    axis.set_ylabel("Patient factor score")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_model_selection_summary(summary: pd.DataFrame, path: Path) -> None:
    """Plot fit and sensitivity cosine scores from the second-level table."""
    required = {
        "method",
        "aggregation",
        "rank",
        "best_fit",
        "second_level_role",
        "feature_cosine_to_primary",
        "time_cosine_to_primary",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Missing columns in model selection summary: {sorted(missing)}")

    methods = list(summary["method"].drop_duplicates())
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = {"primary": "#d95f02", "sensitivity": "#1b9e77"}

    for axis, method in zip(axes[0], methods):
        subset = summary[summary["method"].eq(method)].sort_values("aggregation")
        bars = axis.bar(
            subset["aggregation"],
            subset["best_fit"],
            color=[colors[role] for role in subset["second_level_role"]],
        )
        for bar, (_, row) in zip(bars, subset.iterrows()):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"rank {int(row['rank'])}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axis.set_title(f"{method.upper()} selected fit")
        axis.set_xlabel("Aggregation")
        axis.set_ylabel("Fit")
        axis.set_ylim(0, max(1.0, float(subset["best_fit"].max()) * 1.18))

    if len(methods) < len(axes[0]):
        for axis in axes[0, len(methods):]:
            axis.set_visible(False)

    for axis, column, title in zip(
        axes[1],
        ["feature_cosine_to_primary", "time_cosine_to_primary"],
        ["Feature loading cosine to primary", "Time loading cosine to primary"],
    ):
        for method in methods:
            subset = summary[summary["method"].eq(method)].sort_values("aggregation")
            x = np.arange(len(subset))
            offset = -0.18 if method == methods[0] else 0.18
            bars = axis.bar(
                x + offset,
                subset[column],
                width=0.36,
                label=method.upper(),
                alpha=0.9,
            )
            for bar, value in zip(bars, subset[column]):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        axis.set_xticks(np.arange(len(subset)))
        axis.set_xticklabels(subset["aggregation"])
        axis.set_ylim(0, 1.08)
        axis.set_xlabel("Aggregation")
        axis.set_ylabel("Absolute aligned cosine")
        axis.set_title(title)
        axis.legend()

    figure.suptitle("Second-level model selection and sensitivity summary")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_model_selection_fit_summary(summary: pd.DataFrame, path: Path) -> None:
    """Plot only the CPD and NNCPD selected-fit panels."""
    required = {"method", "aggregation", "rank", "best_fit", "second_level_role"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Missing columns in model selection summary: {sorted(missing)}")

    methods = list(summary["method"].drop_duplicates())
    figure, axes = plt.subplots(1, len(methods), figsize=(12, 5), constrained_layout=True, squeeze=False)
    colors = {"primary": "#d95f02", "sensitivity": "#1b9e77"}

    for axis, method in zip(axes[0], methods):
        subset = summary[summary["method"].eq(method)].sort_values("aggregation")
        bars = axis.bar(
            subset["aggregation"],
            subset["best_fit"],
            color=[colors[role] for role in subset["second_level_role"]],
        )
        for bar, (_, row) in zip(bars, subset.iterrows()):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"rank {int(row['rank'])}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axis.set_title(f"{method.upper()} selected fit")
        axis.set_xlabel("Aggregation")
        axis.set_ylabel("Fit")
        axis.set_ylim(0, max(1.0, float(subset["best_fit"].max()) * 1.18))
        axis.grid(axis="y", alpha=0.25)

    handles = [
        Patch(color=colors["primary"], label="Primary"),
        Patch(color=colors["sensitivity"], label="Sensitivity"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    figure.suptitle("Second-level model selection: selected fit")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_combined_hospital_map(
    models: pd.DataFrame,
    results_root: Path,
    path: Path,
    title: str,
) -> None:
    """Save aligned absolute feature loadings for overall and hospitals."""
    overall = models.loc[models["cohort"].eq("overall")].iloc[0]
    overall_directory = results_root / overall["result_directory"]
    overall_variable, overall_time, _ = load_model(overall_directory, int(overall["rank"]))
    reference_feature = overall_variable.filter(like="comp").to_numpy(float).T
    reference_time = overall_time.filter(like="comp").to_numpy(float).T
    feature_names = overall_variable[overall_variable.columns[0]].astype(str).tolist()
    component_count = reference_feature.shape[0]
    panels: list[tuple[str, np.ndarray]] = []

    def cohort_sort_key(value: str) -> tuple[int, int]:
        if value == "overall":
            return (0, 0)
        return (1, int(value.removeprefix("hospital_")))

    ordered_models = models.sort_values(
        "cohort",
        key=lambda values: values.map(cohort_sort_key),
    )
    for _, model in ordered_models.iterrows():
        directory = results_root / model["result_directory"]
        variable, time, _ = load_model(directory, int(model["rank"]))
        feature = variable.filter(like="comp").to_numpy(float).T
        time_values = time.filter(like="comp").to_numpy(float).T
        alignment = component_alignment(reference_feature, reference_time, feature, time_values)
        aligned = np.full((component_count, len(feature_names)), np.nan)
        for reference_index, other_index in enumerate(alignment):
            if other_index >= 0:
                aligned[reference_index] = np.abs(feature[other_index])
        panels.append((str(model["cohort"]), aligned.T))

    column_count = min(3, len(panels))
    row_count = math.ceil(len(panels) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(4.2 * column_count, 4.8 * row_count),
        squeeze=False,
    )
    figure.subplots_adjust(left=0.16, right=0.88, bottom=0.08, top=0.92, wspace=0.35, hspace=0.55)
    maximum = max(float(np.nanmax(values)) for _, values in panels)
    image = None
    flat_axes = axes.ravel()
    for panel_index, (axis, (cohort, values)) in enumerate(zip(flat_axes, panels)):
        image = axis.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=maximum)
        axis.set_title(cohort)
        axis.set_xticks(range(component_count))
        axis.set_xticklabels([f"comp{i + 1}" for i in range(component_count)], rotation=90)
        axis.set_yticks(range(len(feature_names)))
        if panel_index % column_count == 0:
            axis.set_yticklabels(feature_names)
        else:
            axis.set_yticklabels([])
    for axis in flat_axes[len(panels):]:
        axis.set_visible(False)
    figure.suptitle(title)
    if image is not None:
        colorbar_axis = figure.add_axes([0.90, 0.18, 0.018, 0.64])
        figure.colorbar(image, cax=colorbar_axis, label="Absolute aligned loading")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def visualize_primary_model(
    primary: pd.Series,
    selection: pd.DataFrame,
    results_root: Path,
    output_root: Path,
) -> None:
    """Create all Step 6 figures for one dynamically selected primary model."""
    models = selected_hospital_models(selection, primary)
    output = output_root / f"{primary['method']}_{primary['aggregation']}_rank{int(primary['rank'])}"
    output.mkdir(parents=True, exist_ok=True)
    overall_directory = results_root / primary["result_directory"]
    variable, time, patient = load_model(overall_directory, int(primary["rank"]))
    label = f"{primary['feature_set']} {primary['method']} {primary['aggregation']} rank {int(primary['rank'])}"
    plot_loading_heatmap(variable, "feature", output / "feature_loading.png", f"{label}: feature loading")
    plot_loading_heatmap(time, "hour", output / "time_loading.png", f"{label}: time loading")
    plot_patient_distributions(patient, output / "patient_factor_distribution.png", f"{label}: patient factors")
    plot_combined_hospital_map(models, results_root, output / "combined_hospital_feature_map.png", f"{label}: aligned hospital feature loadings")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--selection", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    results_root = args.results_root
    model_selection_root = results_root / MODEL_SELECTION_NAME
    selection_path = args.selection or model_selection_root / SELECTION_NAME
    summary_path = args.summary or model_selection_root / SUMMARY_NAME
    output_root = args.output_root or results_root / OUTPUT_NAME
    selection = pd.read_csv(selection_path)
    primary_models = selected_primary_models(summary_path)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_path)
    summary_figure = model_selection_root / "model_selection_summary.png"
    fit_summary_figure = model_selection_root / "model_selection_summary_fit_only.png"
    plot_model_selection_summary(summary, summary_figure)
    plot_model_selection_fit_summary(summary, fit_summary_figure)
    print(f"Saved model selection summary to {summary_figure}")
    print(f"Saved fit-only model selection summary to {fit_summary_figure}")
    for _, primary in primary_models.iterrows():
        visualize_primary_model(primary, selection, results_root, output_root)
        print(
            f"Saved visualizations for {primary['method']} {primary['aggregation']} "
            f"rank {int(primary['rank'])}"
        )


if __name__ == "__main__":
    main()
