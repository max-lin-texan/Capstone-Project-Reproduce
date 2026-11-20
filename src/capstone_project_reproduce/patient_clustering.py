"""Cluster patient factors from dynamically selected primary models.

Primary models are read from ``results/model_selection/model_selection_summary.csv``.
Clustering uses all patient-factor components; PCA is used only for plotting.
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_DIR / "results"
MODEL_SELECTION_ROOT_NAME = "model_selection"
OUTPUT_ROOT_NAME = "patient_clustering"
CLUSTER_METHODS = ("kmeans", "hierarchical")
CLUSTER_COUNTS = range(2, 7)
RANDOM_STATE = 42
MIN_CLUSTER_FRACTION = 0.01
HIERARCHICAL_SAMPLE_SIZE = int(
    os.environ.get("PATIENT_CLUSTERING_HIERARCHICAL_SAMPLE_SIZE", "3000")
)
EXPLORATORY_K = {"cpd": 3, "nncpd": 4}
CLUSTER_COLORS = ["#08519C", "#CB181D", "#238B45", "#D94801", "#54278F", "#252525"]


def as_bool(series: pd.Series) -> pd.Series:
    """Parse boolean columns safely from CSV values."""
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().eq("true")


def primary_models(summary_path: Path, cohort: str = "overall") -> pd.DataFrame:
    """Read second-level primary models for one cohort without hard-coded method or rank."""
    summary = pd.read_csv(summary_path)
    required = {
        "second_level_selected",
        "feature_set",
        "cohort",
        "method",
        "aggregation",
        "rank",
        "result_directory",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Missing columns in {summary_path}: {sorted(missing)}")
    selected = summary[as_bool(summary["second_level_selected"])].copy()
    selected = selected[selected["cohort"].eq(cohort)]
    if selected.empty:
        raise ValueError(f"No primary models found in {summary_path}")
    return selected


def load_patient_factors(results_root: Path, model: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """Load patient IDs and component scores for one selected model."""
    path = results_root / model["result_directory"] / f"rank{int(model['rank'])}_patient.csv"
    frame = pd.read_csv(path)
    component_columns = [column for column in frame.columns if column.startswith("comp")]
    if not component_columns:
        raise ValueError(f"No component columns found in {path}")
    if frame.iloc[:, 0].duplicated().any():
        raise ValueError(f"Duplicate patient IDs found in {path}")
    if frame[component_columns].isna().any().any():
        raise ValueError(f"Missing patient factors found in {path}")
    return frame, component_columns


def fit_labels(
    values: np.ndarray,
    method: str,
    cluster_count: int,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Fit one clustering method for one requested cluster count."""
    if method == "kmeans":
        estimator = KMeans(
            n_clusters=cluster_count,
            n_init=20,
            random_state=random_state,
        )
        return estimator.fit_predict(values)
    elif method == "hierarchical":
        sample_size = min(HIERARCHICAL_SAMPLE_SIZE, len(values))
        sample_indices = np.random.default_rng(random_state).choice(
            len(values), size=sample_size, replace=False
        )
        estimator = AgglomerativeClustering(
            n_clusters=cluster_count,
            linkage="ward",
        )
        sample_labels = estimator.fit_predict(values[sample_indices])
        centroids = np.vstack(
            [
                values[sample_indices][sample_labels == label].mean(axis=0)
                for label in range(cluster_count)
            ]
        )
        distances = ((values[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1)
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def evaluate_labels(values: np.ndarray, labels: np.ndarray) -> dict[str, float | int | bool]:
    """Calculate clustering metrics and flag very small clusters."""
    counts = np.bincount(labels)
    minimum_fraction = float(counts.min() / len(labels))
    return {
        "silhouette_score": float(silhouette_score(values, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(values, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(values, labels)),
        "minimum_cluster_fraction": minimum_fraction,
        "has_small_cluster": bool(minimum_fraction < MIN_CLUSTER_FRACTION),
    }


def choose_recommended_k(metrics: pd.DataFrame) -> dict[str, int]:
    """Choose a provisional k using silhouette among non-tiny solutions."""
    recommendations: dict[str, int] = {}
    for method, group in metrics.groupby("clustering_method"):
        eligible = group[~group["has_small_cluster"]]
        if eligible.empty:
            eligible = group
        chosen = eligible.sort_values(
            ["silhouette_score", "calinski_harabasz_score"],
            ascending=[False, False],
        ).iloc[0]
        recommendations[str(method)] = int(chosen["cluster_count"])
    return recommendations


def plot_metrics(metrics: pd.DataFrame, path: Path) -> None:
    """Plot all three quality metrics across k for both methods."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    definitions = [
        ("silhouette_score", "Silhouette score", "higher is better"),
        ("calinski_harabasz_score", "Calinski-Harabasz score", "higher is better"),
        ("davies_bouldin_score", "Davies-Bouldin score", "lower is better"),
    ]
    for axis, (column, title, direction) in zip(axes, definitions):
        for method, group in metrics.groupby("clustering_method"):
            axis.plot(group["cluster_count"], group[column], marker="o", label=method)
        axis.set_title(title)
        axis.set_xlabel("Number of clusters (k)")
        axis.set_ylabel(direction)
        axis.set_xticks(sorted(metrics["cluster_count"].unique()))
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Patient clustering quality metrics")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_cluster_sizes(
    labels_by_method: dict[str, tuple[int, np.ndarray]],
    path: Path,
    title: str = "Recommended cluster sizes",
) -> None:
    """Plot cluster proportions for each provisional recommended solution."""
    figure, axes = plt.subplots(1, len(labels_by_method), figsize=(10, 4), squeeze=False)
    for axis, (method, (cluster_count, labels)) in zip(axes[0], labels_by_method.items()):
        counts = pd.Series(labels).value_counts().sort_index()
        axis.bar([f"C{i + 1}" for i in counts.index], counts / len(labels) * 100)
        axis.set_title(f"{method}, k={cluster_count}")
        axis.set_ylabel("Patients (%)")
        axis.set_ylim(0, 100)
        for index, value in enumerate(counts / len(labels) * 100):
            axis.text(index, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_pca_clusters(
    values: np.ndarray,
    labels_by_method: dict[str, tuple[int, np.ndarray]],
    path: Path,
    title: str = "Patient clusters in PCA space",
) -> None:
    """Plot recommended clusters in PCA space; clustering still uses all dimensions."""
    projection = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(values)
    figure, axes = plt.subplots(1, len(labels_by_method), figsize=(12, 5), squeeze=False)
    for axis, (method, (cluster_count, labels)) in zip(axes[0], labels_by_method.items()):
        axis.scatter(
            projection[:, 0],
            projection[:, 1],
            c=labels,
            s=8,
            alpha=0.9,
            cmap=matplotlib.colors.ListedColormap(CLUSTER_COLORS[:cluster_count]),
            vmin=-0.5,
            vmax=cluster_count - 0.5,
        )
        axis.legend(
            handles=[
                Patch(color=CLUSTER_COLORS[index], label=f"Cluster {index + 1}")
                for index in range(cluster_count)
            ],
            loc="best",
            fontsize=8,
        )
        axis.set_title(f"{method}, k={cluster_count}")
        axis.set_xlabel("PCA 1")
        axis.set_ylabel("PCA 2")
        axis.grid(alpha=0.2)
    explained = PCA(n_components=2, random_state=RANDOM_STATE).fit(values).explained_variance_ratio_.sum()
    figure.suptitle(f"{title} (first 2 PCs explain {explained:.1%})")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_cluster_profiles(profiles: pd.DataFrame, path: Path) -> None:
    """Plot standardized mean component scores by cluster."""
    methods = list(profiles["clustering_method"].unique())
    figure, axes = plt.subplots(1, len(methods), figsize=(14, 5), squeeze=False)
    for axis, method in zip(axes[0], methods):
        subset = profiles[profiles["clustering_method"].eq(method)]
        pivot = subset.pivot(index="cluster", columns="component", values="mean_standardized_score")
        image = axis.imshow(pivot.to_numpy(float), aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
        axis.set_title(method)
        axis.set_xticks(range(len(pivot.columns)))
        axis.set_xticklabels(pivot.columns, rotation=90)
        axis.set_yticks(range(len(pivot.index)))
        axis.set_yticklabels([f"C{int(value) + 1}" for value in pivot.index])
    figure.suptitle("Recommended cluster component profiles")
    figure.colorbar(image, ax=axes.ravel().tolist(), label="Mean standardized factor score")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def profile_rows_for_labels(
    values: np.ndarray,
    component_columns: list[str],
    method: str,
    cluster_count: int,
    labels: np.ndarray,
) -> list[dict[str, object]]:
    """Summarize standardized component scores for one clustering solution."""
    rows: list[dict[str, object]] = []
    for cluster_label in sorted(np.unique(labels)):
        selected = values[labels == cluster_label]
        for component, mean in zip(component_columns, selected.mean(axis=0)):
            rows.append(
                {
                    "clustering_method": method,
                    "cluster_count": cluster_count,
                    "cluster": int(cluster_label),
                    "component": component,
                    "mean_standardized_score": float(mean),
                }
            )
    return rows


def run_model(
    model: pd.Series,
    results_root: Path,
    output_root: Path,
    cluster_counts: tuple[int, ...] = tuple(CLUSTER_COUNTS),
    random_state: int = RANDOM_STATE,
) -> None:
    """Run both clustering methods and write tables and figures for one model."""
    frame, component_columns = load_patient_factors(results_root, model)
    values = StandardScaler().fit_transform(frame[component_columns].to_numpy(float))
    model_name = f"{model['method']}_{model['aggregation']}_rank{int(model['rank'])}"
    output = output_root / model_name
    output.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, object]] = []
    labels_rows: list[pd.DataFrame] = []
    profile_rows: list[dict[str, object]] = []
    labels_by_method: dict[str, tuple[int, np.ndarray]] = {}
    for method in CLUSTER_METHODS:
        for cluster_count in cluster_counts:
            labels = fit_labels(values, method, cluster_count, random_state=random_state)
            metrics = evaluate_labels(values, labels)
            metrics_rows.append(
                {
                    "method": model["method"],
                    "aggregation": model["aggregation"],
                    "rank": int(model["rank"]),
                    "clustering_method": method,
                    "cluster_count": cluster_count,
                    "hierarchical_sample_size": (
                        min(HIERARCHICAL_SAMPLE_SIZE, len(values))
                        if method == "hierarchical"
                        else len(values)
                    ),
                    **metrics,
                }
            )
            labels_rows.append(
                pd.DataFrame(
                    {
                        "patientunitstayid": frame.iloc[:, 0],
                        "clustering_method": method,
                        "cluster_count": cluster_count,
                        "cluster_label": labels,
                    }
                )
            )

    metrics_frame = pd.DataFrame(metrics_rows)
    recommendations = choose_recommended_k(metrics_frame)
    for method, cluster_count in recommendations.items():
        labels = fit_labels(values, method, cluster_count, random_state=random_state)
        labels_by_method[method] = (cluster_count, labels)
        for cluster_label in sorted(np.unique(labels)):
            selected = values[labels == cluster_label]
            for component, mean in zip(component_columns, selected.mean(axis=0)):
                profile_rows.append(
                    {
                        "clustering_method": method,
                        "cluster_count": cluster_count,
                        "cluster": int(cluster_label),
                        "component": component,
                        "mean_standardized_score": float(mean),
                    }
                )

    metrics_frame["recommended_k"] = metrics_frame.apply(
        lambda row: recommendations[row["clustering_method"]], axis=1
    )
    metrics_frame.to_csv(output / "clustering_metrics.csv", index=False, float_format="%.6f")
    pd.concat(labels_rows, ignore_index=True).to_csv(output / "cluster_labels.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(output / "cluster_profiles.csv", index=False, float_format="%.6f")
    plot_metrics(metrics_frame, output / "clustering_metrics.png")
    plot_cluster_sizes(labels_by_method, output / "cluster_sizes.png")
    plot_pca_clusters(values, labels_by_method, output / "pca_clusters.png")
    plot_cluster_profiles(pd.DataFrame(profile_rows), output / "cluster_profiles.png")

    exploratory_k = EXPLORATORY_K.get(str(model["method"]))
    if exploratory_k is not None:
        candidate_output = output / f"candidate_k{exploratory_k}"
        candidate_output.mkdir(parents=True, exist_ok=True)
        candidate_labels_by_method: dict[str, tuple[int, np.ndarray]] = {}
        candidate_profile_rows: list[dict[str, object]] = []
        for method in CLUSTER_METHODS:
            labels = fit_labels(values, method, exploratory_k, random_state=random_state)
            candidate_labels_by_method[method] = (exploratory_k, labels)
            candidate_profile_rows.extend(
                profile_rows_for_labels(
                    values,
                    component_columns,
                    method,
                    exploratory_k,
                    labels,
                )
            )
        candidate_metrics = metrics_frame[
            metrics_frame["cluster_count"].eq(exploratory_k)
        ].copy()
        candidate_metrics["exploratory_candidate"] = True
        candidate_metrics.to_csv(
            candidate_output / "candidate_metrics.csv",
            index=False,
            float_format="%.6f",
        )
        candidate_label_frames = [
            pd.DataFrame(
                {
                    "patientunitstayid": frame.iloc[:, 0],
                    "clustering_method": method,
                    "cluster_count": exploratory_k,
                    "cluster_label": labels,
                }
            )
            for method, (_, labels) in candidate_labels_by_method.items()
        ]
        pd.concat(candidate_label_frames, ignore_index=True).to_csv(
            candidate_output / "candidate_cluster_labels.csv",
            index=False,
        )
        candidate_profiles = pd.DataFrame(candidate_profile_rows)
        candidate_profiles.to_csv(
            candidate_output / "candidate_cluster_profiles.csv",
            index=False,
            float_format="%.6f",
        )
        plot_cluster_sizes(
            candidate_labels_by_method,
            candidate_output / "candidate_cluster_sizes.png",
            title=f"Exploratory candidate cluster sizes (k={exploratory_k})",
        )
        plot_pca_clusters(
            values,
            candidate_labels_by_method,
            candidate_output / "candidate_pca_clusters.png",
            title=f"Exploratory candidate clusters (k={exploratory_k})",
        )
        plot_cluster_profiles(
            candidate_profiles,
            candidate_output / "candidate_cluster_profiles.png",
        )
        print(f"{model_name}: exploratory candidate k={exploratory_k}")
    print(
        f"{model_name}: recommended k="
        + ", ".join(f"{method}={cluster_count}" for method, cluster_count in recommendations.items())
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--cohort", default="overall")
    parser.add_argument("--cluster-counts", default="2,3,4,5,6")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    try:
        cluster_counts = tuple(sorted({int(value.strip()) for value in args.cluster_counts.split(",") if value.strip()}))
    except ValueError as error:
        parser.error(f"--cluster-counts must contain integers: {error}")
    if any(count < 2 for count in cluster_counts):
        parser.error("cluster counts must be at least 2")

    summary_path = args.summary or args.results_root / MODEL_SELECTION_ROOT_NAME / "model_selection_summary.csv"
    output_root = args.output_root or args.results_root / OUTPUT_ROOT_NAME
    output_root.mkdir(parents=True, exist_ok=True)
    for _, model in primary_models(summary_path, cohort=args.cohort).iterrows():
        run_model(
            model,
            args.results_root,
            output_root,
            cluster_counts=cluster_counts,
            random_state=args.random_state,
        )


if __name__ == "__main__":
    main()
