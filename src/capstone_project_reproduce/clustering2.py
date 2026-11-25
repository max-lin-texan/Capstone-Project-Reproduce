"""Run a multi-method patient-factor clustering analysis.

This program compares K-means, hierarchical clustering, GMM, spectral
clustering, and HDBSCAN for the primary overall CPD/NNCPD models.  It
does not use silhouette, Calinski-Harabasz, or Davies-Bouldin scores.
"""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter
from sklearn.cluster import (
    AgglomerativeClustering,
    HDBSCAN,
    KMeans,
    SpectralClustering,
)
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_DIR / "results"
OUTPUT_ROOT = RESULTS_ROOT / "patient_clustering2"
COLORS = [
    "#08519C",
    "#CB181D",
    "#238B45",
    "#D94801",
    "#54278F",
    "#8C6D31",
]


def cluster_color(cluster: int) -> str:
    if cluster == -1:
        return "#808080"
    return COLORS[(cluster - 1) % len(COLORS)]


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().eq("true")


def read_primary_models(results_root: Path) -> pd.DataFrame:
    summary_path = results_root / "model_selection" / "model_selection_summary.csv"
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

    primary = summary[
        as_bool(summary["second_level_selected"])
        & summary["cohort"].eq("overall")
    ].copy()
    if primary.empty:
        raise FileNotFoundError("No primary overall model found")
    return primary.sort_values(["method", "aggregation", "rank"])


def load_factors(results_root: Path, model: pd.Series):
    result_directory = results_root / str(model["result_directory"])
    factor_path = result_directory / f"rank{int(model['rank'])}_patient.csv"
    frame = pd.read_csv(factor_path)
    components = [column for column in frame.columns if column.startswith("comp")]
    if not components:
        raise ValueError(f"No component columns found in {factor_path}")
    if frame.iloc[:, 0].duplicated().any():
        raise ValueError(f"Duplicate patient IDs in {factor_path}")
    if frame[components].isna().any().any():
        raise ValueError(f"Missing patient factors in {factor_path}")

    patient_ids = frame.iloc[:, 0].to_numpy()
    values = StandardScaler().fit_transform(frame[components].to_numpy(float))
    return patient_ids, components, values


def load_overall_scaler_and_aligned_hospital(
    results_root: Path,
    model: pd.Series,
) -> tuple[np.ndarray, StandardScaler, list[str], dict[str, np.ndarray]]:
    overall_ids, overall_components, overall_values = load_factors(results_root, model)
    overall_directory = results_root / str(model["result_directory"])
    overall_patient_path = overall_directory / f"rank{int(model['rank'])}_patient.csv"
    overall_patient = pd.read_csv(overall_patient_path)
    raw_overall = overall_patient[overall_components].to_numpy(float)
    scaler = StandardScaler().fit(raw_overall)

    selection_path = results_root / "model_selection" / "model_selection.csv"
    selection = pd.read_csv(selection_path)
    selection = selection[
        as_bool(selection["selected"])
        & selection["feature_set"].eq(str(model["feature_set"]))
        & selection["method"].eq(str(model["method"]))
        & selection["aggregation"].eq(str(model["aggregation"]))
    ].copy()

    overall_variable = pd.read_csv(
        overall_directory / f"rank{int(model['rank'])}_variable.csv",
        index_col=0,
    )
    overall_time = pd.read_csv(
        overall_directory / f"rank{int(model['rank'])}_time.csv",
        index_col=0,
    )
    overall_feature = np.vstack([
        overall_variable[column].to_numpy(float) for column in overall_components
    ])
    overall_temporal = np.vstack([
        overall_time[column].to_numpy(float) for column in overall_components
    ])

    aligned_hospitals: dict[str, np.ndarray] = {}
    for hospital_number in range(1, 11):
        cohort = f"hospital_{hospital_number}"
        rows = selection[selection["cohort"].eq(cohort)]
        if len(rows) != 1:
            raise ValueError(
                f"Expected one selected model for {cohort}, "
                f"{model['method']}/{model['aggregation']}/{model['feature_set']}; "
                f"found {len(rows)}"
            )
        hospital_row = rows.iloc[0]
        hospital_directory = results_root / str(hospital_row["result_directory"])
        rank = int(hospital_row["rank"])
        hospital_patient = pd.read_csv(hospital_directory / f"rank{rank}_patient.csv")
        hospital_components = [
            column for column in hospital_patient.columns if column.startswith("comp")
        ]
        hospital_variable = pd.read_csv(
            hospital_directory / f"rank{rank}_variable.csv",
            index_col=0,
        )
        hospital_time = pd.read_csv(
            hospital_directory / f"rank{rank}_time.csv",
            index_col=0,
        )
        hospital_feature = np.vstack([
            hospital_variable[column].to_numpy(float) for column in hospital_components
        ])
        hospital_temporal = np.vstack([
            hospital_time[column].to_numpy(float) for column in hospital_components
        ])

        feature_similarity = np.abs(hospital_feature @ overall_feature.T)
        for candidate in range(feature_similarity.shape[0]):
            feature_similarity[candidate] /= (
                np.linalg.norm(hospital_feature[candidate]) *
                np.linalg.norm(overall_feature, axis=1)
            )
        temporal_similarity = np.abs(hospital_temporal @ overall_temporal.T)
        for candidate in range(temporal_similarity.shape[0]):
            temporal_similarity[candidate] /= (
                np.linalg.norm(hospital_temporal[candidate]) *
                np.linalg.norm(overall_temporal, axis=1)
            )
        candidate_indices, reference_indices = linear_sum_assignment(
            -(feature_similarity * temporal_similarity)
        )

        aligned = np.full(
            (len(hospital_patient), len(overall_components)),
            np.nan,
            dtype=float,
        )
        raw_hospital = hospital_patient[hospital_components].to_numpy(float)
        for candidate_component, reference_component in zip(
            candidate_indices, reference_indices
        ):
            feature_cosine = float(
                hospital_feature[candidate_component] @
                overall_feature[reference_component] /
                (
                    np.linalg.norm(hospital_feature[candidate_component]) *
                    np.linalg.norm(overall_feature[reference_component])
                )
            )
            temporal_cosine = float(
                hospital_temporal[candidate_component] @
                overall_temporal[reference_component] /
                (
                    np.linalg.norm(hospital_temporal[candidate_component]) *
                    np.linalg.norm(overall_temporal[reference_component])
                )
            )
            sign = 1.0 if (feature_cosine * temporal_cosine) >= 0 else -1.0
            aligned[:, reference_component] = (
                sign * raw_hospital[:, candidate_component] - scaler.mean_[reference_component]
            ) / scaler.scale_[reference_component]
        if not np.isfinite(aligned).all():
            raise ValueError(f"Could not align all components for {cohort}")
        aligned_hospitals[cohort] = aligned

    return overall_ids, scaler, overall_components, aligned_hospitals


def fit_fixed_k(
    values: np.ndarray,
    method: str,
    cluster_count: int,
    seed: int,
    spectral_neighbors: int,
) -> np.ndarray:
    if method == "kmeans":
        estimator = KMeans(
            n_clusters=cluster_count,
            n_init=20,
            random_state=seed,
        )
    elif method == "hierarchical":
        estimator = AgglomerativeClustering(
            n_clusters=cluster_count,
            linkage="ward",
        )
    elif method == "gmm":
        estimator = GaussianMixture(
            n_components=cluster_count,
            covariance_type="full",
            n_init=5,
            reg_covar=1e-6,
            random_state=seed,
        )
    elif method == "spectral":
        estimator = SpectralClustering(
            n_clusters=cluster_count,
            affinity="nearest_neighbors",
            n_neighbors=spectral_neighbors,
            assign_labels="kmeans",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown fixed-K method: {method}")
    return estimator.fit_predict(values) + 1


def fit_hdbscan(
    values: np.ndarray,
    min_cluster_size: int,
    min_samples: int | None,
) -> np.ndarray:
    estimator = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = estimator.fit_predict(values)
    return np.where(labels == -1, -1, labels + 1)


def cluster_sizes(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    metadata: dict[str, object],
) -> pd.DataFrame:
    counts = pd.Series(labels).value_counts().sort_index()
    total = len(labels)
    return pd.DataFrame(
        [
            {
                **metadata,
                "cluster": int(cluster),
                "patient_count": int(count),
                "patient_fraction": float(count / total),
            }
            for cluster, count in counts.items()
            if cluster != -1
        ]
        + [
            {
                **metadata,
                "cluster": -1,
                "patient_count": int(np.sum(labels == -1)),
                "patient_fraction": float(np.mean(labels == -1)),
            }
        ]
        if np.any(labels == -1)
        else [
            {
                **metadata,
                "cluster": int(cluster),
                "patient_count": int(count),
                "patient_fraction": float(count / total),
            }
            for cluster, count in counts.items()
        ]
    )


def profile_rows(
    values: np.ndarray,
    components: list[str],
    labels: np.ndarray,
    metadata: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster in sorted(set(labels.tolist())):
        if cluster == -1:
            continue
        mean = values[labels == cluster].mean(axis=0)
        for component, score in zip(components, mean):
            rows.append(
                {
                    **metadata,
                    "cluster": int(cluster),
                    "component": component,
                    "mean_standardized_score": float(score),
                }
            )
    return pd.DataFrame(rows)


def write_labels(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    metadata: dict[str, object],
    path: Path,
) -> None:
    frame = pd.DataFrame(
        [
            {
                "patientunitstayid": patient_id,
                "cluster": int(cluster),
                **metadata,
            }
            for patient_id, cluster in zip(patient_ids, labels)
        ]
    )
    frame.to_csv(path, index=False)


def plot_pca(
    values: np.ndarray,
    labels: np.ndarray,
    title: str,
    path: Path,
) -> None:
    pca = PCA(n_components=2, random_state=42)
    projection = pca.fit_transform(values)
    explained = pca.explained_variance_ratio_
    total_explained = explained.sum()

    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    unique = sorted(set(labels.tolist()))
    for cluster in unique:
        mask = labels == cluster
        label = "Noise" if cluster == -1 else f"Cluster {cluster}"
        color = cluster_color(cluster)
        axis.scatter(
            projection[mask, 0],
            projection[mask, 1],
            s=8,
            alpha=0.72,
            label=label,
            color=color,
        )
    axis.set(
        xlabel=f"PCA 1 ({explained[0]:.1%})",
        ylabel=f"PCA 2 ({explained[1]:.1%})",
        title=f"{title}\n(First 2 PCs explain {total_explained:.1%} of variance)",
    )
    axis.grid(alpha=0.2)
    axis.legend(markerscale=2, fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    plot_pca_density(
        projection,
        labels,
        explained,
        title,
        path.with_name("pca_density.png"),
    )


def plot_pca_density(
    projection: np.ndarray,
    labels: np.ndarray,
    explained: np.ndarray,
    title: str,
    path: Path,
) -> None:
    clusters = sorted(set(labels.tolist()))
    x_min, x_max = projection[:, 0].min(), projection[:, 0].max()
    y_min, y_max = projection[:, 1].min(), projection[:, 1].max()
    x_pad = max((x_max - x_min) * 0.02, 1e-6)
    y_pad = max((y_max - y_min) * 0.02, 1e-6)
    x_edges = np.linspace(x_min - x_pad, x_max + x_pad, 201)
    y_edges = np.linspace(y_min - y_pad, y_max + y_pad, 201)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    densities = {
        cluster: gaussian_filter(
            np.histogram2d(
                projection[labels == cluster, 0],
                projection[labels == cluster, 1],
                bins=(x_edges, y_edges),
            )[0],
            sigma=1.8,
            mode="constant",
        )
        for cluster in clusters
    }
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    draw_order = sorted(
        clusters,
        key=lambda cluster: int(np.sum(labels == cluster)),
        reverse=True,
    )
    for cluster in draw_order:
        density = densities[cluster]
        cluster_max = float(density.max())
        relative_density = density / cluster_max
        visible_density = relative_density.copy()
        visible_density[visible_density < 0.10] = 0.0
        alpha = 0.82 * np.power(visible_density, 0.8)
        color = cluster_color(cluster)
        rgba = np.empty((*density.T.shape, 4), dtype=float)
        rgba[:, :, :3] = to_rgba(color)[:3]
        rgba[:, :, 3] = alpha.T
        axis.imshow(
            rgba,
            extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
            origin="lower",
            interpolation="bilinear",
            aspect="auto",
        )
        axis.contour(
            x_centers,
            y_centers,
            relative_density.T,
            levels=[0.12],
            colors=[color],
            linewidths=1.4,
            alpha=0.95,
        )

    handles = [
        Patch(
            facecolor=cluster_color(cluster),
            label="Noise" if cluster == -1 else f"Cluster {cluster}",
        )
        for cluster in clusters
    ]
    axis.set(
        xlabel=f"PCA 1 ({explained[0]:.1%})",
        ylabel=f"PCA 2 ({explained[1]:.1%})",
        xlim=(x_edges[0], x_edges[-1]),
        ylim=(y_edges[0], y_edges[-1]),
        title=(
            f"{title}: PCA cluster density\n"
            f"(first 2 PCs explain {explained.sum():.1%}; opacity shows within-cluster density)"
        ),
    )
    axis.legend(handles=handles, loc="best", fontsize=8)
    axis.grid(alpha=0.15)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_sizes(sizes: pd.DataFrame, title: str, path: Path) -> None:
    plot = sizes[sizes["cluster"] != -1].copy()
    plot = plot.sort_values("cluster")
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.bar(
        [str(cluster) for cluster in plot["cluster"]],
        plot["patient_fraction"] * 100,
        color=[cluster_color(int(cluster)) for cluster in plot["cluster"]],
    )
    axis.set(ylabel="Patients (%)", xlabel="Cluster", title=title)
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.2)
    for index, row in enumerate(plot.itertuples()):
        axis.text(
            index,
            row.patient_fraction * 100,
            f'{row.patient_count:,}\n({row.patient_fraction:.1%})',
            ha="center",
            va="bottom",
            fontsize=8,
        )
    if (sizes["cluster"] == -1).any():
        noise_fraction = float(sizes.loc[sizes["cluster"].eq(-1), "patient_fraction"].iloc[0])
        axis.text(
            0.99,
            0.98,
            f"Noise: {noise_fraction:.1%}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8},
        )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_profile(profiles: pd.DataFrame, title: str, path: Path) -> None:
    pivot = profiles.pivot(
        index="cluster",
        columns="component",
        values="mean_standardized_score",
    )
    components = list(pivot.columns)
    figure, axis = plt.subplots(
        figsize=(max(8, len(components) * 0.45), max(4, len(pivot) * 0.65)),
        constrained_layout=True,
    )
    image = axis.imshow(
        pivot.to_numpy(float),
        aspect="auto",
        cmap="coolwarm",
        vmin=-2,
        vmax=2,
    )
    axis.set_xticks(range(len(components)))
    axis.set_xticklabels(components, rotation=90)
    axis.set_yticks(range(len(pivot.index)))
    axis.set_yticklabels([f"Cluster {cluster}" for cluster in pivot.index])
    axis.set(title=title)
    figure.colorbar(image, ax=axis, label="Mean standardized factor score")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def pairwise_stability(solutions: dict[int, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left_seed, right_seed in combinations(sorted(solutions), 2):
        left = solutions[left_seed]
        right = solutions[right_seed]
        rows.append(
            {
                "seed_a": left_seed,
                "seed_b": right_seed,
                "adjusted_rand_index": float(adjusted_rand_score(left, right)),
                "adjusted_mutual_info": float(adjusted_mutual_info_score(left, right)),
            }
        )
    return pd.DataFrame(rows)


def plot_stability(pairwise: pd.DataFrame, title: str, path: Path) -> None:
    if pairwise.empty:
        return
    seeds = sorted(set(pairwise["seed_a"]) | set(pairwise["seed_b"]))
    ari = pd.DataFrame(np.eye(len(seeds)), index=seeds, columns=seeds)
    ami = pd.DataFrame(np.eye(len(seeds)), index=seeds, columns=seeds)
    for _, row in pairwise.iterrows():
        ari.loc[row["seed_a"], row["seed_b"]] = ari.loc[row["seed_b"], row["seed_a"]] = row["adjusted_rand_index"]
        ami.loc[row["seed_a"], row["seed_b"]] = ami.loc[row["seed_b"], row["seed_a"]] = row["adjusted_mutual_info"]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, matrix, metric in ((axes[0], ari, "ARI"), (axes[1], ami, "AMI")):
        image = axis.imshow(matrix.to_numpy(float), vmin=0, vmax=1, cmap="viridis")
        axis.set_xticks(range(len(seeds)))
        axis.set_yticks(range(len(seeds)))
        axis.set_xticklabels(seeds)
        axis.set_yticklabels(seeds)
        axis.set_xlabel("Seed")
        axis.set_ylabel("Seed")
        axis.set_title(f"{metric}; min={matrix.values[np.triu_indices(len(seeds), 1)].min():.3f}")
        figure.colorbar(image, ax=axis)
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def nearest_centroid_labels(
    values: np.ndarray,
    centroids: np.ndarray,
    cluster_ids: list[int],
) -> np.ndarray:
    if len(centroids) == 0:
        return np.full(len(values), -1, dtype=int)
    distances = (
        (values[:, None, :] - centroids[None, :, :]) ** 2
    ).sum(axis=2)
    return np.asarray(cluster_ids, dtype=int)[distances.argmin(axis=1)]


def plot_hospital_cluster_proportion(
    proportion_table: pd.DataFrame,
    title: str,
    path: Path,
) -> None:
    cohorts = ["overall"] + [f"hospital_{index}" for index in range(1, 11)]
    pivot = (
        proportion_table.pivot(
            index="cohort",
            columns="cluster",
            values="patient_fraction",
        )
        .fillna(0.0)
        .reindex(cohorts)
    )
    clusters = sorted(int(cluster) for cluster in pivot.columns)
    pivot = pivot.reindex(columns=clusters)
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    bottom = np.zeros(len(pivot))
    for cluster in clusters:
        values = pivot[cluster].to_numpy(float)
        axis.bar(
            range(len(pivot)),
            values,
            bottom=bottom,
            color=cluster_color(cluster),
        )
        bottom += values
    axis.set_xticks(range(len(pivot)))
    axis.set_xticklabels(
        ["Overall"] + [f"Hospital {index}" for index in range(1, 11)],
        rotation=45,
        ha="right",
    )
    axis.set_ylabel("Patient fraction")
    axis.set_ylim(0, 1)
    axis.set_title(title)
    handles = [
        Patch(
            facecolor=cluster_color(cluster),
            label="Noise" if cluster == -1 else f"Cluster {cluster}",
        )
        for cluster in clusters
    ]
    ncols = max(1, min(3, len(handles)))
    nrows = (len(handles) + ncols - 1) // ncols
    legend_handles = [
        handles[row * ncols + column]
        for column in range(ncols)
        for row in range(nrows)
        if row * ncols + column < len(handles)
    ]
    axis.legend(handles=legend_handles, title="Cluster", ncol=ncols)
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_hospital_cosine_heatmap(
    proportion_table: pd.DataFrame,
    title: str,
    path: Path,
) -> None:
    cohorts = [f"hospital_{index}" for index in range(1, 11)]
    clusters = sorted(proportion_table["cluster"].unique())
    matrix = np.vstack([
        [
            float(
                proportion_table[
                    (proportion_table["cohort"] == cohort)
                    & (proportion_table["cluster"] == cluster)
                ]["patient_fraction"].iloc[0]
            )
            if not proportion_table[
                (proportion_table["cohort"] == cohort)
                & (proportion_table["cluster"] == cluster)
            ].empty
            else 0.0
            for cluster in clusters
        ]
        for cohort in cohorts
    ])
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(denominator, 1e-12)
    similarity = normalized @ normalized.T
    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    image = axis.imshow(similarity, vmin=0, vmax=1, cmap="viridis")
    labels = [f"Hospital {index}" for index in range(1, 11)]
    axis.set_xticks(range(10))
    axis.set_yticks(range(10))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="Cosine similarity")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def model_label(model: pd.Series) -> str:
    return (
        f"{str(model['method']).lower()}_{str(model['aggregation']).lower()}_"
        f"rank{int(model['rank'])}"
    )


def run_hospital_comparison(
    model: pd.Series,
    overall_values: np.ndarray,
    overall_labels: np.ndarray,
    aligned_hospitals: dict[str, np.ndarray],
    output: Path,
    title: str,
) -> pd.DataFrame:
    clusters = sorted(set(overall_labels.tolist()) - {-1})
    centroids = (
        np.vstack([
            overall_values[overall_labels == cluster].mean(axis=0)
            for cluster in clusters
        ])
        if clusters
        else np.empty((0, overall_values.shape[1]))
    )
    rows: list[dict[str, object]] = []

    cohort_labels: dict[str, np.ndarray] = {"overall": overall_labels}
    cohort_labels.update({
        cohort: nearest_centroid_labels(values, centroids, clusters)
        for cohort, values in aligned_hospitals.items()
    })

    for cohort, labels in cohort_labels.items():
        counts = pd.Series(labels).value_counts()
        for cluster, count in counts.items():
            if int(cluster) == -1:
                continue
            rows.append(
                {
                    "cohort": cohort,
                    "cluster": int(cluster),
                    "patient_count": int(count),
                    "patient_fraction": float(count / len(labels)),
                }
            )
        if np.any(labels == -1):
            rows.append(
                {
                    "cohort": cohort,
                    "cluster": -1,
                    "patient_count": int(np.sum(labels == -1)),
                    "patient_fraction": float(np.mean(labels == -1)),
                }
            )

    proportion_table = pd.DataFrame(rows)
    proportion_table.to_csv(output / "hospital_cluster_proportions.csv", index=False)
    plot_hospital_cluster_proportion(
        proportion_table,
        f"{title}: overall and hospital cluster proportions",
        output / "hospital_cluster_proportion.png",
    )
    plot_hospital_cosine_heatmap(
        proportion_table,
        f"{title}: hospital cluster-profile cosine similarity",
        output / "hospital_cosine_similarity_heatmap.png",
    )
    return proportion_table


def run_fixed_k_methods(
    patient_ids: np.ndarray,
    components: list[str],
    values: np.ndarray,
    model: pd.Series,
    model_output: Path,
    cluster_counts: list[int],
    seeds: list[int],
    spectral_neighbors: int,
    aligned_hospitals: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, object]] = []
    agreement_rows: list[dict[str, object]] = []
    method_labels: dict[str, dict[int, np.ndarray]] = {}

    for method in ("kmeans", "hierarchical", "gmm", "spectral"):
        for cluster_count in cluster_counts:
            output = model_output / method / f"k{cluster_count}"
            output.mkdir(parents=True, exist_ok=True)

            seed_solutions: dict[int, np.ndarray] = {}
            if method == "hierarchical":
                seed_solutions[seeds[0]] = fit_fixed_k(
                    values, method, cluster_count, seeds[0], spectral_neighbors
                )
            else:
                for seed in seeds:
                    seed_solutions[seed] = fit_fixed_k(
                        values, method, cluster_count, seed, spectral_neighbors
                    )

            labels = seed_solutions[seeds[0]]
            if method != "hierarchical":
                labels = seed_solutions[seeds[0]]
            method_labels.setdefault(method, {})[cluster_count] = labels

            metadata = {
                "method": method,
                "aggregation": str(model["aggregation"]),
                "model": str(model["method"]),
                "rank": int(model["rank"]),
                "cluster_count": cluster_count,
            }
            write_labels(
                patient_ids,
                labels,
                metadata,
                output / "cluster_labels.csv",
            )

            sizes = cluster_sizes(patient_ids, labels, metadata)
            sizes.to_csv(output / "cluster_sizes.csv", index=False)
            profiles = profile_rows(values, components, labels, metadata)
            profiles.to_csv(output / "cluster_profiles.csv", index=False)

            title = (
                f"{model_label(model)}: {method}, k={cluster_count}"
            )
            plot_pca(values, labels, title, output / "pca_embedding.png")
            plot_sizes(sizes, title, output / "cluster_sizes.png")
            plot_profile(profiles, title, output / "cluster_profile_heatmap.png")
            run_hospital_comparison(
                model,
                values,
                labels,
                aligned_hospitals,
                output,
                title,
            )

            if method == "gmm":
                gmm = GaussianMixture(
                    n_components=cluster_count,
                    covariance_type="full",
                    n_init=5,
                    reg_covar=1e-6,
                    random_state=seeds[0],
                ).fit(values)
                posterior = gmm.predict_proba(values)
                pd.DataFrame(
                    {
                        "patientunitstayid": patient_ids,
                        "maximum_posterior_probability": posterior.max(axis=1),
                        "uncertain_below_0_70": posterior.max(axis=1) < 0.70,
                        "uncertain_below_0_80": posterior.max(axis=1) < 0.80,
                        "uncertain_below_0_90": posterior.max(axis=1) < 0.90,
                    }
                ).to_csv(output / "posterior_uncertainty.csv", index=False)
                pd.DataFrame(
                    [{"bic": float(gmm.bic(values)), "aic": float(gmm.aic(values))}]
                ).to_csv(output / "bic_aic.csv", index=False)

            if method != "hierarchical":
                pairwise = pairwise_stability(seed_solutions)
                pairwise.to_csv(output / "seed_pairwise_metrics.csv", index=False)
                plot_stability(
                    pairwise,
                    f"{title}: seed stability",
                    output / "seed_stability_heatmap.png",
                )
                metrics_rows.append(
                    {
                        **metadata,
                        "seed_count": len(seed_solutions),
                        "pairwise_ari_min": (
                            float(pairwise["adjusted_rand_index"].min())
                            if not pairwise.empty
                            else np.nan
                        ),
                        "pairwise_ari_median": (
                            float(pairwise["adjusted_rand_index"].median())
                            if not pairwise.empty
                            else np.nan
                        ),
                        "pairwise_ami_min": (
                            float(pairwise["adjusted_mutual_info"].min())
                            if not pairwise.empty
                            else np.nan
                        ),
                        "pairwise_ami_median": (
                            float(pairwise["adjusted_mutual_info"].median())
                            if not pairwise.empty
                            else np.nan
                        ),
                    }
                )

    for (method_a, method_b) in combinations(method_labels, 2):
        for cluster_count in cluster_counts:
            left = method_labels[method_a].get(cluster_count)
            right = method_labels[method_b].get(cluster_count)
            if left is None or right is None:
                continue
            agreement_rows.append(
                {
                    "model": model_label(model),
                    "cluster_count": cluster_count,
                    "method_a": method_a,
                    "method_b": method_b,
                    "adjusted_rand_index": float(adjusted_rand_score(left, right)),
                    "adjusted_mutual_info": float(adjusted_mutual_info_score(left, right)),
                }
            )

    return pd.DataFrame(metrics_rows), pd.DataFrame(agreement_rows)


def run_hdbscan(
    patient_ids: np.ndarray,
    components: list[str],
    values: np.ndarray,
    model: pd.Series,
    model_output: Path,
    min_cluster_sizes: list[int],
    min_samples_values: list[int | None],
    aligned_hospitals: dict[str, np.ndarray],
) -> pd.DataFrame:
    parameter_rows: list[dict[str, object]] = []
    hdbscan_output = model_output / "hdbscan"
    hdbscan_output.mkdir(parents=True, exist_ok=True)

    for min_cluster_size in min_cluster_sizes:
        for min_samples in min_samples_values:
            parameter_name = f"mcs{min_cluster_size}_ms{min_samples if min_samples is not None else 'auto'}"
            output = hdbscan_output / parameter_name
            output.mkdir(parents=True, exist_ok=True)
            labels = fit_hdbscan(values, min_cluster_size, min_samples)
            effective_clusters = sorted(set(labels.tolist()) - {-1})
            metadata = {
                "method": "hdbscan",
                "aggregation": str(model["aggregation"]),
                "model": str(model["method"]),
                "rank": int(model["rank"]),
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples if min_samples is not None else "auto",
            }
            write_labels(patient_ids, labels, metadata, output / "cluster_labels.csv")
            sizes = cluster_sizes(patient_ids, labels, metadata)
            sizes.to_csv(output / "cluster_sizes.csv", index=False)
            profiles = profile_rows(values, components, labels, metadata)
            profiles.to_csv(output / "cluster_profiles.csv", index=False)

            title = (
                f"{model_label(model)}: HDBSCAN "
                f"mcs={min_cluster_size}, ms={min_samples if min_samples is not None else 'auto'}"
            )
            plot_pca(values, labels, title, output / "pca_embedding.png")
            plot_sizes(sizes, title, output / "cluster_sizes.png")
            if not profiles.empty:
                plot_profile(profiles, title, output / "cluster_profile_heatmap.png")
            run_hospital_comparison(
                model,
                values,
                labels,
                aligned_hospitals,
                output,
                title,
            )

            parameter_rows.append(
                {
                    **metadata,
                    "effective_cluster_count": len(effective_clusters),
                    "noise_patient_count": int(np.sum(labels == -1)),
                    "noise_fraction": float(np.mean(labels == -1)),
                    "smallest_cluster_fraction": (
                        float(sizes.loc[sizes["cluster"] != -1, "patient_fraction"].min())
                        if np.any(sizes["cluster"] != -1)
                        else 0.0
                    ),
                }
            )

    parameter_table = pd.DataFrame(parameter_rows)
    parameter_table.to_csv(hdbscan_output / "parameter_search.csv", index=False)
    return parameter_table


def parse_int_list(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def parse_optional_int_list(text: str) -> list[int | None]:
    result: list[int | None] = []
    for value in text.split(","):
        value = value.strip()
        if not value:
            continue
        result.append(None if value.lower() in {"none", "auto"} else int(value))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cluster-counts", default="2,3,4,5,6")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--spectral-neighbors", type=int, default=20)
    parser.add_argument(
        "--only-hdbscan",
        action="store_true",
        help="Run HDBSCAN without rerunning the fixed-k clustering methods.",
    )
    parser.add_argument(
        "--hdbscan-min-cluster-sizes",
        default="5,10",
    )
    parser.add_argument("--hdbscan-min-samples", default="5,10,20,50")
    args = parser.parse_args()

    cluster_counts = parse_int_list(args.cluster_counts)
    seeds = parse_int_list(args.seeds)
    hdbscan_sizes = parse_int_list(args.hdbscan_min_cluster_sizes)
    hdbscan_samples = parse_optional_int_list(args.hdbscan_min_samples)

    primary = read_primary_models(args.results_root)
    all_metrics: list[pd.DataFrame] = []
    all_agreement: list[pd.DataFrame] = []
    all_hdbscan: list[pd.DataFrame] = []

    for _, model in primary.iterrows():
        patient_ids, components, values = load_factors(args.results_root, model)
        model_output = args.output_root / model_label(model)
        model_output.mkdir(parents=True, exist_ok=True)
        _, _, _, aligned_hospitals = load_overall_scaler_and_aligned_hospital(
            args.results_root,
            model,
        )

        if not args.only_hdbscan:
            metrics, agreement = run_fixed_k_methods(
                patient_ids,
                components,
                values,
                model,
                model_output,
                cluster_counts,
                seeds,
                args.spectral_neighbors,
                aligned_hospitals,
            )
            all_metrics.append(metrics)
            all_agreement.append(agreement)

        hdbscan = run_hdbscan(
            patient_ids,
            components,
            values,
            model,
            model_output,
            hdbscan_sizes,
            hdbscan_samples,
            aligned_hospitals,
        )
        all_hdbscan.append(hdbscan)

    args.output_root.mkdir(parents=True, exist_ok=True)
    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(
            args.output_root / "fixed_k_seed_stability.csv",
            index=False,
            float_format="%.6f",
        )
    if all_agreement:
        pd.concat(all_agreement, ignore_index=True).to_csv(
            args.output_root / "cross_method_agreement.csv",
            index=False,
            float_format="%.6f",
        )
    pd.concat(all_hdbscan, ignore_index=True).to_csv(
        args.output_root / "hdbscan_parameter_search.csv",
        index=False,
        float_format="%.6f",
    )


if __name__ == "__main__":
    main()
