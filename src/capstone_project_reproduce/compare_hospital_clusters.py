"""Cluster NNCPD patient factors for overall and hospitals, then align profiles."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_DIR / "results"


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().eq("true")


def abs_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    return np.abs(left @ right.T)


def load_model(results_root: Path, row: pd.Series) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    directory = results_root / row["result_directory"]
    rank = int(row["rank"])
    patient = pd.read_csv(directory / f"rank{rank}_patient.csv")
    variable = pd.read_csv(directory / f"rank{rank}_variable.csv", index_col=0)
    time = pd.read_csv(directory / f"rank{rank}_time.csv", index_col=0)
    components = [column for column in patient.columns if column.startswith("comp")]
    feature = np.vstack([variable[column].to_numpy(float) for column in components])
    temporal = np.vstack([time[column].to_numpy(float) for column in components])
    return patient, feature, temporal, components


def align_components(
    reference_feature: np.ndarray,
    reference_time: np.ndarray,
    candidate_feature: np.ndarray,
    candidate_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    feature_similarity = abs_cosine(candidate_feature, reference_feature)
    time_similarity = abs_cosine(candidate_time, reference_time)
    product = feature_similarity * time_similarity
    candidate_index, reference_index = linear_sum_assignment(product, maximize=True)
    return candidate_index, reference_index


def fit_kmeans(values: np.ndarray, cluster_count: int, seed: int) -> np.ndarray:
    return KMeans(n_clusters=cluster_count, n_init=20, random_state=seed).fit_predict(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--selection", type=Path, default=None)
    parser.add_argument("--method", default="nncpd")
    parser.add_argument("--aggregation", default="last")
    parser.add_argument("--feature-set", default="feature_11")
    parser.add_argument("--cluster-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    selection_path = args.selection or args.results_root / "model_selection" / "model_selection.csv"
    selection = pd.read_csv(selection_path)
    filtered = selection[
        (selection["feature_set"] == args.feature_set)
        & (selection["method"] == args.method)
        & (selection["aggregation"] == args.aggregation)
        & as_bool(selection["selected"])
    ].copy()
    if filtered.empty:
        raise ValueError("No selected models found for the requested setting")

    overall_rows = filtered[filtered["cohort"] == "overall"]
    if len(overall_rows) != 1:
        raise ValueError("Expected exactly one selected overall model")
    overall_row = overall_rows.iloc[0]

    overall_patient, overall_feature, overall_time, overall_components = load_model(args.results_root, overall_row)
    overall_values = StandardScaler().fit_transform(overall_patient[overall_components].to_numpy(float))
    overall_labels = fit_kmeans(overall_values, args.cluster_count, args.seed)
    overall_profiles = np.vstack([
        overall_values[overall_labels == cluster].mean(axis=0)
        for cluster in range(args.cluster_count)
    ])

    overall_output = args.output_root or args.results_root / "patient_clustering" / "hospital_comparison"
    overall_output.mkdir(parents=True, exist_ok=True)

    label_frames = []
    profile_rows = []
    size_rows = []
    alignment_rows = []
    profile_matrices = {"overall": overall_profiles}

    def save_one(cohort: str, row: pd.Series) -> np.ndarray:
        patient, feature, temporal, components = load_model(args.results_root, row)
        values = StandardScaler().fit_transform(patient[components].to_numpy(float))
        labels = fit_kmeans(values, args.cluster_count, args.seed)

        candidate_index, reference_index = align_components(
            overall_feature,
            overall_time,
            feature,
            temporal,
        )

        hospital_to_overall = np.full(len(overall_components), -1, dtype=int)
        for candidate_component, reference_component in zip(candidate_index, reference_index):
            hospital_to_overall[reference_component] = candidate_component

        aligned_profile = np.full((args.cluster_count, len(overall_components)), np.nan)
        for cluster in range(args.cluster_count):
            cluster_mean = values[labels == cluster].mean(axis=0)
            for reference_component in range(len(overall_components)):
                candidate_component = hospital_to_overall[reference_component]
                if candidate_component >= 0:
                    aligned_profile[cluster, reference_component] = cluster_mean[candidate_component]
                    profile_rows.append(
                        {
                            "cohort": cohort,
                            "cluster": cluster,
                            "component": overall_components[reference_component],
                            "mean_standardized_score": float(cluster_mean[candidate_component]),
                        }
                    )

        counts = pd.Series(labels).value_counts().sort_index()
        for cluster, count in counts.items():
            size_rows.append(
                {
                    "cohort": cohort,
                    "cluster": int(cluster),
                    "patient_count": int(count),
                    "patient_fraction": float(count / len(labels)),
                }
            )

        labels_frame = pd.DataFrame(
            {
                "patientunitstayid": patient.iloc[:, 0],
                "cohort": cohort,
                "cluster_count": args.cluster_count,
                "cluster_label": labels,
            }
        )
        label_frames.append(labels_frame)

        reference_profile = overall_profiles
        cosine_matrix = np.full((len(reference_profile), args.cluster_count), np.nan)
        for overall_cluster, overall_profile in enumerate(reference_profile):
            for hospital_cluster in range(args.cluster_count):
                left = overall_profile
                right = aligned_profile[hospital_cluster]
                valid = np.isfinite(left) & np.isfinite(right)
                denominator = np.linalg.norm(left[valid]) * np.linalg.norm(right[valid])
                cosine_matrix[overall_cluster, hospital_cluster] = (
                    float(left[valid] @ right[valid] / denominator) if denominator else 0.0
                )

                alignment_rows.append(
                    {
                        "cohort": cohort,
                        "overall_cluster": overall_cluster,
                        "hospital_cluster": hospital_cluster,
                        "profile_cosine_similarity": cosine_matrix[overall_cluster, hospital_cluster],
                        "selected_alignment": False,
                    }
                )

        overall_index, hospital_index = linear_sum_assignment(cosine_matrix, maximize=True)
        for overall_cluster, hospital_cluster in zip(overall_index, hospital_index):
            alignment_rows.append(
                {
                    "cohort": cohort,
                    "overall_cluster": int(overall_cluster),
                    "hospital_cluster": int(hospital_cluster),
                    "profile_cosine_similarity": float(cosine_matrix[overall_cluster, hospital_cluster]),
                    "selected_alignment": True,
                }
            )

        profile_matrices[cohort] = aligned_profile

        patient_projection = PCA(n_components=2, random_state=args.seed).fit_transform(values)
        return patient_projection

    overall_projection = PCA(n_components=2, random_state=args.seed).fit_transform(overall_values)
    label_frames.append(pd.DataFrame(
        {
            "patientunitstayid": overall_patient.iloc[:, 0],
            "cohort": "overall",
            "cluster_count": args.cluster_count,
            "cluster_label": overall_labels,
        }
    ))
    for cluster in range(args.cluster_count):
        count = int(np.sum(overall_labels == cluster))
        size_rows.append(
            {
                "cohort": "overall",
                "cluster": cluster,
                "patient_count": count,
                "patient_fraction": float(count / len(overall_labels)),
            }
        )
        for component_index, component in enumerate(overall_components):
            profile_rows.append(
                {
                    "cohort": "overall",
                    "cluster": cluster,
                    "component": component,
                    "mean_standardized_score": float(overall_profiles[cluster, component_index]),
                }
            )

    for cohort in [f"hospital_{index}" for index in range(1, 11)]:
        rows = filtered[filtered["cohort"] == cohort]
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one selected row for {cohort}, found {len(rows)}")
        save_one(cohort, rows.iloc[0])

    pd.concat(label_frames, ignore_index=True).to_csv(overall_output / "hospital_cluster_labels.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(overall_output / "hospital_cluster_profiles.csv", index=False, float_format="%.6f")
    pd.DataFrame(size_rows).to_csv(overall_output / "hospital_cluster_sizes.csv", index=False, float_format="%.6f")
    pd.DataFrame(alignment_rows).to_csv(overall_output / "overall_vs_hospital_profile_alignment.csv", index=False, float_format="%.6f")

    ordered_profiles = ["overall"] + [f"hospital_{index}" for index in range(1, 11)]
    profile_matrix = np.vstack([profile_matrices[name] for name in ordered_profiles])
    profile_matrix = np.nan_to_num(profile_matrix, nan=0.0)
    cluster_names = [
        ("Overall" if cohort == "overall" else cohort.removeprefix("hospital_"))
        + f"-C{cluster+1}"
        for cohort in ordered_profiles
        for cluster in range(args.cluster_count)
    ]

    similarity = abs_cosine(profile_matrix, profile_matrix)
    figure, axis = plt.subplots(figsize=(15, 12))
    image = axis.imshow(similarity, vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(cluster_names)))
    axis.set_yticks(range(len(cluster_names)))
    axis.set_xticklabels(cluster_names, rotation=90, fontsize=7)
    axis.xaxis.set_ticks_position("top")
    axis.xaxis.set_label_position("top")
    axis.set_yticklabels(cluster_names, fontsize=7)
    axis.set_title(f"Aligned {args.method.upper()} k={args.cluster_count} cluster profile cosine")
    figure.colorbar(image, ax=axis, label="Absolute cosine similarity")
    figure.tight_layout()
    figure.savefig(overall_output / "hospital_profile_cosine_heatmap.png", dpi=180)
    plt.close(figure)

    sizes = pd.DataFrame(size_rows)
    pivot = sizes.pivot(index="cohort", columns="cluster", values="patient_fraction").reindex(ordered_profiles)
    figure, axis = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", stacked=True, ax=axis, width=0.8)
    axis.set_ylabel("Patient fraction")
    axis.set_xlabel("Cohort")
    axis.set_ylim(0, 1)
    axis.legend([f"C{int(column)+1}" for column in pivot.columns], title="Cluster", ncol=4)
    axis.set_title(f"Aligned {args.method.upper()} k={args.cluster_count} cluster proportions")
    figure.tight_layout()
    figure.savefig(overall_output / "hospital_cluster_proportion.png", dpi=180)
    plt.close(figure)

    print(f"Saved hospital comparison outputs to {overall_output}")


if __name__ == "__main__":
    main()
