"""Measure seed stability of patient-factor K-means solutions."""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_DIR / "results"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().eq("true")


def read_primary_model(summary_path: Path, model_method: str | None) -> pd.Series:
    summary = pd.read_csv(summary_path)
    required = {
        "second_level_selected",
        "cohort",
        "feature_set",
        "method",
        "aggregation",
        "rank",
        "result_directory",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Missing columns in {summary_path}: {sorted(missing)}")

    selected = summary[as_bool(summary["second_level_selected"])].copy()
    selected = selected[selected["cohort"].eq("overall")]
    if model_method:
        selected = selected[selected["method"].str.lower().eq(model_method.lower())]
    if selected.empty:
        raise ValueError("No matching primary overall model found")
    if len(selected) != 1:
        raise ValueError(
            f"Expected one primary model, found {len(selected)}: "
            f"{selected[['feature_set', 'method', 'aggregation', 'rank']].to_dict('records')}"
        )
    return selected.iloc[0]


def load_factors(results_root: Path, model: pd.Series) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    path = results_root / model["result_directory"] / f"rank{int(model['rank'])}_patient.csv"
    frame = pd.read_csv(path)
    components = [column for column in frame.columns if column.startswith("comp")]
    if not components:
        raise ValueError(f"No component columns found in {path}")
    if frame.iloc[:, 0].duplicated().any():
        raise ValueError(f"Duplicate patient IDs found in {path}")
    if frame[components].isna().any().any():
        raise ValueError(f"Missing patient factor values found in {path}")
    values = StandardScaler().fit_transform(frame[components].to_numpy(float))
    return frame, components, values


def fit_seed_solutions(
    values: np.ndarray,
    cluster_count: int,
    seeds: tuple[int, ...],
) -> dict[int, np.ndarray]:
    solutions: dict[int, np.ndarray] = {}
    for seed in seeds:
        estimator = KMeans(
            n_clusters=cluster_count,
            n_init=20,
            random_state=seed,
        )
        solutions[seed] = estimator.fit_predict(values)
    return solutions


def build_pairwise_metrics(solutions: dict[int, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left_seed, right_seed in combinations(solutions, 2):
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


def align_profile_to_reference(
    values: np.ndarray,
    components: list[str],
    reference_labels: np.ndarray,
    candidate_labels: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    reference_profiles: dict[int, np.ndarray] = {}
    candidate_profiles: dict[int, np.ndarray] = {}

    for cluster in np.unique(reference_labels):
        reference_profiles[int(cluster)] = values[reference_labels == cluster].mean(axis=0)
    for cluster in np.unique(candidate_labels):
        candidate_profiles[int(cluster)] = values[candidate_labels == cluster].mean(axis=0)

    aligned: list[dict[str, object]] = []
    mapping: dict[int, int] = {}
    used_reference: set[int] = set()

    similarity_rows: list[dict[str, object]] = []
    for candidate_cluster, candidate_profile in candidate_profiles.items():
        candidate_norm = np.linalg.norm(candidate_profile)
        for reference_cluster, reference_profile in reference_profiles.items():
            denominator = candidate_norm * np.linalg.norm(reference_profile)
            similarity = float(candidate_profile @ reference_profile / denominator) if denominator else 0.0
            similarity_rows.append(
                {
                    "candidate_cluster": candidate_cluster,
                    "reference_cluster": reference_cluster,
                    "profile_cosine_similarity": similarity,
                }
            )

    similarity_frame = pd.DataFrame(similarity_rows)
    for candidate_cluster in candidate_profiles:
        available = similarity_frame[
            similarity_frame["candidate_cluster"].eq(candidate_cluster)
            & ~similarity_frame["reference_cluster"].isin(list(used_reference))
        ]
        if available.empty:
            continue
        match = available.sort_values("profile_cosine_similarity", ascending=False).iloc[0]
        reference_cluster = int(match["reference_cluster"])
        mapping[candidate_cluster] = reference_cluster
        used_reference.add(reference_cluster)
        aligned.append(
            {
                "candidate_cluster": candidate_cluster,
                "reference_cluster": reference_cluster,
                "profile_cosine_similarity": float(match["profile_cosine_similarity"]),
            }
        )

    mapped_labels = np.array([mapping.get(int(label), -1) for label in candidate_labels], dtype=int)
    return mapped_labels, pd.DataFrame(aligned)


def build_consensus(
    frame: pd.DataFrame,
    values: np.ndarray,
    components: list[str],
    solutions: dict[int, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_seed = min(solutions)
    reference_labels = solutions[reference_seed]

    aligned_labels: dict[int, np.ndarray] = {reference_seed: reference_labels}
    alignment_rows: list[dict[str, object]] = []
    for seed in sorted(solutions):
        if seed == reference_seed:
            continue
        mapped, alignment = align_profile_to_reference(
            values,
            components,
            reference_labels,
            solutions[seed],
        )
        aligned_labels[seed] = mapped
        if not alignment.empty:
            alignment["seed"] = seed
            alignment_rows.append(alignment)

    consensus_rows: list[dict[str, object]] = []
    patient_labels = np.vstack([aligned_labels[seed] for seed in sorted(aligned_labels)])
    for row_index, patient_id in enumerate(frame.iloc[:, 0]):
        counts = pd.Series(patient_labels[:, row_index]).value_counts()
        consensus_cluster = int(counts.index[0])
        consensus_fraction = float(counts.iloc[0] / len(aligned_labels))
        consensus_rows.append(
            {
                "patientunitstayid": patient_id,
                "consensus_cluster": consensus_cluster,
                "consensus_fraction": consensus_fraction,
            }
        )

    alignment_frame = pd.concat(alignment_rows, ignore_index=True) if alignment_rows else pd.DataFrame()
    return pd.DataFrame(consensus_rows), alignment_frame


def plot_pairwise_metrics(pairwise: pd.DataFrame, path: Path, title: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    seeds = sorted(set(pairwise["seed_a"]) | set(pairwise["seed_b"]))
    size = len(seeds)
    ari = pd.DataFrame(np.eye(size), index=seeds, columns=seeds)
    ami = pd.DataFrame(np.eye(size), index=seeds, columns=seeds)

    for _, row in pairwise.iterrows():
        ari.loc[row["seed_a"], row["seed_b"]] = ari.loc[row["seed_b"], row["seed_a"]] = row["adjusted_rand_index"]
        ami.loc[row["seed_a"], row["seed_b"]] = ami.loc[row["seed_b"], row["seed_a"]] = row["adjusted_mutual_info"]

    for axis, matrix, metric in ((axes[0], ari, "ARI"), (axes[1], ami, "AMI")):
        image = axis.imshow(matrix.to_numpy(float), vmin=0, vmax=1, cmap="viridis")
        axis.set_xticks(range(size))
        axis.set_yticks(range(size))
        axis.set_xticklabels(seeds)
        axis.set_yticklabels(seeds)
        axis.set_xlabel("Seed")
        axis.set_ylabel("Seed")
        axis.set_title(f"{metric} stability")
        for row in range(size):
            for column in range(size):
                axis.text(column, row, f"{matrix.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
    figure.suptitle(title)
    figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def parse_seeds(text: str) -> tuple[int, ...]:
    try:
        seeds = tuple(sorted({int(value.strip()) for value in text.split(",") if value.strip()}))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"seeds must contain integers: {error}")
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--model-method", default="nncpd")
    parser.add_argument("--cluster-counts", default="2,4")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    args = parser.parse_args()

    try:
        cluster_counts = tuple(sorted({int(value.strip()) for value in args.cluster_counts.split(",") if value.strip()}))
    except ValueError as error:
        parser.error(f"--cluster-counts must contain integers: {error}")
    seeds = parse_seeds(args.seeds)

    summary_path = args.summary or args.results_root / "model_selection" / "model_selection_summary.csv"
    output_root = args.output_root or args.results_root / "patient_clustering_stability"
    model = read_primary_model(summary_path, args.model_method)
    frame, components, values = load_factors(args.results_root, model)

    model_name = f"{model['method']}_{model['aggregation']}_rank{int(model['rank'])}"
    model_output = output_root / model_name
    model_output.mkdir(parents=True, exist_ok=True)

    all_seed_labels: list[pd.DataFrame] = []
    for cluster_count in cluster_counts:
        solutions = fit_seed_solutions(values, cluster_count, seeds)
        pairwise = build_pairwise_metrics(solutions)
        pairwise["cluster_count"] = cluster_count
        pairwise.to_csv(model_output / f"k{cluster_count}_seed_pairwise_metrics.csv", index=False, float_format="%.6f")

        consensus, alignment = build_consensus(frame, values, components, solutions)
        seed_label_frame = frame[[frame.columns[0]]].copy()
        for seed, labels in solutions.items():
            seed_label_frame[f"seed_{seed}"] = labels
        seed_label_frame.to_csv(model_output / f"k{cluster_count}_all_seed_labels.csv", index=False)
        consensus.to_csv(model_output / f"k{cluster_count}_consensus_labels.csv", index=False)
        if not alignment.empty:
            alignment["cluster_count"] = cluster_count
            alignment.to_csv(model_output / f"k{cluster_count}_seed_profile_alignment.csv", index=False, float_format="%.6f")

        plot_pairwise_metrics(
            pairwise,
            model_output / f"k{cluster_count}_seed_stability.png",
            f"{model_name}: K-means k={cluster_count} seed stability",
        )

        all_seed_labels.append(seed_label_frame.assign(cluster_count=cluster_count))

    print(f"Saved stability results to {model_output}")


if __name__ == "__main__":
    main()
