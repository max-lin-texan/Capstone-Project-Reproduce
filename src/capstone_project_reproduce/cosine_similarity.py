"""Pairwise absolute-cosine heatmaps for the saved CP factors.

One experiment and one rank produce two 15 by 15 heatmaps. The feature
heatmap compares feature loadings. The time heatmap compares time loadings.
Components are matched on the absolute product of those two cosines, then
each heatmap summarizes the matched pairs with a geometric mean. The cohorts
are overall and hospital_1 through hospital_14.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cpd import RESULTS_ROOT, cpd_results_dir
from hospitals import MAP_PATH
from nncpd import nncpd_results_dir

HOWS = ("mean", "median", "last")
METHODS = ("cpd", "nncpd")
RANKS = range(1, 11)
OUTPUT_ROOT = RESULTS_ROOT / "cosine_similarity_heatmaps"


def cohort_labels() -> list[tuple[str, str | None]]:
    """Display name and the hospital argument, with overall first."""
    names = pd.read_csv(MAP_PATH)["name"].astype(str).tolist()
    return [("overall", None), *[(name, name) for name in names]]


def results_dir(method: str, how: str, new: bool, hospital: str | None) -> Path:
    """Directory that holds this experiment's factor tables."""
    if method == "cpd":
        return cpd_results_dir(how, new=new, hospital=hospital)
    if method == "nncpd":
        return nncpd_results_dir(how, new=new, hospital=hospital)
    raise ValueError("method must be 'cpd' or 'nncpd'")


def loadings(directory: Path, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Feature loadings and time loadings, one component per row."""
    variables = pd.read_csv(directory / f"rank{rank}_variable.csv", index_col=0)
    hours = pd.read_csv(directory / f"rank{rank}_time.csv", index_col=0)
    components = [name for name in variables.columns if str(name).startswith("comp")]
    feature = np.vstack([variables[name].to_numpy(float) for name in components])
    time = np.vstack([hours[name].to_numpy(float) for name in components])
    return feature, time


def abs_cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Absolute cosine between every pair of rows."""
    left_norm = left / np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = right / np.linalg.norm(right, axis=1, keepdims=True)
    return np.abs(left_norm @ right_norm.T)


def geometric_mean(scores: list[float]) -> float:
    """Geometric mean. A zero score makes the result zero."""
    values = np.asarray(scores, dtype=float)
    if np.any(values <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(values))))


def paired_scores(
    left_feature: np.ndarray,
    left_time: np.ndarray,
    right_feature: np.ndarray,
    right_time: np.ndarray,
) -> tuple[float, float]:
    """Geometric means of matched feature cosines and matched time cosines.

    Matching uses the absolute product of the two cosines, so both heatmaps
    refer to the same component pairs.
    """
    feature_similarity = abs_cosine_matrix(left_feature, right_feature)
    time_similarity = abs_cosine_matrix(left_time, right_time)
    product = feature_similarity * time_similarity
    used_left: set[int] = set()
    used_right: set[int] = set()
    feature_scores = []
    time_scores = []
    for _ in range(product.shape[0]):
        masked = product.copy()
        for index in used_left:
            masked[index, :] = -1
        for index in used_right:
            masked[:, index] = -1
        left_index, right_index = np.unravel_index(np.argmax(masked), masked.shape)
        used_left.add(int(left_index))
        used_right.add(int(right_index))
        feature_scores.append(float(feature_similarity[left_index, right_index]))
        time_scores.append(float(time_similarity[left_index, right_index]))
    return geometric_mean(feature_scores), geometric_mean(time_scores)


def similarity_matrices(
    method: str, how: str, new: bool, rank: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature and time matrices for one experiment and one rank."""
    labels = cohort_labels()
    loaded = [loadings(results_dir(method, how, new, hospital), rank) for _, hospital in labels]
    names = [label for label, _ in labels]
    size = len(loaded)
    feature_matrix = np.eye(size)
    time_matrix = np.eye(size)
    for left in range(size):
        for right in range(left + 1, size):
            feature_score, time_score = paired_scores(*loaded[left], *loaded[right])
            feature_matrix[left, right] = feature_matrix[right, left] = feature_score
            time_matrix[left, right] = time_matrix[right, left] = time_score
    return (
        pd.DataFrame(feature_matrix, index=names, columns=names),
        pd.DataFrame(time_matrix, index=names, columns=names),
    )


def save_heatmap(frame: pd.DataFrame, path: Path, title: str) -> None:
    """Write the annotated 15 by 15 heatmap."""
    values = frame.to_numpy(float)
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(values, vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(frame.columns)))
    axis.set_xticklabels(list(frame.columns), rotation=45, ha="right")
    axis.set_yticks(range(len(frame.index)))
    axis.set_yticklabels(list(frame.index))
    axis.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "white" if values[row, column] < 0.55 else "black"
            axis.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", color=color, fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    for new in (False, True):
        panel = "feature_11" if new else "feature_7"
        for method in METHODS:
            for how in HOWS:
                directory = OUTPUT_ROOT / panel / f"{method}_{how}"
                directory.mkdir(parents=True, exist_ok=True)
                for rank in RANKS:
                    feature, time = similarity_matrices(method, how, new, rank)
                    for kind, frame in (("feature", feature), ("time", time)):
                        stem = directory / f"rank{rank}_{kind}"
                        frame.to_csv(stem.with_suffix(".csv"), float_format="%.5f")
                        title = f"{panel} {method} {how} rank {rank} {kind}"
                        save_heatmap(frame, stem.with_suffix(".png"), title)
                        print(f"saved {stem.with_suffix('.png')}")
                    for suffix in (".png", ".csv"):
                        combined = directory / f"rank{rank}{suffix}"
                        if combined.exists():
                            combined.unlink()


if __name__ == "__main__":
    main()
