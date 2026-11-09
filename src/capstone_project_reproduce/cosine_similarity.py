"""Pairwise absolute-cosine heatmaps for the saved CP factors.

One experiment and one rank produce two heatmaps, one for feature loadings
and one for time loadings. Components are matched on the absolute product of
those two cosines, then each heatmap summarizes the matched pairs with a
geometric mean. The rows are overall and the hospitals in hospital_map.csv.

``--pooled-hospital`` reads the existing feature_11 CSVs and writes the
pooled hospital comparison under cosine_similarity_heatmaps/pooled_hospital/.
"""

import argparse
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
POOLED_PANEL = "feature_11"
POOLED_DIR = OUTPUT_ROOT / "pooled_hospital"
KINDS = ("feature", "time")


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
    """Write the annotated heatmap."""
    values = frame.to_numpy(float)
    if len(frame.index) >= 10 and len(frame.columns) >= 10:
        size = (12, 10)
    else:
        size = (max(6.0, 0.9 * len(frame.columns) + 2), max(4.5, 0.5 * len(frame.index) + 2))
    figure, axis = plt.subplots(figsize=size)
    image = axis.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
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


def save_table(frame: pd.DataFrame, directory: Path, stem: str, title: str) -> None:
    """Write a CSV and the matching heatmap."""
    path = directory / stem
    frame.to_csv(path.with_suffix(".csv"), float_format="%.5f")
    save_heatmap(frame, path.with_suffix(".png"), title)
    print(f"saved {path.with_suffix('.png')}")


def slice_paths(panel: str = POOLED_PANEL) -> list[tuple[Path, str, str, int, str]]:
    """Existing heatmap CSVs for one feature panel."""
    found = []
    for method in METHODS:
        for how in HOWS:
            for rank in RANKS:
                for kind in KINDS:
                    path = OUTPUT_ROOT / panel / f"{method}_{how}" / f"rank{rank}_{kind}.csv"
                    found.append((path, method, how, rank, kind))
    missing = [str(path) for path, *_ in found if not path.exists()]
    if missing:
        preview = "\n".join(missing[:8])
        extra = "" if len(missing) <= 8 else f"\n... and {len(missing) - 8} more"
        raise FileNotFoundError(f"Missing {len(missing)} {panel} heatmaps:\n{preview}{extra}")
    return found


def load_slices(panel: str = POOLED_PANEL) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    """Stack every heatmap and the overall-versus-hospital scores."""
    paths = slice_paths(panel)
    frames = [pd.read_csv(path, index_col=0) for path, *_ in paths]
    labels = list(frames[0].index)
    for frame in frames:
        if list(frame.index) != labels or list(frame.columns) != labels:
            raise ValueError("Heatmap labels do not match across files")
    stack = np.stack([frame.to_numpy(dtype=float) for frame in frames])
    hospitals = [name for name in labels if name != "overall"]
    rows = []
    for (_, method, how, rank, kind), frame in zip(paths, frames):
        for hospital in hospitals:
            rows.append(
                {
                    "method": method,
                    "how": how,
                    "rank": rank,
                    "kind": kind,
                    "hospital": hospital,
                    "vs_overall": float(frame.loc["overall", hospital]),
                }
            )
    return labels, stack, pd.DataFrame(rows)


def disagreement_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean similarity to overall and D(i) = 1 - that mean."""
    grouped = scores.groupby("hospital")["vs_overall"]
    table = pd.DataFrame(
        {
            "S_bar": grouped.mean(),
            "D": 1 - grouped.mean(),
            "n_slices": grouped.size(),
        }
    )
    table = table.sort_values("D", ascending=False)
    table.insert(0, "rank_by_D", range(1, len(table) + 1))
    table.index.name = "hospital"
    return table


def save_disagreement_bar(table: pd.DataFrame, path: Path) -> None:
    """Bar chart of D(i), largest disagreement first."""
    figure, axis = plt.subplots(figsize=(8, 4.5))
    names = list(table.index)
    axis.bar(range(len(names)), table["D"].to_numpy(dtype=float))
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=45, ha="right")
    axis.set_ylabel("D(i)")
    axis.set_xlabel("Hospital")
    axis.set_title("Hospital disagreement versus overall (feature_11)")
    axis.set_ylim(0, max(0.3, float(table["D"].max()) * 1.15))
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"saved {path}")


def save_disagreement_tex(table: pd.DataFrame, path: Path) -> None:
    """LaTeX table of D(i), largest disagreement first."""
    rows = []
    for hospital, row in table.iterrows():
        name = str(hospital).replace("_", r"\_")
        rows.append(
            f"{name} & {int(row['rank_by_D'])} & {row['S_bar']:.4f} & {row['D']:.4f} \\\\"
        )
    body = "\n".join(rows)
    path.write_text(
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{Hospital disagreement versus overall (feature\\_11). "
        "$\\bar S(i)$ is the pooled similarity to overall; $D(i)=1-\\bar S(i)$."
        "}\n"
        "\\begin{tabular}{lrrr}\n"
        "\\hline\n"
        "Hospital & Rank & $\\bar S(i)$ & $D(i)$ \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    print(f"saved {path}")


def factor_table(scores: pd.DataFrame, factor: str, hospitals: list[str]) -> pd.DataFrame:
    """Mean similarity to overall, rows in D(i) order."""
    table = scores.pivot_table("vs_overall", "hospital", factor, aggfunc="mean")
    return table.loc[hospitals]


def run_pooled_hospital(panel: str = POOLED_PANEL) -> None:
    """Average feature_11 heatmaps and write pooled_hospital/."""
    labels, stack, scores = load_slices(panel)
    pooled = pd.DataFrame(stack.mean(axis=0), index=labels, columns=labels)
    table = disagreement_table(scores)
    hospitals = list(table.index)
    POOLED_DIR.mkdir(parents=True, exist_ok=True)
    save_table(
        pooled,
        POOLED_DIR,
        "pooled_heatmap",
        f"{panel} pooled cosine heatmap (method, aggregation, rank, feature/time)",
    )
    table.to_csv(POOLED_DIR / "hospital_disagreement.csv", float_format="%.5f")
    print(f"saved {POOLED_DIR / 'hospital_disagreement.csv'}")
    save_disagreement_bar(table, POOLED_DIR / "disagreement_bar.png")
    save_disagreement_tex(table, POOLED_DIR / "hospital_disagreement.tex")
    for factor, stem, title in (
        ("rank", "hospital_by_rank", f"{panel} mean S(overall, hospital) by rank"),
        ("kind", "hospital_by_kind", f"{panel} mean S(overall, hospital) by feature/time"),
        ("method", "hospital_by_method", f"{panel} mean S(overall, hospital) by method"),
        ("how", "hospital_by_aggregation", f"{panel} mean S(overall, hospital) by aggregation"),
    ):
        save_table(factor_table(scores, factor, hospitals), POOLED_DIR, stem, title)
    print(table.round(4).to_string())
    print(f"wrote {len(stack)} slices to {POOLED_DIR}")


def write_pairwise_heatmaps() -> None:
    """Fit pairwise heatmaps for every panel, method, aggregation, and rank."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pooled-hospital",
        action="store_true",
        help="average existing feature_11 heatmaps; do not refit pairwise maps",
    )
    args = parser.parse_args()
    if args.pooled_hospital:
        run_pooled_hospital()
        return
    write_pairwise_heatmaps()


if __name__ == "__main__":
    main()
