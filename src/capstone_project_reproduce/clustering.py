"""Time-aware soft clustering of the imputed hourly tensor.

The three centroids are the curves in Figure 2 of Jiang et al.: liver, kidney,
and lung. Each feature is z-scored. A patient may belong to more than one
centroid. Single-organ diagnoses pull the matching centroid closer and push the
other two away. The plot smooths those centroids; the saved curves are not
smoothed.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from data_preprocessing import (
    DECIMAL_PLACES,
    FEATURES,
    N_HOURS,
    imputed_table_path,
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "results" / "clustering"
ORGANS = ("liver", "kidney", "lung")
EARLY_FEATURES = ("arterial_bp_systolic", "base_excess", "respiratory_rate")
EARLY_HOURS = 24
ICD9_RANGES = ((570, 573), (580, 589), (510, 519))
ICD10_RANGES = (("K", 72, 76), ("N", 0, 27), ("J", 86, 95))
EPOCHS = 200
ETA = 2.0
BETA_TARGET = 10.0
BETA_OTHER = 0.01
LEARNING_RATE = 1e-5
DISTANCE_EPS = 1e-8
LOWESS_FRAC = 0.25
LOWESS_IT = 20
BOOTSTRAP_DRAWS = 100
BOOTSTRAP_SIZE = 50
COLORS = {"liver": "red", "kidney": "gold", "lung": "green"}
MARKERS = {"liver": "o", "kidney": "v", "lung": "s"}


def organ_code(code: str) -> np.ndarray:
    """Whether one ICD code falls in the liver, kidney, or lung range."""
    flags = np.zeros(len(ORGANS), dtype=bool)
    text = code.strip().upper().replace(".", "")
    if not text:
        return flags
    if text[0].isdigit():
        digits = "".join(character for character in text if character.isdigit())[:3]
        if not digits:
            return flags
        number = int(digits)
        for index, (low, high) in enumerate(ICD9_RANGES):
            flags[index] = low <= number <= high
        return flags
    digits = "".join(character for character in text[1:] if character.isdigit())[:2]
    if len(digits) < 2:
        return flags
    number = int(digits)
    for index, (letter, low, high) in enumerate(ICD10_RANGES):
        flags[index] = text[0] == letter and low <= number <= high
    return flags


def organ_labels(icd_codes: np.ndarray) -> np.ndarray:
    """One row of liver, kidney, and lung indicators per ICU stay."""
    labels = np.zeros((len(icd_codes), len(ORGANS)), dtype=bool)
    for row, text in enumerate(icd_codes):
        for code in str(text).split(","):
            labels[row] |= organ_code(code)
    return labels


def load_imputed(path: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stay ids, the hourly tensor, and the ICD text for each stay."""
    if path is None:
        path = imputed_table_path("mean")
    frame = pd.read_csv(
        path,
        usecols=["patientunitstayid", "icd_code", "hour", *FEATURES],
    )
    ordered = frame.sort_values(["patientunitstayid", "hour"], kind="mergesort")
    ids = ordered["patientunitstayid"].drop_duplicates().to_numpy()
    counts = ordered.groupby("patientunitstayid", sort=False)["hour"].nunique()
    if not counts.eq(N_HOURS).all():
        raise ValueError("Every patient must have exactly one row in each hour")
    values = ordered[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Imputed table contains NaN or infinite values")
    array = values.reshape(len(ids), N_HOURS, len(FEATURES)).transpose(0, 2, 1)
    icd = ordered.drop_duplicates("patientunitstayid")["icd_code"].fillna("").to_numpy()
    return ids, array, icd


def standardize(array: np.ndarray) -> np.ndarray:
    """Z-score each feature over patients and hours, using the sample standard deviation."""
    mean = array.mean(axis=(0, 2), keepdims=True)
    std = array.std(axis=(0, 2), ddof=1, keepdims=True)
    if np.any(std <= 0):
        constant = [FEATURES[k] for k in np.where(std[0, :, 0] <= 0)[0]]
        raise ValueError(f"Constant features: {constant}")
    return (array - mean) / std


def distance_weight() -> np.ndarray:
    """Square-difference weights. Early features use the first 24 hours, scaled by five."""
    weight = np.ones((len(FEATURES), N_HOURS), dtype=float)
    scale = N_HOURS / EARLY_HOURS
    for name in EARLY_FEATURES:
        index = FEATURES.index(name)
        weight[index, :] = 0
        weight[index, :EARLY_HOURS] = scale
    return weight


def weighted_distance(data: np.ndarray, centroids: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Squared distance from each patient curve to each shared centroid."""
    difference = data[:, None, :, :] - centroids[None, :, :, :]
    return np.sum(weight * difference**2, axis=(2, 3))


def curve_distance(data: np.ndarray, curves: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Squared distance when every patient has their own comparison curve."""
    return np.sum(weight * (data - curves) ** 2, axis=(1, 2))


def membership_curves(centroids: np.ndarray, membership: np.ndarray) -> np.ndarray:
    """Average of the centroids assigned to each patient."""
    counts = membership.sum(axis=1, keepdims=True)
    flat = membership @ centroids.reshape(len(ORGANS), -1)
    return (flat / counts).reshape(len(membership), len(FEATURES), N_HOURS)


def assign_membership(
    data: np.ndarray,
    centroids: np.ndarray,
    weight: np.ndarray,
    previous: np.ndarray | None = None,
) -> np.ndarray:
    """Assign each patient to one or more centroids, nearest first."""
    distances = weighted_distance(data, centroids, weight)
    order = np.argsort(distances, axis=1)
    patients = np.arange(len(data))
    membership = np.zeros((len(data), len(ORGANS)), dtype=bool)
    first = order[:, 0]
    membership[patients, first] = True
    current = centroids[first]
    current_distance = curve_distance(data, current, weight)

    second = order[:, 1]
    two_curves = (centroids[first] + centroids[second]) / 2
    two_distance = curve_distance(data, two_curves, weight)
    accept_second = two_distance < current_distance
    reject_second = np.flatnonzero(~accept_second)
    if previous is not None and len(reject_second):
        old_distance = curve_distance(
            data[reject_second],
            membership_curves(centroids, previous[reject_second]),
            weight,
        )
        revert = ~(current_distance[reject_second] < old_distance)
        membership[reject_second[revert]] = previous[reject_second[revert]]

    accepted = np.flatnonzero(accept_second)
    membership[accepted, second[accepted]] = True
    current = two_curves[accepted]
    current_distance = two_distance[accepted]
    third = order[accepted, 2]
    three_curves = (centroids[order[accepted, 0]] + centroids[second[accepted]] + centroids[third]) / 3
    three_distance = curve_distance(data[accepted], three_curves, weight)
    accept_third = three_distance < current_distance
    reject_third = np.flatnonzero(~accept_third)
    if previous is not None and len(reject_third):
        rows = accepted[reject_third]
        old_distance = curve_distance(
            data[rows],
            membership_curves(centroids, previous[rows]),
            weight,
        )
        revert = ~(current_distance[reject_third] < old_distance)
        membership[rows[revert]] = previous[rows[revert]]
    chosen = accepted[accept_third]
    membership[chosen, third[accept_third]] = True
    return membership


def fuzzy_weights(distances: np.ndarray, eta: float) -> np.ndarray:
    """Fuzzy weights treated as constants during the centroid step."""
    safe = np.maximum(distances, DISTANCE_EPS)
    ratio = np.clip(safe[:, :, None] / safe[:, None, :], 0, 1e6)
    total = np.sum(ratio ** (2 / (eta - 1)), axis=2)
    return (1 / total) ** eta


def calibration_gradient(
    data: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Mean gradient and the total loss. Single-organ stays supply the supervised term."""
    distances = weighted_distance(data, centroids, weight)
    weights = fuzzy_weights(distances, ETA)
    residual = data[:, None, :, :] - centroids[None, :, :, :]
    gradient = -2 * np.sum(weights[:, :, None, None] * weight * residual, axis=0)

    single = labels.sum(axis=1) == 1
    supervised = 0.0
    if np.any(single):
        chosen = labels[single].argmax(axis=1)
        single_data = data[single]
        single_distance = weighted_distance(single_data, centroids, weight)
        rows = np.arange(len(chosen))
        target = single_distance[rows, chosen].sum()
        supervised = BETA_TARGET * target - BETA_OTHER * (single_distance.sum() - target)
        for cluster in range(len(ORGANS)):
            matched = single_data[chosen == cluster]
            if len(matched):
                gradient[cluster] += np.sum(
                    -2 * BETA_TARGET * weight * (matched - centroids[cluster]),
                    axis=0,
                )
            unmatched = single_data[chosen != cluster]
            if len(unmatched):
                gradient[cluster] += np.sum(
                    2 * BETA_OTHER * weight * (unmatched - centroids[cluster]),
                    axis=0,
                )
    gradient /= len(data)
    unsupervised = float(np.sum(weights * distances))
    return gradient, unsupervised + float(supervised)


def initial_centroids(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mean curve of patients who have exactly one of the three organ labels."""
    centroids = np.zeros((len(ORGANS), len(FEATURES), N_HOURS), dtype=float)
    single = labels.sum(axis=1) == 1
    for cluster, name in enumerate(ORGANS):
        members = single & labels[:, cluster]
        count = int(members.sum())
        print(f"single-organ {name}: {count:,}")
        if count == 0:
            raise ValueError(f"No patient has only the {name} label")
        centroids[cluster] = data[members].mean(axis=0)
    return centroids


def fit_centroids(
    data: np.ndarray,
    labels: np.ndarray,
    epochs: int = EPOCHS,
) -> np.ndarray:
    """Move centroids by the mean gradient. Stop when the loss increases."""
    weight = distance_weight()
    centroids = initial_centroids(data, labels)
    membership = None
    accepted = centroids
    accepted_loss = None
    for epoch in range(1, epochs + 1):
        membership = assign_membership(data, centroids, weight, membership)
        gradient, loss = calibration_gradient(data, centroids, labels, weight)
        if accepted_loss is not None and loss > accepted_loss:
            print(f"epoch {epoch}: loss={loss:.6e} increased, keep epoch {epoch - 1}")
            return accepted
        accepted = centroids.copy()
        accepted_loss = loss
        centroids = centroids - LEARNING_RATE * gradient
        if not np.isfinite(centroids).all():
            raise ValueError("Centroid update is not finite")
        print(f"epoch {epoch}: loss={loss:.6e}")
    _, final_loss = calibration_gradient(data, centroids, labels, weight)
    if accepted_loss is not None and final_loss > accepted_loss:
        print(f"final step: loss={final_loss:.6e} increased, keep epoch {epochs}")
        return accepted
    return centroids


def smooth_curve(hours: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """LOWESS smooth. Duplicate hours are averaged before smoothing."""
    table = (
        pd.DataFrame({"hour": hours, "value": values})
        .groupby("hour", as_index=False)["value"]
        .mean()
    )
    smoothed = lowess(
        table["value"].to_numpy(),
        table["hour"].to_numpy(),
        frac=LOWESS_FRAC,
        it=LOWESS_IT,
        return_sorted=True,
    )
    return np.interp(grid, smoothed[:, 0], smoothed[:, 1])


def bootstrap_smooth(values: np.ndarray, grid: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean LOWESS curve and the 2.5th to 97.5th percentile band."""
    hours = np.arange(len(values), dtype=float)
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(BOOTSTRAP_DRAWS):
        chosen = rng.choice(len(values), size=BOOTSTRAP_SIZE, replace=True)
        curves.append(smooth_curve(hours[chosen], values[chosen], grid))
    stack = np.vstack(curves)
    return (
        np.nanmean(stack, axis=0),
        np.nanpercentile(stack, 2.5, axis=0),
        np.nanpercentile(stack, 97.5, axis=0),
    )


def plot_centroids(centroids: np.ndarray, path: Path) -> None:
    """Seven panels, one per feature, with smoothed liver, kidney, and lung curves."""
    grid = np.linspace(0, N_HOURS - 1, 50)
    figure, axes = plt.subplots(4, 2, figsize=(12, 16), sharex=True)
    for index, feature in enumerate(FEATURES):
        axis = axes.ravel()[index]
        for cluster, name in enumerate(ORGANS):
            mean, low, high = bootstrap_smooth(centroids[cluster, index], grid, seed=index)
            axis.fill_between(grid, low, high, color=COLORS[name], alpha=0.25)
            axis.plot(
                grid,
                mean,
                color=COLORS[name],
                marker=MARKERS[name],
                markevery=4,
                label=name,
            )
        axis.set_xlim(-1, N_HOURS + 1)
        axis.set_title(feature)
        axis.set_ylabel("Normalized values")
        axis.grid(alpha=0.3)
    axes.ravel()[-1].axis("off")
    for axis in axes[-1, :]:
        axis.set_xlabel("Hours")
    axes.ravel()[0].legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_centroids(centroids: np.ndarray, path: Path) -> None:
    """Unsmoothed centroid value for every cluster, feature, and hour."""
    rows = []
    for cluster, name in enumerate(ORGANS):
        for feature_index, feature in enumerate(FEATURES):
            for hour in range(N_HOURS):
                rows.append(
                    {
                        "cluster": name,
                        "feature": feature,
                        "hour": hour,
                        "value": centroids[cluster, feature_index, hour],
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False, float_format=f"%.{DECIMAL_PLACES}f")


def save_labels(ids: np.ndarray, labels: np.ndarray, path: Path) -> None:
    """Organ indicators used to initialize and calibrate the centroids."""
    frame = pd.DataFrame(labels.astype(int), columns=list(ORGANS))
    frame.insert(0, "patientunitstayid", ids)
    frame.to_csv(path, index=False)


def run(path: Path | None = None, results_dir: Path = RESULTS_DIR, epochs: int = EPOCHS) -> np.ndarray:
    """Fit the three centroids and write the figure, curves, and organ labels."""
    results_dir.mkdir(parents=True, exist_ok=True)
    ids, array, icd = load_imputed(path)
    labels = organ_labels(icd)
    print(f"{len(ids):,} patients, {len(FEATURES)} features, {N_HOURS} hours")
    data = standardize(array)
    centroids = fit_centroids(data, labels, epochs=epochs)
    save_centroids(centroids, results_dir / "centroids.csv")
    save_labels(ids, labels, results_dir / "icd_labels.csv")
    plot_centroids(centroids, results_dir / "figure2.png")
    print(f"Results saved to {results_dir}")
    return centroids


def main() -> None:
    run()


if __name__ == "__main__":
    main()
