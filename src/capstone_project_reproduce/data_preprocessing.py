"""Hourly grid and low-rank imputation for the seven sepsis features.

The measurement table is minute-level and irregular. Imputation follows the
alternating scheme in the paper: bin to hours, then alternate low-rank matrix
completion with a per-patient integer time shift. A positive shift writes hour
t into hour t + shift. The saved table has one row per patient and admission
hour, with every feature cell filled.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MEASUREMENT_PATH = DATA_DIR / "sepsis_measurement.csv"
IMPUTED_PATH = DATA_DIR / "sepsis_measurement_imputed.csv"
N_HOURS = 120
FEATURES = [
    "arterial_bp_systolic",
    "heart_rate",
    "respiratory_rate",
    "base_excess",
    "creatinine",
    "pt_inr",
    "lactate",
]
COLUMNS = ["patientunitstayid", "hour", *FEATURES]
DECIMAL_PLACES = 5


def imputed_table_path(how: str) -> Path:
    """Path of the completed hourly table for this aggregation."""
    if how == "mean":
        return IMPUTED_PATH
    if how not in {"median", "last"}:
        raise ValueError("how must be 'mean', 'median', or 'last'")
    return IMPUTED_PATH.with_name(f"sepsis_measurement_imputed_{how}.csv")


def load_measurements(path: Path = MEASUREMENT_PATH) -> pd.DataFrame:
    """Read the long measurement table written by file_read.py."""
    return pd.read_csv(path)


def aggregate_last(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per patient and hour, using the latest reading of each feature.

    Latest means the largest ``offset`` among non-missing values of that
    feature. Features recorded at different minutes stay in the same hour.
    """
    keys = ["patientunitstayid", "hour"]
    hourly = None
    for feature in FEATURES:
        observed = frame.loc[frame[feature].notna(), [*keys, "offset", feature]]
        if observed.empty:
            continue
        observed = observed.sort_values([*keys, "offset"], kind="mergesort")
        latest = observed.groupby(keys, sort=False, as_index=False).tail(1)
        latest = latest[keys + [feature]]
        hourly = latest if hourly is None else hourly.merge(latest, on=keys, how="outer")
    if hourly is None:
        return pd.DataFrame(columns=[*keys, *FEATURES])
    for feature in FEATURES:
        if feature not in hourly.columns:
            hourly[feature] = pd.NA
    return hourly


def aggregate_hourly(measurements: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """One row per patient and hour.

    Multiple readings of the same patient, feature, and hour are combined with
    ``how``: ``"mean"``, ``"median"``, or ``"last"``. Hours are numbered 0
    through 119. A reading at exactly 120 hours is placed in hour 119.
    """
    if how not in {"mean", "median", "last"}:
        raise ValueError("how must be 'mean', 'median', or 'last'")
    frame = measurements.loc[:, ["patientunitstayid", "offset", *FEATURES]].copy()
    hour = (frame["offset"] // 60).astype("int64")
    keep = (hour >= 0) & (hour <= N_HOURS)
    frame = frame.loc[keep].copy()
    frame["hour"] = hour.loc[keep].clip(upper=N_HOURS - 1).to_numpy()
    if how == "last":
        hourly = aggregate_last(frame)
    else:
        grouped = frame.drop(columns=["offset"]).groupby(
            ["patientunitstayid", "hour"], sort=False, as_index=False
        )
        hourly = getattr(grouped[FEATURES], how)()
    return hourly.sort_values(["patientunitstayid", "hour"], kind="mergesort").reset_index(drop=True)


def hourly_to_tensor(hourly: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pack the hourly table into a tensor of shape (patients, features, hours).

    Missing patient-hours stay NaN. Axis 0 follows ``patientunitstayid`` in
    ascending order, and axis 1 follows ``FEATURES``.
    """
    stays = hourly["patientunitstayid"].drop_duplicates().sort_values().to_numpy()
    index = pd.Index(stays)
    tensor = np.full((len(stays), len(FEATURES), N_HOURS), np.nan, dtype=np.float64)
    rows = index.get_indexer(hourly["patientunitstayid"])
    hours = hourly["hour"].to_numpy()
    for feature_index, feature in enumerate(FEATURES):
        values = pd.to_numeric(hourly[feature], errors="coerce").to_numpy(dtype=np.float64)
        observed = np.isfinite(values)
        tensor[rows[observed], feature_index, hours[observed]] = values[observed]
    return stays, tensor


def standardize_features(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score each feature using its own observed values.

    Squared error would otherwise be dominated by blood pressure and heart rate.
    The returned mean and standard deviation put imputed values back into the
    original units.
    """
    with np.errstate(all="ignore"):
        means = np.nanmean(tensor, axis=(0, 2))
        stds = np.nanstd(tensor, axis=(0, 2))
    stds = np.where(np.isfinite(stds) & (stds > 0), stds, 1.0)
    means = np.where(np.isfinite(means), means, 0.0)
    standardized = (tensor - means[None, :, None]) / stds[None, :, None]
    return standardized, means, stds


def invert_standardize(tensor: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Map standardized feature values back to the original units."""
    return tensor * stds[None, :, None] + means[None, :, None]


def tensor_to_matrix(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reshape (patients, features, hours) to (patients, features * hours).

    Column ``feature * N_HOURS + hour`` is that feature at that hour. The mask
    is True where the matrix is observed.
    """
    n_patients, n_features, n_hours = tensor.shape
    matrix = tensor.reshape(n_patients, n_features * n_hours)
    return matrix, np.isfinite(matrix)


def matrix_to_tensor(matrix: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Inverse of ``tensor_to_matrix``."""
    return matrix.reshape(shape)


def shift_tensor(tensor: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Shift each patient by an integer number of hours.

    A positive shift writes the value at hour t into hour t + shift. Entries
    that leave the recorded window become missing, as do the hours that open
    up at the other end.
    """
    n_patients, n_features, n_hours = tensor.shape
    shifted = np.full((n_patients, n_features, n_hours), np.nan, dtype=np.float64)
    shifts = np.asarray(shifts)
    for shift in np.unique(shifts):
        rows = np.flatnonzero(shifts == shift)
        step = int(shift)
        if step >= n_hours or step <= -n_hours:
            continue
        if step >= 0:
            shifted[rows, :, step:] = tensor[rows, :, : n_hours - step]
        else:
            shifted[rows, :, : n_hours + step] = tensor[rows, :, -step:]
    return shifted


def initialize_factors(
    matrix: np.ndarray,
    mask: np.ndarray,
    rank: int,
    seed: int,
    n_power: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Starting U and V from a few subspace iterations on zero-filled data."""
    filled = np.where(mask, matrix, 0.0)
    n_patients, width = filled.shape
    rank = min(rank, n_patients, width)
    generator = np.random.default_rng(seed)
    feature_factors = generator.standard_normal((width, rank))
    patient_factors = np.zeros((n_patients, rank))
    for _ in range(n_power):
        patient_factors, _ = np.linalg.qr(filled @ feature_factors)
        feature_factors, _ = np.linalg.qr(filled.T @ patient_factors)
    small = patient_factors.T @ filled @ feature_factors
    left, singular, right = np.linalg.svd(small, full_matrices=False)
    patient_factors = patient_factors @ left * singular
    feature_factors = feature_factors @ right.T
    return patient_factors, feature_factors


def update_patient_factors(
    matrix: np.ndarray,
    mask: np.ndarray,
    feature_factors: np.ndarray,
    ridge: float,
    batch_size: int = 512,
) -> np.ndarray:
    """Least squares for U with V held fixed. Missing entries are ignored."""
    observed = np.where(mask, matrix, 0.0)
    n_patients, _ = matrix.shape
    rank = feature_factors.shape[1]
    patient_factors = np.zeros((n_patients, rank))
    diagonal = np.arange(rank)
    for start in range(0, n_patients, batch_size):
        stop = min(start + batch_size, n_patients)
        weighted = feature_factors[None, :, :] * mask[start:stop, :, None]
        gram = np.einsum("bjr,bjs->brs", weighted, weighted)
        gram[:, diagonal, diagonal] += ridge
        rhs = np.einsum("bjr,bj->br", weighted, observed[start:stop])
        patient_factors[start:stop] = np.linalg.solve(gram, rhs[..., None])[..., 0]
    return patient_factors


def update_feature_factors(
    matrix: np.ndarray,
    mask: np.ndarray,
    patient_factors: np.ndarray,
    ridge: float,
    batch_size: int = 64,
) -> np.ndarray:
    """Least squares for V with U held fixed. Missing entries are ignored."""
    observed = np.where(mask, matrix, 0.0)
    _, width = matrix.shape
    rank = patient_factors.shape[1]
    feature_factors = np.zeros((width, rank))
    diagonal = np.arange(rank)
    for start in range(0, width, batch_size):
        stop = min(start + batch_size, width)
        weighted = patient_factors[:, None, :] * mask[:, start:stop, None]
        gram = np.einsum("nbr,nbs->brs", weighted, weighted)
        gram[:, diagonal, diagonal] += ridge
        rhs = np.einsum("nbr,nb->br", weighted, observed[:, start:stop])
        feature_factors[start:stop] = np.linalg.solve(gram, rhs[..., None])[..., 0]
    return feature_factors


def fit_low_rank(
    matrix: np.ndarray,
    mask: np.ndarray,
    rank: int,
    ridge: float,
    n_sweeps: int,
    patient_factors: np.ndarray | None = None,
    feature_factors: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Alternate updates of U and V. Missing entries do not enter the loss."""
    if patient_factors is None or feature_factors is None:
        patient_factors, feature_factors = initialize_factors(matrix, mask, rank, seed)
    for _ in range(n_sweeps):
        patient_factors = update_patient_factors(matrix, mask, feature_factors, ridge)
        feature_factors = update_feature_factors(matrix, mask, patient_factors, ridge)
    return patient_factors, feature_factors


def reconstruction_tensor(
    patient_factors: np.ndarray,
    feature_factors: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    """UV^T reshaped to (patients, features, hours)."""
    return matrix_to_tensor(patient_factors @ feature_factors.T, shape)


def observed_loss(tensor: np.ndarray, reconstruction: np.ndarray) -> float:
    """Squared error on observed entries only."""
    mask = np.isfinite(tensor)
    residual = tensor[mask] - reconstruction[mask]
    return float(np.dot(residual, residual))


def _sse_by_patient(tensor: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    """Per-patient squared error. Patients with no observed entries get +inf."""
    mask = np.isfinite(tensor)
    residual = np.where(mask, tensor - reconstruction, 0.0)
    sse = np.sum(residual * residual, axis=(1, 2))
    return np.where(mask.any(axis=(1, 2)), sse, np.inf)


def search_shifts(
    tensor: np.ndarray,
    reconstruction: np.ndarray,
    max_shift: int,
) -> np.ndarray:
    """Integer shift per patient that best matches the current reconstruction.

    The search covers ``-max_shift`` through ``max_shift``. A shift that moves
    all of a patient's observations out of the window is rejected. Ties stay at
    the current best, which starts at 0.
    """
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")
    n_patients = tensor.shape[0]
    best = np.zeros(n_patients, dtype=np.int64)
    best_sse = _sse_by_patient(tensor, reconstruction)
    for step in range(-max_shift, max_shift + 1):
        if step == 0:
            continue
        shifted = shift_tensor(tensor, np.full(n_patients, step, dtype=np.int64))
        sse = _sse_by_patient(shifted, reconstruction)
        better = sse < best_sse
        best[better] = step
        best_sse[better] = sse[better]
    return best


def impute_missing(
    tensor: np.ndarray,
    rank: int = 10,
    max_shift: int = 12,
    ridge: float = 1e-3,
    max_outer: int = 20,
    n_sweeps: int = 5,
    tol: float = 1e-4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Alternate low-rank completion and per-patient time shifts until convergence.

    Returns the imputed tensor in the original units on the aligned clock, the
    shift of each patient, and the feature means and standard deviations.
    """
    if rank < 1:
        raise ValueError("rank must be at least 1")
    if max_outer < 1:
        raise ValueError("max_outer must be at least 1")
    standardized, means, stds = standardize_features(tensor)
    shape = standardized.shape
    shifts = np.zeros(shape[0], dtype=np.int64)
    patient_factors = None
    feature_factors = None
    previous_loss = None
    reconstruction = None
    used = shifts
    for outer in range(1, max_outer + 1):
        used = shifts.copy()
        shifted = shift_tensor(standardized, used)
        matrix, mask = tensor_to_matrix(shifted)
        patient_factors, feature_factors = fit_low_rank(
            matrix,
            mask,
            rank,
            ridge,
            n_sweeps,
            patient_factors,
            feature_factors,
            seed,
        )
        reconstruction = reconstruction_tensor(patient_factors, feature_factors, shape)
        loss = observed_loss(shifted, reconstruction)
        proposed = search_shifts(standardized, reconstruction, max_shift)
        if previous_loss is None:
            relative = np.inf
        else:
            relative = abs(previous_loss - loss) / max(previous_loss, 1e-12)
        print(
            f"outer {outer}: loss={loss:.6e} relative_change={relative:.3e} "
            f"patients_with_shift={int(np.count_nonzero(proposed))}"
        )
        unchanged = np.array_equal(proposed, used)
        previous_loss = loss
        if unchanged and relative < tol:
            shifts = used
            break
        shifts = proposed
    else:
        if not np.array_equal(shifts, used):
            shifted = shift_tensor(standardized, shifts)
            matrix, mask = tensor_to_matrix(shifted)
            patient_factors, feature_factors = fit_low_rank(
                matrix,
                mask,
                rank,
                ridge,
                n_sweeps,
                patient_factors,
                feature_factors,
                seed,
            )
            reconstruction = reconstruction_tensor(patient_factors, feature_factors, shape)
            loss = observed_loss(shifted, reconstruction)
            print(f"final fit: loss={loss:.6e}")
    imputed = invert_standardize(reconstruction, means, stds)
    if not np.isfinite(imputed).all():
        raise ValueError("Imputation left non-finite values on the aligned clock")
    return imputed, shifts, means, stds


def to_original_clock(imputed: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Map the aligned imputation back onto each patient's admission hour.

    Imputation is fit after a positive shift has moved hour t to hour t + shift.
    This reverses that move. Hours shifted outside the recorded window are left
    empty here and filled by ``fill_outside_window``.
    """
    _, _, n_hours = imputed.shape
    original = np.full_like(imputed, np.nan)
    for shift in np.unique(shifts):
        rows = np.flatnonzero(shifts == shift)
        step = int(shift)
        if step >= n_hours or step <= -n_hours:
            continue
        if step >= 0:
            original[rows, :, : n_hours - step] = imputed[rows, :, step:]
        else:
            original[rows, :, -step:] = imputed[rows, :, : n_hours + step]
    return original


def fill_outside_window(array: np.ndarray) -> np.ndarray:
    """Fill hours that the time shift moved outside the recorded window.

    An empty cell copies the nearest filled hour of the same patient and
    feature. The low-rank model has no value there, because that hour was
    shifted off the 0–119 grid.
    """
    filled = array.copy()
    for hour in range(1, filled.shape[2]):
        gap = ~np.isfinite(filled[:, :, hour])
        filled[:, :, hour] = np.where(gap, filled[:, :, hour - 1], filled[:, :, hour])
    for hour in range(filled.shape[2] - 2, -1, -1):
        gap = ~np.isfinite(filled[:, :, hour])
        filled[:, :, hour] = np.where(gap, filled[:, :, hour + 1], filled[:, :, hour])
    if not np.isfinite(filled).all():
        raise ValueError("Some patient-feature series could not be filled")
    return filled


def fill_hourly_table(
    stays: np.ndarray,
    hourly: np.ndarray,
    imputed: np.ndarray,
    shifts: np.ndarray,
) -> pd.DataFrame:
    """One row per patient and hour, with every feature cell filled.

    A cell that already has an hourly aggregate is kept. An empty cell takes the
    imputed value on the admission clock. Hours shifted off that clock copy the
    nearest filled hour. Every patient has hours 0 through 119.
    """
    on_original = to_original_clock(imputed, shifts)
    filled = hourly.copy()
    missing = ~np.isfinite(hourly)
    filled[missing] = on_original[missing]
    filled = fill_outside_window(filled)
    n_patients, _, n_hours = filled.shape
    frame = pd.DataFrame(
        {
            "patientunitstayid": np.repeat(stays, n_hours),
            "hour": np.tile(np.arange(n_hours), n_patients),
        }
    )
    for feature_index, feature in enumerate(FEATURES):
        frame[feature] = filled[:, feature_index, :].reshape(-1)
    return frame[COLUMNS]


def round_features(filled: pd.DataFrame, places: int = DECIMAL_PLACES) -> pd.DataFrame:
    """Round the seven feature columns. Identifiers and hour stay integers."""
    rounded = filled.copy()
    rounded[FEATURES] = rounded[FEATURES].round(places)
    return rounded


def save_imputed_measurements(
    filled: pd.DataFrame,
    path: Path = IMPUTED_PATH,
) -> Path:
    """Write the filled measurement table with five digits after the decimal point."""
    path.parent.mkdir(parents=True, exist_ok=True)
    round_features(filled).to_csv(path, index=False, float_format=f"%.{DECIMAL_PLACES}f")
    return path


def preprocess(
    measurement_path: Path = MEASUREMENT_PATH,
    output_path: Path = IMPUTED_PATH,
    how: str = "mean",
    rank: int = 10,
    max_shift: int = 12,
    ridge: float = 1e-3,
    max_outer: int = 20,
    n_sweeps: int = 5,
    tol: float = 1e-4,
    seed: int = 0,
) -> pd.DataFrame:
    """Aggregate to hours, impute, and save one row per patient-hour."""
    measurements = load_measurements(measurement_path)
    hourly_table = aggregate_hourly(measurements, how=how)
    stays, hourly = hourly_to_tensor(hourly_table)
    observed = np.isfinite(hourly).mean()
    print(
        f"hourly tensor: {hourly.shape[0]:,} patients, {hourly.shape[1]} features, "
        f"{hourly.shape[2]} hours, observed {observed:.1%}, aggregate={how}"
    )
    imputed, shifts, _, _ = impute_missing(
        hourly,
        rank=rank,
        max_shift=max_shift,
        ridge=ridge,
        max_outer=max_outer,
        n_sweeps=n_sweeps,
        tol=tol,
        seed=seed,
    )
    filled = fill_hourly_table(stays, hourly, imputed, shifts)
    save_imputed_measurements(filled, output_path)
    print(f"saved {len(filled):,} rows to {output_path}")
    return filled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--how", choices=["mean", "median", "last"], default="mean")
    args = parser.parse_args()
    path = imputed_table_path(args.how)
    if path.exists():
        print(f"reusing {path}")
        return
    preprocess(output_path=path, how=args.how)


if __name__ == "__main__":
    main()
