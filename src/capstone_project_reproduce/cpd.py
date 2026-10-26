"""CP decomposition of the imputed hourly tensor (patient x feature x hour).

Each feature is z-scored across patients and hours. Ranks 1-5 are fitted with
several random starts. Factors, a fit summary, and a fit-versus-rank plot are
saved under results/cpd_mean/, results/cpd_median/, or results/cpd_last/.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyttb as ttb

from data_preprocessing import (
    COLUMNS,
    DECIMAL_PLACES,
    FEATURES,
    IMPUTED_PATH,
    N_HOURS,
    imputed_table_path,
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "results" / "cpd"
RANKS = range(1, 6)
SEEDS = range(5)
HOURS = list(range(N_HOURS))


def cpd_results_dir(how: str) -> Path:
    """Result directory for this aggregation."""
    if how not in {"mean", "median", "last"}:
        raise ValueError("how must be 'mean', 'median', or 'last'")
    return RESULTS_DIR.parent / f"cpd_{how}"


def prepare_imputed(how: str = "mean") -> Path:
    """Return the completed hourly table written by data_preprocessing.py."""
    path = imputed_table_path(how)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Run: python data_preprocessing.py --how {how}"
        )
    return path


def load_hourly(path: Path = IMPUTED_PATH) -> pd.DataFrame:
    """Read the imputed hourly table."""
    return pd.read_csv(path, usecols=COLUMNS)


def validate_hourly(df: pd.DataFrame) -> None:
    """Fail if any patient is missing an hour or a feature value."""
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    hours = pd.to_numeric(df["hour"], errors="coerce")
    if hours.isna().any() or not hours.isin(HOURS).all():
        raise ValueError("hour must be an integer from 0 through 119")
    if df.duplicated(["patientunitstayid", "hour"]).any():
        raise ValueError("Duplicate patient-hour rows")
    counts = df.groupby("patientunitstayid")["hour"].nunique()
    if not counts.eq(N_HOURS).all():
        raise ValueError("Every patient must have exactly one row in each hour")
    values = df[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Tensor data contain NaN or infinite values")


def arrange_tensor(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return patient ids and an array of shape (patient, feature, hour)."""
    validate_hourly(df)
    ordered = df.sort_values(["patientunitstayid", "hour"], kind="mergesort")
    ids = ordered["patientunitstayid"].drop_duplicates().to_numpy()
    values = ordered[FEATURES].to_numpy(dtype=float)
    array = values.reshape(len(ids), N_HOURS, len(FEATURES)).transpose(0, 2, 1)
    return ids, array


def build_tensor(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Z-score each feature over all patients and hours."""
    ids, array = arrange_tensor(df)
    mean = array.mean(axis=(0, 2), keepdims=True)
    std = array.std(axis=(0, 2), keepdims=True)
    if np.any(std <= 0):
        constant = [FEATURES[k] for k in np.where(std[0, :, 0] <= 0)[0]]
        raise ValueError(f"Constant features: {constant}")
    return ids, (array - mean) / std


def temporal_diagnostics(array: np.ndarray) -> dict[str, float]:
    """Compare the tensor with a baseline that ignores time."""
    baseline = array.mean(axis=2, keepdims=True)
    total_energy = float(np.sum(array**2))
    temporal_sse = float(np.sum((array - baseline) ** 2))
    ratio = temporal_sse / total_energy if total_energy > 0 else np.nan
    return {"temporal_variation_ratio": ratio, "static_baseline_sse": temporal_sse}


def reconstruction_diagnostics(array: np.ndarray, model: ttb.ktensor) -> dict[str, float]:
    """Squared error of the CP model against the constant-time baseline."""
    predicted = np.asarray(model.full().data)
    baseline = array.mean(axis=2, keepdims=True)
    sse = float(np.sum((array - predicted) ** 2))
    sse_static = float(np.sum((array - baseline) ** 2))
    relative = 1 - sse / sse_static if sse_static > 0 else np.nan
    return {
        "sse": sse,
        "rmse": float(np.sqrt(sse / array.size)),
        "static_relative_score": relative,
        "predicted_above_one_pct": 100 * float(np.mean(predicted > 1)),
    }


def fit_cp(tensor: ttb.tensor, rank: int) -> tuple[tuple[ttb.ktensor, float, int], list[float]]:
    """Fit CP-ALS from several random starts and keep the best fit."""
    best, fits = None, []
    for seed in SEEDS:
        np.random.seed(seed)
        model, _, info = ttb.cp_als(tensor, rank, printitn=0)
        fit = float(info["fit"])
        fits.append(fit)
        if best is None or fit > best[1]:
            best = (model, fit, seed)
    return best, fits


def save_factors(
    model: ttb.ktensor,
    rank: int,
    ids: np.ndarray,
    results_dir: Path,
) -> None:
    """Save patient scores, feature loadings, time loadings, and weights."""
    columns = [f"comp{i + 1}" for i in range(rank)]
    patient, feature, time = model.factor_matrices
    number_format = f"%.{DECIMAL_PLACES}f"

    patient_frame = pd.DataFrame(patient, columns=columns)
    patient_frame.insert(0, "patientunitstayid", ids)
    patient_frame.to_csv(
        results_dir / f"rank{rank}_patient.csv",
        index=False,
        float_format=number_format,
    )
    pd.DataFrame(feature, index=FEATURES, columns=columns).rename_axis("feature").to_csv(
        results_dir / f"rank{rank}_variable.csv",
        float_format=number_format,
    )
    pd.DataFrame(time, index=HOURS, columns=columns).rename_axis("hour").to_csv(
        results_dir / f"rank{rank}_time.csv",
        float_format=number_format,
    )
    pd.DataFrame({"component": columns, "weight": model.weights}).to_csv(
        results_dir / f"rank{rank}_weights.csv",
        index=False,
        float_format=number_format,
    )


def plot_fit(summary: pd.DataFrame, results_dir: Path, title: str | None = None) -> None:
    """Plot the best fit against rank."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(summary["rank"], summary["best_fit"], marker="o")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Fit")
    ax.set_xticks(list(RANKS))
    if title is None:
        title = f"CP-ALS fit vs rank (best of {len(SEEDS)} starts)"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "fit_vs_rank.png", dpi=150)
    plt.close(fig)


def run_cpd(
    how: str = "mean",
    path: Path | None = None,
    results_dir: Path | None = None,
) -> pd.DataFrame:
    """Fit ranks 1-5 and save factors, the fit summary, and the plot."""
    if path is None:
        path = prepare_imputed(how)
    if results_dir is None:
        results_dir = cpd_results_dir(how)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"aggregation={how}")
    frame = load_hourly(path)
    ids, array = build_tensor(frame)
    tensor = ttb.tensor(array)
    temporal = temporal_diagnostics(array)
    print(
        f"{array.shape[0]:,} patients, {array.shape[1]} features, {array.shape[2]} hours"
    )
    print(f"Temporal variation ratio: {temporal['temporal_variation_ratio']:.4%}")

    rows = []
    for rank in RANKS:
        (model, fit, seed), fits = fit_cp(tensor, rank)
        save_factors(model, rank, ids, results_dir)
        diagnostics = reconstruction_diagnostics(array, model)
        rows.append(
            {
                "rank": rank,
                "best_fit": fit,
                "best_seed": seed,
                "fit_min": min(fits),
                "fit_max": max(fits),
                "fit_std": float(np.std(fits)),
                **temporal,
                **diagnostics,
            }
        )
        print(
            f"rank {rank}: best fit {fit:.4f} (seed {seed}), "
            f"range {min(fits):.4f}-{max(fits):.4f}"
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / "fit_summary.csv", index=False, float_format=f"%.{DECIMAL_PLACES}f")
    plot_fit(summary, results_dir)
    print(f"Results saved to {results_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--how", choices=["mean", "median", "last"], default="mean")
    args = parser.parse_args()
    run_cpd(how=args.how)


if __name__ == "__main__":
    main()
