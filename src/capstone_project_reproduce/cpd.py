"""CP decomposition of the imputed hourly tensor (patient x feature x hour).

Each feature is z-scored across patients and hours. Ranks 1-10 are fitted with
several random starts. Factors, a fit summary, and a fit-versus-rank plot are
saved under results/feature_7/overall/cpd_mean/, cpd_median/, or cpd_last/.
``--new`` reads the eleven-feature table and writes the same names under
results/feature_11/. ``--hospital hospital_1`` uses one mapped hospital and
writes under that name instead of overall.
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
    DECIMAL_PLACES,
    FEATURES,
    IMPUTED_PATH,
    N_HOURS,
    NEW_FEATURES,
    imputed_table_path,
)
from hospitals import cohort_name, subset_stays

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_DIR / "results"
RANKS = range(1, 11)
SEEDS = range(5)
HOURS = list(range(N_HOURS))


def _features(features: list[str] | None) -> list[str]:
    return list(FEATURES if features is None else features)


def cpd_results_dir(how: str, new: bool = False, hospital: str | None = None) -> Path:
    """Result directory for this aggregation and cohort."""
    if how not in {"mean", "median", "last"}:
        raise ValueError("how must be 'mean', 'median', or 'last'")
    panel = "feature_11" if new else "feature_7"
    return RESULTS_ROOT / panel / cohort_name(hospital) / f"cpd_{how}"


def decomposition_complete(results_dir: Path) -> bool:
    """True when ranks 1-10 were already saved in this directory."""
    return (results_dir / "fit_summary.csv").exists() and (results_dir / "rank10_patient.csv").exists()


def prepare_imputed(how: str = "mean", new: bool = False) -> Path:
    """Return the completed hourly table written by data_preprocessing.py."""
    path = imputed_table_path(how, new=new)
    if not path.exists():
        flag = " --new" if new else ""
        raise FileNotFoundError(
            f"Missing {path.name}. Run: python data_preprocessing.py --how {how}{flag}"
        )
    return path


def load_hourly(path: Path = IMPUTED_PATH, features: list[str] | None = None) -> pd.DataFrame:
    """Read the imputed hourly table."""
    features = _features(features)
    return pd.read_csv(path, usecols=["patientunitstayid", "hour", *features])


def validate_hourly(df: pd.DataFrame, features: list[str] | None = None) -> None:
    """Fail if any patient is missing an hour or a feature value."""
    features = _features(features)
    columns = ["patientunitstayid", "hour", *features]
    missing = set(columns) - set(df.columns)
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
    values = df[features].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Tensor data contain NaN or infinite values")


def arrange_tensor(
    df: pd.DataFrame,
    features: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return patient ids and an array of shape (patient, feature, hour)."""
    features = _features(features)
    validate_hourly(df, features)
    ordered = df.sort_values(["patientunitstayid", "hour"], kind="mergesort")
    ids = ordered["patientunitstayid"].drop_duplicates().to_numpy()
    values = ordered[features].to_numpy(dtype=float)
    array = values.reshape(len(ids), N_HOURS, len(features)).transpose(0, 2, 1)
    return ids, array


def build_tensor(
    df: pd.DataFrame,
    features: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-score each feature over all patients and hours."""
    features = _features(features)
    ids, array = arrange_tensor(df, features)
    mean = array.mean(axis=(0, 2), keepdims=True)
    std = array.std(axis=(0, 2), keepdims=True)
    if np.any(std <= 0):
        constant = [features[k] for k in np.where(std[0, :, 0] <= 0)[0]]
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
    features: list[str] | None = None,
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
    pd.DataFrame(feature, index=_features(features), columns=columns).rename_axis("feature").to_csv(
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


FEATURE_LABELS = {
    "arterial_bp_systolic": "SBP",
    "heart_rate": "HR",
    "respiratory_rate": "RR",
    "base_excess": "BE",
    "creatinine": "Cr",
    "pt_inr": "INR",
    "lactate": "Lac",
    "wbc": "WBC",
    "platelets": "Plt",
    "oxygen_saturation": "SpO2",
    "urine_output": "UO",
}


def plot_factor_figures(results_dir: Path) -> None:
    """Draw one factor figure per saved rank in this results directory.

    Each row is a component. The patient panel is a bar for each of 50 patients
    drawn with seed 1, in that draw order. The horizontal axis is the patient
    and the vertical axis is the factor score. That draw is not applied to the
    feature bars or the time line, which use the full-cohort loadings. Feature
    panels share a vertical scale, and so do the time panels. Each patient
    panel keeps its own scale. The component weight is left out of all three
    panels.
    """
    colors = [f"C{i}" for i in range(10)]
    for rank in RANKS:
        patient_path = results_dir / f"rank{rank}_patient.csv"
        if not patient_path.exists():
            continue
        patients = pd.read_csv(patient_path)
        variables = pd.read_csv(results_dir / f"rank{rank}_variable.csv", index_col=0)
        hours = pd.read_csv(results_dir / f"rank{rank}_time.csv", index_col=0)
        names = [f"comp{i + 1}" for i in range(rank)]
        feature_span = _padded_span(variables[names].to_numpy(dtype=float))
        time_span = _padded_span(hours[names].to_numpy(dtype=float))
        rng = np.random.default_rng(1)
        sample_size = min(50, len(patients))
        sampled = patients.iloc[rng.choice(len(patients), size=sample_size, replace=False)]

        figure, axes = plt.subplots(rank, 3, figsize=(12, 2.6 * rank), squeeze=False)
        for row, name in enumerate(names):
            color = colors[row % len(colors)]
            score = sampled[name].to_numpy(dtype=float)
            axes[row, 0].bar(np.arange(len(score)), score, color=color)
            axes[row, 0].set_xlim(-1, len(score))
            axes[row, 0].set_ylabel(name)
            axes[row, 0].margins(y=0.05)
            labels = [FEATURE_LABELS.get(str(feature), str(feature)) for feature in variables.index]
            axes[row, 1].bar(labels, variables[name], color=color)
            if len(labels) > 7:
                axes[row, 1].tick_params(axis="x", labelrotation=45)
            axes[row, 1].set_ylim(feature_span)
            axes[row, 2].plot(hours.index.astype(float), hours[name], color=color)
            axes[row, 2].set_ylim(time_span)
            axes[row, 2].set_xlim(-1, N_HOURS)
        axes[0, 0].set_title("Patient")
        axes[0, 1].set_title("Feature")
        axes[0, 2].set_title("Time")
        axes[-1, 0].set_xlabel("Patients")
        axes[-1, 1].set_xlabel("Feature")
        axes[-1, 2].set_xlabel("Hour")
        figure.tight_layout()
        figure.savefig(results_dir / f"rank{rank}_factors.png", dpi=150)
        plt.close(figure)


def _padded_span(values: np.ndarray) -> tuple[float, float]:
    """Vertical range covering every component, with a small margin."""
    low = float(np.min(values))
    high = float(np.max(values))
    if low == high:
        pad = 1.0 if low == 0 else abs(low) * 0.1
    else:
        pad = 0.05 * (high - low)
    return low - pad, high + pad


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
    new: bool = False,
    hospital: str | None = None,
) -> pd.DataFrame:
    """Fit ranks 1-10 and save factors, the fit summary, and the plot."""
    features = list(NEW_FEATURES if new else FEATURES)
    if results_dir is None:
        results_dir = cpd_results_dir(how, new=new, hospital=hospital)
    if decomposition_complete(results_dir):
        print(f"reusing {results_dir}")
        return pd.read_csv(results_dir / "fit_summary.csv")
    if path is None:
        path = prepare_imputed(how, new=new)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"aggregation={how} cohort={cohort_name(hospital)}")
    frame = subset_stays(load_hourly(path, features), hospital)
    ids, array = build_tensor(frame, features)
    tensor = ttb.tensor(array)
    temporal = temporal_diagnostics(array)
    print(
        f"{array.shape[0]:,} patients, {array.shape[1]} features, {array.shape[2]} hours"
    )
    print(f"Temporal variation ratio: {temporal['temporal_variation_ratio']:.4%}")

    rows = []
    for rank in RANKS:
        (model, fit, seed), fits = fit_cp(tensor, rank)
        save_factors(model, rank, ids, results_dir, features)
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
    plot_factor_figures(results_dir)
    print(f"Results saved to {results_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--how", choices=["mean", "median", "last"], default="mean")
    parser.add_argument("--new", action="store_true")
    parser.add_argument("--hospital", default=None)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        results_dir = cpd_results_dir(args.how, new=args.new, hospital=args.hospital)
        plot_factor_figures(results_dir)
        print(f"Factor figures saved to {results_dir}")
        return
    run_cpd(how=args.how, new=args.new, hospital=args.hospital)


if __name__ == "__main__":
    main()
