"""Non-negative CP decomposition of the imputed hourly tensor.

Same ranks, random starts, and output files as cpd.py, with two differences:
  * Each feature is clipped to its 1st-99th percentile and scaled to [0, 1].
    Z-scores contain negatives, which a non-negative factorization cannot fit.
  * Factors are constrained to be at least zero and fitted by L-BFGS-B.
Fits are not comparable to cpd.py because the data are not centered.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyttb as ttb
from pyttb.gcp import handles
from pyttb.gcp.optimizers import LBFGSB

from cpd import (
    RANKS,
    RESULTS_DIR,
    SEEDS,
    arrange_tensor,
    load_hourly,
    plot_factor_figures,
    plot_fit,
    prepare_imputed,
    reconstruction_diagnostics,
    save_factors,
    temporal_diagnostics,
)
from data_preprocessing import DECIMAL_PLACES, FEATURES, NEW_FEATURES

# pyttb logs a harmless factor-matrix layout warning on every iteration.
logging.getLogger().setLevel(logging.ERROR)

RESULTS_DIR_NN = RESULTS_DIR.parent / "nncpd"


def nncpd_results_dir(how: str, new: bool = False) -> Path:
    """Result directory for this aggregation."""
    if how not in {"mean", "median", "last"}:
        raise ValueError("how must be 'mean', 'median', or 'last'")
    prefix = "nncpd_new" if new else "nncpd"
    return RESULTS_DIR_NN.parent / f"{prefix}_{how}"
CLIP_PCT = (1, 99)
NONNEG_GAUSSIAN = (handles.gaussian, handles.gaussian_grad, 0.0)
MAXITER = 2000
PGTOL = 1e-5
FACTR = 1e7


def build_tensor(
    frame: pd.DataFrame,
    features: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip each feature to its 1st-99th percentile, then scale it to [0, 1]."""
    features = list(FEATURES if features is None else features)
    ids, array = arrange_tensor(frame, features)
    low = np.percentile(array, CLIP_PCT[0], axis=(0, 2), keepdims=True)
    high = np.percentile(array, CLIP_PCT[1], axis=(0, 2), keepdims=True)
    array = np.clip(array, low, high)
    scale = high - low
    if np.any(scale <= 0):
        constant = [features[k] for k in np.where(scale[0, :, 0] <= 0)[0]]
        raise ValueError(f"Zero percentile range: {constant}")
    scaled = (array - low) / scale
    if not np.isfinite(scaled).all():
        raise ValueError("Non-finite scaled data")
    return ids, scaled


def fit_of(tensor: ttb.tensor, model: ttb.ktensor) -> float:
    """Same definition as CP-ALS: 1 - ||X - M|| / ||X||."""
    return 1 - (tensor - model.full()).norm() / tensor.norm()


def fit_nncp(
    tensor: ttb.tensor, rank: int
) -> tuple[tuple[ttb.ktensor, float, int], list[float]]:
    """Fit non-negative CP from several random starts and keep the best fit."""
    best, fits = None, []
    for seed in SEEDS:
        np.random.seed(seed)
        model, _, _ = ttb.gcp_opt(
            tensor,
            rank,
            objective=NONNEG_GAUSSIAN,
            optimizer=LBFGSB(maxiter=MAXITER, pgtol=PGTOL, factr=FACTR),
            printitn=0,
        )
        fit = float(fit_of(tensor, model))
        fits.append(fit)
        if best is None or fit > best[1]:
            best = (model, fit, seed)
    model = best[0].normalize(sort=True)
    return (model, best[1], best[2]), fits


def run_nncpd(
    how: str = "mean",
    path: Path | None = None,
    results_dir: Path | None = None,
    new: bool = False,
) -> pd.DataFrame:
    """Fit ranks 1-5 and save factors, the fit summary, and the plot."""
    features = list(NEW_FEATURES if new else FEATURES)
    if path is None:
        path = prepare_imputed(how, new=new)
    if results_dir is None:
        results_dir = nncpd_results_dir(how, new=new)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"aggregation={how}")
    frame = load_hourly(path, features)
    ids, array = build_tensor(frame, features)
    tensor = ttb.tensor(array)
    temporal = temporal_diagnostics(array)
    print(
        f"{array.shape[0]:,} patients, {array.shape[1]} features, {array.shape[2]} hours"
    )
    print(f"Temporal variation ratio: {temporal['temporal_variation_ratio']:.4%}")

    rows = []
    for rank in RANKS:
        (model, fit, seed), fits = fit_nncp(tensor, rank)
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
    summary.to_csv(
        results_dir / "fit_summary.csv",
        index=False,
        float_format=f"%.{DECIMAL_PLACES}f",
    )
    plot_fit(
        summary,
        results_dir,
        title=f"Non-negative CP fit vs rank (best of {len(SEEDS)} starts)",
    )
    plot_factor_figures(results_dir)
    print(f"Results saved to {results_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--how", choices=["mean", "median", "last"], default="mean")
    parser.add_argument("--new", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        results_dir = nncpd_results_dir(args.how, new=args.new)
        plot_factor_figures(results_dir)
        print(f"Factor figures saved to {results_dir}")
        return
    run_nncpd(how=args.how, new=args.new)


if __name__ == "__main__":
    main()
