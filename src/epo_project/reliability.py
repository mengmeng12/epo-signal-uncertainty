"""Step 5: rolling OLS forecasts and uncertainty in the fitted conditional mean.

At formation month t, fit 120 pairs (z[u], excess_return[u+1]), with the
last response observed at t. Resample whole pairs in shared 12-month blocks
across industries, refit, and evaluate every fit at the same current z[t].
The array-level estimator is also the Step 6 calibration interface.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from . import config as cfg
from .data import load_returns


@dataclass(frozen=True)
class ForecastEstimate:
    """Arrays follow the caller's asset order; bootstrap arrays are draw × asset."""

    prediction: np.ndarray
    mean_variance: np.ndarray
    h: np.ndarray
    intercept: np.ndarray
    slope: np.ndarray
    bootstrap_predictions: np.ndarray
    bootstrap_indices: np.ndarray


def moving_block_indices(
    n_observations: int, n_bootstrap: int, block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Non-circular overlapping blocks, concatenated and truncated to window size."""
    if not all(isinstance(v, (int, np.integer)) for v in
               (n_observations, n_bootstrap, block_length)):
        raise ValueError("Window, draws and block length must be integers.")
    if not 1 <= block_length <= n_observations or n_bootstrap < 2:
        raise ValueError("Require 1 <= block_length <= window and at least two draws.")
    n_blocks = (n_observations + block_length - 1) // block_length
    starts = rng.integers(0, n_observations - block_length + 1,
                          size=(n_bootstrap, n_blocks))
    return (starts[..., None] + np.arange(block_length)).reshape(
        n_bootstrap, -1
    )[:, :n_observations]


def relative_uncertainty(
    mean_variance: np.ndarray, asset_variance: np.ndarray,
) -> np.ndarray:
    """Winsorize u / diag(Q) at 5/95%, floor at 1e-8, then normalize to mean 1."""
    u, v = np.asarray(mean_variance, dtype=float), np.asarray(asset_variance, dtype=float)
    if u.ndim != 1 or not u.size or u.shape != v.shape:
        raise ValueError("Uncertainty and asset variance must be matching nonempty vectors.")
    if not np.isfinite(u).all() or not np.isfinite(v).all() or (u < 0).any() or (v <= 0).any():
        raise ValueError("Require finite u >= 0 and finite asset variance > 0.")
    if not u.any():
        return np.ones_like(u)
    ratio = u / v
    if not np.isfinite(ratio).all():
        raise ValueError("Uncertainty / asset variance overflowed.")
    lower, upper = np.quantile(ratio, cfg.H_WINSOR)
    clipped = np.maximum(np.clip(ratio, lower, upper), cfg.H_FLOOR)
    return clipped / clipped.mean()


def _ols(x: np.ndarray, y: np.ndarray, current: np.ndarray):
    """Fit along the penultimate axis; a constant predictor gives an intercept-only fit."""
    x_mean, y_mean = x.mean(axis=-2), y.mean(axis=-2)
    xc, yc = x - x_mean[..., None, :], y - y_mean[..., None, :]
    denominator = (xc * xc).sum(axis=-2)
    slope = np.divide((xc * yc).sum(axis=-2), denominator,
                      out=np.zeros_like(denominator), where=denominator > 1e-14)
    intercept = y_mean - slope * x_mean
    return y_mean + slope * (current - x_mean), intercept, slope


def estimate_forecasts(
    predictors: np.ndarray, responses: np.ndarray, current_signal: np.ndarray,
    asset_variance: np.ndarray, *, seed=cfg.RELIABILITY_SEED,
    n_bootstrap: int = cfg.BOOTSTRAP_DRAWS, block_length: int = cfg.BOOTSTRAP_BLOCK,
    sample_indices: np.ndarray | None = None,
) -> ForecastEstimate:
    """Estimate forecasts for aligned, completed training pairs (time × asset).

    No missing observations are imputed. Empirical eligibility is handled by
    build_forecast_panel. For Step 6, pass simulated completed pairs to this same
    function, then compare prediction with the known conditional mean. The
    variance u excludes future residual/return noise. sample_indices optionally
    supplies a shared draw × time matrix for paired experiments or verification.
    """
    x, y = np.asarray(predictors, dtype=float), np.asarray(responses, dtype=float)
    current, v = np.asarray(current_signal, dtype=float), np.asarray(asset_variance, dtype=float)
    if x.ndim != 2 or x.shape != y.shape or min(x.shape) == 0:
        raise ValueError("Predictors and responses must be matching nonempty time × asset arrays.")
    if current.shape != (x.shape[1],) or v.shape != current.shape:
        raise ValueError("Current signal and variance must match the asset dimension.")
    if not all(np.isfinite(a).all() for a in (x, y, current, v)) or (v <= 0).any():
        raise ValueError("Training pairs and current inputs must be finite; variance must be positive.")
    if sample_indices is None:
        indices = moving_block_indices(len(x), n_bootstrap, block_length, np.random.default_rng(seed))
    else:
        indices = np.asarray(sample_indices)
        if (indices.ndim != 2 or indices.shape[1] != len(x) or indices.shape[0] < 2
                or not np.issubdtype(indices.dtype, np.integer)
                or (indices < 0).any() or (indices >= len(x)).any()):
            raise ValueError("sample_indices must contain at least two complete, in-bounds integer draws.")
    prediction, intercept, slope = _ols(x, y, current)
    boot, _, _ = _ols(x[indices], y[indices], current)
    u = boot.var(axis=0, ddof=1)
    u[np.ptp(boot, axis=0) == 0] = 0.0  # avoid round-off variance for identical fits
    return ForecastEstimate(prediction, u, relative_uncertainty(u, v), intercept,
                            slope, boot, indices)


def _validate_monthly(frame: pd.DataFrame, name: str) -> None:
    idx = frame.index
    if (not isinstance(idx, pd.DatetimeIndex) or idx.empty or idx.hasnans
            or idx.has_duplicates or not idx.is_monotonic_increasing
            or frame.columns.has_duplicates or frame.shape[1] == 0):
        raise ValueError(f"{name} must have unique assets and a sorted, unique monthly DatetimeIndex.")
    expected = pd.date_range(idx[0], idx[-1], freq=pd.offsets.MonthEnd())
    if not idx.equals(expected) or not (idx == idx.normalize()).all():
        raise ValueError(f"{name} must include every calendar month-end; represent missing values with NaN.")
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError(f"{name} contains infinite values.")


def build_forecast_panel(
    momentum: pd.DataFrame, excess: pd.DataFrame, *, start=None, end=None,
    seed: int = cfg.RELIABILITY_SEED, minimum_ready_months: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a long forecast panel and formation-date diagnostics.

    Defaults implement the shared Step 5 workflow: 120 consecutive completed
    pairs plus the current signal and 60-month risk window. minimum_ready_months=60
    reproduces the extra admission delay used in the local quick experiment.
    The delay is distinct from Step 7's 60-month candidate-return selection window.
    Date-keyed seeds make a truncated history or a partial run give identical
    results at the same formation date. Future realized returns are never read.
    """
    _validate_monthly(excess, "excess")
    _validate_monthly(momentum, "momentum")
    if not isinstance(minimum_ready_months, (int, np.integer)) or minimum_ready_months < 1:
        raise ValueError("minimum_ready_months must be a positive integer.")
    if not momentum.columns.isin(excess.columns).all() or not momentum.index.isin(excess.index).all():
        raise ValueError("Momentum assets and dates must be contained in the return panel.")
    z = momentum.reindex(index=excess.index, columns=excess.columns)
    pairs = z.shift(1).notna() & excess.notna()
    variance = excess.rolling(cfg.RISK_WINDOW, min_periods=cfg.RISK_WINDOW).var(ddof=1)
    ready = (pairs.rolling(cfg.REGRESSION_PAIRS, min_periods=cfg.REGRESSION_PAIRS)
             .sum().eq(cfg.REGRESSION_PAIRS) & z.notna() & variance.gt(0))
    streak = np.zeros(excess.shape[1], dtype=int)
    x, y, v = z.to_numpy(), excess.to_numpy(), variance.to_numpy()
    rows, diagnostic_rows, past_h = [], [], {}
    start = pd.Timestamp(start) if start is not None else excess.index[0]
    # Last formation date for a target month in the proposal's subsequent period.
    end = pd.Timestamp(end) if end is not None else pd.Timestamp(cfg.SUBSEQUENT_PERIOD[1]) - pd.offsets.MonthEnd()
    for t, date in enumerate(excess.index):
        if date > end:
            break
        streak = np.where(ready.iloc[t].to_numpy(), streak + 1, 0)
        assets = np.flatnonzero(streak >= minimum_ready_months)
        if date < start or len(assets) < cfg.MIN_INDUSTRIES:
            continue
        first = t - cfg.REGRESSION_PAIRS
        estimate = estimate_forecasts(
            x[first:t, assets], y[first + 1:t + 1, assets], x[t, assets], v[t, assets],
            seed=np.random.SeedSequence([seed, int(date.strftime("%Y%m"))]),
        )
        names = excess.columns[assets]
        h = pd.Series(estimate.h, index=names)
        # Persistence is filled below so restricting output start does not alter forecasts.
        past_h[date] = h
        rows.append(pd.DataFrame({
            "date": date, "target_date": date + pd.offsets.MonthEnd(), "industry": names,
            "z": x[t, assets], "alpha": estimate.intercept, "beta": estimate.slope,
            "s": estimate.prediction, "u": estimate.mean_variance, "sigma2": v[t, assets],
            "h": estimate.h, "regression_ready_months": streak[assets],
            "predictor_start": excess.index[first], "predictor_end": excess.index[t - 1],
            "response_start": excess.index[first + 1], "response_end": date,
            "risk_start": excess.index[t - cfg.RISK_WINDOW + 1], "last_input_date": date,
        }))
        diagnostic_rows.append({
            "date": date, "target_date": date + pd.offsets.MonthEnd(),
            "n_regression_ready": int(ready.iloc[t].sum()), "n_industries": len(assets),
            "h_mean": h.mean(), "h_min": h.min(), "h_max": h.max(),
            "h_cv": h.std(ddof=0) / h.mean(), "mean_u": estimate.mean_variance.mean(),
            "zero_u_count": int((estimate.mean_variance == 0).sum()),
        })
    if not rows:
        raise ValueError("No formation dates have enough complete training histories in the requested range.")
    diagnostics = pd.DataFrame(diagnostic_rows)
    persistence = []
    for date in diagnostics.date:
        previous = past_h.get(date - pd.offsets.MonthEnd(12))
        value = np.nan
        if previous is not None:
            paired = pd.concat([past_h[date], previous], axis=1, join="inner")
            if len(paired) >= 2 and (paired.nunique() > 1).all():
                value = paired.rank().corr().iloc[0, 1]
        persistence.append(value)
    diagnostics["h_rank_corr_12m"] = persistence
    return pd.concat(rows, ignore_index=True), diagnostics


def run(*, start=None, end=None, minimum_ready_months: int = 1) -> dict:
    """Use the tracked Step 1/2 CSVs; do not download data or rebuild Steps 1–4."""
    momentum_path = cfg.INTERIM_DIR / "momentum.csv"
    momentum = pd.read_csv(momentum_path, parse_dates=["date"]).pivot(
        index="date", columns="industry", values="z"
    )
    panel, diagnostics = build_forecast_panel(
        momentum, load_returns("excess_return"), start=start, end=end,
        minimum_ready_months=minimum_ready_months,
    )
    cfg.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_path = cfg.INTERIM_DIR / "forecast_reliability.csv"
    diagnostics_path = cfg.OUTPUT_DIR / "reliability_diagnostics.csv"
    panel.to_csv(panel_path, index=False, date_format="%Y-%m-%d", float_format="%.12g")
    diagnostics.to_csv(diagnostics_path, index=False, date_format="%Y-%m-%d", float_format="%.12g")
    paths = [cfg.PROCESSED_DIR / "returns.csv", momentum_path,
             cfg.ROOT / "src/epo_project/config.py", cfg.ROOT / "src/epo_project/data.py",
             cfg.ROOT / "src/epo_project/reliability.py", panel_path, diagnostics_path]
    manifest = {
        "formation_start": str(panel.date.min().date()), "formation_end": str(panel.date.max().date()),
        "rows": len(panel), "formation_months": len(diagnostics),
        "regression_pairs": cfg.REGRESSION_PAIRS, "bootstrap_draws": cfg.BOOTSTRAP_DRAWS,
        "block_length": cfg.BOOTSTRAP_BLOCK, "seed": cfg.RELIABILITY_SEED,
        "minimum_ready_months": minimum_ready_months,
        "winsor_quantiles": cfg.H_WINSOR, "h_floor": cfg.H_FLOOR,
        "hashes": {str(p.relative_to(cfg.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    (cfg.OUTPUT_DIR / "reliability_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"forecasts": panel, "diagnostics": diagnostics, "manifest": manifest}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", help="First formation month-end (inclusive).")
    parser.add_argument("--end", help="Last formation month-end (inclusive).")
    parser.add_argument("--minimum-ready-months", type=int, default=1)
    args = parser.parse_args()
    result = run(start=args.start, end=args.end, minimum_ready_months=args.minimum_ready_months)
    print(json.dumps({k: v for k, v in result["manifest"].items() if k != "hashes"}, indent=2))
