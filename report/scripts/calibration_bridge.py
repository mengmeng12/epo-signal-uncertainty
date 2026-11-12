"""Calibrate relative-uncertainty error and evaluate targeted trading costs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from epo_project import (
    backtest as bt,
    config as cfg,
    evaluation as ev,
    simulation as sim,
)

DEST = ROOT / "report/generated/calibration_bridge"
INPUTS = [
    ROOT / "output/estimator_calibration.csv",
    ROOT / "output/deployment_returns.csv",
    ROOT / "output/annual_selection.csv",
    ROOT / "data/processed/returns.csv",
    ROOT / "data/interim/forecast_reliability.csv",
    *[
        ROOT / "src/epo_project" / f"{name}.py"
        for name in (
            "simulation",
            "reliability",
            "config",
            "evaluation",
            "backtest",
            "data",
        )
    ],
    Path(__file__).resolve(),
]


def relative_log_error(
    estimated: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Within-history log-error SD removes the irrelevant common log scale."""
    estimated, reference = np.asarray(estimated), np.asarray(reference)
    if (
        estimated.shape != reference.shape
        or estimated.ndim != 2
        or estimated.shape[1] < 2
    ):
        raise ValueError("Require matched history-by-asset matrices.")
    if not (
        np.isfinite(estimated).all()
        and np.isfinite(reference).all()
        and (estimated > 0).all()
        and (reference > 0).all()
    ):
        raise ValueError("Relative uncertainties must be positive and finite.")
    error = np.log(estimated / reference)
    return error, error.std(axis=1, ddof=1)


def calibration_reference(destination: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    with patch.object(cfg, "OUTPUT_DIR", destination / "calibration_control"):
        result = sim.validate_estimator(make_plot=False)
    baseline = pd.read_csv(ROOT / "output/estimator_calibration.csv")
    numeric = baseline.select_dtypes("number").columns
    np.testing.assert_allclose(
        result["table"][numeric],
        baseline[numeric],
        atol=5.1e-7,
        rtol=1e-9,
        equal_nan=True,
    )
    draws = result["draws"].copy()
    # Exact conditional OLS variance only for feedback=0; feedback=.4 is a reference check.
    total_variance = cfg.EST_SLOPE**2 * cfg.EST_SIGNAL_SD**2 + draws.vol**2
    draws["true_relative_raw"] = draws.ols_variance / total_variance
    draws["reference_h"] = draws.true_relative_raw / draws.groupby(
        ["feedback", "replication"]
    ).true_relative_raw.transform("mean")
    rows, histories = [], []
    for feedback, part in draws.groupby("feedback"):
        estimated = part.pivot(
            index="replication", columns="asset", values="h"
        ).to_numpy()
        reference = part.pivot(
            index="replication", columns="asset", values="reference_h"
        ).to_numpy()
        errors, sd = relative_log_error(estimated, reference)
        a = np.sqrt(np.mean(sd**2))
        centered = errors - errors.mean(axis=1, keepdims=True)
        histories.append(
            pd.DataFrame(
                {
                    "feedback": feedback,
                    "replication": np.arange(len(sd)),
                    "log_error_sd": sd,
                }
            )
        )
        rows.append(
            {
                "feedback": feedback,
                "histories": len(sd),
                "assets": estimated.shape[1],
                "implied_a": a,
                "mean_log_error_sd": sd.mean(),
                "median_log_error_sd": np.median(sd),
                "p05_log_error_sd": np.quantile(sd, 0.05),
                "p95_log_error_sd": np.quantile(sd, 0.95),
                "pooled_log_error_sd": errors.std(ddof=1),
                "mean_reference_cv": np.mean(reference.std(axis=1)),
                "mean_estimated_cv": np.mean(estimated.std(axis=1)),
                "mean_centered_log_error_fourth_moment": np.mean(centered**4),
            }
        )
    draws[
        [
            "feedback",
            "replication",
            "asset",
            "h",
            "reference_h",
            "u",
            "ols_variance",
            "asset_variance",
        ]
    ].to_csv(
        destination / "relative_uncertainty_draws.csv",
        index=False,
        float_format="%.12g",
    )
    return pd.DataFrame(rows), pd.concat(histories, ignore_index=True), result["audit"]


def targeted_costs() -> tuple[pd.DataFrame, pd.DataFrame]:
    total, rf, forecasts = bt.load_market()
    selection = pd.read_csv(ROOT / "output/annual_selection.csv")
    stored = pd.read_csv(ROOT / "output/deployment_returns.csv", parse_dates=["date"])
    paths = [stored.assign(cost_bp=10)]
    for cost in (0.0, 0.0025):
        print(f"Replaying common position sizing at cost {cost*1e4:g} bp", flush=True)
        result = bt.simulate_portfolios(
            total,
            rf,
            forecasts,
            fixed_selection=selection,
            deployment=True,
            trade_cost=cost,
        )
        paths.append(result["returns"].assign(cost_bp=cost * 1e4))
    combined = pd.concat(paths, ignore_index=True)
    summaries = []
    for cost, part in combined.groupby("cost_bp"):
        ev.validate_matched(part)
        for period, (start, end) in ev.PERIODS.items():
            frame = part[part.date.between(start, end)]
            paired = frame.pivot(index="date", columns="strategy", values="net_excess")[
                list(ev.STRATEGIES)
            ]
            assert paired.index.equals(
                pd.date_range(start, end, freq=pd.offsets.MonthEnd())
            )
            for block in (6, 12, 24) if cost == 10 else (12,):
                seed = np.random.SeedSequence(
                    [cfg.EVAL_SEED, pd.Timestamp(start).year, block]
                )
                point, samples = ev.paired_ce_bootstrap(
                    paired.to_numpy(), block_length=block, seed=seed
                )
                low, high = np.quantile(samples, [0.025, 0.975])
                ces = ev.certainty_equivalent(paired.to_numpy().T) * 1e4
                summaries.append(
                    {
                        "period": period,
                        "cost_bp": cost,
                        "block_months": block,
                        "delta_ce_bp_year": point * 1e4,
                        "ci_lower_bp_year": low * 1e4,
                        "ci_upper_bp_year": high * 1e4,
                        "scalar_ce_bp_year": ces[0],
                        "heterogeneous_ce_bp_year": ces[1],
                    }
                )
    return pd.DataFrame(summaries), combined


def run(destination: Path = DEST) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    summary, histories, audit = calibration_reference(destination)
    print(summary.to_string(index=False), flush=True)
    costs, paths = targeted_costs()
    for name, frame in [
        ("calibration_log_error", summary),
        ("calibration_log_error_histories", histories),
        ("targeted_cost_summary", costs),
        ("targeted_cost_returns", paths),
    ]:
        frame.to_csv(destination / f"{name}.csv", index=False, float_format="%.12g")
    manifest = {
        "calibration_audit": audit,
        "reference": "OLS variance / population return variance, normalized within history; exact only at feedback zero",
        "implied_a": "root mean within-history sample variance of log(h_hat/h_reference)",
        "input_code_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in INPUTS
        },
        "output_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in destination.glob("*.csv")
        },
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(costs.to_string(index=False), flush=True)
    return manifest


if __name__ == "__main__":
    run()
