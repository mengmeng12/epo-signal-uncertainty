"""Controlled uncertainty-estimation loss and paired empirical comparisons.

All research inputs remain unchanged. Results are generated under report/.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from epo_project import config as cfg, evaluation as ev, simulation as sim
from epo_project.reliability import moving_block_indices

DEST = ROOT / "report/generated/uncertainty_value"
PLACEBO = ROOT / "report/generated/label_placebo"
BRIDGE = ROOT / "report/generated/calibration_bridge"
ERROR_LEVELS = (0.0, 0.5, 1.0)
ERROR_SEED = 20261112
INTERVAL_SEED = cfg.EVAL_SEED
INPUTS = [
    ROOT / "output/simulation_draws.csv",
    ROOT / "output/deployment_returns.csv",
    ROOT / "output/deployment_sensitivity.csv",
    PLACEBO / "label_placebo_returns.csv",
    PLACEBO / "label_placebo_summary.csv",
    BRIDGE / "calibration_log_error.csv",
    *[
        ROOT / "src/epo_project" / f"{name}.py"
        for name in ("config", "simulation", "reliability", "evaluation")
    ],
    Path(__file__).resolve(),
]


def utility(positions: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return np.sum(positions * mean, axis=-1) - cfg.GAMMA / 2 * np.sum(
        (positions @ sigma) * positions, axis=-1
    )


def utility_share(values: np.ndarray, oracle: np.ndarray) -> tuple[float, float]:
    """Ratio of means and paired delta-method Monte Carlo SE, in percent."""
    values, oracle = np.asarray(values), np.asarray(oracle)
    if (
        values.shape != oracle.shape
        or values.ndim != 1
        or len(values) < 2
        or oracle.mean() <= 0
    ):
        raise ValueError("Require paired draws with positive mean oracle utility.")
    ratio = values.mean() / oracle.mean()
    se = (values - ratio * oracle).std(ddof=1) / (np.sqrt(len(values)) * oracle.mean())
    return 100 * ratio, 100 * se


def controlled_error() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Reuse the original fixed profiles/signals; independently perturb h only."""
    rng = np.random.default_rng(cfg.SIM_SEED)
    profiles = {cv: sim.uncertainty_profile(cv, rng) for cv in cfg.SIM_CV_LEVELS}
    z_mu = rng.standard_normal((cfg.SIM_DRAWS, cfg.SIM_N_ASSETS))
    z_eps = rng.standard_normal(z_mu.shape)
    noise = np.random.default_rng(ERROR_SEED).standard_normal(z_mu.shape)
    sigma = sim.model_covariance()
    variances = np.diag(sigma)
    kappa = cfg.SIM_MEAN_LAMBDA / cfg.SIM_TAU
    scalar_matrix = sigma + kappa * np.diag(variances)
    old = pd.read_csv(ROOT / "output/simulation_draws.csv")
    calibration = pd.read_csv(BRIDGE / "calibration_log_error.csv")
    calibrated_a = float(
        calibration.loc[calibration.feedback == 0, "implied_a"].iloc[0]
    )
    error_levels = sorted({*ERROR_LEVELS, calibrated_a})
    frames = []
    max_identity_error = 0.0
    max_control_error = 0.0
    for cv, h in profiles.items():
        lam_diag = cfg.SIM_MEAN_LAMBDA * variances * h
        signal = z_mu @ np.linalg.cholesky(cfg.SIM_TAU * sigma).T + z_eps * np.sqrt(
            lam_diag
        )
        mean = signal @ np.linalg.solve(
            cfg.SIM_TAU * sigma + np.diag(lam_diag), cfg.SIM_TAU * sigma
        )
        oracle = np.linalg.solve(sigma, mean.T).T / cfg.GAMMA
        scalar = np.linalg.solve(scalar_matrix, signal.T).T / cfg.GAMMA
        u_oracle, u_scalar = utility(oracle, mean, sigma), utility(scalar, mean, sigma)
        loss = u_oracle - u_scalar
        baseline = old[old.target_cv == cv].sort_values("draw")
        for actual, expected in [
            (u_oracle, baseline.utility_het),
            (u_scalar, baseline.utility_mean_scalar),
        ]:
            np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)
            max_control_error = max(
                max_control_error, float(np.max(np.abs(actual - expected)))
            )
        for a in error_levels:
            estimates = h * np.exp(a * noise - a * a / 2)
            estimates /= estimates.mean(axis=1, keepdims=True)
            assert (estimates > 0).all()
            np.testing.assert_allclose(estimates.mean(axis=1), 1, atol=1e-14)
            matrices = np.broadcast_to(sigma, (len(signal), *sigma.shape)).copy()
            diagonal = np.arange(len(h))
            matrices[:, diagonal, diagonal] += kappa * variances * estimates
            estimated = np.linalg.solve(matrices, signal[..., None])[..., 0] / cfg.GAMMA
            delta = estimated - oracle
            penalty = cfg.GAMMA / 2 * np.sum((delta @ sigma) * delta, axis=1)
            gain = utility(estimated, mean, sigma) - u_scalar
            error = gain - (loss - penalty)
            max_identity_error = max(max_identity_error, float(np.max(np.abs(error))))
            np.testing.assert_allclose(gain, loss - penalty, atol=2e-13, rtol=1e-10)
            if a == 0:
                np.testing.assert_allclose(penalty, 0, atol=1e-25)
            if cv == 0:
                np.testing.assert_allclose(loss, 0, atol=1e-13)
                assert gain.max() < 1e-12
            frames.append(
                pd.DataFrame(
                    {
                        "target_cv": cv,
                        "realized_cv": h.std(),
                        "error_sd": a,
                        "draw": np.arange(len(signal)),
                        "oracle_utility": u_oracle,
                        "scalar_loss": loss,
                        "estimation_penalty": penalty,
                        "net_gain": gain,
                        "identity_error": error,
                    }
                )
            )
    draws = pd.concat(frames, ignore_index=True)
    rows = []
    for (cv, actual_cv, a), part in draws.groupby(
        ["target_cv", "realized_cv", "error_sd"]
    ):
        row = {
            "target_cv": cv,
            "realized_cv": actual_cv,
            "error_sd": a,
            "draws": len(part),
            "kappa": kappa,
            "probability_gain_positive": float((part.net_gain > 1e-12).mean()),
        }
        for name in ("oracle_utility", "scalar_loss", "estimation_penalty", "net_gain"):
            values = part[name].to_numpy()
            mean, se = values.mean(), values.std(ddof=1) / np.sqrt(len(values))
            share, share_se = utility_share(values, part.oracle_utility.to_numpy())
            row.update(
                {
                    f"{name}_bp_year": 12e4 * mean,
                    f"{name}_se_bp_year": 12e4 * se,
                    f"{name}_lower_bp_year": 12e4 * (mean - 1.96 * se),
                    f"{name}_upper_bp_year": 12e4 * (mean + 1.96 * se),
                    f"{name}_share_pct": share,
                    f"{name}_share_se_pct": share_se,
                }
            )
        rows.append(row)
    audit = {
        "calibration_implied_a": calibrated_a,
        "max_identity_error_monthly": max_identity_error,
        "max_original_utility_control_error_monthly": max_control_error,
    }
    return draws, pd.DataFrame(rows), audit


def ce_contrasts(
    panel: np.ndarray, samples: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Columns: scalar, observed h, then each random assignment. Return delta/D."""
    if panel.ndim != 2 or panel.shape[1] < 3 or not np.isfinite(panel).all():
        raise ValueError("Require complete finite time-by-strategy returns.")
    ce = ev.certainty_equivalent(panel.T)
    observed = np.array([ce[1] - ce[0], ce[1] - ce[2:].mean()])
    boot = np.empty((len(samples), 2))
    # Batch to keep memory bounded; each statistic uses the same dates.
    for start in range(0, len(samples), 50):
        idx = samples[start : start + 50]
        values = ev.certainty_equivalent(panel[idx].transpose(0, 2, 1))
        boot[start : start + len(idx), 0] = values[:, 1] - values[:, 0]
        boot[start : start + len(idx), 1] = values[:, 1] - values[:, 2:].mean(axis=1)
    return observed, boot


def empirical_intervals() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    paths = pd.read_csv(PLACEBO / "label_placebo_returns.csv", parse_dates=["date"])
    deployed = pd.read_csv(ROOT / "output/deployment_returns.csv", parse_dates=["date"])
    original_summary = pd.read_csv(PLACEBO / "label_placebo_summary.csv")
    random_labels = sorted(
        name for name in paths.strategy.unique() if name.startswith("shuffle_")
    )
    assert len(random_labels) == 100
    columns = ["scalar", "observed_h", *random_labels]
    rows, boot_frames = [], []
    max_pair_error = 0.0
    for period, (start, end) in ev.PERIODS.items():
        expected_dates = pd.date_range(start, end, freq=pd.offsets.MonthEnd())
        seed = np.random.SeedSequence(
            [INTERVAL_SEED, pd.Timestamp(start).year, cfg.EVAL_BLOCK]
        )
        samples = moving_block_indices(
            len(expected_dates),
            cfg.EVAL_BOOTSTRAP_DRAWS,
            cfg.EVAL_BLOCK,
            np.random.default_rng(seed),
        )
        for policy in ("fixed_gamma", "targeted_capped"):
            part = paths[(paths.policy == policy) & paths.date.between(start, end)]
            wide = part.pivot(index="date", columns="strategy", values="net_excess")[
                columns
            ]
            assert wide.index.equals(expected_dates)
            panel = wide.to_numpy()
            observed, bootstrap = ce_contrasts(panel, samples)
            paired_observed, paired_samples = ev.paired_ce_bootstrap(
                panel[:, :2], sample_indices=samples
            )
            np.testing.assert_allclose(bootstrap[:, 0], paired_samples, atol=1e-13)
            max_pair_error = max(
                max_pair_error, float(np.max(np.abs(bootstrap[:, 0] - paired_samples)))
            )
            np.testing.assert_allclose(observed[0], paired_observed, atol=1e-13)
            reference = original_summary[
                (original_summary.period == period)
                & (original_summary.policy == policy)
            ].iloc[0]
            np.testing.assert_allclose(
                observed[0] * 1e4,
                reference.observed_minus_scalar_net_bp_year,
                atol=1e-6,
            )
            if policy == "targeted_capped":
                saved = deployed[deployed.date.between(start, end)].pivot(
                    index="date", columns="strategy", values="net_excess"
                )[["scalar", "heterogeneous"]]
                np.testing.assert_allclose(
                    panel[:, :2], saved.to_numpy(), atol=1e-10, rtol=0
                )
            ce = ev.certainty_equivalent(panel.T)
            lower, upper = np.quantile(bootstrap, [0.025, 0.975], axis=0)
            rows.append(
                {
                    "period": period,
                    "policy": policy,
                    "months": len(wide),
                    "random_assignments": len(random_labels),
                    "resamples": len(samples),
                    "block_months": cfg.EVAL_BLOCK,
                    "delta_ce_bp_year": observed[0] * 1e4,
                    "delta_lower_bp_year": lower[0] * 1e4,
                    "delta_upper_bp_year": upper[0] * 1e4,
                    "random_mean_delta_bp_year": (ce[2:].mean() - ce[0]) * 1e4,
                    "random_median_delta_bp_year": (np.median(ce[2:]) - ce[0]) * 1e4,
                    "assignment_advantage_bp_year": observed[1] * 1e4,
                    "assignment_lower_bp_year": lower[1] * 1e4,
                    "assignment_upper_bp_year": upper[1] * 1e4,
                    "below_original_pct": 100 * float((ce[2:] < ce[1]).mean()),
                }
            )
            boot_frames.append(
                pd.DataFrame(
                    {
                        "period": period,
                        "policy": policy,
                        "draw": np.arange(len(samples)),
                        "delta_ce_bp_year": bootstrap[:, 0] * 1e4,
                        "assignment_advantage_bp_year": bootstrap[:, 1] * 1e4,
                    }
                )
            )
    return (
        pd.DataFrame(rows),
        pd.concat(boot_frames, ignore_index=True),
        {"max_paired_bootstrap_difference": max_pair_error},
    )


def run(destination: Path = DEST) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    draws, summary, simulation_audit = controlled_error()
    intervals, bootstrap, empirical_audit = empirical_intervals()
    for name, frame in [
        ("error_simulation_draws", draws),
        ("error_simulation_summary", summary),
        ("empirical_intervals", intervals),
        ("empirical_bootstrap", bootstrap),
    ]:
        frame.to_csv(destination / f"{name}.csv", index=False, float_format="%.12g")
    manifest = {
        "error_levels": sorted(summary.error_sd.unique().tolist()),
        "error_seed": ERROR_SEED,
        "interval_seed": INTERVAL_SEED,
        "simulation_seed": cfg.SIM_SEED,
        "simulation_audit": simulation_audit,
        "empirical_audit": empirical_audit,
        "input_code_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in INPUTS
        },
        "output_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(destination.glob("*.csv"))
        },
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        summary[
            [
                "target_cv",
                "error_sd",
                "scalar_loss_bp_year",
                "estimation_penalty_bp_year",
                "net_gain_bp_year",
            ]
        ].to_string(index=False)
    )
    print(intervals.to_string(index=False))
    return manifest


if __name__ == "__main__":
    run()
