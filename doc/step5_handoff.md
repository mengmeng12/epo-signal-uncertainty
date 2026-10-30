# Step 5 handoff to Yadan

Owner: Shuchen Meng / Isabella. Branch: `feature/reliability-and-oos`, based on your Step 4 commit `667e8d6` (including Yuzhen's `f789590`). Steps 1–4 code and result tables have been preserved. Step 6 can start with the array-level estimator below; it does not depend on resolving the empirical missing-return convention.

## Use the same estimator in Step 6

```python
import numpy as np
from epo_project.reliability import estimate_forecasts

# Supply arrays from your fixed linear forecasting DGP:
# x_history, y_history: (120, N), already paired as x[u] -> y[u+1].
# current_z: (N,), observed at the forecast origin.
# asset_variance: (N,), strictly positive, using the DGP's documented risk rule.
estimate = estimate_forecasts(
    x_history, y_history, current_z, asset_variance,
    seed=np.random.SeedSequence([20261030, replication_id]),
)
forecast = estimate.prediction                   # (N,)
u = estimate.mean_variance                       # (N,)
h = estimate.h                                  # (N,), positive, mean 1
bootstrap_means = estimate.bootstrap_predictions # (200, N)
sampled_dates = estimate.bootstrap_indices       # (200, 120), shared across N

# The DGP supplies true_mean_at_current_z; it is never an estimator input.
squared_error = (forecast - true_mean_at_current_z) ** 2
lower, upper = np.quantile(bootstrap_means, [0.025, 0.975], axis=0)
covered = (lower <= true_mean_at_current_z) & (true_mean_at_current_z <= upper)
```

The percentile interval above is an interface example, not a coverage result. Aggregate squared errors, u and coverage across independent DGP repetitions, and report Monte Carlo uncertainty. Record the DGP, seeds and interval convention. Use a new bootstrap seed per repetition. Residual return noise belongs in the simulated training responses; do not add it to u or compare u with realized future-return variance. Under a linear DGP the true conditional mean at the current signal is alpha + beta*current_z.

The estimator uses OLS with an intercept, 200 shared-date moving-block draws (block length 12), and refits each draw at the fixed current signal. It returns full-precision arrays in input asset order. The defaults are the empirical procedure; changing draws or block length should be reported as sensitivity analysis. `sample_indices` can supply an explicit shared resampling matrix for paired comparisons. Inputs must be finite; missing histories are excluded by the empirical wrapper, not filled inside the core estimator.

Step 4's h describes known uncertainty in its Gaussian model. Step 5's h is an estimated relative uncertainty proxy. Step 6 evaluates the latter's calibration; the stored empirical h values cannot supply simulation ground truth.

## Rebuild and inspect empirical inputs

From the repository root, in the existing Python environment:

```bash
PYTHONPATH=src python -m epo_project.reliability
PYTHONPATH=src python -m pytest tests/test_reliability.py -q
```

The notebook's new Step 5 section calls `reliability.run()` and displays diagnostics. Expected files:

| File | Contents |
|---|---|
| `data/interim/forecast_reliability.csv` | 46,662 date-industry rows; 1,014 formation dates, 1941-06 to 2025-11; 40–49 industries per date |
| `output/reliability_diagnostics.csv` | Counts, h dispersion and 12-month rank persistence |
| `output/reliability_manifest.json` | Parameters and SHA256 hashes of inputs, estimator code and outputs |

Dates refer to **formation month t**; `target_date` is t+1. At 2018-12, responses run from 2009-01 through 2018-12; predictors run from 2008-12 through 2018-11. Forecasts target 2019-01. See [data schema](../data/README.md) for all columns.

## Decisions before Step 7

- Step 3 fills one missing next-month **excess** return with zero. The local walkthrough marks that portfolio month unavailable. See the [alignment review](../output/step5_alignment_review.md) for the quantified impact. Do not drop a holding using future return availability.
- The tracked Step 5 panel uses 120 completed pairs without an additional admission delay. The local quick experiment adds 60 consecutive regression-ready months. To reproduce that convention in memory, call `build_forecast_panel(z, excess, minimum_ready_months=60)`. Calling `run(minimum_ready_months=60)` overwrites the standard output files; treat that as an explicit sensitivity run.
- After changing the eligible universe, recompute h from u and sigma2 on that universe. Join covariance matrices and forecasts by industry labels. Both strategies must share the resulting universe and common w.
- The local experiment permanently excludes bankrupt fixed-w candidates and leaves their subsequent returns missing. If either selected strategy becomes insolvent, execution stops with an error. The selected paths start from cash in 1970 and continue through 2019 without restarting. Step 7 still needs these accounting rules to be implemented and documented. No portfolio backtest or performance claim is part of Step 5.
- Local results for 2019–2025 have already been inspected. Describe this period as subsequent evaluation, not an untouched holdout.

Verification includes independent least-squares comparisons for every bootstrap draw, shared cross-asset sampling, strict consecutive-history eligibility, time truncation/future perturbation, industry-order invariance, and degenerate inputs. At four historical dates, the local walkthrough's estimator and this module produce identical s and u on the same inputs/universe; h differs by less than 1.5e−13. Full CSV reruns are byte-identical in the working environment.
