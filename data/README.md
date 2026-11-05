# Data

## What is tracked

| Tracked by Git | Not tracked |
|---|---|
| `raw/`: both French zip files, the extracted CSVs and `source_manifest.json` | `interim/risk_models.npz` (25 MB) |
| `processed/`: `returns.csv`, `eligibility.csv` | |
| `interim/`: `momentum.csv`, `risk_index.csv`, `forecast_reliability.csv` | |

The data vintage is fixed by the zips: downloaded 2026-10-23, CRSP 202608. `risk_models.npz` is not tracked because its numbers are reproducible but its zip headers depend on the Python version that writes it, so Git would report a change after every run in another environment.

## What running the pipeline should do

Run `end_to_end_pipeline.ipynb` from the top (or `PYTHONPATH=src python -m epo_project.data`, then `python -m epo_project.replication`) from the repository root, in an environment with `requirements.txt` installed. Expected behaviour:

1. **No download.** The tracked zips are reused; `source_manifest.json` keeps its download time and hashes.
2. **Every file is rewritten**, including the untracked `risk_models.npz`.
3. **Inspect numerical changes with `git diff`.** The original data/benchmark CSVs were byte-identical under Python 3.10 / NumPy 1.26 and Python 3.14 / NumPy 2.4. Other environments can differ at floating-point precision (for example minimum eigenvalues or oracle optimizer coordinates). Extracted raw CSVs may also restore CRLF line endings from the unchanged zip. Compare parsed values and source zip hashes before attributing such differences to changed inputs. Unexplained material changes require investigation.
4. **`output/problem_pc.png` may show as changed** after a run with another Matplotlib version, because the image file records rendering details. The figure content is the same; do not commit it unless the diagnostics table changed.
5. **The notebook file may show as changed** because executing it stores new run metadata; commit it only when its code or results change.

To move to a newer data vintage, delete the zips (or call `prepare_returns(force_download=True)`) and commit the new zips, CSVs and manifest together.

All dates are ISO month-ends and all returns are decimal simple returns. Missing values in stored data are left empty. Step 3 currently fills a missing next-month excess return with zero when evaluating portfolios; this proposed convention is under review. Step 5 does not fill missing observations.

## raw/ (Step 1)

Unmodified French Data Library downloads (zip files and the extracted CSVs) and `source_manifest.json` with URLs, download time, SHA256 hashes, CRSP vintage and a summary of the cleaned panel.

## processed/ (Step 1)

| File | One row per | Columns |
|---|---|---|
| `returns.csv` | date × industry | `date`, `industry`, `total_return`, `rf`, `excess_return` |
| `eligibility.csv` | date × industry | `date`, `industry`, `consecutive_months` (non-missing months ending at the date), `has_return`, `momentum_ready` (≥ 12), `risk_ready` (≥ 60), `benchmark_eligible` |

Loaders: `epo_project.data.load_returns(kind)` and `load_eligibility(column)` return wide date × industry tables in the French column order.

## interim/ (Step 2)

| File | Content |
|---|---|
| `momentum.csv` | `date`, `industry`, `trailing_12m_return` (months t−11..t), `z` (momentum score; legs sum to +1 and −1), benchmark-eligible industries only |
| `risk_models.npz` | `dates`, `industries` (all 49), `sigma_hat` and `q`, each of shape (dates, 49, 49), NaN outside the date's universe |
| `risk_index.csv` | `formation_date`, `window_start`, `last_input_date`, `n_industries`, `min_eigenvalue_q`, `industries` (pipe-separated order of the matrix) |

Loader: `epo_project.data.load_inputs()` returns z (wide), the risk index, and `{formation_date: (industries, Sigma_hat, Q)}` with each matrix cut to its own universe and order.

## interim/ (Step 5)

`forecast_reliability.csv` has one row per formation date and eligible industry. Run `PYTHONPATH=src python -m epo_project.reliability` to rebuild it from `returns.csv` and `momentum.csv` without rebuilding earlier steps.

| Columns | Definition |
|---|---|
| `date`, `target_date`, `industry` | Formation month t, target month t+1, industry label; join by label, not row position |
| `z`, `alpha`, `beta`, `s` | Current Step 2 momentum, fitted intercept/slope, forecast `s = alpha + beta*z` of monthly excess return (decimal) |
| `u` | Bootstrap variance of the fitted conditional mean, decimal return squared; excludes future residual noise |
| `sigma2` | Sample variance of the latest 60 monthly excess returns, ddof=1; equals diag(Q) |
| `h` | Winsorized and floored `u/sigma2`, normalized to mean 1 within the formation-date forecast universe |
| `regression_ready_months` | Consecutive months satisfying the 120-pair regression, current signal and risk-history requirements |
| `predictor_start`, `predictor_end` | t−120 through t−1 (120 observations) |
| `response_start`, `response_end` | t−119 through t (120 observations) |
| `risk_start`, `last_input_date` | t−59 and t; every estimator input is known by t |

Default eligibility requires 120 **consecutive, completed** pairs and current risk/signal inputs. Missing data exclude that industry at formation time until its complete history is restored. The optional `minimum_ready_months=60` adds the local walkthrough's admission delay; tracked Steps 5 and 7 use the shared workflow default of 1.

If the investment universe changes, recompute winsorization and normalization from `u` and `sigma2` on that universe using `relative_uncertainty`. Filtering stored `h` alone is insufficient. Keep the Step 2 definition of `z`; it is normalized on the benchmark universe.

`output/reliability_diagnostics.csv` records the industry count, mean/min/max/CV of h (CV uses population standard deviation), mean u, zero-u count, and 12-month Spearman rank persistence using common industry labels. Rank persistence is blank until both dates are present, or if ranks are constant. A partial run can therefore have fewer persistence observations than the full run.

`output/reliability_manifest.json` records input/code/output hashes and estimator parameters. CSVs use 12 significant digits; the in-memory API retains full precision. The array-level `estimate_forecasts` API in `src/epo_project/reliability.py` returns forecasts, fitted-mean variance, h and the bootstrap forecast distribution for Step 6.

## Empirical outputs (Steps 7–9)

These files live under `output/`. Dates in returns and weights are **realization months**; `formation_date` is the preceding month-end. Returns are decimal monthly values; holdings are fractions of beginning-of-month total wealth. CSVs retain 12 significant digits.

| File | Grain and content |
|---|---|
| `annual_selection.csv` | Year × candidate w: prior 60-month bounds, alive/eligible flags, annualized net CE in decimals, selected flag and common selected_w |
| `candidate_returns.csv` | Month × fixed w, 1960–2025: returns, costs and status (`active`, `failed`, `inactive`); failed/inactive net returns remain missing |
| `candidate_exits.csv` | Permanent exit date, w, nonpositive total wealth factor and gross exposure at failure |
| `matched_returns.csv` | Month × strategy, 1970–2025: common w/universe size, gross and net excess return, RF, net total return, wealth factor, full-L1 turnover, gross/short/cash exposures, trading and borrowing costs, predicted annual volatility |
| `weights.csv` | Month × strategy × industry: target weight and drift-adjusted pretrade weight; zero positions retained so industry exits and liquidation trades are visible |
| `performance_table.csv` | Period × strategy: annualized net CE (bp), mean excess return/volatility (%), Sharpe, monthly 95% ES loss (%), maximum drawdown (%), average monthly turnover and exposure, annualized mean costs (bp). Paired delta CE and its 12-month-block 95% interval appear on the heterogeneous row |
| `delta_ce_summary.csv` | Period × block length (12, 6, 24 months): observed heterogeneous-minus-scalar net CE and 2.5/97.5 percentile bounds, all bp/year |
| `delta_ce_bootstrap.csv` | Period × block length × draw: 1,000 paired delta-CE draws for each of six comparisons |
| `equity_curves.csv`, `equity_curves.png` | Net total wealth and drawdown, rebased within each displayed period; this plotting convention does not restart the underlying portfolio in 2019 |
| `deployment_sensitivity.csv` | Same performance definitions after targeting and capping, plus cap-bound months/frequency and mean predicted volatility; delta CE is descriptive |
| `deployment_returns.csv`, `deployment_weights.csv` | Full deployed paths, with the primary selected w but their own positions, drift, turnover and costs |
| `empirical_manifest.json` | Input/code/output SHA256 hashes and parameters for Steps 7–9 |

CE = 12 mean(net excess return) − (gamma/2)·12 sample variance, with gamma=3. Sharpe and volatility use sample standard deviation (ddof=1). ES is the average net-excess loss in exactly the worst 5% empirical probability mass, with a fractional boundary observation when needed; it is monthly and is not scaled by sqrt(12). Drawdown uses compounded **net total** wealth and includes starting wealth of 1 when identifying the peak. Maximum drawdown is reported as a positive loss magnitude. Costs in bp/year are 12 times average monthly costs, not compounded returns.

Bootstrap blocks are non-circular overlapping blocks, concatenated and trimmed to the original sample length. Both strategies share every sampled month. Inference is conditional on the realized selected paths; it does not rerun annual selection inside each draw. The 2019–2025 panel has 84 observations, so especially the 24-month-block sensitivity uses few blocks per draw.
