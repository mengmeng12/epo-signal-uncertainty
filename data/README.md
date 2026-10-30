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
3. **`git status` shows no changes** to the tracked files under `data/` and to the CSV files under `output/`: they are rebuilt byte for byte. This was checked with Python 3.10 / NumPy 1.26 and Python 3.14 / NumPy 2.4. Any change there means the code or the inputs differ; inspect it with `git diff`.
4. **`output/problem_pc.png` may show as changed** after a run with another Matplotlib version, because the image file records rendering details. The figure content is the same; do not commit it unless the diagnostics table changed.
5. **The notebook file may show as changed** because executing it stores new run metadata; commit it only when its code or results change.

To move to a newer data vintage, delete the zips (or call `prepare_returns(force_download=True)`) and commit the new zips, CSVs and manifest together.

All dates are ISO month-ends and all returns are decimal simple returns. Missing values in stored data are left empty. Step 3 currently fills a missing next-month excess return with zero when evaluating portfolios; this proposed convention is under review (see `output/step5_alignment_review.md`). Step 5 does not fill missing observations.

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

Default eligibility requires 120 **consecutive, completed** pairs and current risk/signal inputs. Missing data exclude that industry at formation time until its complete history is restored. The optional `minimum_ready_months=60` adds the local walkthrough's admission delay; tracked results use the shared workflow default of 1. See the handoff before choosing the Step 7 universe.

If the investment universe changes, recompute winsorization and normalization from `u` and `sigma2` on that universe using `relative_uncertainty`. Filtering stored `h` alone is insufficient. Keep the Step 2 definition of `z`; it is normalized on the benchmark universe.

`output/reliability_diagnostics.csv` records the industry count, mean/min/max/CV of h (CV uses population standard deviation), mean u, zero-u count, and 12-month Spearman rank persistence using common industry labels. Rank persistence is blank until both dates are present, or if ranks are constant. A partial run can therefore have fewer persistence observations than the full run.

`output/reliability_manifest.json` records input/code/output hashes and estimator parameters. CSVs use 12 significant digits; the in-memory API retains full precision. See `doc/step5_handoff.md` for the array-level Step 6 interface.
