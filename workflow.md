# Project workflow

Research question: when does one scalar preserve decision-relevant uncertainty information, and when does even the best scalar lose economic value? Simulations identify the mechanism; the empirical study evaluates an implementable uncertainty proxy.

Steps 0–5 are implemented on `feature/reliability-and-oos`; Steps 6–10 remain planned work. The revised proposal is the design authority.

Each step has one accountable owner. Other members can review or provide inputs without sharing responsibility for completing that step. Step numbers describe the workflow; independent steps can run in parallel.

## Step overview

| Step | Task | Owner | Depends on |
|---|---|---|---|
| 0 | Establish shared conventions | Yuzhen Chen | None |
| 1 | Prepare industry returns | Yuzhen Chen | Step 0 |
| 2 | Build momentum and risk inputs | Yuzhen Chen | Step 1 |
| 3 | Validate the EPO benchmark | Yuzhen Chen | Step 2 |
| 4 | Measure scalar compression loss | Yadan Zhan | Step 0 |
| 5 | Estimate forecasts and relative uncertainty | Shuchen Meng | Step 2 |
| 6 | Validate the uncertainty estimator | Yadan Zhan | Steps 4 and 5 |
| 7 | Run matched empirical portfolios | Shuchen Meng | Step 5 |
| 8 | Evaluate economic value and uncertainty | Shuchen Meng | Step 7 |
| 9 | Check deployment sensitivity | Shuchen Meng | Step 7 |
| 10 | Assemble the final report and presentation | Shuchen Meng | Steps 3, 6, 8, and 9 |

## Step 0 - Establish shared conventions

**Owner:** Yuzhen Chen

Confirm the revised proposal, Python environment, shared parameters, information timing, and artifact names with the team. Record whether the 2019-2025 sample has already informed any design choices. Define the common data interfaces before implementation.

**Deliverables:** `src/epo_project/config.py`, an environment dependency file, `data/README.md`, and `output/design_decisions.md`.

**Completion criteria:** the common shrinkage parameter, risk aversion `gamma = 3`, windows, costs, and sample periods match the proposal. Returns use decimal units; each covariance matrix has an explicit date and industry order. Document any agreed design changes and the status of the proposed holdout.

## Step 1 - Prepare industry returns

**Owner:** Yuzhen Chen

Download the monthly value-weighted French 49 Industry Portfolios and risk-free series. Clean missing-value codes and units, preserve the source files, and determine which industries are eligible at each historical date.

**Deliverables:** `data/processed/returns.csv`, `data/processed/eligibility.csv`, and `data/raw/source_manifest.json`.

**Completion criteria:** each date-industry pair is unique; returns are decimals; missing observations are not filled using future information. Record source URLs, download dates, file hashes, and the histories required for eligibility.

**Planned entry point:** `data.prepare_returns`.

## Step 2 - Build momentum and risk inputs

**Owner:** Yuzhen Chen

Implement the trailing 12-month momentum score, the 60-month covariance model, and the fixed 5% preliminary covariance shrinkage. Cache the inputs with their formation dates, last input dates, and industry ordering.

**Deliverables:** `data/interim/momentum.csv`, `data/interim/risk_models.npz`, and `data/interim/risk_index.csv`.

**Completion criteria:** the positive and negative momentum legs sum to +1 and -1. Each corrected covariance matrix `Q` is positive definite and preserves the original diagonal. Every input used to form positions at month-end `t` is available by `t`.

**Planned entry point:** `data.build_inputs`.

## Step 3 - Validate the EPO benchmark

**Owner:** Yuzhen Chen

Implement only the Equity 1 industry benchmark: raw MVO, MVO with the 5% preliminary correction, and scalar EPO. Select the scalar shrinkage parameter each January using candidate Sharpe ratios from the preceding 60 months. Form principal-component portfolios at `t` and evaluate them at `t+1`.

**Deliverables:** `output/replication_table.csv`, `output/problem_pc_diagnostics.csv`, and `output/problem_pc.png`. Prepare the data and replication sections of the report and the corresponding presentation material.

**Completion criteria:** evaluate 1942-2018 where the required histories are available. Document implementation differences from the published paper and compare predicted versus realized risk and return. Treat this as implementation validation, not a separate research hypothesis or a claim of exact replication.

**Planned entry point:** `replication.run`.

## Step 4 - Measure scalar compression loss

**Owner:** Yadan Zhan

Implement the known-covariance Gaussian model and compare the heterogeneous allocation with the best scalar allocation under the same information set. Separate loss from choosing a scalar parameter poorly from the irreducible loss of the scalar family. Describe exposure alignment through conditional comparisons.

**Deliverables:** `output/simulation_draws.csv`, `output/compression_loss.csv`, and `output/compression_loss.png`.

**Completion criteria:** use the proposal's 49-asset design and 1,000 paired draws per heterogeneity level. Uniform relative uncertainty must recover scalar EPO with approximately zero compression loss. Neither allocation may observe the sampled true mean. Check numerical convergence, boundary solutions, and the zero-position limit. Set uncertainty profiles before drawing the mean and signal noise.

**Planned entry point:** `simulation.run`.

## Step 5 - Estimate forecasts and relative uncertainty

**Owner:** Shuchen Meng

Fit OLS forecasts using 120 completed monthly observation pairs. Refit 200 times using shared sampled dates in 12-month blocks. At the current signal, estimate the variance of the fitted conditional mean, divide by asset-return variance, winsorize at the cross-sectional 5th and 95th percentiles, apply the numerical floor, and normalize to mean one.

**Deliverables:** a reusable forecasting and bootstrap function in `src/epo_project/reliability.py`, `data/interim/forecast_reliability.csv`, and `output/reliability_diagnostics.csv`.

**Completion criteria:** relative uncertainty weights are positive and have cross-sectional mean one. If all fitted-mean variances vanish, use `h = 1`. Forecasts and uncertainty estimates use information available by `t`; regression pairs satisfy `u+1 <= t`. Do not add regression residual variance to the fitted-mean uncertainty proxy.

**Planned entry point:** `reliability.run`.

## Step 6 - Validate the uncertainty estimator

**Owner:** Yadan Zhan

Define one fixed linear forecasting data-generating process and use the exact estimator delivered in Step 5. Compare conditional-mean estimation errors with bootstrap variance estimates and assess interval coverage. Combine this check with Step 4 when interpreting the simulation evidence.

**Deliverables:** `output/estimator_calibration.csv` and `output/estimator_calibration.png`. Prepare the theory and simulation sections of the report and the corresponding presentation material.

**Completion criteria:** record all simulation parameters and seeds. Reuse the implemented estimation procedure rather than replacing it with artificially added reliability noise. Distinguish conditional-mean estimation error from realized return variance. Step 5 is an interface dependency; it does not supply a true mean for the simulation.

**Planned entry point:** `simulation.validate_estimator`.

## Step 7 - Run matched empirical portfolios

**Owner:** Shuchen Meng

Construct scalar and heterogeneous portfolios with the same forecasts, covariance model, eligible industries, and `gamma = 3`. Each January, use the scalar candidates' preceding 60-month net certainty-equivalent returns to choose one common `w`. The heterogeneous rule changes only the relative uncertainty weights. Apply 10 basis points per dollar traded and a 50-basis-point annual charge on short exposure.

**Deliverables:** `output/annual_selection.csv`, `output/weights.csv`, and `output/matched_returns.csv`.

**Completion criteria:** no additional heterogeneity tuning parameter is selected. Both portfolios share the same `w`, universe, and dates. The primary comparison uses fixed risk aversion without independent volatility targeting. Audit historical selection, drift-adjusted turnover, cash financing, and costs. Freeze the 2019-2025 design before using its results for evaluation.

**Planned entry point:** `backtest.run`.

## Step 8 - Evaluate economic value and uncertainty

**Owner:** Shuchen Meng

Estimate net `delta CE` and a two-sided 95% interval using 1,000 paired moving-block bootstrap samples. Summarize Sharpe ratio, volatility, 95% expected shortfall, maximum drawdown, turnover, and gross exposure. Report historical and subsequent evaluation periods separately.

**Deliverables:** `output/performance_table.csv`, `output/delta_ce_bootstrap.csv`, and `output/equity_curves.png`. Prepare the empirical findings and limitations sections of the report and the corresponding presentation material.

**Completion criteria:** use 12-month blocks for the primary interval and 6- and 24-month blocks for sensitivity checks. Resample the two strategies together. Do not interpret statistical insignificance as economic equivalence or confuse theoretical compression loss with empirical net `delta CE`.

**Planned entry point:** `evaluation.run`.

## Step 9 - Check deployment sensitivity

**Owner:** Shuchen Meng

Reuse the shrinkage parameters selected in the primary empirical analysis. Apply the same 10% predicted-volatility targeting rule to each portfolio, then reduce gross exposure to at most 3. Report how often the cap binds. Additional cost scenarios, factor attribution, and reliability-history permutations remain optional follow-ups.

**Deliverable:** `output/deployment_sensitivity.csv`.

**Completion criteria:** do not retune `w` separately for the heterogeneous strategy. Interpret the scaled and capped comparison separately from the unconstrained theoretical optimality result.

**Planned entry point:** `evaluation.deployment_sensitivity`.

## Step 10 - Assemble the final report and presentation

**Owner:** Shuchen Meng

Integrate Yuzhen's data and replication material, Yadan's theory and simulation material, and the empirical findings. Coordinate feedback, align the conclusions with the evidence, and assemble the final report and presentation. Each contributor supplies the material assigned in their earlier steps; Shuchen owns the final integration.

**Deliverables:** `output/result_manifest.json`, `output/report_summary.md`, the final report, and the presentation deck.

**Completion criteria:** every reported result is traceable to its input version, parameters, seed, code commit, and saved artifact. Resolve inconsistencies across sections and verify all tables and figures. Clearly distinguish synthetic experiments from empirical findings and theoretical guarantees from implementation results.

**Planned entry point:** `reporting.run`.

## Development and handoffs

- Yuzhen uses `feature/data-and-replication`; Yadan uses `feature/theory-and-simulation`; Shuchen uses `feature/reliability-and-oos`.
- Step 4 can proceed alongside Steps 1-3 after Step 0. Step 6 starts when the reusable Step 5 estimator is available.
- Submit small pull requests and request a teammate's review before merging. The step owner remains accountable for resolving feedback and completing the deliverables.
- Fill the notebook gradually with calls and explanations. Keep substantial calculations in source modules to reduce notebook conflicts.
- The repository owner adds collaborators. Agree on deadlines within the team rather than treating this workflow as a fixed schedule.

## Data and output conventions

- Use ISO month-end dates and decimal simple returns. Distinguish total returns, the risk-free rate, and excess returns; preserve missing observations.
- Positions formed at `t` earn returns in `t+1`. Store information cutoffs explicitly and avoid shifting returns twice.
- Store source data and reusable intermediate inputs under future `data/raw`, `data/processed`, and `data/interim` directories. Store generated tables, figures, diagnostics, and report artifacts under `output`.
- Add modules and notebook sections as their assigned development steps begin. Agree on data schemas, output formats, and version-control rules before adding generated files.
- Record input versions, parameters, seeds, code commits, and generation dates with the results. The tracked-data policy is documented in `data/README.md`; `risk_models.npz` is rebuilt locally.

## Interpretation boundaries

The nonnegative theoretical loss follows from the specified model and optimality criterion. Empirical net `delta CE` can be negative. The common-parameter empirical comparison evaluates the prescribed implementable rules; it does not establish superiority over every scalar rule. Report 1970-2018 and 2019-2025 separately. If the latter informed design choices, do not label it an untouched holdout.
