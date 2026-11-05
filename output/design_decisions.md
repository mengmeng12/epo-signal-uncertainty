# Design decisions

Step 0 record of the shared conventions. Parameter values live in `src/epo_project/config.py` and follow the revised proposal (`doc/EPO_Revised_Proposal.docx`, October 23, 2026). Items marked **proposed** were chosen while implementing Steps 1–3 and await team confirmation.

## Environment

- Python 3.10 with the packages pinned in `requirements.txt`.
- Run the modules from the repository root with `src` on the path, e.g. `PYTHONPATH=src python -m epo_project.data`.

## Data

| Decision | Choice | Status |
|---|---|---|
| Industry returns | French 49 Industry Portfolios, average value-weighted, monthly | proposal |
| Risk-free rate | Monthly `RF` from `F-F_Research_Data_Factors` (starts 1926; the five-factor file starts in 1963) | proposed |
| Vintage | Downloaded 2026-10-23; both files state "created using the 202608 CRSP database" | recorded |
| Units | Percent converted to decimal simple returns | proposal |
| Missing values | French codes (≤ −99.99) set to missing and never filled; 2,868 industry-months are missing | proposal |
| Excess return | Industry total return minus RF of the same month | proposed |
| Dates | ISO month-end | workflow |
| Tracked data | All files under `data/` except `data/interim/risk_models.npz` (its zip headers depend on the Python version); see `data/README.md` for what a rerun should change | decided (Yuzhen, data upload owner) |

## Signal, risk model and eligibility

| Decision | Choice | Status |
|---|---|---|
| Trailing return | Compounded total return over months t−11..t; all 12 months required | proposed |
| Momentum score z | Trailing return minus its cross-sectional mean; positive and negative legs scaled to +1 and −1 | proposal |
| Universe for z | Industries eligible for the benchmark at t (below) | proposed |
| Covariance | Sample covariance (ddof = 1) of excess returns over months t−59..t | proposal |
| Preliminary correction | Q = 0.95 Σ̂ + 0.05 V, V = diag(Σ̂); diagonal unchanged | proposal |
| Benchmark eligibility | At least 60 consecutive non-missing months ending at t (implies the 12 months for momentum) | proposed |
| Minimum universe | Skip a formation date with fewer than 5 eligible industries (never binds: 40–49 industries) | proposed |
| Missing month-t+1 return | An eligible industry whose t+1 return is missing earns 0 that month (1 case) instead of being dropped with hindsight | proposed |

The extension adds its own requirement of 120 completed regression pairs; `eligibility.csv` stores `consecutive_months` so any history rule can be derived without recomputing.

## Timing and selection

- Positions formed at month-end t use data through t and earn month t+1.
- Candidate-return tables are indexed by the realization month t+1.
- Each January of year Y the parameter is chosen from candidate returns realized in Jan Y−5 .. Dec Y−1 (60 complete, contiguous months) and applied to the realizations of Jan Y .. Dec Y, i.e. positions formed Dec Y−1 .. Nov Y. Ties go to the higher w.
- Sharpe ratio: mean / standard deviation (ddof = 1) × √12. Certainty equivalent: 12·mean − (γ/2)·12·Var with γ = 3.

## Benchmark (Equity 1) conventions

- Portfolios: raw MVO on Σ̂, MVO on Q (EPO w = 0), and scalar EPO with (1 − w)Q + wV for w in the proposal grid, all on the same z.
- Each portfolio is scaled to 10% annualized predicted volatility under Q_t (proposal's common benchmark convention).
- `replication_table.csv` lists the published Equity 1 Sharpe ratios next to ours to document differences; the published grid has no w = 0.95.

### Differences from the published results

Over 1942–2018 the out-of-sample EPO earns a Sharpe ratio of 0.751, against 0.79 in Pedersen, Babu and Levine (2021, Table 5, Equity 1). The data, signal, 60-month risk model and 5% correction are the same; the benchmark differs from the published procedure in three conventions. A one-off diagnostic (not part of the pipeline) changed them one at a time:

| Positions | Selection of w | Published grid | Proposal grid |
|---|---|---|---|
| Unscaled | Every month, all earlier months (published procedure) | 0.797 | 0.797 |
| Unscaled | Each January, preceding 60 months | 0.709 | 0.730 |
| Scaled to 10% predicted volatility | Every month, all earlier months | 0.807 | 0.807 |
| Scaled to 10% predicted volatility | Each January, preceding 60 months (proposal) | 0.745 | 0.751 |

- **Selection rule (about −0.06).** A Sharpe ratio estimated from 60 months has a standard error near 0.5, larger than the true differences across candidates (0.57 at w = 0 to 0.80 at w = 0.90 for fixed w). The rolling rule therefore often picks weak values: w = 0 in 11 of the 77 selection years 1942–2018, against never below 0.25 with the expanding window.
- **Volatility scaling (about +0.01).** Equalizing predicted risk across months changes how much each month contributes to the Sharpe ratio; raw MVO is affected most (0.22 unscaled, 0.30 scaled) because near-singular Σ̂ produces extreme unscaled positions in some months.
- **Grid (about 0.02 at most).** Dropping 0.10 and 1.00 and adding 0.95 matters only under the rolling rule.
- **Anchor.** With a single λ, the signal-based anchor and a zero anchor give positions that differ by the positive factor (1 − w), which the volatility scaling removes; the benchmark results do not depend on this choice.

With the published procedure our implementation matches the published Sharpe ratios within 0.03 for every reported portfolio (out-of-sample EPO 0.797 against 0.79).

## Evaluation periods

Report 1970–2018 and 2019–2025 separately. Each forecast and annual parameter selection uses information available at its formation date; portfolio holdings continue across the period boundary.

### Step 5 update (2026-10-30)

| Step 5 decision | Implementation / status |
|---|---|
| Training pairs at formation t | 120 consecutive pairs, z[t−120:t−1] against excess returns[t−119:t]; no imputation |
| OLS | Separate intercept and slope per industry; constant predictor (centered sum of squares ≤ 1e−14) uses intercept-only prediction |
| Bootstrap | 200 non-circular moving-block samples, length 12, same sampled pair dates for every industry; refit and evaluate at fixed z[t] |
| u and h | u = sample variance of bootstrap fitted means (ddof=1); no residual variance added. h uses u/diag(Q), cross-sectional 5/95% winsorization, floor 1e−8, then mean-one normalization; all u=0 gives h=1 |
| Risk variance | Latest 60 excess returns with ddof=1; equals diag(Q), so Step 5 does not need the full covariance cache |
| Seed | 20261029 combined with formation YYYYMM through SeedSequence; independent of Step 4's seed |
| Default admission | 120-pair eligibility, minimum_ready_months=1, following the shared Step 5 wording |
| Local admission sensitivity | minimum_ready_months=60 reproduces the local experiment's extra 60-month eligibility streak; this is separate from the 60-month candidate selection history. Step 7 uses the shared default of 1 |
| Formation range | First eligible date through 2025-11, supporting target returns through 2025-12; earlier dates are cached for warmup |

Steps 1–4 calculations and tracked results were unchanged by Step 5. The implemented empirical accounting and candidate-exit rules are recorded under Steps 7–9 below.

### Step 6 estimator check (2026-11-01)

| Step 6 decision | Implementation / status |
|---|---|
| Model | z[t+1] = 0.92 z[t] + 0.056·√(1−0.92²)·η, r[t+1] = 0.005 + 0.10 z[t] + e[t+1]; true mean a + b z[t] is known. Persistence and signal s.d. match the tracked momentum panel |
| Cross-section | 49 assets, residual vols evenly spaced 4%–8% a month, residual correlation 0.3; predictor innovations independent across assets |
| Feedback | corr(η[t], e[t]) = 0 (the estimator's model) and 0.4 (near the empirical median of 0.39 between the momentum innovation and the same-month excess return), on common random numbers |
| Estimator | `reliability.estimate_forecasts` unchanged: 120 pairs, 200 moving blocks of 12, asset variance from the latest 60 training returns |
| Replications and seed | 1,000 histories, one formation date each; seed 20261101 split by SeedSequence into a data stream and per-history bootstrap streams |
| Inference | Monte Carlo standard errors clustered by history, since assets share shocks and bootstrap dates |

Results (`output/estimator_calibration.csv`, `output/estimator_calibration.png`), feedback 0 with feedback 0.4 in brackets:

- The simulation is consistent: actual MSE / analytic OLS variance = 1.002 ± 0.011, and intervals built from the exact variance cover 95.2%.
- u is close to unbiased on average: actual MSE / mean u = 0.94 ± 0.01 (0.91), so u slightly overstates error, most at high-leverage points. Forecast bias is −0.4 ± 1.0 bp (−0.5).
- u is noisy: the s.d. of u / true variance is 0.49 (0.55). Resampling noise from 200 draws accounts for only about 0.10 (√(2/199)); the rest is variation of the bootstrap variance across histories, so more draws would not remove it. Normal 95% intervals cover 92.4% (92.8%), and percentile intervals 89.8% (88.8%). Sorting on u exaggerates both tails: the lowest u decile covers 85%, the highest 97%.
- Ranking: Spearman correlation of u with the true variance across assets is 0.84 (0.83); of h with the true relative uncertainty, 0.77 (0.75). h therefore carries real but imperfect information about relative uncertainty, which Step 4 treats as known.
- Estimation error is small next to return noise: u is about 2% of the realized forecast MSE.
- Feedback 0.4 changes these numbers little.

These check the implemented estimator under a known linear model; they do not show that u measures forecast uncertainty in market data.

### Steps 7–9 empirical implementation (2026-11-05)

| Decision | Implementation |
|---|---|
| Universe | Shared Step 5 default: 120 completed consecutive pairs and current signal/risk history; no additional 60-month industry admission delay. h is recomputed within the chosen universe from u/diag(Q) |
| Primary weights | x = (Q + kappa diag(diag(Q)·h))⁻¹s / 3; scalar uses h=1. kappa=w/(1−w). No sum-to-one normalization, cap or volatility targeting |
| Candidate warmup | Fixed-w portfolios start from cash for realized January 1960; each has its own drifting holdings and costs. Permanent exits depend on this fixed start date |
| Candidate insolvency | Total wealth factor ≤0 permanently excludes that fixed-w candidate; the failed month and all later candidate returns remain missing. The supplied data produce exits at w=0 in June 1960 and w=0.25 in December 1961 |
| Annual selection | Before earning each January return, compare surviving scalar candidates with all 60 preceding calendar months available; choose maximum annualized net CE, ties within 1e−12 favor higher w. Use that w for both strategies for the calendar year |
| Actual paths | Start from cash in January 1970 and run continuously through December 2025; retain December 2018 holdings for January 2019. Insolvency on either selected path raises an error |
| Cash and costs | Residual cash = 1−sum(target weights), earning/paying RF. Full L1 turnover from net-wealth drifted holdings costs 10 bp per dollar; shorts cost 50 bp/year divided by 12. Fees are deducted at month end. No extra leverage financing spread |
| Missing next returns | A missing return on a held industry raises an error; unheld missing returns have no portfolio effect. No held missing return occurs in the completed 1960–2025 run |
| Risk summaries | Net excess returns for CE/Sharpe/volatility and monthly ES; compounded net total returns for drawdown. Detailed units and ES boundary convention are in data/README.md |
| Paired inference | 1,000 moving-block draws, seed 20261105 keyed by period start year and block length. 12-month blocks primary; 6/24-month sensitivity. Percentile 95% intervals for annualized net delta CE; no parameter retuning in bootstrap draws |
| Deployment | Reuse primary w and the same monthly inputs. Scale each allocation to 10% annual predicted volatility under Q, then reduce gross exposure to at most 3. Recompute trading, drift and short fees on deployed holdings; report cap frequency |

Steps 7–9 can run directly from the tracked forecast and return CSVs. Step 10 report/presentation integration is deferred. Tests cover accounting against a hand-worked cash/long/short example, missing data, insolvency, permanent candidate exits, selection timing, allocation scaling, paired resampling, and the risk metrics. A separate local cross-check with minimum_ready_months=60 reproduces the original walkthrough's 672 monthly returns and annual selections to numerical precision; the tracked primary results use minimum_ready_months=1.

Step 4 reproducibility: alignment terciles now use stable sorting and equal-count splits (334/333/333 for 1,000 draws), retaining the original saved group convention. This removes pandas/NumPy quantile-boundary differences; profiles, draws and the utility calculation are unchanged. Ties are split in original draw order.
