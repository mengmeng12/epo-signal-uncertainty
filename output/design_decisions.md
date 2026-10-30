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

## Holdout status

Earlier exploratory work by Yuzhen (outside this repository) downloaded data through August 2026 but computed and inspected results only for 1942–2018; no design choice used 2019 or later. The 2019–2025 period has not informed the design so far.

### Step 5 handoff update (2026-10-30)

The preceding paragraph records Yuzhen's earlier work. Isabella's separate local quick experiment has since computed and inspected 2019–2025 empirical results. The team should now call this a **subsequent evaluation period**, and should not describe it as an untouched holdout. No Step 7 portfolio results are added by this commit.

| Step 5 decision | Implementation / status |
|---|---|
| Training pairs at formation t | 120 consecutive pairs, z[t−120:t−1] against excess returns[t−119:t]; no imputation |
| OLS | Separate intercept and slope per industry; constant predictor (centered sum of squares ≤ 1e−14) uses intercept-only prediction |
| Bootstrap | 200 non-circular moving-block samples, length 12, same sampled pair dates for every industry; refit and evaluate at fixed z[t] |
| u and h | u = sample variance of bootstrap fitted means (ddof=1); no residual variance added. h uses u/diag(Q), cross-sectional 5/95% winsorization, floor 1e−8, then mean-one normalization; all u=0 gives h=1 |
| Risk variance | Latest 60 excess returns with ddof=1; equals diag(Q), so Step 5 does not need the full covariance cache |
| Seed | 20261029 combined with formation YYYYMM through SeedSequence; independent of Step 4's seed |
| Default admission | 120-pair eligibility, minimum_ready_months=1, following the shared Step 5 wording |
| Local admission sensitivity | minimum_ready_months=60 reproduces the local experiment's extra 60-month eligibility streak. Team alignment is required before Step 7; this is separate from the 60-month candidate selection history |
| Formation range | First eligible date through 2025-11, supporting target returns through 2025-12; earlier dates are cached for warmup |

Steps 1–4 calculations and tracked results are unchanged by Step 5. The [alignment review](step5_alignment_review.md) records the unresolved missing-next-return convention, simulation reporting units, and the local bankruptcy-exit rule that will need explicit treatment in Step 7.
