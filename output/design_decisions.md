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
| Missing month-t+1 return | If an industry held at t has no return in t+1, that month's portfolio returns stay missing; selection years whose 60-month window contains the month are skipped; all portfolios are evaluated over the same common months. One case (Rubbr, 1943-07) removes 1943-07 and 1944–1948, leaving 863 of 924 months | decided (team convention, proposal Appendix A) |

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

Over the 863 common months in 1942–2018 the out-of-sample EPO earns a Sharpe ratio of 0.770, against 0.79 in Pedersen, Babu and Levine (2021, Table 5, Equity 1). The data, signal, 60-month risk model and 5% correction are the same; the benchmark differs from the published procedure in three conventions and in the evaluation months. A one-off diagnostic (not part of the pipeline) changed the conventions one at a time over the same 863 months:

| Positions | Selection of w | Published grid | Proposal grid |
|---|---|---|---|
| Unscaled | Every month, all earlier months (published procedure) | 0.824 | 0.824 |
| Unscaled | Each January, preceding 60 months | 0.725 | 0.747 |
| Scaled to 10% predicted volatility | Every month, all earlier months | 0.829 | 0.831 |
| Scaled to 10% predicted volatility | Each January, preceding 60 months (proposal) | 0.765 | 0.770 |

- **Selection rule (about −0.06 to −0.08).** A Sharpe ratio estimated from 60 months has a standard error near 0.5, larger than the true differences across candidates (0.59 at w = 0 to 0.83 at w = 0.95 for fixed w). The rolling rule therefore often picks weak values: w = 0 in 11 of the 72 selection years 1942–2018 (1944–1948 have no selection), against never below 0.25 with the expanding window.
- **Volatility scaling (about +0.01 to +0.02).** Equalizing predicted risk across months changes how much each month contributes to the Sharpe ratio; raw MVO is affected most (0.23 unscaled, 0.32 scaled) because near-singular Σ̂ produces extreme unscaled positions in some months.
- **Grid (about 0.02 at most).** Dropping 0.10 and 1.00 and adding 0.95 matters only under the rolling rule.
- **Anchor.** With a single λ, the signal-based anchor and a zero anchor give positions that differ by the positive factor (1 − w), which the volatility scaling removes; the benchmark results do not depend on this choice.
- **Evaluation months.** With the published procedure, the 863 common months give Sharpe ratios 0.02–0.07 above the published values (out-of-sample EPO 0.824 against 0.79). Evaluating all 924 months instead, with the single missing return set to zero, brings every published-procedure row within 0.03 of the paper (out-of-sample EPO 0.797). Dropping 1944–1948 therefore accounts for part of the gap to the published numbers.
