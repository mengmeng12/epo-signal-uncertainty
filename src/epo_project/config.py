"""Shared parameters, paths and timing conventions for the whole project.

Every module imports its settings from here so that all steps use the same
values. The numbers follow the revised proposal (doc/EPO_Revised_Proposal.docx);
changes must be agreed by the team and recorded in output/design_decisions.md.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INTERIM_DIR = DATA_DIR / "interim"
OUTPUT_DIR = ROOT / "output"

# ---------------------------------------------------------------- data sources
FRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
# key -> (zip file on the French site, csv inside the zip, marker of the table to read)
SOURCES = {
    "industry49_monthly": ("49_Industry_Portfolios_CSV.zip", "49_Industry_Portfolios.csv",
                           "Average Value Weighted Returns -- Monthly"),
    "ff3_monthly": ("F-F_Research_Data_Factors_CSV.zip", "F-F_Research_Data_Factors.csv", "Mkt-RF"),
}
MISSING_CODE_THRESHOLD = -99.99   # French uses -99.99 and -999 for missing returns (in percent)

# ---------------------------------------------------------------- sample periods (month-end, inclusive)
BENCHMARK_PERIOD = ("1942-01-31", "2018-12-31")    # Equity 1 implementation validation
EXTENSION_PERIOD = ("1970-01-31", "2018-12-31")    # historical evaluation of the extension
SUBSEQUENT_PERIOD = ("2019-01-31", "2025-12-31")   # frozen-design subsequent evaluation

# ---------------------------------------------------------------- signal and risk model
MOMENTUM_MONTHS = 12      # trailing compounded total return, months t-11..t
RISK_WINDOW = 60          # monthly excess returns t-59..t for the sample covariance
PRELIM_SHRINK = 0.05      # Q_t = 0.95 * Sigma_hat_t + 0.05 * V_t
MIN_INDUSTRIES = 5        # skip a formation date with fewer eligible industries

# ---------------------------------------------------------------- portfolio construction and selection
GAMMA = 3.0
W_GRID = (0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
SELECTION_MONTH = 1       # choose w each January ...
SELECTION_WINDOW = 60     # ... from the preceding 60 months of candidate returns ...
HOLD_MONTHS = 12          # ... and keep it for 12 months
TIE_BREAK = "higher_w"
TARGET_VOL = 0.10         # annualized predicted volatility (benchmark convention, deployment sensitivity)
GROSS_CAP = 3.0           # deployment sensitivity only

# ---------------------------------------------------------------- costs (extension)
COST_PER_DOLLAR = 0.0010          # 10 bp per dollar traded, drift-adjusted turnover
SHORT_FEE_ANNUAL = 0.0050         # 50 bp a year on short exposure, accrued monthly
COST_SENSITIVITY = (0.0, 0.0025)  # optional trading-cost scenarios

# ---------------------------------------------------------------- reliability estimates (extension)
REGRESSION_PAIRS = 120
BOOTSTRAP_DRAWS = 200
BOOTSTRAP_BLOCK = 12
H_WINSOR = (0.05, 0.95)
H_FLOOR = 1e-8

# ---------------------------------------------------------------- evaluation
EVAL_BOOTSTRAP_DRAWS = 1000
EVAL_BLOCK = 12
EVAL_BLOCK_SENSITIVITY = (6, 24)
ES_LEVEL = 0.95

# ---------------------------------------------------------------- simulation
SIM_N_ASSETS = 49
SIM_MONTHLY_VOL = 0.05
SIM_CORRELATION = 0.3
SIM_MEAN_LAMBDA = 0.1
SIM_TAU = 0.04            # mu ~ N(0, tau * Sigma): 1% marginal monthly s.d.
SIM_DRAWS = 1000
SIM_CV_LEVELS = (0.0, 0.5, 1.0)

# ---------------------------------------------------------------- benchmark diagnostics
PC_GROUPS = 5             # eigenvalue-rank groups, group 1 = smallest eigenvalues
