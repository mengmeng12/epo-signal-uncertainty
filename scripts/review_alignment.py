"""Reproduce the Steps 3/4 review without overwriting their tracked outputs.

Run from the repository root with PYTHONPATH=src. Optionally pass
--local-raw /path/to/EPO_Local_Walkthrough/data/raw to compare data vintages.
The review uses the implementation inherited from 667e8d6.
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from epo_project import config as cfg, data, replication, rolling, simulation


def review(local_raw=None):
    original_interim = cfg.INTERIM_DIR
    try:
        with TemporaryDirectory(prefix="epo-alignment-") as temporary:
            cfg.INTERIM_DIR = Path(temporary)
            data.build_inputs()
            panel, _, _, _ = replication.candidate_panel()
            _, _, models = data.load_inputs()
    finally:
        cfg.INTERIM_DIR = original_interim
    excess = data.load_returns("excess_return")
    missing = []
    for date, (members, _, _) in models.items():
        target = date + pd.offsets.MonthEnd()
        if target in excess.index:
            bad = excess.loc[target, members].isna()
            if bad.any():
                missing.append({"formation_date": str(date.date()),
                                "return_date": str(target.date()),
                                "industries": list(bad[bad].index)})
    masked = panel.copy()
    for row in missing:
        masked.loc[pd.Timestamp(row["return_date"])] = np.nan
    selections, comparisons = {}, {}
    for label, returns in [("zero", panel), ("missing", masked)]:
        selected = rolling.annual_selection(returns[list(cfg.W_GRID)])
        selections[label] = selected
        oos = rolling.apply_selection(returns[list(cfg.W_GRID)], selected)
        dates = oos.loc[slice(*cfg.BENCHMARK_PERIOD)].dropna().index
        comparisons[label] = {
            "months": len(dates),
            "raw": rolling.annualized_sharpe(returns.loc[dates, "raw_mvo"]),
            "corrected": rolling.annualized_sharpe(returns.loc[dates, 0.0]),
            "oos": rolling.annualized_sharpe(oos.loc[dates, "return"]),
        }
    result = {
        "reviewed_commits": {"data": "f789590", "theory": "667e8d6"},
        "missing_next_returns": missing, "same_vintage_missing_comparison": comparisons,
        "excluded_selection_years": sorted(set(selections["zero"].loc[1942:2018].index)
                                            - set(selections["missing"].loc[1942:2018].index)),
    }
    if local_raw is not None:
        original_raw = cfg.RAW_DIR
        try:
            cfg.RAW_DIR = Path(local_raw)
            local = data.read_french_table("49_Industry_Portfolios.csv",
                                           "Average Value Weighted Returns -- Monthly")
            local_rf = data.read_french_table("F-F_Research_Data_Factors.csv", "Mkt-RF")["RF"]
        finally:
            cfg.RAW_DIR = original_raw
        raw = data.load_returns("total_return")
        common = raw.index.intersection(local.index)
        diff = (raw.loc[common] - local.loc[common]).loc[:"2025-12-31"]
        rf_diff = data.load_returns("rf").loc[diff.index] - local_rf.loc[diff.index]
        assert raw.loc[diff.index].isna().equals(local.loc[diff.index].isna())
        result["vintage_comparison_through_2025"] = {
            "common_months": len(diff), "missing_masks_equal": True,
            "changed_industry_months": int((diff.abs() > 1e-12).sum().sum()),
            "max_abs_return_change": float(diff.abs().max().max()),
            "changed_rf_months": int((rf_diff.abs() > 1e-12).sum()),
        }
    rng = np.random.default_rng(cfg.SIM_SEED)
    profiles = {cv: simulation.uncertainty_profile(cv, rng) for cv in cfg.SIM_CV_LEVELS}
    z_mean = rng.standard_normal((cfg.SIM_DRAWS, cfg.SIM_N_ASSETS))
    z_error = rng.standard_normal((cfg.SIM_DRAWS, cfg.SIM_N_ASSETS))
    sigma = simulation.model_covariance()
    family = simulation.ScalarFamily(sigma)
    convergence = []
    for cv, h in profiles.items():
        uncertainty = cfg.SIM_MEAN_LAMBDA * np.diag(sigma) * h
        mu = z_mean[:30] @ np.linalg.cholesky(cfg.SIM_TAU * sigma).T
        signal = mu + z_error[:30] * np.sqrt(uncertainty)
        posterior = signal @ np.linalg.solve(cfg.SIM_TAU * sigma + np.diag(uncertainty),
                                             cfg.SIM_TAU * sigma)
        b, g = family.rotate(signal, posterior)
        _, coarse = simulation.oracle_w(family, b, g)
        _, dense = simulation.oracle_w(family, b, g, points=4001)
        convergence.append({"target_cv": cv,
                            "max_utility_change": float(np.abs(dense - coarse).max())})
    result["oracle_grid_check_30_draws_per_level"] = convergence
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-raw", type=Path)
    args = parser.parse_args()
    findings = review(args.local_raw)
    destination = cfg.OUTPUT_DIR / "step5_alignment_audit.json"
    destination.write_text(json.dumps(findings, indent=2) + "\n")
    print(json.dumps(findings, indent=2))
