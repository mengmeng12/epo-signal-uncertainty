"""Intermediate evidence for the EPO research report; supplements the fixed primary design.

Run ``PYTHONPATH=src python -m epo_project.diagnostics`` after Steps 7–9.
All CSVs and figures are generated locally in output/diagnostics. No new raw
data are downloaded and primary forecasts/results are never overwritten.
Forecast-error diagnostics concern realized returns, not latent mean uncertainty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as bt, config as cfg, evaluation as ev
from .reliability import moving_block_indices


SEED = 20261105


def rank_correlation(x, y) -> float:
    pair = pd.DataFrame({"x": np.asarray(x), "y": np.asarray(y)}).dropna()
    if len(pair) < 3 or (pair.nunique() < 2).any():
        return np.nan
    return float(pair.rank().corr().iloc[0, 1])


def formation_mean(excess: pd.DataFrame, window: int = cfg.REGRESSION_PAIRS) -> pd.DataFrame:
    """Benchmark for target t+1: the last 120 completed returns, ending at formation t."""
    return excess.rolling(window, min_periods=window).mean()


def prepare_forecasts(total: pd.DataFrame, rf: pd.Series, forecasts: pd.DataFrame,
                      start=cfg.EXTENSION_PERIOD[0], end=cfg.SUBSEQUENT_PERIOD[1],
                      window: int = cfg.REGRESSION_PAIRS) -> pd.DataFrame:
    """Join each forecast to a past-only mean benchmark and its next-month realization."""
    if forecasts.duplicated(["date", "industry"]).any():
        raise ValueError("Duplicate formation/industry forecast.")
    frame = forecasts[forecasts.target_date.between(start, end)].copy()
    if frame.empty:
        raise ValueError("No forecasts in the requested evaluation period.")
    if (not frame.last_input_date.eq(frame.date).all() or not frame.response_end.eq(frame.date).all()
            or not frame.target_date.eq(frame.date + pd.offsets.MonthEnd()).all()):
        raise ValueError("Forecast has an invalid information cutoff or target date.")
    excess = total.sub(rf, axis=0)
    historical_mean = formation_mean(excess, window)
    columns = excess.columns.get_indexer(frame.industry)
    formation = excess.index.get_indexer(frame.date)
    target = excess.index.get_indexer(frame.target_date)
    if min(columns.min(), formation.min(), target.min()) < 0:
        raise ValueError("Unknown forecast industry/date or unavailable target month.")
    frame["historical_mean"] = historical_mean.to_numpy()[formation, columns]
    frame["realized_excess"] = excess.to_numpy()[target, columns]
    if not np.isfinite(frame[["s", "historical_mean", "realized_excess", "h", "u", "sigma2"]]).all().all():
        raise ValueError("Incomplete benchmark, forecast or evaluation return; no silent dropping/imputation.")
    if (frame.sigma2 <= 0).any() or (frame.h <= 0).any() or (frame.u < 0).any():
        raise ValueError("Invalid return or forecast variance / h.")
    frame["squared_error"] = (frame.realized_excess - frame.s) ** 2
    frame["benchmark_squared_error"] = (frame.realized_excess - frame.historical_mean) ** 2
    return frame.sort_values(["target_date", "industry"]).reset_index(drop=True)


def balanced_bins(values: pd.Series, bins: int = 10) -> pd.Series:
    """Nearly equal counts; caller supplies deterministic industry order to break ties."""
    if not np.isfinite(values).all() or bins < 1 or len(values) < bins:
        raise ValueError("Sorting bins require finite observations and at least one asset per bin.")
    return ((values.rank(method="first") - 1) * bins // len(values) + 1).astype(int)


def block_indices(length: int, key: int, year: int) -> np.ndarray:
    return moving_block_indices(length, cfg.EVAL_BOOTSTRAP_DRAWS, cfg.EVAL_BLOCK,
                                np.random.default_rng(np.random.SeedSequence([SEED, key, year])))


def sample_diagnostics(total: pd.DataFrame, rf: pd.Series, panel: pd.DataFrame) -> dict:
    excess = total.sub(rf, axis=0)
    rows, industries, monthly = [], [], []
    for period, (start, end) in ev.PERIODS.items():
        raw = excess.loc[start:end]
        sub = panel[panel.target_date.between(start, end)]
        counts = sub.groupby("target_date").size().reindex(raw.index)
        if counts.isna().any():
            raise ValueError(f"Missing forecast calendar months in {period}.")
        rows.append({"period": period, "months": len(raw), "raw_industry_months": raw.size,
                     "raw_missing_industry_months": int(raw.isna().sum().sum()),
                     "eligible_industry_months": len(sub), "min_eligible": counts.min(),
                     "mean_eligible": counts.mean(), "max_eligible": counts.max(),
                     "unique_eligible_industries": sub.industry.nunique()})
        monthly.append(pd.DataFrame({"date": raw.index, "period": period,
                                      "raw_available_industries": raw.notna().sum(axis=1).to_numpy(),
                                      "eligible_industries": counts.to_numpy()}))
        for industry, values in sub.groupby("industry", sort=True):
            returns = values.realized_excess
            industries.append({"period": period, "industry": industry, "months": len(values),
                                "first_target": values.target_date.min(), "last_target": values.target_date.max(),
                                "excess_mean_pct_year": 1200 * returns.mean(),
                                "volatility_pct_year": 100 * np.sqrt(12) * returns.std(ddof=1)})
    return {"sample_summary": pd.DataFrame(rows), "sample_monthly": pd.concat(monthly, ignore_index=True),
            "industry_descriptives": pd.DataFrame(industries)}


def forecast_diagnostics(panel: pd.DataFrame) -> dict:
    """Time-block inference treats each month's whole cross-section as one observation."""
    rows, sort_rows = [], []
    for date, frame in panel.groupby("target_date", sort=True):
        frame = frame.sort_values("industry")
        rows.append({"date": date, "n_assets": len(frame),
                     "sse_forecast": frame.squared_error.sum(), "sse_benchmark": frame.benchmark_squared_error.sum(),
                     "mse_forecast": frame.squared_error.mean(), "mse_benchmark": frame.benchmark_squared_error.mean(),
                     "rank_ic": rank_correlation(frame.s, frame.realized_excess),
                     "mean_forecast": frame.s.mean(), "mean_realized": frame.realized_excess.mean()})
        assigned = frame.assign(decile=balanced_bins(frame.s))
        for decile, group in assigned.groupby("decile"):
            sort_rows.append({"date": date, "decile": decile, "n_assets": len(group),
                              "mean_forecast": group.s.mean(), "mean_realized": group.realized_excess.mean()})
    monthly = pd.DataFrame(rows)
    sorts = pd.DataFrame(sort_rows)
    summaries, bins = [], []
    for period, (start, end) in ev.PERIODS.items():
        f = monthly[monthly.date.between(start, end)]
        indices = block_indices(len(f), 1, pd.Timestamp(start).year)
        numerator, denominator = f.sse_forecast.to_numpy(), f.sse_benchmark.to_numpy()
        if denominator.sum() <= 0:
            raise ValueError("OOS R-squared is undefined for a zero-error benchmark.")
        r2 = 1 - numerator.sum() / denominator.sum()
        boot_r2 = 1 - numerator[indices].sum(axis=1) / denominator[indices].sum(axis=1)
        ic = f.rank_ic.to_numpy()
        boot_ic = np.nanmean(ic[indices], axis=1)
        summaries.append({"period": period, "months": len(f), "industry_months": f.n_assets.sum(),
                          "rmse_forecast_bp_month": 1e4 * np.sqrt(numerator.sum() / f.n_assets.sum()),
                          "rmse_historical_mean_bp_month": 1e4 * np.sqrt(denominator.sum() / f.n_assets.sum()),
                          "pooled_oos_r2": r2, "oos_r2_ci_lower": np.quantile(boot_r2, .025),
                          "oos_r2_ci_upper": np.quantile(boot_r2, .975),
                          "equal_month_oos_r2": 1 - f.mse_forecast.mean() / f.mse_benchmark.mean(),
                          "mean_rank_ic": np.nanmean(ic), "ic_valid_months": np.isfinite(ic).sum(),
                          "rank_ic_ci_lower": np.quantile(boot_ic, .025), "rank_ic_ci_upper": np.quantile(boot_ic, .975)})
        subset = sorts[sorts.date.between(start, end)]
        wide = subset.pivot(index="date", columns="decile", values="mean_realized")
        expected_dates = pd.DatetimeIndex(f.date)
        if not wide.index.equals(expected_dates) or wide.isna().any().any():
            raise ValueError("Forecast sorts do not have a complete paired calendar.")
        for decile in wide:
            value = wide[decile].to_numpy()
            samples = value[indices].mean(axis=1)
            bins.append({"period": period, "decile": str(decile), "months": len(value),
                         "mean_realized_pct_month": 100 * value.mean(),
                         "ci_lower_pct_month": 100 * np.quantile(samples, .025),
                         "ci_upper_pct_month": 100 * np.quantile(samples, .975),
                         "mean_forecast_pct_month": 100 * subset[subset.decile == decile].mean_forecast.mean()})
        spread = (wide[10] - wide[1]).to_numpy()
        samples = spread[indices].mean(axis=1)
        bins.append({"period": period, "decile": "10-1", "months": len(spread),
                     "mean_realized_pct_month": 100 * spread.mean(), "ci_lower_pct_month": 100 * np.quantile(samples, .025),
                     "ci_upper_pct_month": 100 * np.quantile(samples, .975), "mean_forecast_pct_month": np.nan})
    return {"forecast_summary": pd.DataFrame(summaries), "forecast_monthly": monthly,
            "forecast_deciles": pd.DataFrame(bins), "forecast_decile_monthly": sorts}


def reliability_diagnostics(panel: pd.DataFrame, full_forecasts: pd.DataFrame) -> dict:
    # Use pre-evaluation histories for first-year lag correlations, rather than resetting in 1970/2019.
    h_history = full_forecasts.pivot(index="date", columns="industry", values="h")
    rows = []
    for target, frame in panel.groupby("target_date"):
        date = frame.date.iloc[0]
        h, ratio = frame.h, frame.u / frame.sigma2
        lo, hi = np.quantile(ratio, cfg.H_WINSOR)
        clipped = ratio.clip(lo, hi)
        row = {"date": target, "formation_date": date, "n_assets": len(frame),
               "h_mean": h.mean(), "h_cv": h.std(ddof=0) / h.mean(), "h_p05": h.quantile(.05),
               "h_p50": h.median(), "h_p95": h.quantile(.95), "u_mean": frame.u.mean(),
               "ratio_mean": ratio.mean(), "winsor_changed_fraction": ((ratio < lo) | (ratio > hi)).mean(),
               "floor_fraction": (clipped < cfg.H_FLOOR).mean(),
               "rank_corr_h_abs_signal": rank_correlation(h, frame.s.abs()),
               "rank_corr_h_asset_vol": rank_correlation(h, np.sqrt(frame.sigma2))}
        for label, values in (("u", frame.u), ("ratio", ratio)):
            for quantile, suffix in ((.05, "p05"), (.5, "p50"), (.95, "p95")):
                row[f"{label}_{suffix}"] = values.quantile(quantile)
        for lag in (1, 12):
            previous = date - pd.offsets.MonthEnd(lag)
            row[f"h_rank_corr_{lag}m"] = np.nan
            if previous in h_history.index:
                past = h_history.loc[previous].reindex(frame.industry).to_numpy()
                row[f"h_rank_corr_{lag}m"] = rank_correlation(h, past)
        rows.append(row)
    monthly = pd.DataFrame(rows)
    summary = []
    for period, (start, end) in ev.PERIODS.items():
        sub = monthly[monthly.date.between(start, end)]
        row = {"period": period, "months": len(sub)}
        for col in [c for c in monthly if c not in ("date", "formation_date", "n_assets")]:
            row["mean_" + col] = sub[col].mean()
        summary.append(row)
    annual = panel.assign(year=panel.target_date.dt.year).groupby(["industry", "year"]).agg(
        mean_h=("h", "mean"), months=("h", "size"), mean_u=("u", "mean"), mean_sigma2=("sigma2", "mean")
    ).reset_index()
    return {"reliability_summary": pd.DataFrame(summary), "reliability_monthly": monthly,
            "reliability_industry_year": annual}


def selection_diagnostics(selection: pd.DataFrame) -> pd.DataFrame:
    selected = selection[selection.selected]
    if selected.year.duplicated().any():
        raise ValueError("Multiple selected parameters in a year.")
    rows = []
    for period, (start, end) in ev.PERIODS.items():
        years = selected[selected.year.between(pd.Timestamp(start).year, pd.Timestamp(end).year)]
        for w in cfg.W_GRID:
            candidates = selection[(selection.candidate_w == w) & selection.year.isin(years.year)]
            count = int((years.selected_w == w).sum())
            rows.append({"period": period, "w": w, "selected_years": count, "fraction_years": count / len(years),
                         "eligible_years": int(candidates.eligible.sum()), "evaluation_years": len(years)})
    return pd.DataFrame(rows)


def portfolio_geometry(scalar: np.ndarray, heterogeneous: np.ndarray, q: np.ndarray) -> dict:
    gs, gh = np.abs(scalar).sum(), np.abs(heterogeneous).sum()
    vs, vh = float(scalar @ q @ scalar), float(heterogeneous @ q @ heterogeneous)
    return {"gross_ratio_het_scalar": gh / gs if gs > 0 else np.nan,
            "predicted_vol_ratio_het_scalar": np.sqrt(vh / vs) if vs > 0 else np.nan,
            "raw_weight_l1_distance": np.abs(heterogeneous - scalar).sum(),
            "gross_normalized_l1_distance": np.abs(heterogeneous / gh - scalar / gs).sum() if min(gs, gh) > 0 else np.nan,
            "risk_direction_correlation": scalar @ q @ heterogeneous / np.sqrt(vs * vh) if min(vs, vh) > 0 else np.nan}


def portfolio_diagnostics(total, rf, forecasts, returns, weights) -> dict:
    returns = ev.validate_matched(returns)
    by_month = {key: group.set_index("industry").reindex(total.columns) for key, group in weights.groupby(["date", "strategy"])}
    monthly, industry_rows = [], []
    lookup = returns.set_index(["date", "strategy"])
    for month in bt.formation_inputs(total, rf, forecasts, cfg.EXTENSION_PERIOD[0], cfg.SUBSEQUENT_PERIOD[1]):
        x = {name: by_month[(month.target_date, name)].weight.to_numpy() for name in ev.STRATEGIES}
        if not all(np.isfinite(values).all() for values in x.values()):
            raise ValueError("Incomplete saved holdings.")
        row = {"date": month.target_date, **portfolio_geometry(x["scalar"][month.ids], x["heterogeneous"][month.ids], month.q)}
        for name in ev.STRATEGIES:
            position = x[name]
            local = position[month.ids]
            gross, variance = np.abs(position).sum(), float(local @ month.q @ local)
            stored = lookup.loc[(month.target_date, name)]
            np.testing.assert_allclose(np.sqrt(12 * variance), stored.predicted_vol, rtol=1e-9, atol=1e-12)
            for col in ("gross", "short", "cash", "turnover", "predicted_vol"):
                row[f"{name}_{col}"] = stored[col]
            row[f"{name}_net_exposure"] = position.sum()
            row[f"{name}_effective_holdings"] = 1 / np.sum((position / gross) ** 2) if gross > 0 else np.nan
            row[f"{name}_largest_abs_weight_share"] = np.abs(position).max() / gross if gross > 0 else np.nan
            held = position != 0
            if not np.isfinite(month.realized_total[held]).all():
                raise ValueError("Held asset has no realized return.")
            contribution = np.zeros(len(position))
            contribution[held] = position[held] * (month.realized_total[held] - month.rf)
            np.testing.assert_allclose(contribution.sum(), stored.gross_excess, rtol=1e-9, atol=1e-12)
            risk = np.zeros(len(position))
            if variance > 0:
                risk[month.ids] = local * (month.q @ local) / variance
            industry_rows.append(pd.DataFrame({"date": month.target_date, "strategy": name, "industry": total.columns,
                                                "weight": position, "gross_excess_contribution": contribution,
                                                "predicted_variance_share": risk}))
        monthly.append(row)
    monthly = pd.DataFrame(monthly)
    industry_month = pd.concat(industry_rows, ignore_index=True)
    summaries, industries = [], []
    for period, (start, end) in ev.PERIODS.items():
        frame = monthly[monthly.date.between(start, end)]
        summaries.append({"period": period, "months": len(frame), **{f"mean_{col}": frame[col].mean() for col in frame if col != "date"}})
        part = industry_month[industry_month.date.between(start, end)]
        aggregate = part.groupby(["strategy", "industry"]).agg(
            mean_weight=("weight", "mean"), mean_abs_weight=("weight", lambda x: x.abs().mean()),
            mean_predicted_variance_share=("predicted_variance_share", "mean"),
            gross_excess_contribution_bp_year=("gross_excess_contribution", lambda x: 12e4 * x.mean()), months=("date", "size")
        ).reset_index()
        industries.append(aggregate.assign(period=period))
    return {"portfolio_monthly": monthly, "portfolio_summary": pd.DataFrame(summaries),
            "industry_contributions": pd.concat(industries, ignore_index=True)}


def ce_decomposition(returns: pd.DataFrame) -> pd.DataFrame:
    returns = ev.validate_matched(returns)
    rows = []
    for period, (start, end) in ev.PERIODS.items():
        sub = returns[returns.date.between(start, end)]
        scalar, het = (sub[sub.strategy == name] for name in ev.STRATEGIES)
        gross = 12e4 * (het.gross_excess.mean() - scalar.gross_excess.mean())
        trading = -12e4 * (het.trading_cost.mean() - scalar.trading_cost.mean())
        borrowing = -12e4 * (het.borrow_cost.mean() - scalar.borrow_cost.mean())
        variance = -cfg.GAMMA / 2 * 12e4 * (het.net_excess.var(ddof=1) - scalar.net_excess.var(ddof=1))
        delta = 1e4 * (ev.certainty_equivalent(het.net_excess.to_numpy()) - ev.certainty_equivalent(scalar.net_excess.to_numpy()))
        np.testing.assert_allclose(gross + trading + borrowing + variance, delta, atol=1e-8, rtol=1e-9)
        rows.append({"period": period, "gross_mean_contribution_bp_year": gross,
                     "trading_cost_contribution_bp_year": trading, "short_cost_contribution_bp_year": borrowing,
                     "net_variance_contribution_bp_year": variance, "delta_ce_bp_year": float(delta),
                     "reconciliation_error_bp_year": gross + trading + borrowing + variance - delta})
    return pd.DataFrame(rows)


def robustness_diagnostics(total, rf, forecasts, primary, selection) -> dict:
    paths = [primary.assign(scenario="primary_10bp")]
    settings = [("primary_10bp", 10., 1, "annual_scalar_ce")]
    for cost in cfg.COST_SENSITIVITY:
        name = f"frozen_w_{cost * 1e4:g}bp"
        result = bt.simulate_portfolios(total, rf, forecasts, fixed_selection=selection, trade_cost=cost)
        paths.append(result["returns"].assign(scenario=name))
        settings.append((name, cost * 1e4, 1, "frozen_primary_w"))
    delayed = bt.simulate_portfolios(total, rf, forecasts, minimum_ready_months=60)
    paths.append(delayed["returns"].assign(scenario="admission_60"))
    settings.append(("admission_60", 10., 60, "annual_scalar_ce"))
    combined = pd.concat(paths, ignore_index=True)
    rows = []
    for scenario, cost, admission, rule in settings:
        sub = ev.validate_matched(combined[combined.scenario == scenario])
        for period, (start, end) in ev.PERIODS.items():
            frame = sub[sub.date.between(start, end)]
            paired = frame.pivot(index="date", columns="strategy", values="net_excess")[list(ev.STRATEGIES)]
            seed = np.random.SeedSequence([cfg.EVAL_SEED, pd.Timestamp(start).year, cfg.EVAL_BLOCK])
            point, samples = ev.paired_ce_bootstrap(paired.to_numpy(), seed=seed)
            lo, hi = 1e4 * np.quantile(samples, [.025, .975])
            for name in ev.STRATEGIES:
                part = frame[frame.strategy == name]
                rows.append({"scenario": scenario, "period": period, "strategy": name,
                             "trading_cost_bp_per_dollar": cost, "short_fee_bp_year": cfg.SHORT_FEE_ANNUAL * 1e4,
                             "minimum_ready_months": admission, "selection_rule": rule, **ev.performance_metrics(part),
                             "delta_ce_bp_year": point * 1e4, "ci_lower_bp_year": lo, "ci_upper_bp_year": hi})
    return {"robustness_summary": pd.DataFrame(rows), "robustness_returns": combined,
            "admission_60_selection": delayed["selection"], "admission_60_exits": delayed["exits"]}


def run(output_dir: Path | None = None, make_plots: bool = True) -> dict:
    destination = Path(output_dir) if output_dir is not None else cfg.OUTPUT_DIR / "diagnostics"
    destination.mkdir(parents=True, exist_ok=True)
    total, rf, forecasts = bt.load_market()
    returns = pd.read_csv(cfg.OUTPUT_DIR / "matched_returns.csv", parse_dates=["date", "formation_date"])
    weights = pd.read_csv(cfg.OUTPUT_DIR / "weights.csv", parse_dates=["date", "formation_date"])
    selection = pd.read_csv(cfg.OUTPUT_DIR / "annual_selection.csv")
    panel = prepare_forecasts(total, rf, forecasts)
    tables = sample_diagnostics(total, rf, panel)
    tables.update(forecast_diagnostics(panel))
    tables.update(reliability_diagnostics(panel, forecasts))
    tables["selection_frequency"] = selection_diagnostics(selection)
    tables.update(portfolio_diagnostics(total, rf, forecasts, returns, weights))
    tables["ce_decomposition"] = ce_decomposition(returns)
    tables.update(robustness_diagnostics(total, rf, forecasts, returns, selection))
    for name, frame in tables.items():
        frame.to_csv(destination / f"{name}.csv", index=False, date_format="%Y-%m-%d", float_format="%.12g")
    plots = []
    if make_plots:
        from .diagnostic_plots import plot_all
        intervals = pd.read_csv(cfg.OUTPUT_DIR / "delta_ce_summary.csv")
        plots = plot_all(tables, selection, intervals, destination)
    inputs = [cfg.PROCESSED_DIR / "returns.csv", cfg.INTERIM_DIR / "forecast_reliability.csv",
              *[cfg.OUTPUT_DIR / name for name in ("matched_returns.csv", "weights.csv", "annual_selection.csv", "delta_ce_summary.csv")]]
    code = [cfg.ROOT / f"src/epo_project/{name}.py" for name in ("config", "data", "backtest", "evaluation", "reliability", "diagnostics", "diagnostic_plots")]
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"periods": ev.PERIODS, "benchmark_window_months": cfg.REGRESSION_PAIRS,
                "diagnostic_bootstrap_seed": SEED, "draws": cfg.EVAL_BOOTSTRAP_DRAWS, "block_months": cfg.EVAL_BLOCK,
                "cost_scenarios_bp": [0, 10, 25], "cost_selection": "frozen primary w",
                "admission_sensitivity": "60 ready months; fully rerun candidate warmup and common-w selection",
                "input_code_sha256": {str(p.relative_to(cfg.ROOT)): sha(p) for p in inputs + code},
                "output_sha256": {p.name: sha(p) for p in [*[destination / f"{name}.csv" for name in tables], *plots]}}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"tables": tables, "figures": plots, "manifest": manifest}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = run(args.output_dir, make_plots=not args.no_plots)
    print(f"Generated {len(result['tables'])} tables and {len(result['figures'])} figures.")
