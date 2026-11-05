"""Steps 8–9: paired net-CE inference, risk metrics and deployment sensitivity.

CE uses monthly net excess returns and sample variance, annualized arithmetically.
Expected shortfall is a one-month net-excess loss; drawdown uses net total wealth.
Bootstrap intervals resample the two realized strategies together; the strategy
selection process is not rerun within a bootstrap draw.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from . import backtest, config as cfg
from .reliability import moving_block_indices


PERIODS = {"1970-2018": cfg.EXTENSION_PERIOD, "2019-2025": cfg.SUBSEQUENT_PERIOD}
STRATEGIES = ("scalar", "heterogeneous")


def certainty_equivalent(returns: np.ndarray, gamma: float = cfg.GAMMA) -> np.ndarray:
    """Annualized CE along the final (time) axis, in decimal return units."""
    r = np.asarray(returns, dtype=float)
    if r.ndim == 0 or r.shape[-1] < 2 or not np.isfinite(r).all():
        raise ValueError("CE requires at least two finite monthly returns.")
    return 12 * r.mean(axis=-1) - gamma / 2 * 12 * r.var(axis=-1, ddof=1)


def expected_shortfall(returns: np.ndarray, level: float = cfg.ES_LEVEL) -> float:
    """Mean loss in the worst 1-level probability mass of the empirical distribution.

    Fractional weight at the boundary gives exactly 5% tail mass even when
    0.05*n is not an integer. Positive values denote losses; no annualization.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or not r.size or not np.isfinite(r).all() or not 0 < level < 1:
        raise ValueError("ES needs a finite return vector and a level between zero and one.")
    losses = np.sort(-r)[::-1]
    mass = (1 - level) * len(r)
    full = int(np.floor(mass))
    return float((losses[:full].sum() + (mass - full) * losses[full]) / mass)


def wealth_and_drawdown(total_returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(total_returns, dtype=float)
    if r.ndim != 1 or not r.size or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError("Wealth requires finite returns with strictly positive growth.")
    wealth = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return wealth, wealth / peak - 1


def validate_matched(returns: pd.DataFrame) -> pd.DataFrame:
    """Require a complete paired monthly panel; never silently drop missing months."""
    if returns.duplicated(["date", "strategy"]).any() or set(returns.strategy) != set(STRATEGIES):
        raise ValueError("Require exactly two unique strategy observations per date.")
    ordered = returns.sort_values(["date", "strategy"]).copy()
    wide = ordered.pivot(index="date", columns="strategy", values="net_excess")
    expected = pd.date_range(wide.index[0], wide.index[-1], freq=pd.offsets.MonthEnd())
    if not wide.index.equals(expected) or not np.isfinite(wide.to_numpy()).all():
        raise ValueError("Matched returns must have every month and both strategies, without NaN.")
    numeric = ["rf", "gross_excess", "net_excess", "net_total", "turnover", "gross", "short",
               "trading_cost", "borrow_cost", "w", "n_assets"]
    if not np.isfinite(ordered[numeric].to_numpy()).all():
        raise ValueError("Return/position metrics must be finite.")
    for name in ("w", "n_assets", "rf"):
        if not ordered.groupby("date")[name].nunique().eq(1).all():
            raise ValueError(f"Strategies do not share {name} at every date.")
    if not np.allclose(ordered.net_total, ordered.net_excess + ordered.rf, rtol=0, atol=1e-10):
        raise ValueError("Total return does not reconcile to excess return plus RF.")
    if not np.allclose(ordered.net_excess, ordered.gross_excess - ordered.trading_cost - ordered.borrow_cost,
                       rtol=0, atol=1e-10):
        raise ValueError("Net returns do not reconcile to costs.")
    if (ordered.net_total <= -1).any():
        raise ValueError("An evaluated strategy became insolvent.")
    return ordered


def performance_metrics(frame: pd.DataFrame) -> dict:
    r = frame.net_excess.to_numpy()
    _, drawdown = wealth_and_drawdown(frame.net_total.to_numpy())
    sd = r.std(ddof=1)
    return {"months": len(frame), "net_ce_bp_year": float(1e4 * certainty_equivalent(r)),
            "net_excess_mean_pct_year": float(1200 * r.mean()),
            "net_sharpe": float(np.sqrt(12) * r.mean() / sd) if sd > 0 else np.nan,
            "volatility_pct_year": float(100 * np.sqrt(12) * sd),
            "es95_net_excess_loss_pct_month": 100 * expected_shortfall(r),
            "max_drawdown_pct": float(-100 * drawdown.min()),
            "mean_turnover_per_month": frame.turnover.mean(), "mean_gross": frame.gross.mean(),
            "max_gross": frame.gross.max(), "mean_short": frame.short.mean(),
            "trading_cost_bp_year": 12e4 * frame.trading_cost.mean(),
            "borrow_cost_bp_year": 12e4 * frame.borrow_cost.mean()}


def paired_ce_bootstrap(paired: np.ndarray, *, block_length: int = cfg.EVAL_BLOCK,
                        draws: int = cfg.EVAL_BOOTSTRAP_DRAWS, seed=cfg.EVAL_SEED,
                        sample_indices: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Input time × [scalar, heterogeneous]; return observed and bootstrap delta CE (decimal/year)."""
    paired = np.asarray(paired, dtype=float)
    if paired.ndim != 2 or paired.shape[1] != 2 or len(paired) < 2 or not np.isfinite(paired).all():
        raise ValueError("Paired bootstrap requires a finite time × two-strategy array.")
    if sample_indices is None:
        indices = moving_block_indices(len(paired), draws, block_length, np.random.default_rng(seed))
    else:
        indices = np.asarray(sample_indices)
        if (indices.ndim != 2 or indices.shape[1] != len(paired) or len(indices) < 2
                or not np.issubdtype(indices.dtype, np.integer)
                or (indices < 0).any() or (indices >= len(paired)).any()):
            raise ValueError("Invalid paired bootstrap indices.")
    observed = certainty_equivalent(paired.T)
    ce = certainty_equivalent(paired[indices].transpose(0, 2, 1))
    return float(observed[1] - observed[0]), ce[:, 1] - ce[:, 0]


def evaluate(returns: pd.DataFrame) -> dict:
    returns = validate_matched(returns)
    metrics, interval_rows, sample_rows, curves = [], [], [], []
    for label, (start, end) in PERIODS.items():
        part = returns[returns.date.between(start, end)]
        expected = pd.date_range(start, end, freq=pd.offsets.MonthEnd())
        if not pd.DatetimeIndex(part.date.unique()).equals(expected):
            raise ValueError(f"Incomplete evaluation period {label}.")
        paired = part.pivot(index="date", columns="strategy", values="net_excess")[list(STRATEGIES)]
        for block in (cfg.EVAL_BLOCK, *cfg.EVAL_BLOCK_SENSITIVITY):
            seed = np.random.SeedSequence([cfg.EVAL_SEED, pd.Timestamp(start).year, block])
            observed, samples = paired_ce_bootstrap(paired.to_numpy(), block_length=block, seed=seed)
            lower, upper = np.quantile(samples, [0.025, 0.975])
            interval_rows.append({"period": label, "months": len(paired), "block_months": block,
                                  "draws": len(samples), "delta_ce_bp_year": 1e4 * observed,
                                  "ci_lower_bp_year": 1e4 * lower, "ci_upper_bp_year": 1e4 * upper})
            sample_rows.append(pd.DataFrame({"period": label, "block_months": block,
                                             "draw": np.arange(len(samples)), "delta_ce_bp_year": samples * 1e4}))
        primary = next(row for row in interval_rows if row["period"] == label and row["block_months"] == cfg.EVAL_BLOCK)
        for strategy in STRATEGIES:
            frame = part[part.strategy == strategy]
            row = {"period": label, "strategy": strategy, **performance_metrics(frame)}
            # Delta and its interval are a pair comparison, shown only on the heterogeneous row.
            if strategy == "heterogeneous":
                row.update({k: primary[k] for k in ("delta_ce_bp_year", "ci_lower_bp_year", "ci_upper_bp_year")})
            metrics.append(row)
            wealth, drawdown = wealth_and_drawdown(frame.net_total.to_numpy())
            curves.append(pd.DataFrame({"period": label, "date": frame.date.to_numpy(), "strategy": strategy,
                                        "net_total": frame.net_total.to_numpy(), "wealth": wealth, "drawdown": drawdown}))
    return {"performance": pd.DataFrame(metrics), "intervals": pd.DataFrame(interval_rows),
            "bootstrap": pd.concat(sample_rows, ignore_index=True), "curves": pd.concat(curves, ignore_index=True)}


def plot_equity(curves: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, PercentFormatter, ScalarFormatter

    colors = {"scalar": "#2a78d6", "heterogeneous": "#eb6834"}
    styles = {"scalar": "-", "heterogeneous": "--"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), height_ratios=(1.4, 1), layout="constrained")
    for column, (period, (start, end)) in enumerate(PERIODS.items()):
        part = curves[curves.period == period]
        first = pd.Timestamp(start) - pd.offsets.MonthEnd()
        for name in STRATEGIES:
            series = part[part.strategy == name]
            dates = pd.DatetimeIndex([first, *series.date])
            axes[0, column].plot(dates, np.r_[1, series.wealth], color=colors[name],
                                 linestyle=styles[name], linewidth=1.5, label=name.capitalize())
            axes[1, column].plot(dates, np.r_[0, series.drawdown], color=colors[name],
                                 linestyle=styles[name], linewidth=1.5)
        axes[0, column].set(title=period, yscale="log", ylabel="Net total wealth (log scale; start = 1)")
        ticks_y = [1, 2, 5, 10, 20, 50] if column == 0 else [0.8, 1, 1.2, 1.4, 1.6]
        axes[0, column].set_yticks(ticks_y)
        axes[0, column].yaxis.set_major_formatter(ScalarFormatter())
        axes[0, column].yaxis.set_minor_formatter(NullFormatter())
        axes[1, column].set(ylabel="Drawdown from running peak")
        axes[1, column].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1, column].set_ylim(min(-0.05, curves.drawdown.min() * 1.08), 0.02)
        axes[0, column].legend(frameon=False, fontsize=9)
        ticks = ([pd.Timestamp("1970-01-01"), *[pd.Timestamp(f"{year}-01-01") for year in (1980, 1990, 2000, 2010)],
                  pd.Timestamp(end)] if column == 0 else
                 [pd.Timestamp(f"{year}-01-01") for year in (2019, 2021, 2023)] + [pd.Timestamp(end)])
        for ax in axes[:, column]:
            ax.set_xlim(first, pd.Timestamp(end))
            ax.set_xticks(ticks, [str(t.year) for t in ticks])
            ax.grid(axis="y", color="#e4e3df", linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=9)
    fig.suptitle("Matched portfolios: fixed gamma = 3, common selected w\n"
                 "10 bp trading cost; 50 bp annual short fee. Panels rebased for display; holdings continue in 2019.",
                 fontsize=11)
    fig.savefig(cfg.OUTPUT_DIR / "equity_curves.png", dpi=160)
    plt.close(fig)


def run(make_plot: bool = True) -> dict:
    returns = pd.read_csv(cfg.OUTPUT_DIR / "matched_returns.csv", parse_dates=["date", "formation_date"])
    result = evaluate(returns)
    for key, filename in [("performance", "performance_table.csv"), ("intervals", "delta_ce_summary.csv"),
                           ("bootstrap", "delta_ce_bootstrap.csv"), ("curves", "equity_curves.csv")]:
        backtest.save_csv(result[key], filename)
    if make_plot:
        plot_equity(result["curves"])
    return result


def deployment_sensitivity() -> dict:
    """Replay the primary selection, with its own drift/costs after targeting and capping."""
    selection = pd.read_csv(cfg.OUTPUT_DIR / "annual_selection.csv")
    result = backtest.simulate_portfolios(*backtest.load_market(), fixed_selection=selection, deployment=True)
    returns = validate_matched(result["returns"])
    primary = pd.read_csv(cfg.OUTPUT_DIR / "matched_returns.csv", parse_dates=["date"])
    matched = returns.merge(primary[["date", "strategy", "w", "n_assets"]], on=["date", "strategy"],
                            suffixes=("", "_primary"), validate="one_to_one")
    if len(matched) != len(primary) or not (matched.w == matched.w_primary).all() or not (matched.n_assets == matched.n_assets_primary).all():
        raise ValueError("Deployment must reuse primary dates, universe and selected w.")
    if returns.gross.max() > cfg.GROSS_CAP + 1e-10 or returns.predicted_vol.max() > cfg.TARGET_VOL + 1e-10:
        raise ValueError("Deployment constraints violated.")
    metrics = []
    for period, (start, end) in PERIODS.items():
        part = returns[returns.date.between(start, end)]
        ce = {name: float(certainty_equivalent(part[part.strategy == name].net_excess.to_numpy())) for name in STRATEGIES}
        for name in STRATEGIES:
            frame = part[part.strategy == name]
            metrics.append({"period": period, "strategy": name, **performance_metrics(frame),
                            "cap_months": int(frame.cap_bound.sum()), "cap_frequency": frame.cap_bound.mean(),
                            "mean_predicted_vol_pct_year": 100 * frame.predicted_vol.mean(),
                            "delta_ce_bp_year": 1e4 * (ce["heterogeneous"] - ce["scalar"]) if name == "heterogeneous" else np.nan})
    table = pd.DataFrame(metrics)
    backtest.save_csv(table, "deployment_sensitivity.csv")
    backtest.save_csv(returns, "deployment_returns.csv")
    backtest.save_csv(result["weights"], "deployment_weights.csv")
    write_manifest()
    return {"table": table, "returns": returns, "weights": result["weights"]}


def write_manifest() -> None:
    """Record the empirical inputs, code and outputs without changing earlier step manifests."""
    files = [cfg.PROCESSED_DIR / "returns.csv", cfg.INTERIM_DIR / "forecast_reliability.csv"]
    files += [cfg.ROOT / f"src/epo_project/{name}.py" for name in ("config", "backtest", "evaluation", "reliability", "data")]
    files += [cfg.OUTPUT_DIR / name for name in (
        "annual_selection.csv", "candidate_returns.csv", "candidate_exits.csv", "weights.csv", "matched_returns.csv",
        "performance_table.csv", "delta_ce_summary.csv", "delta_ce_bootstrap.csv", "equity_curves.csv", "equity_curves.png",
        "deployment_sensitivity.csv", "deployment_returns.csv", "deployment_weights.csv")]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Run Steps 7 and 8 before Step 9: {missing}")
    manifest = {"candidate_start": cfg.CANDIDATE_START, "periods": PERIODS, "gamma": cfg.GAMMA,
                "w_grid": cfg.W_GRID, "minimum_ready_months": cfg.EMPIRICAL_MIN_READY_MONTHS,
                "trading_cost": cfg.COST_PER_DOLLAR, "short_fee_annual": cfg.SHORT_FEE_ANNUAL,
                "bootstrap_draws": cfg.EVAL_BOOTSTRAP_DRAWS, "bootstrap_seed": cfg.EVAL_SEED,
                "block_months": [cfg.EVAL_BLOCK, *cfg.EVAL_BLOCK_SENSITIVITY],
                "deployment_target_vol": cfg.TARGET_VOL, "deployment_gross_cap": cfg.GROSS_CAP,
                "hashes": {str(p.relative_to(cfg.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (cfg.OUTPUT_DIR / "empirical_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    result = run()
    print(result["performance"].round(4).to_string(index=False))
    deployment = deployment_sensitivity()
    print(deployment["table"].round(4).to_string(index=False))
