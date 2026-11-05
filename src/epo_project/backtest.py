"""Step 7: matched fixed-gamma portfolios, scalar-selected common w, and costs.

Weights are risky dollar positions per unit of beginning-of-month wealth.
The residual cash position earns/pays RF. Trading and short fees are deducted
at month end, matching the local experiment; subsequent holdings drift using
net total wealth. Fixed-w candidates begin in 1960; actual strategies begin
from cash in 1970 and continue through 2025 without a 2019 reset.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from . import config as cfg
from .data import load_returns
from .reliability import _validate_monthly, relative_uncertainty


class PortfolioInsolvency(ValueError):
    """End-of-month total wealth is nonpositive; continuing requires new capital."""

    def __init__(self, growth: float):
        self.growth = growth
        super().__init__(f"Nonpositive total wealth factor: {growth:.8g}")


def allocate(q: np.ndarray, signal: np.ndarray, w: float,
             h: np.ndarray | None = None, gamma: float = cfg.GAMMA) -> np.ndarray:
    """Absolute zero-anchor positions (Q + kappa diag(diag(Q)*h))^-1 s / gamma."""
    q, signal = np.asarray(q, dtype=float), np.asarray(signal, dtype=float)
    h = np.ones_like(signal) if h is None else np.asarray(h, dtype=float)
    if (signal.ndim != 1 or q.shape != (signal.size, signal.size) or h.shape != signal.shape
            or not all(np.isfinite(a).all() for a in (q, signal, h))
            or (h <= 0).any() or gamma <= 0 or not np.isfinite(gamma) or not 0 <= w < 1):
        raise ValueError("Invalid allocation inputs, gamma or w.")
    if not np.allclose(q, q.T) or (np.diag(q) <= 0).any():
        raise ValueError("Q must be symmetric with positive diagonal.")
    penalty = q + w / (1 - w) * np.diag(np.diag(q) * h)
    np.linalg.cholesky(penalty)
    return np.linalg.solve(penalty, signal) / gamma


def deploy_weights(weights: np.ndarray, q: np.ndarray,
                   target_vol: float = cfg.TARGET_VOL, gross_cap: float = cfg.GROSS_CAP) -> tuple[np.ndarray, bool]:
    """Apply 10% predicted annual volatility first, then the gross cap; never re-expand."""
    if target_vol <= 0 or gross_cap <= 0:
        raise ValueError("Target volatility and gross cap must be positive.")
    predicted = np.sqrt(12 * weights @ q @ weights)
    scaled = weights * target_vol / predicted if predicted > 0 else np.zeros_like(weights)
    gross = np.abs(scaled).sum()
    capped = bool(gross > gross_cap)
    return scaled * min(1, gross_cap / gross) if gross > 0 else scaled, capped


def portfolio_month(target: np.ndarray, pretrade: np.ndarray, asset_returns: np.ndarray,
                    risk_free: float, trade_cost: float = cfg.COST_PER_DOLLAR,
                    short_fee_annual: float = cfg.SHORT_FEE_ANNUAL) -> tuple[np.ndarray, dict]:
    """Full L1 turnover, not half turnover. Missing returns on held assets raise an error."""
    target, pretrade, r = (np.asarray(a, dtype=float) for a in (target, pretrade, asset_returns))
    if target.ndim != 1 or target.shape != pretrade.shape or target.shape != r.shape:
        raise ValueError("Targets, pretrade holdings and returns must have matching vector shapes.")
    if (not np.isfinite(target).all() or not np.isfinite(pretrade).all()
            or not np.isfinite([risk_free, trade_cost, short_fee_annual]).all()
            or min(trade_cost, short_fee_annual) < 0 or risk_free <= -1):
        raise ValueError("Invalid holdings, RF or costs.")
    held = target != 0
    if not np.isfinite(r[held]).all():
        raise ValueError("Missing realized return on a held asset; no zero-fill is permitted.")
    if (r[held] < -1).any():
        raise ValueError("Asset total returns cannot be below -100%.")
    turnover = float(np.abs(target - pretrade).sum())
    gross, short = float(np.abs(target).sum()), float(np.maximum(-target, 0).sum())
    gross_excess = float(target[held] @ (r[held] - risk_free))
    trading, borrowing = trade_cost * turnover, short_fee_annual / 12 * short
    net_excess = gross_excess - trading - borrowing
    growth = 1 + risk_free + net_excess
    if not np.isfinite(growth):
        raise ValueError("Portfolio wealth overflowed.")
    if growth <= 0:
        raise PortfolioInsolvency(growth)
    drift = np.zeros_like(target)
    drift[held] = target[held] * (1 + r[held]) / growth
    return drift, {"rf": risk_free, "gross_excess": gross_excess, "net_excess": net_excess,
                   "net_total": risk_free + net_excess, "wealth_factor": growth,
                   "turnover": turnover, "gross": gross, "short": short,
                   "cash": float(1 - target.sum()), "trading_cost": trading, "borrow_cost": borrowing}


def choose_candidate(history: pd.DataFrame, alive: np.ndarray,
                     cutoff: pd.Timestamp, window: int = cfg.SELECTION_WINDOW) -> tuple[float, pd.DataFrame]:
    """Use exactly the preceding 60 realized months, and only still-alive complete candidates."""
    expected = pd.date_range(end=cutoff, periods=window, freq=pd.offsets.MonthEnd())
    if history.index.has_duplicates or not history.index.is_monotonic_increasing:
        raise ValueError("Candidate history needs sorted unique realization dates.")
    past = history.reindex(expected)
    alive = np.asarray(alive, dtype=bool)
    if alive.shape != (history.shape[1],):
        raise ValueError("Candidate alive mask does not match the grid.")
    eligible = alive & np.isfinite(past.to_numpy()).all(axis=0)
    scores = np.full(history.shape[1], np.nan)
    values = past.to_numpy()[:, eligible]
    scores[eligible] = 12 * values.mean(axis=0) - cfg.GAMMA / 2 * 12 * values.var(axis=0, ddof=1)
    if not eligible.any():
        raise ValueError(f"No surviving candidate has a complete history at {cutoff.date()}.")
    best = np.nanmax(scores)
    chosen = max(w for w, score in zip(history.columns, scores) if score >= best - 1e-12)
    audit = pd.DataFrame({"year": cutoff.year + 1, "cutoff": cutoff,
                          "window_start": expected[0], "window_end": expected[-1],
                          "candidate_w": history.columns, "alive": alive, "eligible": eligible,
                          "net_ce": scores, "selected": history.columns == chosen, "selected_w": chosen})
    return float(chosen), audit


@dataclass
class Formation:
    date: pd.Timestamp
    target_date: pd.Timestamp
    ids: np.ndarray
    q: np.ndarray
    signal: np.ndarray
    h: np.ndarray
    realized_total: np.ndarray
    rf: float


def formation_inputs(total: pd.DataFrame, rf: pd.Series, forecasts: pd.DataFrame,
                     start, end, minimum_ready_months: int = cfg.EMPIRICAL_MIN_READY_MONTHS):
    """Build monthly matched inputs; start/end are realization dates. No covariance cache required."""
    _validate_monthly(total, "total returns")
    if not rf.index.equals(total.index) or not np.isfinite(rf.to_numpy()).all():
        raise ValueError("RF must be finite and aligned to the total-return calendar.")
    if forecasts.duplicated(["date", "industry"]).any():
        raise ValueError("Duplicate forecast date/industry.")
    if minimum_ready_months < 1:
        raise ValueError("minimum_ready_months must be positive.")
    grouped = {date: frame.set_index("industry") for date, frame in forecasts.groupby("date")}
    excess = total.sub(rf, axis=0)
    for target in pd.date_range(start, end, freq=pd.offsets.MonthEnd()):
        date = target - pd.offsets.MonthEnd()
        if date not in grouped or target not in total.index:
            raise ValueError(f"Missing input/target month: {date.date()} / {target.date()}")
        frame = grouped[date]
        frame = frame[frame.regression_ready_months >= minimum_ready_months]
        if not frame.index.isin(total.columns).all():
            raise ValueError("Forecast contains unknown industry labels.")
        names = total.columns[total.columns.isin(frame.index)]
        frame = frame.loc[names]
        if len(names) < cfg.MIN_INDUSTRIES:
            raise ValueError(f"Too few eligible industries at {date.date()}")
        if (not frame.target_date.eq(target).all() or not frame.last_input_date.eq(date).all()
                or not frame.response_end.eq(date).all()):
            raise ValueError("Forecast target date or information cutoff does not match its formation date.")
        risk_dates = pd.date_range(end=date, periods=cfg.RISK_WINDOW, freq=pd.offsets.MonthEnd())
        history = excess.reindex(index=risk_dates, columns=names).to_numpy()
        if not np.isfinite(history).all():
            raise ValueError(f"Incomplete risk history at {date.date()}")
        sigma = np.cov(history, rowvar=False, ddof=1)
        variance = np.diag(sigma)
        if not np.allclose(frame.sigma2, variance, rtol=1e-8, atol=1e-14):
            raise ValueError("Forecast risk variance does not match the market inputs.")
        q = (1 - cfg.PRELIM_SHRINK) * sigma + cfg.PRELIM_SHRINK * np.diag(variance)
        h = relative_uncertainty(frame.u.to_numpy(), variance)
        yield Formation(date, target, total.columns.get_indexer(names), q, frame.s.to_numpy(), h,
                        total.loc[target].to_numpy(), float(rf.loc[target]))


def simulate_portfolios(total: pd.DataFrame, rf: pd.Series, forecasts: pd.DataFrame, *,
                        fixed_selection: pd.DataFrame | None = None, deployment: bool = False,
                        candidate_start=cfg.CANDIDATE_START, evaluation_start=cfg.EXTENSION_PERIOD[0],
                        end=cfg.SUBSEQUENT_PERIOD[1],
                        minimum_ready_months: int = cfg.EMPIRICAL_MIN_READY_MONTHS,
                        trade_cost: float = cfg.COST_PER_DOLLAR,
                        short_fee_annual: float = cfg.SHORT_FEE_ANNUAL) -> dict:
    """Run candidates/strategies or replay frozen w with optional costs/deployment.

    Explicit fee parameters affect both candidate histories and actual strategies.
    Each replay carries its own net-wealth drift, including under frozen selection.
    """
    first, last = pd.Timestamp(evaluation_start), pd.Timestamp(end)
    if first.month != 1 or first != first + pd.offsets.MonthEnd(0) or last < first:
        raise ValueError("Evaluation must start at a January month-end and end after its start.")
    if deployment and fixed_selection is None:
        raise ValueError("Deployment requires primary selected w; it cannot retune.")
    years = range(first.year, last.year + 1)
    schedule = None
    if fixed_selection is not None:
        chosen_rows = fixed_selection[fixed_selection.selected]
        if chosen_rows.year.duplicated().any() or set(chosen_rows.year) != set(years):
            raise ValueError("A frozen selection must contain exactly one selected w per evaluation year.")
        schedule = chosen_rows.set_index("year").selected_w.to_dict()
        if not all(w in cfg.W_GRID for w in schedule.values()):
            raise ValueError("Frozen selection includes w outside the proposal grid.")
    start = pd.Timestamp(candidate_start) if schedule is None else first
    months = pd.date_range(start, last, freq=pd.offsets.MonthEnd())
    history = pd.DataFrame(np.nan, index=months, columns=cfg.W_GRID)
    alive = np.ones(len(cfg.W_GRID), dtype=bool)
    candidate_pretrade = np.zeros((len(cfg.W_GRID), total.shape[1]))
    strategies = ("scalar", "heterogeneous")
    pretrade = {name: np.zeros(total.shape[1]) for name in strategies}
    selections, returns, weights, candidates, exits = [], [], [], [], []
    selected_w = None
    for month in formation_inputs(total, rf, forecasts, start, last, minimum_ready_months):
        date = month.target_date
        # Selection precedes evaluation of the current month's returns.
        if date >= first and date.month == 1:
            if schedule is None:
                selected_w, selected_rows = choose_candidate(history, alive, month.date)
                selections.append(selected_rows)
            else:
                selected_w = schedule[date.year]
        if schedule is None:
            for j, w in enumerate(cfg.W_GRID):
                record = {"date": date, "formation_date": month.date, "w": w, "status": "inactive"}
                if alive[j]:
                    target = np.zeros(total.shape[1])
                    target[month.ids] = allocate(month.q, month.signal, w)
                    try:
                        candidate_pretrade[j], row = portfolio_month(
                            target, candidate_pretrade[j], month.realized_total, month.rf,
                            trade_cost=trade_cost, short_fee_annual=short_fee_annual)
                        history.loc[date, w] = row["net_excess"]
                        record.update(row, status="active")
                    except PortfolioInsolvency as error:
                        alive[j] = False
                        record.update(status="failed", wealth_factor=error.growth)
                        exits.append({"date": date, "w": w, "wealth_factor": error.growth,
                                      "gross_at_failure": float(np.abs(target).sum())})
                candidates.append(record)
        if date < first:
            continue
        for name in strategies:
            local = allocate(month.q, month.signal, selected_w,
                             month.h if name == "heterogeneous" else None)
            capped = False
            if deployment:
                local, capped = deploy_weights(local, month.q)
            target = np.zeros(total.shape[1])
            target[month.ids] = local
            try:
                drift, row = portfolio_month(target, pretrade[name], month.realized_total, month.rf,
                                             trade_cost=trade_cost, short_fee_annual=short_fee_annual)
            except PortfolioInsolvency as error:
                error.args = (f"Selected {name} at {date.date()}, w={selected_w}: {error}",)
                raise
            returns.append({"date": date, "formation_date": month.date, "strategy": name,
                            "w": selected_w, "n_assets": len(month.ids), "cap_bound": capped,
                            "predicted_vol": float(np.sqrt(12 * local @ month.q @ local)), **row})
            weights.append(pd.DataFrame({"date": date, "formation_date": month.date, "strategy": name,
                                         "industry": total.columns, "weight": target,
                                         "pretrade_weight": pretrade[name]}))
            pretrade[name] = drift
    return {"returns": pd.DataFrame(returns), "weights": pd.concat(weights, ignore_index=True),
            "selection": pd.concat(selections, ignore_index=True) if selections else fixed_selection.copy(),
            "candidates": pd.DataFrame(candidates),
            "exits": pd.DataFrame(exits, columns=["date", "w", "wealth_factor", "gross_at_failure"])}


def load_market() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    forecasts = pd.read_csv(cfg.INTERIM_DIR / "forecast_reliability.csv",
                           parse_dates=["date", "target_date", "response_end", "last_input_date"])
    return load_returns("total_return"), load_returns("rf"), forecasts


def save_csv(frame: pd.DataFrame, name: str) -> None:
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cfg.OUTPUT_DIR / name, index=False, date_format="%Y-%m-%d", float_format="%.12g")


def run() -> dict:
    result = simulate_portfolios(*load_market())
    for key, name in [("returns", "matched_returns.csv"), ("weights", "weights.csv"),
                      ("selection", "annual_selection.csv"), ("candidates", "candidate_returns.csv"),
                      ("exits", "candidate_exits.csv")]:
        save_csv(result[key], name)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"months": result["returns"].date.nunique(),
                      "selected_years": result["selection"].year.nunique(),
                      "candidate_exits": len(result["exits"])}, indent=2))
