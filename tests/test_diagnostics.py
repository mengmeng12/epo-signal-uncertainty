"""Past-only forecasts, cross-sectional aggregation and economic decompositions."""

import numpy as np
import pandas as pd
import pytest

from epo_project import diagnostics as d, evaluation as ev


def test_benchmark_and_target_alignment_exclude_future_information():
    dates = pd.date_range("2000-01-31", periods=8, freq=pd.offsets.MonthEnd())
    total = pd.DataFrame({"A": np.arange(8) / 100, "B": np.arange(8, 16) / 100}, index=dates)
    rf = pd.Series(.001, index=dates)
    formation = dates[4]
    forecasts = pd.DataFrame({"date": formation, "target_date": dates[5], "industry": ["B", "A"],
                              "last_input_date": formation, "response_end": formation,
                              "s": [.03, .02], "h": [1.1, .9], "u": .001, "sigma2": .02})
    original = d.prepare_forecasts(total, rf, forecasts, start=dates[5], end=dates[5], window=3)
    np.testing.assert_allclose(original.historical_mean, [.029, .109])
    np.testing.assert_allclose(original.realized_excess, [.049, .129])
    changed = total.copy()
    changed.loc[dates[5]:] = .9
    after = d.prepare_forecasts(changed, rf, forecasts, start=dates[5], end=dates[5], window=3)
    np.testing.assert_array_equal(original.historical_mean, after.historical_mean)
    assert not np.allclose(original.realized_excess, after.realized_excess)
    forecasts["last_input_date"] = dates[5]
    with pytest.raises(ValueError, match="cutoff"):
        d.prepare_forecasts(total, rf, forecasts, start=dates[5], end=dates[5], window=3)


def test_benchmark_does_not_fill_missing_history():
    frame = pd.DataFrame({"A": [1., 2., np.nan, 4., 5., 6.]})
    mean = d.formation_mean(frame, window=3)
    assert mean.iloc[:5].isna().all().all()
    assert mean.iloc[5, 0] == 5


def test_decile_ties_have_stable_balanced_counts_and_reject_infinite():
    assigned = d.balanced_bins(pd.Series(np.ones(49)))
    assert set(assigned) == set(range(1, 11))
    assert assigned.value_counts().max() - assigned.value_counts().min() == 1
    assert assigned.iloc[0] == 1 and assigned.iloc[-1] == 10
    with pytest.raises(ValueError, match="finite"):
        d.balanced_bins(pd.Series([np.inf] * 10))


@pytest.fixture
def forecast_panel(monkeypatch):
    dates = pd.date_range("2000-01-31", periods=48, freq=pd.offsets.MonthEnd())
    monkeypatch.setattr(ev, "PERIODS", {"first": (dates[0], dates[23]), "second": (dates[24], dates[-1])})
    rows = []
    for t, date in enumerate(dates):
        n = 10 if t % 2 == 0 else 20
        signal = np.arange(n) / 1000
        realized = signal + (t + 1) / 1000
        rows.append(pd.DataFrame({"target_date": date, "industry": [f"I{i:02}" for i in range(n)],
                                  "s": signal, "realized_excess": realized,
                                  "squared_error": (realized - signal) ** 2,
                                  "benchmark_squared_error": realized ** 2}))
    return pd.concat(rows, ignore_index=True)


def test_pooled_r2_and_month_block_inference_match_manual_calculation(forecast_panel):
    result = d.forecast_diagnostics(forecast_panel)
    row = result["forecast_summary"].iloc[0]
    sub = forecast_panel[forecast_panel.target_date.dt.year <= 2001]
    assert row.pooled_oos_r2 == pytest.approx(1 - sub.squared_error.sum() / sub.benchmark_squared_error.sum())
    # Exact ranking in every month, despite changing universe and common time shocks.
    assert row.mean_rank_ic == pytest.approx(1)
    assert row.rank_ic_ci_lower == pytest.approx(1)
    assert row.rank_ic_ci_upper == pytest.approx(1)
    monthly = result["forecast_monthly"].iloc[:24]
    indices = d.block_indices(24, 1, 2000)
    manual = 1 - monthly.sse_forecast.to_numpy()[indices].sum(axis=1) / monthly.sse_benchmark.to_numpy()[indices].sum(axis=1)
    assert row.oos_r2_ci_lower == pytest.approx(np.quantile(manual, .025))
    assert row.oos_r2_ci_upper == pytest.approx(np.quantile(manual, .975))
    # Decile spread is paired within a month, not two independently sampled groups.
    sorts = result["forecast_decile_monthly"].query("date.dt.year <= 2001")
    wide = sorts.pivot(index="date", columns="decile", values="mean_realized")
    spread = (wide[10] - wide[1]).to_numpy()
    output = result["forecast_deciles"].query("period == 'first' and decile == '10-1'").iloc[0]
    assert output.mean_realized_pct_month == pytest.approx(100 * spread.mean())
    assert output.ci_lower_pct_month == pytest.approx(100 * np.quantile(spread[indices].mean(axis=1), .025))


def test_h_persistence_uses_pre_period_history_and_common_industries(monkeypatch):
    date = pd.Timestamp("1969-12-31")
    target = date + pd.offsets.MonthEnd()
    monkeypatch.setattr(ev, "PERIODS", {"first": (target, target)})
    h = np.array([.4, .8, 1.2, 1.6])
    panel = pd.DataFrame({"date": date, "target_date": target, "industry": list("ABCD"),
                          "h": h, "u": h / 1000, "sigma2": .02, "s": [1., 2., 3., 4.]})
    history = pd.concat([panel[["date", "industry", "h"]],
                         pd.DataFrame({"date": date - pd.offsets.MonthEnd(12), "industry": list("ABCE"), "h": [3., 2., 1., 4.]})])
    result = d.reliability_diagnostics(panel, history)
    row = result["reliability_monthly"].iloc[0]
    assert row.h_rank_corr_12m == pytest.approx(-1)
    assert np.isnan(row.h_rank_corr_1m)
    assert row.h_mean == 1
    assert row.h_cv == pytest.approx(np.std(h, ddof=0))


def test_scale_and_direction_are_separately_identified():
    x = np.array([.6, -.4])
    q = np.array([[.04, .01], [.01, .09]])
    scaled = d.portfolio_geometry(x, 2 * x, q)
    assert scaled["gross_ratio_het_scalar"] == 2
    assert scaled["predicted_vol_ratio_het_scalar"] == 2
    assert scaled["gross_normalized_l1_distance"] == 0
    assert scaled["risk_direction_correlation"] == pytest.approx(1)
    reversed_direction = d.portfolio_geometry(x, -x, q)
    assert reversed_direction["gross_normalized_l1_distance"] == 2
    assert reversed_direction["risk_direction_correlation"] == pytest.approx(-1)


def test_ce_decomposition_includes_cost_covariance(monkeypatch):
    dates = pd.date_range("2000-01-31", periods=3, freq=pd.offsets.MonthEnd())
    monkeypatch.setattr(ev, "PERIODS", {"toy": (dates[0], dates[-1])})
    frames = []
    for name, gross, trading, borrowing in (
        ("scalar", np.array([-.02, .01, .04]), np.array([.001, .002, .003]), np.array([.0001, .0002, .0003])),
        ("heterogeneous", np.array([-.03, .02, .07]), np.array([.003, .004, .005]), np.array([.0002, .0004, .0006])),
    ):
        net = gross - trading - borrowing
        frames.append(pd.DataFrame({"date": dates, "strategy": name, "rf": .001, "gross_excess": gross,
                                    "net_excess": net, "net_total": net + .001, "trading_cost": trading,
                                    "borrow_cost": borrowing, "turnover": 1., "gross": 2., "short": .1, "w": .9, "n_assets": 2}))
    result = d.ce_decomposition(pd.concat(frames, ignore_index=True)).iloc[0]
    assert result.gross_mean_contribution_bp_year == pytest.approx(1200)
    assert result.trading_cost_contribution_bp_year == pytest.approx(-240)
    assert result.short_cost_contribution_bp_year == pytest.approx(-24)
    # Net deviations are +/- .0289 (scalar) and +/- .0488 (heterogeneous).
    risk = -18e4 * (.0488 ** 2 - .0289 ** 2)
    assert result.net_variance_contribution_bp_year == pytest.approx(risk)
    assert result.delta_ce_bp_year == pytest.approx(1200 - 240 - 24 + risk)
    assert abs(result.reconciliation_error_bp_year) < 1e-9
