"""Accounting, ex-ante selection and complete-path tests for Steps 7 and 9."""

import numpy as np
import pandas as pd
import pytest

from epo_project import backtest as bt, config as cfg


def test_self_financing_two_asset_cash_costs_and_drift():
    # $100: $60 long, $20 short, $60 cash. Asset P&L $7, cash interest $0.60.
    target, returns = np.array([0.6, -0.2]), np.array([0.1, -0.05])
    drift, row = bt.portfolio_month(target, np.zeros(2), returns, 0.01)
    total_pnl = 7 + 0.6 - 0.08 - 20 * 0.005 / 12
    assert row["net_total"] == pytest.approx(total_pnl / 100)
    assert row["net_excess"] == pytest.approx(total_pnl / 100 - 0.01)
    assert row["cash"] == pytest.approx(0.6)
    np.testing.assert_allclose(drift, [66 / (100 + total_pnl), -19 / (100 + total_pnl)])
    _, next_row = bt.portfolio_month(drift, drift, np.zeros(2), 0)
    assert next_row["turnover"] == 0
    # End-of-month cash includes interest and fees; risky positions plus cash sum to wealth.
    cash_dollars = 60 * 1.01 - 0.08 - 20 * 0.005 / 12
    assert drift.sum() + cash_dollars / (100 + total_pnl) == pytest.approx(1)


def test_cash_liquidation_missing_unheld_and_no_fees():
    _, cash = bt.portfolio_month(np.zeros(2), np.zeros(2), np.array([np.nan, np.nan]), 0.01)
    assert cash["net_excess"] == 0 and cash["net_total"] == 0.01
    _, liquidation = bt.portfolio_month(np.zeros(2), np.array([0.6, -0.2]), np.zeros(2), 0.01)
    assert liquidation["turnover"] == pytest.approx(0.8)
    _, free = bt.portfolio_month(np.array([0.6, -0.2]), np.zeros(2), np.array([0.1, -0.05]), 0.01, 0, 0)
    assert free["net_excess"] == free["gross_excess"]
    with pytest.raises(ValueError, match="Missing realized"):
        bt.portfolio_month(np.ones(2), np.zeros(2), np.array([0.1, np.nan]), 0)
    with pytest.raises(bt.PortfolioInsolvency):
        bt.portfolio_month(np.array([20.0]), np.zeros(1), np.array([-0.1]), 0)


def test_fixed_gamma_allocation_scaling_and_uniform_h():
    q = np.array([[0.04, 0.01], [0.01, 0.09]])
    signal, w = np.array([0.02, -0.01]), 0.75
    scalar = bt.allocate(q, signal, w)
    np.testing.assert_allclose(bt.allocate(q, signal, w, np.ones(2)), scalar)
    expected = np.linalg.solve(q + 3 * np.diag(np.diag(q)), signal) / 3
    np.testing.assert_allclose(scalar, expected)
    benchmark_style = np.linalg.solve((1 - w) * q + w * np.diag(np.diag(q)), signal) / 3
    np.testing.assert_allclose(scalar, (1 - w) * benchmark_style)
    np.testing.assert_allclose(bt.allocate(q, signal, 0, np.array([0.2, 1.8])), bt.allocate(q, signal, 0))


def test_deployment_order_target_then_cap_and_zero_signal():
    q = np.eye(2) * 1e-5
    scaled, capped = bt.deploy_weights(np.array([1.0, -1.0]), q)
    assert capped and np.abs(scaled).sum() == pytest.approx(3)
    assert np.sqrt(12 * scaled @ q @ scaled) < 0.1
    q = np.eye(2) * 0.01
    scaled, capped = bt.deploy_weights(np.array([1.0, -1.0]), q)
    assert not capped
    assert np.sqrt(12 * scaled @ q @ scaled) == pytest.approx(0.1)
    zero, capped = bt.deploy_weights(np.zeros(2), q)
    np.testing.assert_array_equal(zero, np.zeros(2))
    assert not capped


def test_selection_uses_past_complete_alive_candidates_and_higher_w_tie():
    dates = pd.date_range("2000-01-31", periods=72, freq=pd.offsets.MonthEnd())
    history = pd.DataFrame(0.01, index=dates, columns=[0.0, 0.5, 0.9])
    cutoff = pd.Timestamp("2004-12-31")
    chosen, rows = bt.choose_candidate(history, [False, True, True], cutoff)
    assert chosen == 0.9
    assert (rows.window_start == pd.Timestamp("2000-01-31")).all()
    history.loc[dates > cutoff, 0.5] = 1000
    assert bt.choose_candidate(history, [False, True, True], cutoff)[0] == 0.9
    history.loc[dates[5], 0.9] = np.nan
    assert bt.choose_candidate(history, [False, True, True], cutoff)[0] == 0.5
    with pytest.raises(ValueError, match="No surviving"):
        bt.choose_candidate(history.drop(dates[10]), [False, True, True], cutoff)


def test_ce_selection_can_reject_a_higher_mean_riskier_candidate():
    dates = pd.date_range("2000-01-31", periods=60, freq=pd.offsets.MonthEnd())
    history = pd.DataFrame({0.5: np.tile([-0.09, 0.11], 30), 0.9: 0.008}, index=dates)
    assert bt.choose_candidate(history, [True, True], dates[-1])[0] == 0.9


@pytest.fixture
def market():
    rng = np.random.default_rng(17)
    dates = pd.date_range("1994-01-31", "2006-12-31", freq=pd.offsets.MonthEnd())
    total = pd.DataFrame(rng.normal(0.004, 0.03, (len(dates), 5)), index=dates, columns=list("ABCDE"))
    rf = pd.Series(0.002, index=dates)
    variance = total.sub(rf, axis=0).rolling(60).var()
    rows = []
    for date in dates[dates >= "1999-12-31"][:-1]:
        rows.append(pd.DataFrame({"date": date, "target_date": date + pd.offsets.MonthEnd(),
                                  "industry": total.columns, "s": np.arange(1, 6) * 0.0001,
                                  "u": np.arange(1, 6) * 1e-5, "sigma2": variance.loc[date].to_numpy(),
                                  "regression_ready_months": 100, "response_end": date, "last_input_date": date}))
    return total, rf, pd.concat(rows, ignore_index=True)


def simulate(market, **kwargs):
    return bt.simulate_portfolios(*market, candidate_start="2000-01-31", evaluation_start="2005-01-31",
                                 end=kwargs.pop("end", "2006-12-31"), **kwargs)


def test_full_path_cutoff_invariance_and_label_alignment(market):
    total, rf, forecasts = market
    original = simulate(market)
    cutoff = pd.Timestamp("2005-12-31")
    truncated = simulate((total.loc[:cutoff], rf.loc[:cutoff], forecasts[forecasts.date < cutoff]), end=cutoff)
    for key in ("returns", "weights", "candidates"):
        expected = original[key][original[key].date <= cutoff].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, truncated[key])
    moved = forecasts.copy()
    moved.loc[moved.date >= cutoff, "s"] *= 100
    # Future forecast values cannot affect prior selected portfolios.
    before_future = simulate((total, rf, moved), end=cutoff)
    pd.testing.assert_frame_equal(truncated["returns"], before_future["returns"])
    reordered = simulate((total[total.columns[::-1]], rf, forecasts.sample(frac=1, random_state=3)))
    pd.testing.assert_frame_equal(original["returns"], reordered["returns"], check_exact=False, atol=1e-12, rtol=1e-10)


def test_bankrupt_candidates_never_reenter(market):
    total, rf, forecasts = market
    total = total.copy()
    forecasts = forecasts.copy()
    forecasts.loc[forecasts.date == "1999-12-31", "s"] = 0.03
    total.loc["2000-01-31"] = -0.1
    variances = total.sub(rf, axis=0).rolling(60).var()
    forecasts["sigma2"] = [variances.loc[row.date, row.industry] for row in forecasts.itertuples()]
    result = simulate((total, rf, forecasts))
    assert not result["exits"].empty
    for exit_row in result["exits"].itertuples():
        candidate = result["candidates"].query("w == @exit_row.w")
        assert candidate.loc[candidate.date > exit_row.date, "status"].eq("inactive").all()
        assert candidate.loc[candidate.date >= exit_row.date, "net_excess"].isna().all()
        assert not result["selection"].query("candidate_w == @exit_row.w").eligible.any()


def test_deployment_reuses_w_and_recomputes_its_own_drift(market):
    main = simulate(market)
    deployed = simulate(market, fixed_selection=main["selection"], deployment=True)
    np.testing.assert_array_equal(main["returns"].w, deployed["returns"].w)
    assert deployed["returns"].gross.max() <= 3 + 1e-12
    assert deployed["returns"].predicted_vol.max() <= 0.1 + 1e-12
    weights = deployed["weights"]
    for (date, name), frame in weights.groupby(["date", "strategy"]):
        actual = deployed["returns"].query("date == @date and strategy == @name").iloc[0]
        assert actual.turnover == pytest.approx(np.abs(frame.weight - frame.pretrade_weight).sum())
    january = weights[(weights.date == "2006-01-31") & (weights.strategy == "scalar")]
    assert (january.pretrade_weight != 0).any()  # no annual cash reset
    assert not np.allclose(main["returns"].trading_cost, deployed["returns"].trading_cost)


def test_information_cutoff_mismatch_rejected(market):
    total, rf, forecasts = market
    forecasts = forecasts.copy()
    forecasts.loc[forecasts.date == "1999-12-31", "last_input_date"] = pd.Timestamp("2000-01-31")
    with pytest.raises(ValueError, match="information cutoff"):
        simulate((total, rf, forecasts))


def test_cost_replay_keeps_targets_and_recomputes_net_wealth_drift(market):
    main = simulate(market)
    free = simulate(market, fixed_selection=main["selection"], trade_cost=0, short_fee_annual=0)
    pd.testing.assert_series_equal(main["weights"].weight, free["weights"].weight)
    np.testing.assert_array_equal(main["returns"].w, free["returns"].w)
    assert free["returns"].trading_cost.eq(0).all()
    assert free["returns"].borrow_cost.eq(0).all()
    np.testing.assert_array_equal(free["returns"].net_excess, free["returns"].gross_excess)
    assert not np.allclose(main["weights"].pretrade_weight, free["weights"].pretrade_weight)
    # Independently reconstruct all fee-free drifts from previous target positions.
    total, rf, _ = market
    for strategy in ("scalar", "heterogeneous"):
        previous = None
        previous_date = None
        for date, group in free["weights"].query("strategy == @strategy").groupby("date"):
            if previous is not None:
                returns = total.loc[previous_date].to_numpy()
                growth = 1 + rf.loc[previous_date] + previous @ (returns - rf.loc[previous_date])
                np.testing.assert_allclose(group.pretrade_weight, previous * (1 + returns) / growth)
            previous, previous_date = group.weight.to_numpy(), date


def test_explicit_fees_reach_candidate_histories(market):
    result = simulate(market, trade_cost=.0025, short_fee_annual=.0075)
    for key in ("returns", "candidates"):
        frame = result[key]
        if key == "candidates":
            frame = frame[frame.status == "active"]
        np.testing.assert_allclose(frame.trading_cost, .0025 * frame.turnover)
        np.testing.assert_allclose(frame.borrow_cost, .0075 / 12 * frame.short)


def test_selected_portfolio_insolvency_stops_execution(market):
    total, rf, forecasts = market
    total, forecasts = total.copy(), forecasts.copy()
    forecasts.loc[forecasts.date == "2004-12-31", "s"] = 0.03
    total.loc["2005-01-31"] = -0.2
    schedule = pd.DataFrame({"year": [2005, 2006], "selected": True, "selected_w": 0.0})
    with pytest.raises(bt.PortfolioInsolvency, match="Selected scalar"):
        simulate((total, rf, forecasts), fixed_selection=schedule)
