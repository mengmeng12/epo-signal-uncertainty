"""Independent metric arithmetic and paired-inference checks."""

import numpy as np
import pandas as pd
import pytest

from epo_project import evaluation as ev


def test_annual_ce_matches_hand_calculation():
    r = np.array([-0.02, 0.01, 0.04])
    # Mean=.01; sample variance=(.03^2 + 0 + .03^2)/2=.0009.
    assert ev.certainty_equivalent(r) == pytest.approx(0.12 - 1.5 * 12 * 0.0009)


def test_drawdown_includes_initial_capital():
    wealth, drawdown = ev.wealth_and_drawdown(np.array([-0.2, 0.1, -0.1, 0.5]))
    np.testing.assert_allclose(wealth, [0.8, 0.88, 0.792, 1.188])
    np.testing.assert_allclose(drawdown, [-0.2, -0.12, -0.208, 0])
    with pytest.raises(ValueError):
        ev.wealth_and_drawdown(np.array([-1.0]))


def test_es_uses_exact_fractional_tail_mass():
    # 95% ES of 30 observations: worst 1.5 observations, with losses 10% and 4%.
    r = np.r_[-0.1, -0.04, np.full(28, 0.01)]
    assert ev.expected_shortfall(r) == pytest.approx((0.1 + 0.5 * 0.04) / 1.5)
    assert ev.expected_shortfall(np.full(10, 0.01)) == pytest.approx(-0.01)


def test_paired_bootstrap_matches_independent_sample_calculation():
    paired = np.array([[0.01, 0.02], [-0.03, -0.01], [0.02, 0.01], [0.0, -0.02]])
    indices = np.array([[0, 1, 0, 1], [2, 3, 1, 2], [3, 3, 2, 1]])
    observed, draws = ev.paired_ce_bootstrap(paired, sample_indices=indices)
    def ce(values):
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return 12 * mean - 18 * variance
    expected = [ce(paired[idx, 1]) - ce(paired[idx, 0]) for idx in indices]
    np.testing.assert_allclose(draws, expected)
    assert observed == pytest.approx(ce(paired[:, 1]) - ce(paired[:, 0]))


def test_pairing_identical_strategies_and_constant_shift():
    rng = np.random.default_rng(17)
    r = rng.normal(0.005, 0.04, 84)
    point, samples = ev.paired_ce_bootstrap(np.column_stack([r, r]))
    assert point == 0
    np.testing.assert_array_equal(samples, np.zeros(1000))
    point, samples = ev.paired_ce_bootstrap(np.column_stack([r, r + 0.001]))
    assert point == pytest.approx(0.012)
    np.testing.assert_allclose(samples, 0.012, atol=1e-14)


def test_delta_sign_reverses_when_swapping_strategies():
    paired = np.random.default_rng(31).normal(0.001, 0.03, (84, 2))
    point, samples = ev.paired_ce_bootstrap(paired)
    swapped_point, swapped = ev.paired_ce_bootstrap(paired[:, ::-1])
    assert swapped_point == pytest.approx(-point)
    np.testing.assert_array_equal(swapped, -samples)


@pytest.fixture
def matched():
    dates = pd.date_range("1970-01-31", "2025-12-31", freq=pd.offsets.MonthEnd())
    rng = np.random.default_rng(55)
    r = rng.normal(0.004, 0.03, len(dates))
    frames = []
    for strategy, shift in [("scalar", 0), ("heterogeneous", 0.001)]:
        frames.append(pd.DataFrame({"date": dates, "strategy": strategy, "rf": 0.002,
                                    "net_excess": r + shift, "net_total": r + shift + 0.002,
                                    "gross_excess": r + shift + 0.0012, "trading_cost": 0.001,
                                    "borrow_cost": 0.0002, "turnover": 1.0, "gross": 2.0,
                                    "short": 0.3, "w": 0.9, "n_assets": 49}))
    return pd.concat(frames, ignore_index=True)


def test_complete_evaluation_units_windows_and_confidence_interval(matched):
    result = ev.evaluate(matched)
    table = result["performance"]
    assert table.months.tolist() == [588, 588, 84, 84]
    assert len(result["bootstrap"]) == 6000
    assert len(result["intervals"]) == 6
    # Constant +10 bp/month with unchanged variance yields +120 bp/year delta CE.
    np.testing.assert_allclose(result["intervals"][["delta_ce_bp_year", "ci_lower_bp_year", "ci_upper_bp_year"]], 120, atol=1e-10)
    np.testing.assert_allclose(table.trading_cost_bp_year, 120, atol=1e-12)
    np.testing.assert_allclose(table.borrow_cost_bp_year, 24, atol=1e-12)


@pytest.mark.parametrize("problem", ["missing_strategy", "missing_month", "duplicate", "nan", "different_w", "costs"])
def test_invalid_paired_results_rejected(matched, problem):
    frame = matched.copy()
    if problem == "missing_strategy":
        frame = frame.drop(index=0)
    elif problem == "missing_month":
        frame = frame[frame.date != "1980-01-31"]
    elif problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]])
    elif problem == "nan":
        frame.loc[0, "net_excess"] = np.nan
    elif problem == "different_w":
        frame.loc[0, "w"] = 0.5
    else:
        frame.loc[0, "trading_cost"] = 0.5
    with pytest.raises(ValueError):
        ev.validate_matched(frame)
