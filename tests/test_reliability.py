"""Independent numerical references and timing/missing-data checks for Step 5."""

import numpy as np
import pandas as pd
import pytest

from epo_project import config as cfg
from epo_project.reliability import (
    build_forecast_panel, estimate_forecasts, moving_block_indices, relative_uncertainty,
)


def test_ols_and_every_bootstrap_fit_match_lstsq():
    rng = np.random.default_rng(72)
    x = rng.normal(size=(120, 6))
    y = 0.03 + x * np.arange(1, 7) / 10 + rng.normal(0, 0.1, x.shape)
    current = rng.normal(size=6)
    result = estimate_forecasts(x, y, current, np.full(6, 0.02), n_bootstrap=25)
    expected = []
    for indices in [np.arange(120), *result.bootstrap_indices]:
        expected.append([
            np.array([1, current[j]]) @ np.linalg.lstsq(
                np.column_stack([np.ones(120), x[indices, j]]), y[indices, j], rcond=None
            )[0] for j in range(6)
        ])
    np.testing.assert_allclose(result.prediction, expected[0], atol=1e-14)
    np.testing.assert_allclose(result.bootstrap_predictions, expected[1:], atol=1e-14)
    np.testing.assert_allclose(result.mean_variance, np.var(expected[1:], axis=0, ddof=1))


def test_shared_dates_preserve_cross_asset_dependence():
    rng = np.random.default_rng(42)
    x = rng.normal(size=120)
    y = x + rng.normal(size=120)
    result = estimate_forecasts(np.column_stack([x, x]), np.column_stack([y, 3 * y + 2]),
                                np.array([0.3, 0.3]), np.array([1.0, 9.0]))
    np.testing.assert_allclose(result.bootstrap_predictions[:, 1],
                               3 * result.bootstrap_predictions[:, 0] + 2)
    np.testing.assert_allclose(result.mean_variance[1], 9 * result.mean_variance[0])
    np.testing.assert_allclose(result.h, [1, 1])


def test_overlapping_blocks_are_contiguous_bounded_and_truncated():
    indices = moving_block_indices(125, 200, 12, np.random.default_rng(1))
    assert indices.shape == (200, 125)
    assert indices.min() == 0 and indices.max() == 124
    for start in range(0, 125, 12):
        assert (np.diff(indices[:, start:start + 12], axis=1) == 1).all()
    assert len(np.unique(indices[:, ::12])) == 114  # every valid block start is reachable


def test_degenerate_predictor_and_zero_uncertainty():
    rng = np.random.default_rng(4)
    y = rng.normal(size=(120, 5))
    result = estimate_forecasts(np.ones_like(y), y, np.ones(5), np.ones(5))
    np.testing.assert_array_equal(result.slope, np.zeros(5))
    np.testing.assert_allclose(result.prediction, y.mean(axis=0))
    np.testing.assert_allclose(result.bootstrap_predictions,
                               y[result.bootstrap_indices].mean(axis=1))
    zero = estimate_forecasts(y, np.zeros_like(y), np.ones(5), np.ones(5))
    np.testing.assert_array_equal(zero.mean_variance, np.zeros(5))
    np.testing.assert_array_equal(zero.h, np.ones(5))


def test_winsor_floor_and_normalization():
    u = np.array([0.0, 1, 2, 3, 100])
    # Five equally spaced quantile positions: 5% = 0.2 and 95% = 80.6.
    expected = np.array([0.2, 1, 2, 3, 80.6])
    np.testing.assert_allclose(relative_uncertainty(u, np.ones(5)), expected / expected.mean())
    np.testing.assert_array_equal(relative_uncertainty(np.full(5, 1e-12), np.ones(5)), np.ones(5))
    np.testing.assert_array_equal(relative_uncertainty(np.zeros(5), np.ones(5)), np.ones(5))


@pytest.mark.parametrize("change", ["missing", "zero_variance", "wrong_shape", "bad_indices"])
def test_invalid_training_inputs_rejected(change):
    x, y, current, v = np.ones((120, 5)), np.zeros((120, 5)), np.ones(5), np.ones(5)
    kwargs = {}
    if change == "missing":
        y[0, 0] = np.nan
    elif change == "zero_variance":
        v[0] = 0
    elif change == "wrong_shape":
        current = current[:-1]
    else:
        kwargs["sample_indices"] = np.full((2, 120), 120)
    with pytest.raises(ValueError):
        estimate_forecasts(x, y, current, v, **kwargs)


@pytest.fixture
def monthly_panel():
    rng = np.random.default_rng(43)
    dates = pd.date_range("1980-01-31", periods=300, freq=pd.offsets.MonthEnd())
    z = pd.DataFrame(rng.normal(0, 0.05, (300, 6)), index=dates, columns=list("ABCDEF"))
    excess = z.shift(1).fillna(0) * 0.4 + rng.normal(0, 0.03, z.shape)
    return z, excess


def test_pair_boundaries_and_independent_first_forecast(monthly_panel):
    z, excess = monthly_panel
    t = z.index[120]
    panel, diagnostics = build_forecast_panel(z, excess, end=t)
    assert panel.date.unique().tolist() == [t]
    assert (panel.predictor_start == z.index[0]).all()
    assert (panel.predictor_end == z.index[119]).all()
    assert (panel.response_start == z.index[1]).all()
    assert (panel.response_end == t).all()
    assert (panel.target_date == z.index[121]).all()
    assert (panel.last_input_date == t).all()
    reference = estimate_forecasts(z.iloc[:120].to_numpy(), excess.iloc[1:121].to_numpy(),
                                   z.iloc[120].to_numpy(), excess.iloc[61:121].var().to_numpy(),
                                   seed=np.random.SeedSequence([cfg.RELIABILITY_SEED, int(t.strftime("%Y%m"))]))
    np.testing.assert_allclose(panel.s, reference.prediction)
    np.testing.assert_allclose(panel.u, reference.mean_variance)
    np.testing.assert_allclose(panel.h, reference.h)
    assert diagnostics.n_industries.iloc[0] == 6


def test_future_perturbation_truncation_and_partial_run_do_not_change_forecasts(monthly_panel):
    z, excess = monthly_panel
    date = z.index[180]
    expected, _ = build_forecast_panel(z, excess, start=date, end=date)
    truncated, _ = build_forecast_panel(z.loc[:date], excess.loc[:date], start=date)
    pd.testing.assert_frame_equal(expected, truncated)
    z_changed, excess_changed = z.copy(), excess.copy()
    z_changed.loc[z.index > date] = 999
    excess_changed.loc[excess.index > date] = -999
    changed, _ = build_forecast_panel(z_changed, excess_changed, start=date, end=date)
    pd.testing.assert_frame_equal(expected, changed)
    longer, _ = build_forecast_panel(z, excess, start=z.index[179], end=date)
    pd.testing.assert_frame_equal(expected, longer[longer.date == date].reset_index(drop=True))


def test_industry_order_only_reorders_results(monthly_panel):
    z, excess = monthly_panel
    date = z.index[180]
    original, _ = build_forecast_panel(z, excess, start=date, end=date)
    reordered, _ = build_forecast_panel(z[z.columns[::-1]], excess[excess.columns[::-1]], start=date, end=date)
    pd.testing.assert_frame_equal(original.sort_values("industry").reset_index(drop=True),
                                  reordered.sort_values("industry").reset_index(drop=True),
                                  check_exact=False, rtol=1e-12, atol=1e-14)


def test_missing_response_requires_120_new_consecutive_pairs(monthly_panel):
    z, excess = monthly_panel
    excess.loc[excess.index[125], "A"] = np.nan
    panel, _ = build_forecast_panel(z, excess, start=z.index[244], end=z.index[245])
    assert "A" not in panel[panel.date == z.index[244]].industry.tolist()
    assert "A" in panel[panel.date == z.index[245]].industry.tolist()
    assert panel.loc[panel.industry == "A", "regression_ready_months"].item() == 1
    np.testing.assert_allclose(panel.groupby("date").h.mean(), 1)


def test_optional_local_admission_delay_is_separate_from_training_window(monthly_panel):
    z, excess = monthly_panel
    panel, _ = build_forecast_panel(z, excess, minimum_ready_months=60, end=z.index[179])
    assert (panel.date == z.index[179]).all()
    assert (panel.regression_ready_months == 60).all()
    assert (panel.predictor_start == z.index[59]).all()


@pytest.mark.parametrize("bad", ["gap", "duplicate", "non_month_end", "infinite"])
def test_invalid_monthly_panel_rejected(monthly_panel, bad):
    z, excess = monthly_panel
    if bad == "gap":
        excess = excess.drop(excess.index[80])
    elif bad == "duplicate":
        excess = pd.concat([excess, excess.iloc[[-1]]])
    elif bad == "non_month_end":
        excess.index = excess.index - pd.Timedelta(days=1)
    else:
        excess.iloc[70, 1] = np.inf
    with pytest.raises(ValueError):
        build_forecast_panel(z, excess)


def test_twelve_month_rank_persistence_uses_common_assets(monthly_panel):
    z, excess = monthly_panel
    panel, diagnostics = build_forecast_panel(z, excess, start=z.index[180], end=z.index[192])
    h = panel.pivot(index="date", columns="industry", values="h")
    expected = np.corrcoef(h.iloc[0].rank(), h.iloc[-1].rank())[0, 1]
    assert diagnostics.h_rank_corr_12m.iloc[-1] == pytest.approx(expected)
    assert diagnostics.h_rank_corr_12m.iloc[:12].isna().all()
