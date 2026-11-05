"""Timing and invariance checks for the Step 6 estimator simulation."""

import numpy as np
import pandas as pd

from epo_project import config as cfg
from epo_project.simulation import alignment_terciles, estimator_draws, estimator_history


def _shocks(n_rep=3, n_assets=5, seed=7):
    rng = np.random.default_rng(seed)
    months = cfg.REGRESSION_PAIRS + 1
    return (rng.standard_normal((n_rep, months, n_assets)), rng.standard_normal((n_rep, months, n_assets)),
            rng.standard_normal((n_rep, n_assets)), np.linspace(0.04, 0.08, n_assets))


def _seeds(n_rep):
    return np.random.SeedSequence(1).spawn(n_rep)


def test_alignment_groups_have_stable_boundaries_and_ties():
    values = pd.Series(np.arange(1000, dtype=float), index=np.arange(1000) + 2000)
    groups = alignment_terciles(values)
    assert groups.value_counts(sort=False).tolist() == [334, 333, 333]
    assert groups.loc[2333] == "low" and groups.loc[2334] == "middle"
    assert groups.loc[2666] == "middle" and groups.loc[2667] == "high"
    tied = alignment_terciles(pd.Series(np.zeros(1000)))
    assert tied.value_counts(sort=False).tolist() == [334, 333, 333]


def test_history_follows_the_stated_recursion():
    e, xi, z0, vols = _shocks()
    z, r = estimator_history(e, xi, z0, vols, feedback=0.4, phi=0.9, sd=0.05, a=0.01, b=0.2)
    eta = 0.4 * e + np.sqrt(1 - 0.16) * xi
    np.testing.assert_allclose(z[:, 0], 0.05 * z0)
    np.testing.assert_allclose(z[:, 1:], 0.9 * z[:, :-1] + 0.05 * np.sqrt(1 - 0.81) * eta[:, :-1])
    np.testing.assert_allclose(r, 0.01 + 0.2 * z + vols * e)


def test_noise_free_returns_recover_the_true_mean_exactly():
    e, xi, z0, vols = _shocks()
    z, r = estimator_history(np.zeros_like(e), xi, z0, vols, feedback=0.0)
    out = estimator_draws(z, r, vols, _seeds(len(z)))
    np.testing.assert_allclose(out["error"], 0, atol=1e-15)
    np.testing.assert_allclose(out["true_mean"], cfg.EST_INTERCEPT + cfg.EST_SLOPE * z[:, -1].ravel())


def test_target_return_is_not_a_training_input():
    e, xi, z0, vols = _shocks()
    z, r = estimator_history(e, xi, z0, vols, feedback=0.0)
    base = estimator_draws(z, r, vols, _seeds(len(z)))
    r_moved = r.copy()
    r_moved[:, -1] += 1.0
    moved = estimator_draws(z, r_moved, vols, _seeds(len(z)))
    np.testing.assert_array_equal(base["prediction"], moved["prediction"])
    np.testing.assert_array_equal(base["u"], moved["u"])
    np.testing.assert_allclose(moved["realized_return"] - base["realized_return"], 1.0)


def test_errors_do_not_depend_on_intercept_or_slope():
    e, xi, z0, vols = _shocks()
    z, r1 = estimator_history(e, xi, z0, vols, feedback=0.0, a=0.0, b=0.0)
    _, r2 = estimator_history(e, xi, z0, vols, feedback=0.0, a=0.02, b=0.5)
    out1 = estimator_draws(z, r1, vols, _seeds(len(z)), a=0.0, b=0.0)
    out2 = estimator_draws(z, r2, vols, _seeds(len(z)), a=0.02, b=0.5)
    np.testing.assert_allclose(out1["error"], out2["error"], atol=1e-13)
    np.testing.assert_allclose(out1["u"], out2["u"], rtol=1e-9)
