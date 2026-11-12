"""Independent arithmetic for utility/assignment comparisons used in the report."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "uncertainty_value",
    Path(__file__).parents[1] / "report/scripts/uncertainty_value.py",
)
value = importlib.util.module_from_spec(spec)
spec.loader.exec_module(value)


def hand_ce(returns):
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return 12 * mean - 18 * variance


def test_assignment_contrast_averages_utilities_not_returns():
    panel = np.array(
        [
            [0, 0.01, 0.10, -0.10],
            [0, -0.01, -0.10, 0.10],
            [0, 0.02, 0.10, -0.10],
            [0, 0, -0.10, 0.10],
        ]
    )
    indices = np.array([[0, 1, 0, 1], [2, 3, 1, 2], [3, 3, 2, 1]])
    point, draws = value.ce_contrasts(panel, indices)

    def contrasts(part):
        ce = [hand_ce(part[:, j]) for j in range(4)]
        return [ce[1] - ce[0], ce[1] - (ce[2] + ce[3]) / 2]

    np.testing.assert_allclose(point, contrasts(panel))
    np.testing.assert_allclose(draws, [contrasts(panel[idx]) for idx in indices])
    assert point[1] != pytest.approx(
        hand_ce(panel[:, 1]) - hand_ce(panel[:, 2:].mean(axis=1))
    )


def test_assignment_contrast_is_zero_for_identical_strategies():
    r = np.array([0.01, -0.02, 0.03, 0.02])
    panel = np.tile(r[:, None], (1, 102))
    indices = np.array([[0, 1, 2, 3], [1, 1, 2, 2]])
    point, draws = value.ce_contrasts(panel, indices)
    np.testing.assert_allclose(point, 0, atol=1e-15)
    np.testing.assert_allclose(draws, 0, atol=1e-15)


def test_loss_identity_for_correlated_risk_and_arbitrary_estimation_error():
    # Directly specified conditional mean, independent of Gaussian-conditioning code.
    sigma = np.array([[0.003, 0.001], [0.001, 0.002]])
    mean = np.array([[0.01, -0.003], [-0.005, 0.008]])
    optimal = np.linalg.solve(sigma, mean.T).T / 3
    scalar = np.array([[0.2, -0.1], [-0.1, 0.3]])
    estimated = np.array([[0.1, -0.2], [-0.5, 0.9]])
    u_optimal = value.utility(optimal, mean, sigma)
    loss = u_optimal - value.utility(scalar, mean, sigma)
    gap = estimated - optimal
    penalty = np.array([1.5 * d @ sigma @ d for d in gap])
    np.testing.assert_allclose(
        value.utility(estimated, mean, sigma) - value.utility(scalar, mean, sigma),
        loss - penalty,
        atol=1e-15,
    )
    assert (penalty > 0).all()
