"""MVO and scalar EPO weights on a given covariance matrix.

Q is the covariance after the fixed 5% preliminary correction and V = diag(Q).
Scalar EPO (Pedersen, Babu and Levine, 2021, Simple EPO) uses
    Sigma_w = (1 - w) Q + w V,   x proportional to Sigma_w^{-1} s.
"""

from __future__ import annotations

import numpy as np

from . import config as cfg


def solve(cov: np.ndarray, s: np.ndarray) -> np.ndarray:
    """cov^{-1} s, with a least-squares fallback if cov is numerically singular."""
    try:
        return np.linalg.solve(cov, s)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(cov, s, rcond=None)[0]


def shrunk_covariance(q: np.ndarray, w: float) -> np.ndarray:
    """Sigma_w = (1 - w) Q + w diag(Q): every correlation multiplied by (1 - w)."""
    return (1 - w) * q + w * np.diag(np.diag(q))


def epo_weights(q: np.ndarray, s: np.ndarray, w: float) -> np.ndarray:
    """Scalar EPO direction Sigma_w^{-1} s (w = 0 is MVO on Q)."""
    return solve(shrunk_covariance(q, w), s)


def scale_to_target_vol(x: np.ndarray, cov: np.ndarray, target: float = cfg.TARGET_VOL) -> np.ndarray:
    """Rescale x to an annualized predicted volatility of `target` under `cov` (monthly)."""
    vol = np.sqrt(12 * x @ cov @ x)
    return x * (target / vol) if vol > 0 else x
