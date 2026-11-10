"""Numerical checks of Appendix A identities, bound and local expansion.

These are algebra checks, not additional portfolio experiments or empirical
estimates of the sufficient condition. The proofs are in scalar_sufficiency.tex.
"""

from pathlib import Path
import json
import numpy as np

GAMMA = 3.0


def utility(x, mean, sigma):
    return float(x @ mean - GAMMA / 2 * x @ sigma @ x)


def powers(sigma):
    values, vectors = np.linalg.eigh(sigma)
    return (
        (vectors * np.sqrt(values)) @ vectors.T,
        (vectors / np.sqrt(values)) @ vectors.T,
    )


def main():
    rng = np.random.default_rng(20261110)
    checks = 0
    maximum_identity_error = 0.0
    maximum_local_relative_error = 0.0
    for n in (3, 7, 20):
        for _ in range(40):
            a = rng.normal(size=(n, n))
            sigma = a @ a.T / n + 0.5 * np.eye(n)
            v = np.diag(np.diag(sigma))
            h = np.exp(rng.normal(scale=0.7, size=n))
            h /= h.mean()
            penalty = 1.4 * np.diag(np.diag(v) * h)
            kappa = rng.uniform(0, 3)
            b = sigma + kappa * v
            d = penalty - kappa * v
            m_matrix = sigma + penalty
            signal = rng.normal(size=n)
            q = np.linalg.solve(b, signal) / GAMMA
            x = np.linalg.solve(m_matrix, signal) / GAMMA
            mean = GAMMA * sigma @ x
            gap = q - x
            np.testing.assert_allclose(gap, np.linalg.solve(b, d @ x), atol=1e-12)
            np.testing.assert_allclose(
                gap, np.linalg.solve(m_matrix, d @ q), atol=1e-12
            )
            loss = utility(x, mean, sigma) - utility(q, mean, sigma)
            quadratic_loss = GAMMA / 2 * gap @ sigma @ gap
            np.testing.assert_allclose(loss, quadratic_loss, atol=1e-12)
            maximum_identity_error = max(
                maximum_identity_error, abs(loss - quadratic_loss)
            )
            root, inverse_root = powers(sigma)
            t = root @ np.linalg.solve(b, d) @ inverse_root
            y = root @ x
            ratio = loss / utility(x, mean, sigma)
            np.testing.assert_allclose(
                ratio, np.linalg.norm(t @ y) ** 2 / (y @ y), atol=1e-11
            )
            assert ratio <= np.linalg.norm(t, 2) ** 2 + 1e-11

            # Estimation trade-off, evaluated at the true conditional mean.
            estimated_h = np.exp(rng.normal(size=n))
            estimated_h /= estimated_h.mean()
            estimated_matrix = sigma + 1.4 * np.diag(np.diag(v) * estimated_h)
            estimated_x = np.linalg.solve(estimated_matrix, signal) / GAMMA
            estimation_gap = estimated_x - x
            estimation_penalty = GAMMA / 2 * estimation_gap @ sigma @ estimation_gap
            np.testing.assert_allclose(
                utility(estimated_x, mean, sigma) - utility(q, mean, sigma),
                loss - estimation_penalty,
                atol=1e-12,
            )

            # Fixed-signal perturbation around uniform relative uncertainty.
            b0 = sigma + 1.4 * v
            q0 = np.linalg.solve(b0, signal) / GAMMA
            profile_direction = rng.normal(size=n)
            profile_direction -= profile_direction.mean()
            d0 = 1.4 * np.diag(np.diag(v) * profile_direction)
            first_order_gap = np.linalg.solve(b0, d0 @ q0)
            coefficient = GAMMA / 2 * first_order_gap @ sigma @ first_order_gap
            theta = 1e-5
            assert np.all(1 + theta * profile_direction > 0)
            x_theta = np.linalg.solve(b0 + theta * d0, signal) / GAMMA
            error = x_theta - q0
            local_loss = GAMMA / 2 * error @ sigma @ error
            relative_error = abs(local_loss / (theta**2 * coefficient) - 1)
            assert relative_error < 5e-4, relative_error
            maximum_local_relative_error = max(
                maximum_local_relative_error, relative_error
            )
            scale = 0.1
            np.testing.assert_allclose(
                utility(scale * x, scale * mean, sigma)
                - utility(scale * q, scale * mean, sigma),
                scale**2 * loss,
                atol=1e-12,
            )
            checks += 1

    # Exact agreement with heterogeneous uncertainty outside the held support.
    sigma = np.array([[1.0, 0.2, 0.1], [0.2, 1.4, 0.3], [0.1, 0.3, 0.8]])
    v = np.diag(np.diag(sigma))
    h = np.array([1.25, 1.25, 0.5])
    q = np.array([0.3, -0.2, 0.0])
    mean_penalty = 1.4
    kappa = mean_penalty * 1.25
    signal = GAMMA * (sigma + kappa * v) @ q
    x = np.linalg.solve(sigma + mean_penalty * np.diag(np.diag(v) * h), signal) / GAMMA
    np.testing.assert_allclose(x, q, atol=1e-12)
    np.testing.assert_array_equal(np.linalg.solve(sigma, np.zeros(3)), np.zeros(3))
    result = {
        "random_matrix_cases": checks,
        "exact_support_case": "passed",
        "zero_signal_case": "passed",
        "maximum_utility_identity_absolute_error": maximum_identity_error,
        "maximum_local_expansion_relative_error_at_theta_1e_minus5": maximum_local_relative_error,
    }
    destination = Path(__file__).resolve().parents[1] / "build"
    destination.mkdir(exist_ok=True)
    (destination / "scalar_sufficiency_checks.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
