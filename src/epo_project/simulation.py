"""Step 4: scalar compression loss under a known Gaussian model.

Model (revised proposal, Section 2): known Sigma, V = diag(Sigma), gamma = 3,
    mu ~ N(0, tau Sigma),  s = mu + eps,  eps ~ N(0, Lambda) independent of mu,
    Lambda = lambda * diag(sigma_i^2 h_i),  mean(h) = 1.
The criterion is expected conditional mean-variance utility
    U(x | s) = x' m(s) - (gamma/2) x' Sigma x,   m(s) = E[mu | s] = tau Sigma (tau Sigma + Lambda)^{-1} s.

Allocations, all formed from the same observed s (none observes mu):
    heterogeneous   x_het      = (1/gamma) Sigma^{-1} m(s) = (tau/gamma) (tau Sigma + Lambda)^{-1} s
    scalar family   x(kappa)   = (1/gamma) (Sigma + kappa V)^{-1} s,  kappa >= 0
    mean scalar     kappa_bar  = lambda / tau   (x_het when h = 1)
    oracle scalar   argmax_kappa U(x(kappa) | s), including kappa -> infinity (x = 0)
The scalar family is searched on w = kappa / (1 + kappa) in [0, 1], the same
parameter as the EPO grid: (1 - w) Sigma + w V = (1 - w)(Sigma + kappa V).

Losses per draw, in monthly utility:
    compression loss  L = U(x_het) - U(x_oracle) = (gamma/2) dx' Sigma dx >= 0
    choice loss         = U(x_oracle) - U(x(kappa_bar)) >= 0
Exposure alignment A = sum sigma^2 q^2 (h - 1)^2 / sum sigma^2 q^2 with q = x(kappa_bar).

Outputs: output/simulation_draws.csv, output/compression_loss.csv,
output/compression_loss.png.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

GOLDEN = (np.sqrt(5) - 1) / 2


def model_covariance(n: int = cfg.SIM_N_ASSETS, vol: float = cfg.SIM_MONTHLY_VOL,
                     corr: float = cfg.SIM_CORRELATION) -> np.ndarray:
    """Equicorrelated monthly return covariance Sigma."""
    r = np.full((n, n), corr)
    np.fill_diagonal(r, 1.0)
    return vol ** 2 * r


def uncertainty_profile(cv: float, rng: np.random.Generator, n: int = cfg.SIM_N_ASSETS) -> np.ndarray:
    """Positive lognormal relative uncertainties h with target coefficient of
    variation cv, normalized to mean one (cv = 0 gives h = 1 exactly)."""
    if cv == 0:
        return np.ones(n)
    s2 = np.log(1 + cv ** 2)
    h = np.exp(np.sqrt(s2) * rng.standard_normal(n) - s2 / 2)
    return h / h.mean()


class ScalarFamily:
    """Utility of x(kappa) = (1/gamma)(Sigma + kappa V)^{-1} s for many kappas at once.

    With R = V^{-1/2} Sigma V^{-1/2} = P D P', (Sigma + kappa V)^{-1} =
    V^{-1/2} P (D + kappa I)^{-1} P' V^{-1/2}, so one eigendecomposition serves
    every kappa and every draw.
    """

    def __init__(self, sigma: np.ndarray, gamma: float = cfg.GAMMA):
        self.gamma = gamma
        self.sd = np.sqrt(np.diag(sigma))
        self.d, self.p = np.linalg.eigh(sigma / np.outer(self.sd, self.sd))

    def rotate(self, s: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """b = P' V^{-1/2} s and g = P' V^{-1/2} m, one row per draw."""
        return (s / self.sd) @ self.p, (m / self.sd) @ self.p

    def scale(self, w: np.ndarray) -> np.ndarray:
        """1 / (gamma (d + kappa)) written in w = kappa / (1 + kappa), so it stays
        finite at w = 1, where x = 0. Rows follow w, columns follow d."""
        w = np.asarray(w, dtype=float)[..., None]
        return (1 - w) / (self.gamma * ((1 - w) * self.d + w))

    def utility(self, w: np.ndarray, b: np.ndarray, g: np.ndarray) -> np.ndarray:
        """U(x(kappa) | s) in rotated units u = b / (gamma (d + kappa)):
        U = u'g - (gamma/2) u'Du, as a draws x len(w) table for a 1-D grid w;
        see utility_per_draw for one w per draw."""
        c = self.scale(w)
        return (b * g) @ c.T - self.gamma / 2 * (self.d * b ** 2) @ (c ** 2).T

    def utility_per_draw(self, w: np.ndarray, b: np.ndarray, g: np.ndarray) -> np.ndarray:
        """U at one w per draw."""
        c = self.scale(w)
        return np.sum(b * g * c, axis=1) - self.gamma / 2 * np.sum(self.d * b ** 2 * c ** 2, axis=1)

    def weights(self, w: np.ndarray, s: np.ndarray) -> np.ndarray:
        """x(kappa) for one w per row of s (or a single w for all rows)."""
        return ((s / self.sd) @ self.p * self.scale(w)) @ self.p.T / self.sd


def oracle_w(family: ScalarFamily, b: np.ndarray, g: np.ndarray,
             points: int = cfg.SIM_W_GRID_POINTS, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Best w per draw: coarse grid on [0, 1], then golden-section search
    between the neighbours of the best grid point. The refined point is kept
    only if it beats the grid point and both ends (w = 0 and w = 1)."""
    grid = np.linspace(0.0, 1.0, points)
    k = family.utility(grid, b, g).argmax(axis=1)
    lo, hi = grid[np.maximum(k - 1, 0)], grid[np.minimum(k + 1, points - 1)]

    def f(w):
        return family.utility_per_draw(w, b, g)

    x1, x2 = hi - GOLDEN * (hi - lo), lo + GOLDEN * (hi - lo)
    f1, f2 = f(x1), f(x2)
    while np.max(hi - lo) > tol:
        left = f1 > f2                                               # maximum lies in [lo, x2]
        lo, hi = np.where(left, lo, x1), np.where(left, x2, hi)
        x1, x2 = hi - GOLDEN * (hi - lo), lo + GOLDEN * (hi - lo)
        f1, f2 = f(x1), f(x2)

    n = len(b)
    candidates = np.column_stack([(lo + hi) / 2, grid[k], np.zeros(n), np.ones(n)])
    values = np.column_stack([f(c) for c in candidates.T])
    best = values.argmax(axis=1)
    return candidates[np.arange(n), best], values[np.arange(n), best]


def simulate_level(family: ScalarFamily, sigma: np.ndarray, h: np.ndarray, z_mu: np.ndarray,
                   z_eps: np.ndarray, tau: float = cfg.SIM_TAU, lam: float = cfg.SIM_MEAN_LAMBDA,
                   gamma: float = cfg.GAMMA) -> pd.DataFrame:
    """All comparators for one uncertainty profile h on shared standard-normal draws."""
    var = np.diag(sigma)
    lam_diag = lam * var * h
    mu = z_mu @ np.linalg.cholesky(tau * sigma).T
    s = mu + z_eps * np.sqrt(lam_diag)

    m = s @ np.linalg.solve(tau * sigma + np.diag(lam_diag), tau * sigma)    # E[mu | s], one row per draw
    x_het = np.linalg.solve(sigma, m.T).T / gamma
    u_het = np.sum(x_het * m, axis=1) - gamma / 2 * np.sum((x_het @ sigma) * x_het, axis=1)

    b, g = family.rotate(s, m)
    w_or, u_or = oracle_w(family, b, g)
    x_or = family.weights(w_or, s)

    kappa_bar = lam / tau
    w_bar = np.full(len(s), kappa_bar / (1 + kappa_bar))
    u_bar = family.utility_per_draw(w_bar, b, g)
    q = family.weights(w_bar, s)

    dx = x_het - x_or
    loss_check = gamma / 2 * np.sum((dx @ sigma) * dx, axis=1)
    sq = var * q ** 2
    with np.errstate(divide="ignore"):
        kappa_or = np.where(w_or < 1, w_or / (1 - w_or), np.inf)
    return pd.DataFrame({
        "draw": np.arange(len(s)),
        "w_oracle": w_or,
        "kappa_oracle": kappa_or,
        "kappa_equivalent": (sq @ (lam * h)) / sq.sum(axis=1) / tau,   # (q'Lambda q)/(q'Vq) in kappa units
        "alignment": (sq @ (h - 1) ** 2) / sq.sum(axis=1),
        "utility_het": u_het,
        "utility_oracle": u_or,
        "utility_mean_scalar": u_bar,
        "compression_loss": u_het - u_or,
        "choice_loss": u_or - u_bar,
        "loss_check_error": (u_het - u_or) - loss_check,
    })


def _mean_row(sub: pd.DataFrame, **labels) -> dict:
    """Mean losses in basis points of monthly utility with Monte Carlo 95% intervals."""
    row = dict(labels, draws=len(sub), mean_alignment=sub["alignment"].mean(),
               utility_het_bp=1e4 * sub["utility_het"].mean(),
               compression_loss_share=sub["compression_loss"].mean() / sub["utility_het"].mean(),
               median_kappa_oracle=sub["kappa_oracle"].median(),
               median_kappa_equivalent=sub["kappa_equivalent"].median())
    for col in ("compression_loss", "choice_loss"):
        bp = 1e4 * sub[col]
        se = bp.std(ddof=1) / np.sqrt(len(bp)) if len(bp) > 1 else np.nan
        row.update({f"{col}_bp": bp.mean(), f"{col}_se_bp": se,
                    f"{col}_lo_bp": bp.mean() - 1.96 * se, f"{col}_hi_bp": bp.mean() + 1.96 * se})
    return row


def run(make_plot: bool = True) -> dict:
    rng = np.random.default_rng(cfg.SIM_SEED)
    profiles = {cv: uncertainty_profile(cv, rng) for cv in cfg.SIM_CV_LEVELS}   # fixed before mu and eps
    z_mu = rng.standard_normal((cfg.SIM_DRAWS, cfg.SIM_N_ASSETS))
    z_eps = rng.standard_normal((cfg.SIM_DRAWS, cfg.SIM_N_ASSETS))

    sigma = model_covariance()
    family = ScalarFamily(sigma)
    draws = []
    for cv, h in profiles.items():
        level = simulate_level(family, sigma, h, z_mu, z_eps)
        level.insert(0, "realized_cv", h.std() / h.mean())
        level.insert(0, "target_cv", cv)
        draws.append(level)
    draws = pd.concat(draws, ignore_index=True)

    # ---------------------------------------------------------------- completion checks
    audit = {
        "max_loss_uniform_h": float(draws.loc[draws["target_cv"] == 0, "compression_loss"].abs().max()),
        "min_compression_loss": float(draws["compression_loss"].min()),
        "min_choice_loss": float(draws["choice_loss"].min()),
        "max_loss_check_error": float(draws["loss_check_error"].abs().max()),
        "oracle_at_w0": int((draws["w_oracle"] == 0).sum()),
        "oracle_at_zero_position": int((draws["w_oracle"] == 1).sum()),
    }
    assert audit["max_loss_uniform_h"] < 1e-12, "uniform h must recover scalar EPO"
    assert audit["min_compression_loss"] > -1e-12 and audit["min_choice_loss"] > -1e-12
    assert audit["max_loss_check_error"] < 1e-12

    # ---------------------------------------------------------------- summary by level and alignment tercile
    rows = []
    for (cv, rcv), sub in draws.groupby(["target_cv", "realized_cv"]):
        rows.append(_mean_row(sub, target_cv=cv, realized_cv=rcv, subset="all"))
        if cv > 0:
            tercile = pd.qcut(sub["alignment"], 3, labels=["low", "middle", "high"])
            for name, part in sub.groupby(tercile, observed=True):
                rows.append(_mean_row(part, target_cv=cv, realized_cv=rcv, subset=f"alignment_{name}"))
    table = pd.DataFrame(rows)

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    draws.to_csv(cfg.OUTPUT_DIR / "simulation_draws.csv", index=False, float_format="%.10g")
    table.to_csv(cfg.OUTPUT_DIR / "compression_loss.csv", index=False, float_format="%.6f")

    if make_plot:
        plot_loss(table)
    return {"draws": draws, "table": table, "profiles": profiles, "audit": audit}


def plot_loss(table: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange = "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    allrows = table[table["subset"] == "all"]
    for col, color, label in [("compression_loss", blue, "Compression loss (best scalar)"),
                              ("choice_loss", orange, "Choice loss (mean-uncertainty scalar)")]:
        mean = allrows[f"{col}_bp"]
        err = np.vstack([mean - allrows[f"{col}_lo_bp"], allrows[f"{col}_hi_bp"] - mean])
        ax.errorbar(allrows["realized_cv"], mean, yerr=err, color=color, marker="o", capsize=3, label=label)
    ax.set_xlabel("Realized coefficient of variation of h", fontsize=9, color=muted)
    ax.set_title("Expected utility loss (bp per month)", fontsize=10, color=ink, loc="left")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    parts = table[table["subset"] != "all"]
    levels = sorted(parts["target_cv"].unique())
    names = ["alignment_low", "alignment_middle", "alignment_high"]
    width = 0.8 / len(levels)
    for i, cv in enumerate(levels):
        sub = parts[parts["target_cv"] == cv].set_index("subset").loc[names]
        pos = np.arange(len(names)) + (i - (len(levels) - 1) / 2) * width
        err = np.vstack([sub["compression_loss_bp"] - sub["compression_loss_lo_bp"],
                         sub["compression_loss_hi_bp"] - sub["compression_loss_bp"]])
        ax.bar(pos, sub["compression_loss_bp"], width=width * 0.95, yerr=err, capsize=2,
               color=[blue, orange][i % 2], label=f"target CV {cv:g}")
    ax.set_xticks(np.arange(len(names)), ["Low A", "Middle A", "High A"], fontsize=8, color=muted)
    ax.set_title("Compression loss by exposure-alignment tercile (bp per month)", fontsize=10, color=ink, loc="left")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=8, colors=muted)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(grid)
        ax.spines["bottom"].set_color(muted)
    fig.suptitle(f"Scalar compression loss, {cfg.SIM_N_ASSETS} assets, {cfg.SIM_DRAWS} paired draws per level "
                 "(error bars: Monte Carlo 95% intervals)", fontsize=10, color=ink)
    fig.tight_layout()
    fig.savefig(cfg.OUTPUT_DIR / "compression_loss.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out = run()
    pd.set_option("display.width", 200)
    print(out["table"].round(4).to_string(index=False))
    print("\naudit:", out["audit"])
