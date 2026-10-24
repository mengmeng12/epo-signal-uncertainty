"""Step 3: Equity 1 benchmark validation and problem-portfolio diagnostics.

Portfolios formed at month-end t from the Step 2 inputs (momentum score z_t,
Sigma_hat_t, Q_t) earn the industries' excess returns of month t+1:
    raw MVO      Sigma_hat_t^{-1} z_t
    MVO with Q   Q_t^{-1} z_t                    (= scalar EPO at w = 0)
    scalar EPO   [(1 - w) Q_t + w V_t]^{-1} z_t,  w in the proposal grid
Each portfolio is scaled to 10% annualized predicted volatility under Q_t, and
w is chosen each January from the preceding 60 months of candidate Sharpe
ratios. Published Equity 1 values are shown to document differences.

Outputs: output/replication_table.csv, output/problem_pc_diagnostics.csv,
output/problem_pc.png.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .data import load_inputs, load_returns
from .epo import epo_weights, scale_to_target_vol, solve
from .rolling import annual_selection, annualized_sharpe, apply_selection

# Pedersen, Babu and Levine (2021), Table 5, Equity 1, realized gross Sharpe ratios 1942-2018
# (the published grid has no 0.95, so that row has no published value)
PUBLISHED_EQUITY1 = {"raw_mvo": 0.19, 0.0: 0.56, 0.25: 0.75, 0.50: 0.79, 0.75: 0.80,
                     0.90: 0.79, 0.99: 0.73, "oos": 0.79}


def _pc_stats(q: np.ndarray, s: np.ndarray, r: np.ndarray, weights: dict) -> tuple[list, dict]:
    """Principal components of the correlation matrix of Q at t, evaluated at t+1.

    Assets are standardized by their volatility sd_t, so component k is a
    portfolio with predicted monthly variance d_k (its eigenvalue). The signal
    is projected the same way, which gives each component's ex ante signal
    direction and Sharpe ratio up to a common scale.
    """
    sd = np.sqrt(np.diag(q))
    d, p = np.linalg.eigh(q / np.outer(sd, sd))                 # ascending eigenvalues
    n = len(d)
    group = np.floor(cfg.PC_GROUPS * np.arange(n) / n).astype(int) + 1

    pc_ret = p.T @ (r / sd)                                    # realized component returns
    pc_sig = p.T @ (s / sd)                                    # signal in component space
    ex_ante = np.abs(pc_sig) / np.sqrt(d)
    oriented = np.sign(pc_sig) * pc_ret / np.sqrt(d)           # unit predicted vol, signal direction

    rows = []
    for g in range(1, cfg.PC_GROUPS + 1):
        k = group == g
        rows.append({"group": g, "eigenvalue": d[k].mean(),
                     "realized_to_predicted_var": np.mean(pc_ret[k] ** 2 / d[k]),
                     "ex_ante_sharpe_rel": ex_ante[k].mean() / ex_ante.mean(),
                     "oriented_return": oriented[k].mean()})

    shares = {}
    for label, x in weights.items():                           # share of predicted variance by group
        contrib = d * (p.T @ (sd * x)) ** 2
        shares[label] = np.array([contrib[group == g].sum() for g in range(1, cfg.PC_GROUPS + 1)]) / contrib.sum()
    return rows, shares


def candidate_panel() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Scaled candidate returns indexed by realization month t+1, PC diagnostics
    per month, and audit counts."""
    excess = load_returns("excess_return")
    z, _, models = load_inputs()
    dates = excess.index

    scaled, pc_rows, share_rows = [], [], []
    audit = {"formation_dates": 0, "missing_next_returns_set_to_zero": 0}
    for t, (members, sigma_hat, q) in models.items():
        pos = dates.get_loc(t)
        if pos + 1 >= len(dates):
            continue                                            # no realized month yet
        t1 = dates[pos + 1]
        s = z.loc[t, members].to_numpy()
        r_next = excess.loc[t1, members]
        audit["missing_next_returns_set_to_zero"] += int(r_next.isna().sum())
        r = r_next.fillna(0.0).to_numpy()                       # not dropped with hindsight
        audit["formation_dates"] += 1

        weights = {"raw_mvo": solve(sigma_hat, s)}
        weights.update({w: epo_weights(q, s, w) for w in cfg.W_GRID})
        scaled.append({"date": t1, **{k: scale_to_target_vol(x, q) @ r for k, x in weights.items()}})

        rows, shares = _pc_stats(q, s, r, weights)
        pc_rows += [{"date": t1, **row} for row in rows]
        share_rows.append({"date": t1, **shares})

    shares = {row["date"]: {k: v for k, v in row.items() if k != "date"} for row in share_rows}
    return pd.DataFrame(scaled).set_index("date"), pd.DataFrame(pc_rows), shares, audit


def _label(key) -> str:
    names = {"raw_mvo": "Raw MVO (Sigma_hat)", 0.0: "MVO with Q (EPO w=0)", "oos": "OOS EPO"}
    return names[key] if key in names else f"EPO w={key:.2f}"


def run(make_plot: bool = True) -> dict:
    start, end = cfg.BENCHMARK_PERIOD
    scaled, pc, shares, audit = candidate_panel()

    grid = list(cfg.W_GRID)
    selection = annual_selection(scaled[grid])
    oos = apply_selection(scaled[grid], selection)

    # ---------------------------------------------------------------- replication table
    rows = []
    for key in ["raw_mvo", *grid, "oos"]:
        series = (oos["return"] if key == "oos" else scaled[key]).loc[start:end]
        rows.append({"portfolio": _label(key), "sharpe": annualized_sharpe(series),
                     "sharpe_published": PUBLISHED_EQUITY1.get(key, np.nan), "months": len(series)})
    table = pd.DataFrame(rows)
    table["difference"] = table["sharpe"] - table["sharpe_published"]

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.OUTPUT_DIR / "replication_table.csv", index=False, float_format="%.4f")

    # ---------------------------------------------------------------- problem-portfolio diagnostics
    pc = pc[(pc["date"] >= start) & (pc["date"] <= end)]
    by_group = pc.groupby("group")
    oriented = pc.pivot(index="date", columns="group", values="oriented_return")
    diag = pd.DataFrame({
        "mean_eigenvalue": by_group["eigenvalue"].mean(),
        "realized_to_predicted_vol": np.sqrt(by_group["realized_to_predicted_var"].mean()),
        "ex_ante_sharpe_relative": by_group["ex_ante_sharpe_rel"].mean(),
        "realized_sharpe_signal_direction": oriented.apply(annualized_sharpe),
    })
    months = list(oos.loc[start:end].index)
    diag["risk_share_mvo_q"] = np.mean([shares[m][0.0] for m in months], axis=0)
    diag["risk_share_oos_epo"] = np.mean([shares[m][oos.at[m, "selected"]] for m in months], axis=0)
    diag.index.name = "eigenvalue_group"
    diag.to_csv(cfg.OUTPUT_DIR / "problem_pc_diagnostics.csv", float_format="%.4f")

    if make_plot:
        plot_diagnostics(diag)
    return {"table": table, "selection": selection, "diagnostics": diag, "audit": audit}


def plot_diagnostics(diag: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange = "#2a78d6", "#eb6834"
    groups = diag.index.to_numpy()
    ticks = [f"G{g}" + (" (smallest)" if g == groups.min() else " (largest)" if g == groups.max() else "")
             for g in groups]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [
        (axes[0, 0], "realized_to_predicted_vol", "Realized / predicted volatility", 1.0),
        (axes[0, 1], "ex_ante_sharpe_relative", "Ex ante signal Sharpe ratio (relative to average)", 1.0),
        (axes[1, 0], "realized_sharpe_signal_direction", "Realized Sharpe ratio in the signal direction", 0.0),
    ]
    for ax, col, title, ref in panels:
        ax.bar(groups, diag[col], width=0.6, color=blue)
        ax.axhline(ref, color=muted, linewidth=1)
        ax.set_title(title, fontsize=10, color=ink, loc="left")

    ax = axes[1, 1]
    width = 0.38
    ax.bar(groups - width / 2 - 0.01, diag["risk_share_mvo_q"], width=width, color=blue, label="MVO with Q")
    ax.bar(groups + width / 2 + 0.01, diag["risk_share_oos_epo"], width=width, color=orange, label="OOS EPO")
    ax.set_title("Share of predicted portfolio variance", fontsize=10, color=ink, loc="left")
    ax.legend(frameon=False, fontsize=9)

    for ax in axes.flat:
        ax.set_xticks(groups, ticks, fontsize=8, color=muted)
        ax.tick_params(axis="y", labelsize=8, colors=muted)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(grid)
        ax.spines["bottom"].set_color(muted)
    fig.suptitle("Principal-component portfolios by eigenvalue group, formed at t and evaluated at t+1 (1942-2018)",
                 fontsize=10, color=ink)
    fig.tight_layout()
    fig.savefig(cfg.OUTPUT_DIR / "problem_pc.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out = run()
    pd.set_option("display.width", 200)
    print(out["table"].round(3).to_string(index=False))
    print("\nselected w (years):\n", out["selection"]["selected"].value_counts().sort_index().to_string())
    print("\n", out["diagnostics"].round(3).to_string())
    print("\naudit:", out["audit"])
