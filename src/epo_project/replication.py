"""Step 3: Equity 1 benchmark validation and problem-portfolio diagnostics.

Portfolios formed at month-end t from the Step 2 inputs (momentum score z_t,
Sigma_hat_t, Q_t) earn the industries' excess returns of month t+1:
    raw MVO      Sigma_hat_t^{-1} z_t
    MVO with Q   Q_t^{-1} z_t                    (= scalar EPO at w = 0)
    scalar EPO   [(1 - w) Q_t + w V_t]^{-1} z_t,  w in the proposal grid
Each portfolio is scaled to 10% annualized predicted volatility under Q_t, and
w is chosen each January from the preceding 60 months of candidate Sharpe
ratios. Published Equity 1 values are shown to document differences.

Missing returns: if an industry held at t has no return in month t+1, that
month's portfolio returns stay missing (nothing is filled). A selection year
whose 60-month window contains such a month is skipped, and every portfolio is
evaluated over the same common months.

Outputs: output/replication_table.csv, output/problem_pc_diagnostics.csv,
output/problem_pc.png.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .data import load_inputs, load_returns
from .epo import epo_weights, scale_to_target_vol, shrunk_covariance, solve
from .rolling import annual_selection, annualized_sharpe, apply_selection

# Pedersen, Babu and Levine (2021), Table 5, Equity 1, realized gross Sharpe ratios 1942-2018
# (the published grid has no 0.95, so that row has no published value)
PUBLISHED_EQUITY1 = {"raw_mvo": 0.19, 0.0: 0.56, 0.25: 0.75, 0.50: 0.79, 0.75: 0.80,
                     0.90: 0.79, 0.99: 0.73, "oos": 0.79}


def _components(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Volatilities, ascending eigenvalues, eigenvectors and eigenvalue groups of cov's correlation matrix."""
    sd = np.sqrt(np.diag(cov))
    d, p = np.linalg.eigh(cov / np.outer(sd, sd))
    n = len(d)
    group = np.floor(cfg.PC_GROUPS * np.arange(n) / n).astype(int) + 1
    return sd, d, p, group


def _component_stats(cov: np.ndarray, s: np.ndarray, r: np.ndarray) -> list:
    """Principal components of one risk matrix at t, evaluated at t+1.

    Assets are standardized by their volatility sd_t, so component k is a
    portfolio with predicted monthly variance d_k (its eigenvalue). The signal
    is projected the same way, which gives each component's ex ante signal
    direction and Sharpe ratio up to a common scale. Sigma_hat, Q and every
    Sigma_w share the volatilities and eigenvectors; only the eigenvalues differ.
    """
    sd, d, p, group = _components(cov)
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
    return rows


def _pc_stats(q: np.ndarray, sigma_hat: np.ndarray, s: np.ndarray, r: np.ndarray,
              weights: dict) -> tuple[dict, dict]:
    """Component statistics under each portfolio's own risk matrix, and risk shares under Q.

    Raw MVO uses Sigma_hat and EPO(w) uses (1 - w) Q + w V (w = 0 is Q). Risk
    shares measure every portfolio with Q, the matrix used to scale all of them.
    """
    matrices = {"raw_mvo": sigma_hat, **{w: shrunk_covariance(q, w) for w in cfg.W_GRID}}
    stats = {label: _component_stats(cov, s, r) for label, cov in matrices.items()}

    sd, d, p, group = _components(q)
    shares = {}
    for label, x in weights.items():                           # share of predicted variance by group
        contrib = d * (p.T @ (sd * x)) ** 2
        shares[label] = np.array([contrib[group == g].sum() for g in range(1, cfg.PC_GROUPS + 1)]) / contrib.sum()
    return stats, shares


def candidate_panel() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Scaled candidate returns indexed by realization month t+1, PC diagnostics
    per month, and audit counts."""
    excess = load_returns("excess_return")
    z, _, models = load_inputs()
    dates = excess.index

    scaled, pc_rows, share_rows = [], [], []
    audit = {"formation_dates": 0, "months_with_missing_return": []}
    for t, (members, sigma_hat, q) in models.items():
        pos = dates.get_loc(t)
        if pos + 1 >= len(dates):
            continue                                            # no realized month yet
        t1 = dates[pos + 1]
        s = z.loc[t, members].to_numpy()
        r_next = excess.loc[t1, members]
        audit["formation_dates"] += 1

        weights = {"raw_mvo": solve(sigma_hat, s)}
        weights.update({w: epo_weights(q, s, w) for w in cfg.W_GRID})
        if r_next.isna().any():                                 # leave the month's returns missing
            audit["months_with_missing_return"].append((str(t1.date()), list(r_next[r_next.isna()].index)))
            scaled.append({"date": t1, **{k: np.nan for k in weights}})
            continue
        r = r_next.to_numpy()
        scaled.append({"date": t1, **{k: scale_to_target_vol(x, q) @ r for k, x in weights.items()}})

        stats, shares = _pc_stats(q, sigma_hat, s, r, weights)
        pc_rows += [{"date": t1, "model": label, **row} for label, rows in stats.items() for row in rows]
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
    selection = annual_selection(scaled[grid])                 # skips windows containing a missing month
    oos = apply_selection(scaled[grid], selection)

    # common evaluation months: every portfolio, including OOS EPO, has a return
    panel = scaled.assign(oos=oos["return"]).loc[start:end]
    common = panel.dropna().index
    audit["evaluation_months"] = len(common)
    audit["selection_years_skipped"] = [y for y in range(int(start[:4]), int(end[:4]) + 1) if y not in selection.index]

    # ---------------------------------------------------------------- replication table
    rows = []
    for key in ["raw_mvo", *grid, "oos"]:
        series = panel.loc[common, key]
        rows.append({"portfolio": _label(key), "sharpe": annualized_sharpe(series),
                     "sharpe_published": PUBLISHED_EQUITY1.get(key, np.nan), "months": len(series)})
    table = pd.DataFrame(rows)
    table["difference"] = table["sharpe"] - table["sharpe_published"]

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.OUTPUT_DIR / "replication_table.csv", index=False, float_format="%.4f")

    # ---------------------------------------------------------------- problem-portfolio diagnostics
    # Component statistics use each portfolio's own risk matrix: Sigma_hat for raw
    # MVO, Q for corrected MVO, and Sigma_w at the selected w for OOS EPO.
    pc = pc[pc["date"].isin(common)]
    selected = oos["selected"]
    frames = {"": pc[pc["model"].apply(lambda m: m == 0.0)],
              "_raw_mvo": pc[pc["model"].apply(lambda m: m == "raw_mvo")],
              "_oos_epo": pc[[m == selected.at[t] for t, m in zip(pc["date"], pc["model"])]]}
    columns = {}
    for suffix, frame in frames.items():
        by_group = frame.groupby("group")
        oriented = frame.pivot(index="date", columns="group", values="oriented_return")
        columns.update({
            "mean_eigenvalue" + suffix: by_group["eigenvalue"].mean(),
            "realized_to_predicted_vol" + suffix: np.sqrt(by_group["realized_to_predicted_var"].mean()),
            "ex_ante_sharpe_relative" + suffix: by_group["ex_ante_sharpe_rel"].mean(),
            "realized_sharpe_signal_direction" + suffix: oriented.apply(annualized_sharpe),
        })
    diag = pd.DataFrame(columns)
    months = list(common)
    diag["risk_share_raw_mvo"] = np.mean([shares[m]["raw_mvo"] for m in months], axis=0)
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
    models = [("Raw MVO", "_raw_mvo", "raw_mvo", "#8a8f98"),      # same colors as the slides
              ("Corrected MVO", "", "mvo_q", "#2a78d6"),
              ("OOS EPO", "_oos_epo", "oos_epo", "#eb6834")]
    groups = diag.index.to_numpy()
    ticks = [f"G{g}" + (" (smallest)" if g == groups.min() else " (largest)" if g == groups.max() else "")
             for g in groups]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [
        (axes[0, 0], lambda sfx, key: "realized_to_predicted_vol" + sfx, "Realized / predicted volatility", 1.0),
        (axes[0, 1], lambda sfx, key: "ex_ante_sharpe_relative" + sfx, "Ex ante signal Sharpe ratio (relative to average)", 1.0),
        (axes[1, 0], lambda sfx, key: "realized_sharpe_signal_direction" + sfx, "Realized Sharpe ratio in the signal direction", 0.0),
        (axes[1, 1], lambda sfx, key: "risk_share_" + key, "Share of predicted portfolio variance under Q", None),
    ]
    width = 0.27
    for ax, column, title, ref in panels:
        for i, (label, suffix, key, color) in enumerate(models):
            ax.bar(groups + (i - 1) * (width + 0.01), diag[column(suffix, key)], width=width, color=color, label=label)
        if ref is not None:
            ax.axhline(ref, color=muted, linewidth=1)
        ax.set_title(title, fontsize=10, color=ink, loc="left")
    axes[0, 0].legend(frameon=False, fontsize=9)

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
