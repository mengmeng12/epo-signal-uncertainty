"""Export report diagnostics from inspectable tables, without recalculating results."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE, ORANGE, GRAY = "#2878b5", "#d76532", "#727b85"


def plot_all(tables: dict, selection: pd.DataFrame, intervals: pd.DataFrame,
             destination: Path) -> list[Path]:
    """Create nine standalone PNG figures; units and inference are labeled on each."""
    paths = []
    periods = list(tables["sample_summary"].period)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold",
                         "axes.grid": True, "grid.alpha": .18, "savefig.dpi": 160}):

        def save(fig, name, note):
            fig.text(.02, .015, note, fontsize=8, color="#4b5055", va="bottom")
            fig.tight_layout(rect=(0, .07, 1, .97))
            path = destination / f"{name}.png"
            fig.savefig(path, facecolor="white")
            plt.close(fig)
            paths.append(path)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        data = tables["sample_monthly"]
        axes[0].plot(data.date, data.raw_available_industries, color=GRAY, label="Observed returns", lw=2)
        axes[0].plot(data.date, data.eligible_industries, color=BLUE, label="Eligible forecasts", ls="--")
        axes[0].set(title="Monthly sample coverage", ylabel="Number of industries", xlabel="Realization month")
        axes[0].legend(loc="lower right")
        industry = tables["industry_descriptives"]
        for period, color in zip(periods, (BLUE, ORANGE)):
            values = industry.loc[industry.period == period, "volatility_pct_year"]
            axes[1].hist(values, bins=np.arange(0, 61, 5), alpha=.5, color=color, label=period)
        axes[1].set(title="Industry return volatility", ylabel="Number of industries", xlabel="Annualized volatility (%)")
        axes[1].legend()
        save(fig, "sample_coverage", "Notes: Eligibility follows the primary 120-pair forecast rule. Industry volatility uses eligible months within each period.")

        fig = plt.figure(figsize=(11, 7.5))
        grid = fig.add_gridspec(2, 2)
        for i, period in enumerate(periods):
            ax = fig.add_subplot(grid[0, i])
            sub = tables["forecast_deciles"].query("period == @period and decile != '10-1'").copy()
            sub["decile"] = sub.decile.astype(int)
            sub = sub.sort_values("decile")
            ax.vlines(sub.decile, sub.ci_lower_pct_month, sub.ci_upper_pct_month, color=BLUE, lw=1.8)
            ax.plot(sub.decile, sub.mean_realized_pct_month, "o-", color=BLUE, label="Realized excess return")
            ax.plot(sub.decile, sub.mean_forecast_pct_month, "--", color=ORANGE, label="Forecast")
            ax.axhline(0, color=GRAY, lw=.8)
            ax.set(title=period, xlabel="Forecast decile (low to high)", ylabel="Monthly excess return (%)", xticks=range(1, 11))
            if i == 0:
                ax.legend(fontsize=9)
        ax = fig.add_subplot(grid[1, :])
        data = tables["forecast_monthly"]
        ax.plot(data.date, data.rank_ic.rolling(60, min_periods=60).mean(), color=BLUE)
        ax.axhline(0, color=GRAY, lw=.8)
        ax.set(title="Trailing 60-month mean rank IC", ylabel="Spearman correlation", xlabel="Realization month")
        save(fig, "forecast_quality", "Notes: Equal-weight decile returns before costs. Bars: 95% moving-block bootstrap intervals (12 months, 1,000 draws).\nIC: cross-sectional rank correlation between forecasts and next-month returns. Bands do not adjust for multiple comparisons.")

        fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
        data = tables["reliability_monthly"]
        axes[0].plot(data.date, data.h_cv, color=BLUE)
        axes[0].set(title="Cross-sectional dispersion of h", ylabel="Coefficient of variation")
        for lag, color in ((1, BLUE), (12, ORANGE)):
            axes[1].plot(data.date, data[f"h_rank_corr_{lag}m"].rolling(12).mean(), label=f"{lag}-month lag", color=color)
        axes[1].set(title="Persistence of industry h rankings", ylabel="Rank correlation")
        axes[1].legend()
        for col, label, color in (("rank_corr_h_abs_signal", "Absolute forecast", BLUE), ("rank_corr_h_asset_vol", "Asset volatility", ORANGE)):
            axes[2].plot(data.date, data[col].rolling(12).mean(), label=label, color=color)
        axes[2].set(title="Association of h with signal magnitude and asset risk", ylabel="Rank correlation", xlabel="Realization month")
        axes[2].legend()
        for ax in axes[1:]:
            ax.axhline(0, color=GRAY, lw=.8)
        save(fig, "reliability_diagnostics", "Notes: h is normalized to cross-sectional mean 1; larger h implies a larger uncertainty penalty. CV uses population dispersion.\nRank correlations use common observed industries. Bottom two panels show trailing 12-month averages.")

        wide = tables["reliability_industry_year"].pivot(index="industry", columns="year", values="mean_h").sort_index()
        fig, ax = plt.subplots(figsize=(12, 11))
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("h_ratio", [BLUE, "#fafafa", ORANGE])
        cmap.set_bad("#d9d9d9")
        mesh = ax.imshow(np.log2(wide.to_numpy()), aspect="auto", vmin=-2, vmax=2, cmap=cmap, interpolation="nearest")
        positions = [i for i, year in enumerate(wide.columns) if year % 10 == 0 or year == wide.columns[-1]]
        ax.set(yticks=np.arange(len(wide)), yticklabels=wide.index, xticks=positions,
               xticklabels=wide.columns[positions], xlabel="Realization year", title="Annual mean h by industry")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(False)
        bar = fig.colorbar(mesh, ax=ax, shrink=.6, pad=.02, extend="both")
        bar.set_ticks([-2, -1, 0, 1, 2], labels=["0.25", "0.5", "1", "2", "4"])
        bar.set_label("Mean h (logarithmic color scale)")
        save(fig, "reliability_heatmap", "Notes: Arithmetic mean of monthly h in each realization year. Gray denotes unavailable industries. Colors saturate below 0.25 / above 4.")

        chosen = selection[selection.selected].sort_values("year")
        score = selection[selection.eligible].pivot(index="candidate_w", columns="year", values="net_ce").sort_index()
        fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
        axes[0].step(np.r_[chosen.year, chosen.year.max() + 1],
                     np.r_[chosen.selected_w, chosen.selected_w.iloc[-1]], where="post", color=BLUE)
        axes[0].set(title="Annual common-w selection", ylabel="Selected w", ylim=(.45, 1.01))
        scale = np.nanquantile(score.to_numpy() * 1e4, [.05, .95])
        mesh = axes[1].imshow(score.to_numpy() * 1e4, origin="lower", aspect="auto", cmap="viridis",
                              extent=(score.columns.min(), score.columns.max() + 1, -.5, len(score) - .5),
                              vmin=scale[0], vmax=scale[1], interpolation="nearest")
        axes[1].set(yticks=np.arange(len(score)), yticklabels=score.index,
                    title="Candidate net CE known at each January selection", ylabel="Candidate w", xlabel="Selection year")
        axes[1].grid(False)
        fig.colorbar(mesh, ax=axes, label="Annual net CE (bp)", extend="both", shrink=.75, pad=.02)
        # Reserve the right edge for the shared colorbar; tight_layout cannot place it reliably.
        fig.text(.02, .015, "Notes: Scores use the preceding 60 complete months. Failed candidates are excluded. Color limits: 5th/95th score percentiles.", fontsize=8, color=GRAY)
        fig.subplots_adjust(left=.09, right=.78, bottom=.13, top=.94, hspace=.32)
        path = destination / "selection_diagnostics.png"
        fig.savefig(path, facecolor="white")
        plt.close(fig)
        paths.append(path)

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        data = tables["portfolio_monthly"]
        for name, color in (("scalar", BLUE), ("heterogeneous", ORANGE)):
            axes[0, 0].plot(data.date, data[f"{name}_gross"], color=color, label=name, lw=.8)
            axes[0, 1].plot(data.date, data[f"{name}_predicted_vol"] * 100, color=color, label=name, lw=.8)
        axes[0, 0].set(title="Gross exposure", ylabel="Multiple of portfolio wealth")
        axes[0, 1].set(title="Predicted portfolio volatility", ylabel="Annualized volatility (%)")
        axes[0, 0].legend()
        axes[1, 0].plot(data.date, data.gross_normalized_l1_distance, color=BLUE, lw=.8)
        axes[1, 0].set(title="Composition difference after gross normalization", ylabel="L1 distance", ylim=(0, 2), xlabel="Realization month")
        axes[1, 1].plot(data.date, data.risk_direction_correlation, color=BLUE, lw=.8)
        axes[1, 1].set(title="Correlation of portfolio directions under Q", ylabel="Risk-direction correlation",
                      ylim=(max(-1, data.risk_direction_correlation.min() - .03), 1.01), xlabel="Realization month")
        save(fig, "portfolio_transmission", "Notes: Monthly primary portfolios with fixed risk aversion. Composition distance is 0 and direction correlation is 1 for positive rescaling alone.")

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        columns = ["gross_mean_contribution_bp_year", "trading_cost_contribution_bp_year", "short_cost_contribution_bp_year", "net_variance_contribution_bp_year", "delta_ce_bp_year"]
        for ax, period in zip(axes, periods):
            row = tables["ce_decomposition"].set_index("period").loc[period]
            values = row[columns].to_numpy(dtype=float)
            ax.bar(range(5), values, color=[BLUE if v >= 0 else ORANGE for v in values[:4]] + [GRAY])
            for i, value in enumerate(values):
                ax.annotate(f"{value:+.1f}", (i, value), xytext=(0, 4 if value >= 0 else -4), textcoords="offset points", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
            ax.axhline(0, color=GRAY, lw=.8)
            ax.set(title=period, ylabel="Contribution to annual ΔCE (bp)", xticks=range(5), xticklabels=["Gross\nmean", "Trading\ncost", "Short\nfee", "Net-return\nvariance", "Net ΔCE"])
            ax.margins(y=.15)
        save(fig, "ce_decomposition", "Notes: Heterogeneous minus scalar. The first four terms sum to net ΔCE; the variance term uses net returns, including cost covariance.")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
        for ax, period in zip(axes, periods):
            sub = intervals[intervals.period == period].set_index("block_months").loc[[6, 12, 24]]
            for i, (_, row) in enumerate(sub.iterrows()):
                color = BLUE if i == 1 else GRAY
                ax.hlines(i, row.ci_lower_bp_year, row.ci_upper_bp_year, color=color, lw=2)
                ax.scatter(row.delta_ce_bp_year, i, color=color)
            ax.axvline(0, color=GRAY, ls="--", lw=1)
            ax.set(title=period, yticks=range(3), yticklabels=["6 months", "12 months (primary)", "24 months"], xlabel="Annual ΔCE (bp)", ylim=(-.6, 2.6))
        save(fig, "delta_ce_intervals", "Notes: Heterogeneous minus scalar; 95% paired moving-block bootstrap intervals, 1,000 draws. Both portfolios use identical sampled months.")

        fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
        scenarios = ["frozen_w_0bp", "primary_10bp", "frozen_w_25bp", "admission_60"]
        for ax, period in zip(axes, periods):
            sub = tables["robustness_summary"].query("period == @period and strategy == 'scalar'").set_index("scenario").loc[scenarios]
            for i, (_, row) in enumerate(sub.iterrows()):
                color = BLUE if i == 1 else GRAY
                ax.hlines(i, row.ci_lower_bp_year, row.ci_upper_bp_year, color=color, lw=2)
                ax.scatter(row.delta_ce_bp_year, i, color=color)
            ax.axvline(0, color=GRAY, ls="--", lw=1)
            ax.set(title=period, yticks=range(4), yticklabels=["0 bp; fixed w", "10 bp; primary", "25 bp; fixed w", "60 ready months; reselected w"], xlabel="Annual ΔCE (bp)", ylim=(3.6, -.6))
        save(fig, "robustness", "Notes: 95% paired 12-month block intervals, 1,000 draws. Cost scenarios preserve primary w and recompute drift.\nAdmission sensitivity reruns candidate histories and annual selection; all scenarios retain the 50 bp/year short fee.")
    return paths
