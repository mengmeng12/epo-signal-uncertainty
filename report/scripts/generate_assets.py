"""Generate manuscript tables and vector figures from the project's result files."""

from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import platform
import re
import sys
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "report"
OUT = ROOT / "output"
DEST = REPORT / "generated"
sys.path.insert(0, str(ROOT / "src"))
from epo_project import (
    config as cfg,
    replication,
    simulation,
    evaluation,
    diagnostic_plots,
    diagnostics,
)

from label_placebo import run as run_label_placebo, DEST as PLACEBO_DIR
from uncertainty_value import (
    run as run_uncertainty_value,
    DEST as VALUE_DIR,
    utility_share,
)
from calibration_bridge import run as run_calibration_bridge, DEST as BRIDGE_DIR

INPUTS: set[Path] = set()


def read(name, directory=OUT):
    path = directory / f"{name}.csv"
    INPUTS.add(path)
    frame = pd.read_csv(path)
    for column in ("date", "formation_date"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column])
    return frame


def annual_simulation_report() -> pd.DataFrame:
    """Convert monthly utility summaries once, at the report boundary."""
    frame = read("compression_loss").copy()
    utility_columns = [column for column in frame if column.endswith("_bp")]
    frame[utility_columns] = 12 * frame[utility_columns]
    return frame


def display_year_ranges(text: str, separator: str = "--") -> str:
    return re.sub(r"(?<!\d)(\d{4})-(\d{4})(?!\d)", rf"\1{separator}\2", text)


def table(name, caption, label, headers, rows, note="", widths=None, header_detail=""):
    # Table values are formatted here, never edited in generated TeX.
    spec = widths or ("l" + "r" * (len(headers) - 1))
    body = "\n".join(" & ".join(str(cell) for cell in row) + r" \\" for row in rows)
    text = (
        r"\begin{table}[!htbp]\centering"
        + "\n"
        + r"\caption{"
        + caption
        + "}"
        + r"\label{"
        + label
        + "}\n"
        + r"\small\setlength{\tabcolsep}{5pt}"
        + "\n"
        + r"\begin{tabular}{"
        + spec
        + "}\n"
        + r"\toprule"
        + "\n"
        + " & ".join(headers)
        + r" \\"
        + "\n"
        + header_detail
        + r"\midrule"
        + "\n"
        + body
        + "\n"
        + r"\bottomrule\end{tabular}"
        + "\n"
    )
    if note:
        text += (
            r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize "
            + note
            + r"\end{minipage}"
            + "\n"
        )
    text += r"\end{table}" + "\n"
    (DEST / f"table_{name}.tex").write_text(display_year_ranges(text))


def make_tables(d):
    s = d["sample_summary"]
    benchmark = read("benchmark/sample_construction").set_index("item")["value"]

    def span(lo, hi):
        return str(int(lo)) if int(lo) == int(hi) else f"{int(lo)}--{int(hi)}"

    extension_spans = [span(r.min_eligible, r.max_eligible) for r in s.itertuples()]
    risk_window = f"{cfg.RISK_WINDOW} months"
    selection_window = f"{cfg.SELECTION_WINDOW} months"
    regression_window = f"{cfg.REGRESSION_PAIRS} pairs"
    gamma = rf"$\gamma={cfg.GAMMA:g}$"
    costs = f"{cfg.COST_PER_DOLLAR*1e4:g} bp / {cfg.SHORT_FEE_ANNUAL*1e4:g} bp"
    rows = [
        [
            "Evaluation months",
            benchmark["Common evaluation months"],
            *s.months.astype(str),
        ],
        [
            "Eligible industries",
            span(
                benchmark["Eligible industries at a formation month, minimum"],
                benchmark["Eligible industries at a formation month, maximum"],
            ),
            *extension_spans,
        ],
        ["Covariance window", risk_window, risk_window, risk_window],
        ["Forecast training", "Momentum score", regression_window, regression_window],
        ["Selection objective", "Gross Sharpe", "Net utility", "Net utility"],
        ["Selection window", selection_window, selection_window, selection_window],
        ["Risk scaling", rf"{cfg.TARGET_VOL*100:g}\% target", gamma, gamma],
        ["Trading cost / short fee", "Gross returns", costs, costs],
    ]
    table(
        "design",
        "Samples and common design settings.",
        "tab:design",
        ["Setting", "Replication", "1970--2018", "2019--2025"],
        rows,
        "Trading cost is per dollar traded; short fee is annual. The benchmark is evaluated in 1942--2018 on the common valid-month sample. Both extension portfolios use identical eligible industries and selected penalty levels.",
        "lccc",
    )
    r = read("replication_table")
    labels = [
        "Raw mean--variance",
        "Corrected mean--variance",
        r"EPO, $w=0.25$",
        r"EPO, $w=0.50$",
        r"EPO, $w=0.75$",
        r"EPO, $w=0.90$",
        r"EPO, $w=0.95$",
        r"EPO, $w=0.99$",
        "EPO, annual selection",
    ]
    table(
        "replication",
        "Replication of the industry benchmark.",
        "tab:replication",
        ["Portfolio", "Published Sharpe", "Replication Sharpe"],
        [
            [
                label,
                (
                    "---"
                    if pd.isna(row.sharpe_published)
                    else f"{row.sharpe_published:.2f}"
                ),
                f"{row.sharpe:.3f}",
            ]
            for label, (_, row) in zip(labels, r.iterrows())
        ],
        r"Published values: \citet{pedersen2021}, Table 5, Equity 1. Replication values use 863 common months in 1942--2018 and gross excess returns. The published grid has no $w=0.95$ entry. Appendix~\ref{app:data} analyzes the numerical differences.",
    )
    f = d["forecast_summary"].set_index("period")
    h = d["reliability_summary"].set_index("period")
    periods = list(f.index)
    specs = [
        ("Forecast error, root mean square (bp/month)", f, "rmse_forecast_bp_month", 1),
        ("Historical-mean error (bp/month)", f, "rmse_historical_mean_bp_month", 1),
        ("Pooled predictive $R^2$", f, "pooled_oos_r2", 4),
        ("Mean rank correlation", f, "mean_rank_ic", 4),
        ("Mean uncertainty dispersion", h, "mean_h_cv", 3),
        ("One-month uncertainty rank persistence", h, "mean_h_rank_corr_1m", 3),
        ("Twelve-month uncertainty rank persistence", h, "mean_h_rank_corr_12m", 3),
    ]
    rows = [
        [label] + [f"{frame.loc[p,col]:.{digits}f}" for p in periods]
        for label, frame, col, digits in specs
    ]
    rows.insert(
        4,
        ["95\\% interval for mean rank correlation"]
        + [
            f'[{f.loc[p,"rank_ic_ci_lower"]:.3f}, {f.loc[p,"rank_ic_ci_upper"]:.3f}]'
            for p in periods
        ],
    )
    table(
        "forecast",
        "Forecast accuracy and the estimated uncertainty profile.",
        "tab:forecast",
        ["Statistic", *periods],
        rows,
        r"Forecast accuracy uses 28,104 and 4,116 eligible industry-months, respectively. $R^2$ compares pooled squared errors with the rolling historical-mean forecast. Rank correlation compares forecasts with next-month excess returns. Uncertainty dispersion is the monthly population coefficient of variation of $\widehat h$, averaged over time. Intervals use 1,000 moving-block bootstrap samples with 12 consecutive months per block.",
    )
    p = read("performance_table")
    dep = read("deployment_sensitivity")
    deployment_intervals = read("empirical_intervals", VALUE_DIR)
    deployment_intervals = deployment_intervals[
        deployment_intervals.policy == "targeted_capped"
    ].set_index("period")
    rows = []
    order = [
        (period, strategy)
        for period in periods
        for strategy in ("scalar", "heterogeneous")
    ]
    metrics = [
        (r"Annual net mean (\%)", "net_excess_mean_pct_year", 2),
        (r"Annual volatility (\%)", "volatility_pct_year", 2),
        ("Net Sharpe ratio", "net_sharpe", 3),
        ("Annual net utility (bp)", "net_ce_bp_year", 2),
        (r"Monthly tail loss, 95\% (\%)", "es95_net_excess_loss_pct_month", 2),
        (r"Maximum drawdown (\%)", "max_drawdown_pct", 2),
        ("Mean monthly turnover", "mean_turnover_per_month", 3),
        ("Mean gross exposure", "mean_gross", 3),
    ]
    for title, frame, primary in [
        (r"A. Direct allocation: fixed $\gamma=3$", p, True),
        (r"B. Common position sizing: 10\% volatility target; gross cap 3", dep, False),
    ]:
        indexed = frame.set_index(["period", "strategy"])
        rows.append([r"\addlinespace\multicolumn{5}{l}{\textit{" + title + "}}"])
        for label, col, digits in metrics:
            rows.append(
                [label] + [f"{indexed.loc[key,col]:.{digits}f}" for key in order]
            )
        delta = [indexed.loc[(period, "heterogeneous")] for period in periods]
        rows.append(
            [r"Annual $\Delta\mathrm{CE}$ (bp)"]
            + [
                r"\multicolumn{2}{c}{" + f"{row.delta_ce_bp_year:.2f}" + "}"
                for row in delta
            ]
        )
        if primary:
            rows.append(
                [r"95\% paired block interval"]
                + [
                    r"\multicolumn{2}{c}{["
                    + f"{row.ci_lower_bp_year:.2f}, {row.ci_upper_bp_year:.2f}"
                    + "]}"
                    for row in delta
                ]
            )
        else:
            rows.append(
                [r"95\% paired block interval"]
                + [
                    r"\multicolumn{2}{c}{["
                    + f"{deployment_intervals.loc[period, 'delta_lower_bp_year']:.2f}, {deployment_intervals.loc[period, 'delta_upper_bp_year']:.2f}"
                    + "]}"
                    for period in periods
                ]
            )
            rows.append(
                ["Months with binding cap"]
                + [str(int(indexed.loc[key, "cap_months"])) for key in order]
            )
    table(
        "performance",
        "Matched net performance under two position-sizing rules.",
        "tab:performance",
        [
            "Statistic",
            r"\multicolumn{2}{c}{1970--2018}",
            r"\multicolumn{2}{c}{2019--2025}",
        ],
        rows,
        r"Each period contains 588 or 84 months. Certainty equivalents use risk aversion 3; cash has zero excess-return certainty equivalent. $\Delta\mathrm{CE}$ is heterogeneous minus scalar. Trading cost is 10 bp per dollar of full traded notional; the annual short fee is 50 bp. Both panels use 1,000 paired moving-block bootstrap samples with 12-month blocks. Panel B retains Panel A's selected penalties and recomputes its own position drift and costs. The target is predicted volatility under the common risk matrix, before the gross cap. Tail loss is monthly expected shortfall; drawdown uses total net wealth. Holdings continue through the 2019 boundary.",
        "lrrrr",
        header_detail=(
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}"
            + "\n"
            + r" & Scalar & Heterog. & Scalar & Heterog. \\"
            + "\n"
        ),
    )
    rob = d["robustness_summary"]
    targeted = read("targeted_cost_summary", BRIDGE_DIR)
    rows = []
    for scenario, cost in [
        ("frozen_w_0bp", 0),
        ("primary_10bp", 10),
        ("frozen_w_25bp", 25),
    ]:
        fixed = rob[(rob.scenario == scenario) & (rob.strategy == "scalar")].set_index(
            "period"
        )
        sized = targeted[
            (targeted.cost_bp == cost) & (targeted.block_months == 12)
        ].set_index("period")
        rows.append(
            [f"Trading cost: {cost} bp"]
            + [
                f"{frame.loc[period, 'delta_ce_bp_year']:.2f}"
                for period in periods
                for frame in (fixed, sized)
            ]
        )
    admission = rob[
        (rob.scenario == "admission_60") & (rob.strategy == "scalar")
    ].set_index("period")
    rows.append(
        [r"\addlinespace\multicolumn{5}{l}{\textit{Supporting eligibility comparison}}"]
    )
    rows.append(
        ["60 ready months"]
        + [
            cell
            for period in periods
            for cell in (f"{admission.loc[period, 'delta_ce_bp_year']:.2f}", "---")
        ]
    )
    table(
        "robustness",
        "Trading-cost sensitivity under both position-sizing rules.",
        "tab:robustness",
        [
            "Specification",
            r"\multicolumn{2}{c}{1970--2018}",
            r"\multicolumn{2}{c}{2019--2025}",
        ],
        rows,
        r"Entries are heterogeneous-minus-scalar net certainty equivalent differences in annual bp. Target/cap applies the 10\% predicted-volatility target and gross cap of three. Cost scenarios retain the selected penalty sequence and recompute drift, turnover and costs separately under each rule. The annual short fee remains 50 bp. The supporting 60-ready-month exercise uses fixed $\gamma$ and reselects the penalty; dashes identify the position-sizing rule outside that comparison. Appendix~\ref{app:performance} reports the targeted cost intervals.",
        "lrrrr",
        header_detail=r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}"
        + "\n"
        + r" & Fixed $\gamma$ & Target/cap & Fixed $\gamma$ & Target/cap \\"
        + "\n",
    )
    cost_rows = [
        [
            r.period,
            f"{r.cost_bp:g}",
            f"{r.scalar_ce_bp_year:.2f}",
            f"{r.heterogeneous_ce_bp_year:.2f}",
            f"{r.delta_ce_bp_year:.2f}",
            f"[{r.ci_lower_bp_year:.2f}, {r.ci_upper_bp_year:.2f}]",
        ]
        for r in targeted[targeted.block_months == 12].itertuples()
    ]
    table(
        "targeted_costs",
        "Net utility under targeted positions at alternative trading costs.",
        "tab:targeted-costs",
        [
            "Period",
            "Cost",
            "Scalar",
            "Heterog.",
            r"$\Delta\mathrm{CE}$",
            r"95\% interval",
        ],
        cost_rows,
        r"Cost is bp per dollar traded. Utilities and differences are annual bp. Intervals use 1,000 paired moving-block bootstrap samples with 12-month blocks. Selection is held fixed across cost rates, with each strategy's turnover recomputed from its own drifted positions.",
        "lrrrrr",
    )
    calibration = read("calibration_log_error", BRIDGE_DIR)
    table(
        "calibration_error",
        "Relative-uncertainty log error in estimator calibration.",
        "tab:calibration-error",
        [
            "Feedback",
            "$a_{\\mathrm{cal}}$",
            "Median SD",
            "5th--95th percentile",
            "True/reference CV",
        ],
        [
            [
                f"{r.feedback:g}",
                f"{r.implied_a:.3f}",
                f"{r.median_log_error_sd:.3f}",
                f"[{r.p05_log_error_sd:.3f}, {r.p95_log_error_sd:.3f}]",
                f"{100*r.mean_reference_cv:.1f}\\%",
            ]
            for r in calibration.itertuples()
        ],
        r"Each design contains 1,000 histories of 49 assets. SD is the within-history sample standard deviation of $\log(\widehat h/h^\star)$. $a_{\mathrm{cal}}$ is the square root of its mean squared value. Quantiles summarize history-specific SDs, not confidence limits for $a_{\mathrm{cal}}$. The zero-feedback reference uses the exact conditional OLS variance; feedback 0.4 uses an analytical reference. CV is the mean within-history coefficient of variation of normalized reference uncertainty. Figure~\ref{fig:estimation-tradeoff} uses the zero-feedback value.",
        "lrrrr",
    )
    c = annual_simulation_report()
    rows = []
    for _, r in c.iterrows():
        rows.append(
            [
                f"{r.target_cv:g}",
                r["subset"].replace("alignment_", "").replace("all", "All draws"),
                f"{r.realized_cv:.3f}",
                f"{r.compression_loss_bp:.0f} ({r.compression_loss_se_bp:.0f})",
                f"{max(0,r.choice_loss_bp):.0f} ({r.choice_loss_se_bp:.0f})",
                f"{100*r.compression_loss_share:.1f}",
            ]
        )
    intervals = []
    for row in c[(c["subset"] == "all") & (c.target_cv > 0)].itertuples():
        intervals.append(
            f"For target CV {row.target_cv:g}, the 95\\% compression interval is "
            f"[{row.compression_loss_lo_bp:.0f}, {row.compression_loss_hi_bp:.0f}] "
            f"and the choice interval is [{row.choice_loss_lo_bp:.0f}, {row.choice_loss_hi_bp:.0f}]."
        )
    table(
        "simulation",
        "Simulation loss by uncertainty profile and exposure alignment.",
        "tab:simulation",
        [
            "Target CV",
            "Draw group",
            "Actual CV",
            "Compression (SE)",
            "Choice (SE)",
            "Share (\\%)",
        ],
        rows,
        "Compression and choice losses and their Monte Carlo standard errors (SE) are in annual utility basis points: twelve times the monthly quantities. Share is mean compression loss divided by mean heterogeneous utility. Each profile has 1,000 draws; low, middle and high alignment groups contain 334, 333 and 333 draws. CV denotes coefficient of variation. "
        + " ".join(intervals),
        "llrrrr",
    )
    c = read("estimator_calibration")
    sub = c[c["subset"] == "all"].set_index("feedback")
    specs = [
        ("Conditional-mean estimation error (bp)", "rmse_bp", 2, 1),
        ("Bootstrap root mean variance (bp)", "root_mean_u_bp", 2, 1),
        ("Mean squared estimation error / mean variance", "mse_to_u", 3, 1),
        ("Normal interval coverage, 90\\%", "coverage_normal_0.9", 2, 100),
        ("Percentile interval coverage, 90\\%", "coverage_percentile_0.9", 2, 100),
        ("Normal interval coverage, 95\\%", "coverage_normal_0.95", 2, 100),
        ("Percentile interval coverage, 95\\%", "coverage_percentile_0.95", 2, 100),
        ("Variance rank correlation", "spearman_u_vs_ols", 3, 1),
        ("Relative-uncertainty rank correlation", "spearman_h_vs_relative", 3, 1),
    ]
    rows = [
        [l] + [f"{sub.loc[k,c]*scale:.{digits}f}" for k in (0, 0.4)]
        for l, c, digits, scale in specs
    ]
    table(
        "calibration",
        "Aggregate calibration of the conditional-mean uncertainty estimator.",
        "tab:calibration",
        ["Statistic", "Feedback 0", "Feedback 0.4"],
        rows,
        r"Each specification has 1,000 histories of 49 industries. Coverage is in percent; errors are monthly. Rank comparisons use the analytical regression-variance expression, which is exact conditional on predictors only in the zero-feedback design. Under feedback, it is a diagnostic reference. Both coverage calculations evaluate the known conditional mean.",
    )
    c = read("benchmark/convention_steps")
    labels = [
        "Published Equity 1 result",
        "Expanding selector; zero-filled month",
        "Common evaluation months",
        "Seven-value candidate grid",
        "10\\% predicted-volatility scaling",
        "Annual trailing-window selection",
    ]
    table(
        "conventions",
        "Sequential accounting for the benchmark difference.",
        "tab:conventions",
        ["Specification", "Months", "Sharpe"],
        [
            [
                l,
                "---" if pd.isna(r.months) else str(int(r.months)),
                f"{r.oos_epo_sharpe:.4f}",
            ]
            for l, (_, r) in zip(labels, c.iterrows())
        ],
        r"The sequence changes one implemented convention at a time on the current data vintage. It is an order-dependent accounting exercise. The expanding selector follows the monthly rule described by \citet[p.~138, Figure~2 and accompanying text]{pedersen2021}; applying it to Equity~1 is the diagnostic interpretation explained above. The zero-filled row is a diagnostic scenario; the primary replication does not fill a held missing return. The first row is the published Equity~1 result (p.~142, Table~5).",
    )
    f = read("benchmark/convention_steps_fixed")
    labels = [
        "Published Equity 1 result",
        "Unscaled positions; zero-filled month",
        "Common evaluation months",
        "10\\% predicted-volatility scaling",
    ]
    table(
        "conventions_fixed",
        "Sequential accounting for raw mean--variance and EPO with $w=0.99$.",
        "tab:conventions-fixed",
        ["Specification", "Months", "Raw mean--variance", r"EPO, $w=0.99$"],
        [
            [
                l,
                "---" if pd.isna(r.months) else str(int(r.months)),
                f"{r.raw_mvo_sharpe:.4f}",
                f"{r.epo_099_sharpe:.4f}",
            ]
            for l, (_, r) in zip(labels, f.iterrows())
        ],
        r"Rows follow Table~\ref{tab:conventions} without the selection steps. The second row uses unscaled positions, as the published values are invariant to the paper's constant ex post volatility scaling. The first row is from \citet{pedersen2021}.",
        "lrrr",
    )
    dce = read("delta_ce_summary").assign(policy=r"Fixed $\gamma$")
    sized_intervals = targeted[targeted.cost_bp == 10].assign(policy="Target/cap")
    combined = pd.concat([dce, sized_intervals], ignore_index=True)
    table(
        "intervals",
        "Block-length sensitivity under both position-sizing rules.",
        "tab:intervals",
        ["Period", "Position sizing", "Block months", "Difference", r"95\% interval"],
        [
            [
                r.period,
                r.policy,
                str(r.block_months),
                f"{r.delta_ce_bp_year:.2f}",
                f"[{r.ci_lower_bp_year:.2f}, {r.ci_upper_bp_year:.2f}]",
            ]
            for r in combined.sort_values(
                ["period", "policy", "block_months"]
            ).itertuples()
        ],
        "Differences and interval endpoints are annual bp at a trading-cost rate of 10 bp per dollar. Each interval uses 1,000 paired moving-block bootstrap samples. Target/cap uses a 10\\% predicted-volatility target and gross cap of three.",
        "llrrr",
    )


def make_figures(d):
    original = Figure.savefig

    def vector_save(fig, filename, *args, **kwargs):
        name = Path(filename).stem
        for text in list(fig.texts):
            text.set_visible(
                False
            )  # Retain layout ownership; the manuscript supplies captions.
        for artist in fig.findobj(Text):
            artist.set_text(
                display_year_ranges(
                    artist.get_text()
                    .replace("12 months (primary)", "12 months (baseline)")
                    .replace("10 bp; primary", "10 bp; baseline"),
                    "–",
                )
            )
            if artist.get_visible():
                artist.set_fontsize(max(12, artist.get_fontsize()))
        if name in ("delta_ce_intervals", "robustness"):
            # FixedFormatter regenerates tick text when drawing; update its labels.
            for ax in fig.axes:
                labels = [
                    tick.get_text().replace("primary", "baseline")
                    for tick in ax.get_yticklabels()
                ]
                ax.set_yticks(ax.get_yticks(), labels, fontsize=12)
        if name == "estimator_calibration":
            fig.set_size_inches(9.5, 7.7)
            for ax in fig.axes[:2]:
                ax.set_ylabel("Conditional-mean error (bp)", fontsize=12)
                ax.set_xlabel("Bootstrap standard error (bp)", fontsize=12)
            for ax, box in zip(
                fig.axes,
                [
                    (0.10, 0.60, 0.35, 0.32),
                    (0.62, 0.60, 0.35, 0.32),
                    (0.10, 0.09, 0.87, 0.34),
                ],
            ):
                ax.set_position(box)
            fig.axes[0].set_title(
                "Grouped by analytical variance", loc="left", fontsize=12
            )
            fig.axes[1].set_title(
                "Grouped by estimated variance", loc="left", fontsize=12
            )
        if name == "portfolio_transmission":
            # Reuse the exact plotted series, placing composition in the main text
            # and exposure levels in the appendix without duplicating panels.
            axes = list(fig.axes)
            fig.set_size_inches(11, 3.8)
            boxes = [(0.08, 0.18, 0.39, 0.70), (0.59, 0.18, 0.39, 0.70)]
            for ax in axes[:2]:
                ax.set_visible(False)
            for ax, box in zip(axes[2:], boxes):
                ax.set_position(box)
            axes[2].set_title("Gross-normalized composition distance", fontsize=12)
            axes[3].set_title("Risk-direction correlation", fontsize=12)
            original(
                fig,
                DEST / "portfolio_composition.pdf",
                format="pdf",
                bbox_inches="tight",
                facecolor="white",
            )
            for ax in axes[2:]:
                ax.set_visible(False)
            for ax, box in zip(axes[:2], boxes):
                ax.set_visible(True)
                ax.set_position(box)
                ax.set_xlabel("Realization month")
        if name == "problem_pc":
            fig.subplots_adjust(hspace=0.48, bottom=0.09, top=0.96)
            sharpe = fig.axes[2]
            sharpe.set_ylim(-0.12, max(0.65, sharpe.get_ylim()[1]))
            sharpe.set_yticks(np.arange(-0.1, 0.7, 0.1))
            sharpe.axhline(0, color="#52514e", linewidth=0.8)
            for ax in fig.axes:
                ax.set_xticks(
                    range(1, 6), ["G1\n(smallest)", "G2", "G3", "G4", "G5\n(largest)"]
                )
        return original(
            fig,
            DEST / f"{name}.pdf",
            format="pdf",
            bbox_inches="tight",
            facecolor="white",
        )

    with patch.object(Figure, "savefig", vector_save):
        replication.plot_diagnostics(
            read("problem_pc_diagnostics").set_index("eigenvalue_group")
        )
        # Main simulation figures are generated directly in utility-share units.
        simulation.plot_calibration(read("estimator_calibration"))
        evaluation.plot_equity(read("equity_curves"))
        diagnostic_plots.plot_all(
            d, read("annual_selection"), read("delta_ce_summary"), DEST
        )
        spec = importlib.util.spec_from_file_location(
            "benchmark_report", ROOT / "scripts/benchmark_report.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monthly = read("benchmark/sample_monthly")
        monthly.iloc[:, 0] = pd.to_datetime(monthly.iloc[:, 0])
        counts = pd.Series(
            monthly.iloc[:, 1].to_numpy(), index=pd.DatetimeIndex(monthly.iloc[:, 0])
        )
        module.plot_sample(
            counts,
            read("benchmark/industry_descriptives"),
            list(range(1944, 1949)),
            [pd.Timestamp("1943-07-31")],
            range(863),
        )
        module.plot_selection(
            read("benchmark/selected_w").set_index("year")["selected"],
            list(range(1944, 1949)),
        )
    plt.close("all")


def make_compression_figure():
    draws = read("simulation_draws")
    summaries = []
    for cv, part in draws.groupby("target_cv", sort=True):
        groups = [("all", part)]
        if cv > 0:
            tercile = simulation.alignment_terciles(part.alignment)
            groups += [
                (name, part.loc[tercile == name]) for name in ("low", "middle", "high")
            ]
        for name, group in groups:
            row = {
                "target_cv": cv,
                "realized_cv": group.realized_cv.iloc[0],
                "subset": name,
            }
            for col in ("compression_loss", "choice_loss"):
                share, se = utility_share(
                    group[col].to_numpy(), group.utility_het.to_numpy()
                )
                row[col] = max(0, share)
                row[col + "_se"] = se
            summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(DEST / "simulation_shares.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    all_draws = summary[summary.subset == "all"]
    positions = np.arange(len(all_draws))
    for offset, col, label, color in [
        (-0.19, "compression_loss", "Compression (best scalar)", "#2a78d6"),
        (0.19, "choice_loss", "Additional choice loss", "#eb6834"),
    ]:
        axes[0].bar(
            positions + offset,
            all_draws[col],
            width=0.35,
            yerr=1.96 * all_draws[col + "_se"],
            capsize=3,
            color=color,
            label=label,
        )
    axes[0].set_xticks(positions, [f"{100*x:.0f}%" for x in all_draws.realized_cv])
    axes[0].set(xlabel="True uncertainty CV", title="Loss from scalar restriction")
    for offset, cv, color in [(-0.19, 0.5, "#2a78d6"), (0.19, 1, "#eb6834")]:
        part = summary[(summary.target_cv == cv) & (summary.subset != "all")]
        axes[1].bar(
            np.arange(3) + offset,
            part.compression_loss,
            width=0.35,
            yerr=1.96 * part.compression_loss_se,
            capsize=3,
            color=color,
            label=f"True CV: {100*part.realized_cv.iloc[0]:.0f}%",
        )
    axes[1].set_xticks(range(3), ["Low", "Middle", "High"])
    axes[1].set(
        xlabel="Exposure-alignment group", title="Compression by exposure alignment"
    )
    for ax in axes:
        ax.set_ylabel("Share of mean oracle utility (%)", fontsize=12)
        ax.set_ylim(bottom=0)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=12)
        ax.xaxis.label.set_fontsize(12)
        ax.title.set_fontsize(14)
        ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.savefig(DEST / "compression_loss.pdf", bbox_inches="tight")
    plt.close(fig)


def make_value_assets():
    intervals = read("empirical_intervals", VALUE_DIR)
    rows = []
    for period, part in intervals.groupby("period", sort=False):
        rows.append([r"\addlinespace\multicolumn{6}{l}{\textit{" + period + "}}"])
        for r in part.itertuples():
            rows.append(
                [
                    (
                        r"Fixed $\gamma$"
                        if r.policy == "fixed_gamma"
                        else r"10\% target; cap 3"
                    ),
                    f"{r.delta_ce_bp_year:.2f}",
                    f"{r.random_mean_delta_bp_year:.2f}",
                    f"{r.assignment_advantage_bp_year:.2f}",
                    f"[{r.assignment_lower_bp_year:.2f}, {r.assignment_upper_bp_year:.2f}]",
                    f"{r.below_original_pct:.0f}",
                ]
            )
    table(
        "label_placebo",
        "The value of the original uncertainty-to-asset assignment.",
        "tab:label-placebo",
        [
            "Position sizing",
            "Original",
            "Random mean",
            "$D$",
            r"95\% interval for $D$",
            r"Rank (\%)",
        ],
        rows,
        r"Original and Random mean are annual net certainty equivalent differences from scalar, in bp. Random mean averages the certainty equivalents of 100 fixed random assignments. $D$ is Original minus Random mean. Intervals use 1,000 paired moving-block bootstrap samples of all strategy returns together, with 12-month blocks and the assignments held fixed. Rank is the percentage of random assignments with lower net certainty equivalent than the original. Appendix~\ref{app:label-placebo} gives the assignment algorithm and distribution summaries.",
        "lrrrrr",
    )
    rows = []
    for r in intervals.itertuples():
        rows.append(
            [
                r.period,
                (
                    r"Fixed $\gamma$"
                    if r.policy == "fixed_gamma"
                    else r"10\% target; cap 3"
                ),
                f"{r.random_mean_delta_bp_year:.2f}",
                f"{r.random_median_delta_bp_year:.2f}",
                f"{r.below_original_pct:.0f}",
            ]
        )
    table(
        "assignment_distribution",
        "Random-assignment net utility summaries.",
        "tab:assignment-distribution",
        ["Period", "Position sizing", "Mean", "Median", r"Below original (\%)"],
        rows,
        r"Mean and median are annual net certainty equivalent differences from scalar in bp, across the same 100 assignments. The percentage below the original is a descriptive rank. All strategies remain solvent.",
        "llrrr",
    )

    summary = read("error_simulation_summary", VALUE_DIR)
    rows = []
    for r in summary.itertuples():

        def rounded(value):
            return f"{0 if abs(value) < 0.5 else value:,.0f}"

        rows.append(
            [
                f"{100*r.realized_cv:.0f}",
                f"{r.error_sd:.3f}".rstrip("0").rstrip("."),
                rounded(r.scalar_loss_bp_year),
                rounded(r.estimation_penalty_bp_year),
                rounded(r.net_gain_bp_year),
                f"[{rounded(r.net_gain_lower_bp_year)}, {rounded(r.net_gain_upper_bp_year)}]",
                f"{100*r.probability_gain_positive:.1f}",
            ]
        )
    table(
        "estimation_tradeoff",
        "Scalar restriction cost and uncertainty-estimation loss.",
        "tab:estimation-tradeoff",
        [
            r"CV (\%)",
            "$a$",
            r"$L_{\bar\kappa}$",
            "$P$",
            r"$L_{\bar\kappa}-P$",
            r"95\% interval",
            r"$P<L_{\bar\kappa}$ (\%)",
        ],
        rows,
        r"CV is the realized coefficient of variation of true relative uncertainty. $a$ is the log standard deviation of the multiplicative estimation error. Utilities and interval endpoints are annual bp; entries are means across 1,000 paired draws. Intervals equal the mean plus or minus 1.96 Monte Carlo standard errors. The last column reports the fraction of draws with a positive gain, using a monthly numerical tolerance of $10^{-12}$. The common scalar is $\bar\kappa=\lambda/\tau=2.5$. Each error level uses the same true uncertainty, signals and standard-normal estimation errors. The additional level $a=0.414$ matches calibration log-error dispersion; calculations use its unrounded value.",
        "rrrrrrr",
    )

    calibration = read("calibration_log_error", BRIDGE_DIR)
    calibrated_a = float(
        calibration.loc[calibration.feedback == 0, "implied_a"].iloc[0]
    )
    colors = ["#2a78d6", "#eb6834", "#23866b"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.4), layout="constrained", sharey=True)
    for ax, (_, part) in zip(axes, summary.groupby("target_cv", sort=True)):
        for col, label, color, marker in zip(
            ["scalar_loss", "estimation_penalty", "net_gain"],
            [
                r"Restriction cost $L_{\bar\kappa}$",
                r"Estimation loss $P$",
                r"Net gain $L_{\bar\kappa}-P$",
            ],
            colors,
            ["o", "s", "^"],
        ):
            ax.errorbar(
                part.error_sd,
                part[f"{col}_share_pct"],
                yerr=1.96 * part[f"{col}_share_se_pct"],
                label=label,
                color=color,
                marker=marker,
                capsize=3,
                linewidth=1.8,
            )
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.axvline(
            calibrated_a,
            color="#555555",
            linestyle="--",
            linewidth=1.3,
            label=rf"Calibration $a={calibrated_a:.3f}$",
        )
        ax.set(
            title=f"True uncertainty CV: {100*part.realized_cv.iloc[0]:.0f}%",
            xlabel=r"Error log SD $a$",
            xticks=[0, 0.5, 1],
        )
        ax.title.set_fontsize(13)
        ax.xaxis.label.set_fontsize(14)
        ax.tick_params(labelsize=13)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Share of mean oracle utility (%)", fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=12
    )
    fig.savefig(DEST / "estimation_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)


def diagnostics_current(directory, required):
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest.get("input_code_sha256", {}).items():
        path = ROOT / name
        if (
            not path.exists()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            return False
    for name in required:
        path = directory / f"{name}.csv"
        expected = manifest.get("output_sha256", {}).get(path.name)
        if (
            not path.exists()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            return False
    return True


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    diagnostic_dir = OUT / "diagnostics"
    required = [
        "sample_summary",
        "sample_monthly",
        "industry_descriptives",
        "forecast_deciles",
        "forecast_monthly",
        "forecast_summary",
        "reliability_monthly",
        "reliability_industry_year",
        "reliability_summary",
        "portfolio_monthly",
        "portfolio_summary",
        "ce_decomposition",
        "robustness_summary",
    ]
    if not diagnostics_current(diagnostic_dir, required):
        diagnostic_dir = DEST / "diagnostic_data"
        if not diagnostics_current(diagnostic_dir, required):
            print(
                "Regenerating diagnostics from tracked empirical inputs...", flush=True
            )
            diagnostics.run(diagnostic_dir, make_plots=False)
    INPUTS.add(diagnostic_dir / "manifest.json")
    data = {name: read(name, diagnostic_dir) for name in required}
    if not diagnostics_current(
        PLACEBO_DIR,
        ["label_placebo_summary", "label_placebo_metrics", "label_placebo_returns"],
    ):
        run_label_placebo()
    INPUTS.add(PLACEBO_DIR / "manifest.json")
    if not diagnostics_current(
        BRIDGE_DIR,
        [
            "calibration_log_error",
            "targeted_cost_summary",
            "relative_uncertainty_draws",
            "targeted_cost_returns",
        ],
    ):
        run_calibration_bridge()
    INPUTS.add(BRIDGE_DIR / "manifest.json")
    if not diagnostics_current(
        VALUE_DIR,
        [
            "error_simulation_summary",
            "empirical_intervals",
            "empirical_bootstrap",
            "error_simulation_draws",
        ],
    ):
        run_uncertainty_value()
    INPUTS.add(VALUE_DIR / "manifest.json")
    make_compression_figure()
    make_value_assets()
    make_tables(data)
    make_figures(data)
    code = (
        list((ROOT / "src/epo_project").glob("*.py"))
        + list(REPORT.rglob("*.py"))
        + list((REPORT / "sections").glob("*.tex"))
        + [REPORT / "main.tex", REPORT / "references.bib"]
    )
    files = sorted(INPUTS | set(code))
    manifest = {
        "report_units": {
            "simulation_utility": "main figures/text: percent of mean oracle utility; appendix: annual basis points, 12 times monthly utility",
            "simulation_intervals_and_standard_errors": "same factor of 12",
            "forecast_error_and_bootstrap_standard_error": "monthly basis points",
            "simulation_loss_shares": "ratio of means; paired delta-method Monte Carlo standard errors",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "inputs_and_code_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
        "outputs_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(DEST.glob("*"))
            if p.suffix in (".pdf", ".tex", ".csv")
        },
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f'Generated {len(list(DEST.glob("*.pdf")))} figures and {len(list(DEST.glob("*.tex")))} tables.',
        flush=True,
    )


if __name__ == "__main__":
    main()
