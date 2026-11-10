"""Generate manuscript tables and vector figures from the project's result files."""

from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import platform
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

INPUTS: set[Path] = set()


def read(name, directory=OUT):
    path = directory / f"{name}.csv"
    INPUTS.add(path)
    frame = pd.read_csv(path)
    for column in ("date", "formation_date"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column])
    return frame


def table(name, caption, label, headers, rows, note="", widths=None):
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
        + r" \\ \midrule"
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
    (DEST / f"table_{name}.tex").write_text(text)


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
        r"Forecast accuracy uses 28,104 and 4,116 eligible industry-months, respectively. $R^2$ compares pooled squared errors with the rolling historical-mean forecast. Rank correlation compares forecasts with next-month excess returns. Uncertainty dispersion is the monthly population coefficient of variation of $\widehat h$, averaged over time. Intervals use 1,000 whole-month block resamples of length 12.",
    )
    p = read("performance_table")
    rows = []
    for period in periods:
        sub = p[p.period == period].set_index("strategy")
        rows.append([r"\multicolumn{3}{l}{\textit{" + period.replace("-", "--") + "}}"])
        for label, col, digits in [
            ("Annual net mean (\\%)", "net_excess_mean_pct_year", 2),
            ("Annual volatility (\\%)", "volatility_pct_year", 2),
            ("Net Sharpe ratio", "net_sharpe", 3),
            ("Annual net certainty equivalent (bp)", "net_ce_bp_year", 2),
            (
                "Monthly expected shortfall, 95\\% (\\%)",
                "es95_net_excess_loss_pct_month",
                2,
            ),
            ("Maximum drawdown (\\%)", "max_drawdown_pct", 2),
            ("Mean monthly turnover", "mean_turnover_per_month", 3),
            ("Mean gross exposure", "mean_gross", 3),
        ]:
            rows.append(
                [label]
                + [f"{sub.loc[k,col]:.{digits}f}" for k in ("scalar", "heterogeneous")]
            )
        v = sub.loc["heterogeneous"]
        rows.append(
            [
                r"Heterogeneous minus scalar, annual utility (bp)",
                "",
                f"{v.delta_ce_bp_year:.2f}",
            ]
        )
        rows.append(
            [
                "95\\% paired block interval",
                "",
                f"[{v.ci_lower_bp_year:.2f}, {v.ci_upper_bp_year:.2f}]",
            ]
        )
    table(
        "performance",
        "Matched portfolio performance after costs.",
        "tab:performance",
        ["Statistic", "Scalar", "Heterogeneous"],
        rows,
        r"Risk aversion is 3. Trading cost is 10 bp per dollar of full traded notional; the short fee is 50 bp per year. Expected shortfall is a monthly net-excess-return loss. Drawdown uses total net wealth. Confidence intervals use 1,000 paired 12-month block resamples. Positions continue through the 2019 boundary.",
    )
    rob = d["robustness_summary"]
    dep = read("deployment_sensitivity")
    rows = []
    for scenario, label in [
        ("primary_10bp", "Primary: 10 bp trading cost"),
        ("frozen_w_0bp", "Zero trading cost; fixed selection"),
        ("frozen_w_25bp", "25 bp trading cost; fixed selection"),
        ("admission_60", "60 ready months; reselected penalty"),
    ]:
        sub = rob[(rob.scenario == scenario) & (rob.strategy == "scalar")].set_index(
            "period"
        )
        rows.append([label] + [f'{sub.loc[p,"delta_ce_bp_year"]:.2f}' for p in periods])
    sub = dep[dep.strategy == "heterogeneous"].set_index("period")
    rows.append(
        ["10\\% volatility target; gross cap 3"]
        + [f'{sub.loc[p,"delta_ce_bp_year"]:.2f}' for p in periods]
    )
    table(
        "robustness",
        "Sensitivity of the annual net utility difference.",
        "tab:robustness",
        ["Specification", *periods],
        rows,
        r"Entries are heterogeneous minus scalar certainty equivalents, in annual basis points. Cost scenarios retain the primary selected penalty sequence and recompute position drift. The entry-rule exercise reruns selection. Deployment retains primary selection but rescales positions; its uncertainty interval is not estimated. Appendix~\ref{app:performance} reports intervals for the other scenarios.",
    )
    c = read("compression_loss")
    rows = []
    for _, r in c.iterrows():
        rows.append(
            [
                f"{r.target_cv:g}",
                r["subset"].replace("alignment_", "").replace("all", "All draws"),
                f"{r.realized_cv:.3f}",
                f"{r.compression_loss_bp:.2f}",
                f"{max(0,r.choice_loss_bp):.2f}",
                f"{100*r.compression_loss_share:.1f}",
            ]
        )
    table(
        "simulation",
        "Simulation loss by uncertainty profile and exposure alignment.",
        "tab:simulation",
        [
            "Target CV",
            "Draw group",
            "Actual CV",
            "Compression",
            "Choice",
            "Share (\\%)",
        ],
        rows,
        "Compression and choice losses are in monthly utility basis points. Share is mean compression loss divided by mean heterogeneous utility. Each profile has 1,000 draws; low, middle and high alignment groups contain 334, 333 and 333 draws. CV denotes coefficient of variation.",
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
        "Published procedure; zero-filled month",
        "Common evaluation months",
        "Project candidate grid",
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
        r"The sequence changes one implemented convention at a time on the current data vintage. It is an order-dependent accounting exercise. The zero-filled row is a diagnostic scenario; the primary replication does not fill a held missing return. The first row is from \citet{pedersen2021}.",
    )
    dce = read("delta_ce_summary")
    table(
        "intervals",
        "Block-length sensitivity of the net utility difference.",
        "tab:intervals",
        ["Period", "Block months", "Difference", "95\\% interval"],
        [
            [
                r.period,
                str(r.block_months),
                f"{r.delta_ce_bp_year:.2f}",
                f"[{r.ci_lower_bp_year:.2f}, {r.ci_upper_bp_year:.2f}]",
            ]
            for r in dce.itertuples()
        ],
        "Differences and interval endpoints are annual basis points; each interval uses 1,000 paired resamples.",
        "lrrr",
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
            if artist.get_visible():
                artist.set_fontsize(max(12, artist.get_fontsize()))
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
        if name == "problem_pc":
            fig.subplots_adjust(hspace=0.48, bottom=0.09, top=0.96)
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
        simulation.plot_loss(read("compression_loss"))
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
            if p.suffix in (".pdf", ".tex")
        },
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f'Generated {len(list(DEST.glob("*.pdf")))} figures and {len(list(DEST.glob("*.tex")))} tables.',
        flush=True,
    )


if __name__ == "__main__":
    main()
