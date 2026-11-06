"""Report tables and figures for the data/sample and benchmark layers (Steps 0-3).

Reads the Step 1-2 inputs and writes only to output/benchmark:
    sample_construction.csv     how the 1942-2018 benchmark sample is built
    sample_monthly.csv          eligible industries at each formation month
    industry_descriptives.csv   per-industry excess return and volatility
    industry_summary.csv        cross-section of those industry statistics
    sample_overview.png         eligible industries and industry distributions
    convention_grid.csv         Sharpe under each combination of conventions
    convention_steps.csv        one path from the published procedure to ours
    selected_w.csv              w chosen each January
    selected_w.png              the same as a step chart

Run from the repository root: PYTHONPATH=src python scripts/benchmark_report.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from epo_project import config as cfg
from epo_project.data import load_inputs, load_returns
from epo_project.epo import epo_weights, scale_to_target_vol
from epo_project.rolling import annual_selection, annualized_sharpe, apply_selection

OUT = cfg.OUTPUT_DIR / "benchmark"
START, END = cfg.BENCHMARK_PERIOD
PROPOSAL_GRID = tuple(cfg.W_GRID)
PUBLISHED_GRID = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.00)   # Pedersen, Babu and Levine (2021)
PUBLISHED_OOS = 0.79

INK, MUTED, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"


def candidate_tables():
    """Candidate returns by realization month, plus the eligible set at each formation month.

    scaled and unscaled keep a month missing when a held industry has no return;
    zero_filled sets that return to zero and is used only for the 924-month row.
    """
    excess = load_returns("excess_return")
    z, _, models = load_inputs()
    dates = excess.index
    grid = sorted(set(PROPOSAL_GRID + PUBLISHED_GRID))
    scaled, unscaled, zero_filled, members_by_t = [], [], [], {}
    for t, (members, _, q) in models.items():
        members_by_t[t] = list(members)
        pos = dates.get_loc(t)
        if pos + 1 >= len(dates):
            continue
        t1 = dates[pos + 1]
        s = z.loc[t, members].to_numpy()
        r_next = excess.loc[t1, members]
        missing = r_next.isna().any()
        r = r_next.fillna(0.0).to_numpy()
        x = {w: epo_weights(q, s, w) for w in grid}
        scaled.append({"date": t1, **{w: np.nan if missing else scale_to_target_vol(v, q) @ r for w, v in x.items()}})
        unscaled.append({"date": t1, **{w: np.nan if missing else v @ r for w, v in x.items()}})
        zero_filled.append({"date": t1, **{w: v @ r for w, v in x.items()}})
    tables = [pd.DataFrame(rows).set_index("date") for rows in (scaled, unscaled, zero_filled)]
    return (*tables, members_by_t, excess)


def annual_returns(candidates: pd.DataFrame, grid) -> pd.Series:
    """Proposal rule: choose w each January from the preceding 60 months."""
    columns = list(grid)
    return apply_selection(candidates[columns], annual_selection(candidates[columns]))["return"]


def expanding_returns(candidates: pd.DataFrame, grid) -> pd.Series:
    """Published rule: choose w every month from all earlier observed months."""
    table = candidates[list(grid)]
    rows = {}
    for i, month in enumerate(table.index):
        past = table.iloc[:i].dropna()
        if len(past) >= 2:
            rows[month] = table.at[month, past.apply(annualized_sharpe).idxmax()]
    return pd.Series(rows)


def sample_tables(scaled, members_by_t, excess, common):
    start, end = pd.Timestamp(START), pd.Timestamp(END)
    window = excess.loc[start:end]
    formation = excess.index[excess.index.get_indexer(window.index) - 1]
    counts = pd.Series({t: len(members_by_t[t]) for t in formation}, name="eligible_industries").rename_axis("formation_date")
    missing_held = [m for m in window.index if scaled.loc[m, list(PROPOSAL_GRID)].isna().any()]
    skipped = [y for y in range(start.year, end.year + 1) if y not in annual_selection(scaled[list(PROPOSAL_GRID)]).index]

    rows = []
    for t, month in zip(formation, window.index):
        if month in common:
            rows += [(name, month, excess.at[month, name]) for name in members_by_t[t]]
    panel = pd.DataFrame(rows, columns=["industry", "date", "excess"])

    descriptives = panel.groupby("industry", sort=False).excess.agg(
        months="count",
        mean_excess_pct_year=lambda r: 1200 * r.mean(),
        volatility_pct_year=lambda r: 100 * np.sqrt(12) * r.std(ddof=1),
    )
    descriptives["sharpe"] = descriptives.mean_excess_pct_year / descriptives.volatility_pct_year
    summary = descriptives.drop(columns="months").describe(percentiles=[.25, .5, .75]).T[
        ["min", "25%", "50%", "75%", "max"]].rename_axis("statistic")

    construction = pd.DataFrame([
        ("Industries in the data file", window.shape[1]),
        ("First realization month", window.index[0].date()),
        ("Last realization month", window.index[-1].date()),
        ("Realization months in the period", len(window)),
        ("Missing industry-months in the period (all industries)", int(window.isna().sum().sum())),
        ("Months with a missing return on a held industry", len(missing_held)),
        ("That month", "; ".join(str(m.date()) for m in missing_held)),
        ("Selection years skipped", f"{min(skipped)}-{max(skipped)}"),
        ("Months in the skipped years", int(window.index.year.isin(skipped).sum())),
        ("Common evaluation months", len(common)),
        ("Eligible industries at a formation month, minimum", int(counts.min())),
        ("Eligible industries at a formation month, mean", round(float(counts.mean()), 1)),
        ("Eligible industries at a formation month, maximum", int(counts.max())),
        ("First formation month with all industries eligible thereafter", counts[counts < counts.max()].index[-1] + pd.offsets.MonthEnd(1)),
        ("Industries in the descriptive statistics", len(descriptives)),
        ("Eligible industry-months in the common evaluation months", len(panel)),
    ], columns=["item", "value"])
    construction["value"] = construction["value"].map(lambda v: v.date() if isinstance(v, pd.Timestamp) else v)
    return construction, counts, descriptives, summary, skipped, missing_held


def convention_tables(scaled, unscaled, zero_filled, common):
    grid_rows = []
    for positions, table in (("Unscaled", unscaled), ("Scaled to 10% predicted volatility", scaled)):
        for selection, rule in (("Every month, all earlier months", expanding_returns),
                                ("Each January, preceding 60 months", annual_returns)):
            grid_rows.append({"positions": positions, "selection_of_w": selection,
                              "sharpe_published_grid": annualized_sharpe(rule(table, PUBLISHED_GRID).reindex(common)),
                              "sharpe_proposal_grid": annualized_sharpe(rule(table, PROPOSAL_GRID).reindex(common))})
    grid = pd.DataFrame(grid_rows)

    published_924 = annualized_sharpe(expanding_returns(zero_filled, PUBLISHED_GRID).loc[START:END])
    path = [("Published result (Table 5, Equity 1)", np.nan, PUBLISHED_OOS),
            ("Published procedure, all months, missing return set to zero", 924, published_924),
            ("Drop 1943-07 and 1944-1948", len(common), grid.sharpe_published_grid[0]),
            ("Use the proposal grid", len(common), grid.sharpe_proposal_grid[0]),
            ("Scale to 10% predicted volatility", len(common), grid.sharpe_proposal_grid[2]),
            ("Choose w each January from the preceding 60 months", len(common), grid.sharpe_proposal_grid[3])]
    steps = pd.DataFrame(path, columns=["step", "months", "oos_epo_sharpe"])
    steps["change"] = steps.oos_epo_sharpe.diff()
    return grid, steps


def style(ax):
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(MUTED)


def distribution(ax, values, bins, title, xlabel):
    from matplotlib.ticker import MaxNLocator

    ax.hist(values, bins=bins, color=BLUE, rwidth=.92)
    median = values.median()
    ax.axvline(median, color=INK, linewidth=1)
    ax.annotate(f"median {median:.1f}%", (median, 1), xycoords=("data", "axes fraction"), xytext=(4, -2),
                textcoords="offset points", fontsize=8, color=INK, va="top")
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set(xlabel=xlabel, ylabel="Number of industries")
    ax.xaxis.label.set(color=MUTED, size=8)
    ax.yaxis.label.set(color=MUTED, size=8)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color=GRID, linewidth=.8)
    style(ax)


def plot_sample(counts, descriptives, skipped, missing_held, common):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    ax = axes[0]
    ax.axvspan(pd.Timestamp(min(skipped), 1, 1), pd.Timestamp(max(skipped), 12, 31), color=GRID, zorder=0)
    ax.step(counts.index, counts.to_numpy(), where="post", color=BLUE, linewidth=2)
    ax.axvline(missing_held[0], color=ORANGE, linewidth=1.2)
    ax.annotate(f"{missing_held[0]:%Y-%m}: Rubbr\nreturn missing", (missing_held[0], 45.3),
                xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
    ax.annotate(f"{min(skipped)}-{max(skipped)}: no selection,\nnot evaluated", (pd.Timestamp(min(skipped), 1, 1), 40.75),
                xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK, va="bottom")
    ax.set_ylim(40, 50)
    ax.set_title("Eligible industries at each formation month", fontsize=10, color=INK, loc="left")
    ax.set_ylabel("Number of industries", fontsize=8, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=.8)
    style(ax)

    distribution(axes[1], descriptives.mean_excess_pct_year, np.arange(2, 15, 2),
                 "Industry mean excess return", "Annualized mean excess return (%)")
    distribution(axes[2], descriptives.volatility_pct_year, np.arange(12, 41, 4),
                 "Industry return volatility", "Annualized volatility (%)")

    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.text(.01, .015, f"Notes: Benchmark sample, 1942-2018. An industry is eligible with at least 60 consecutive monthly returns ending at the formation month.\n"
                        f"Return and volatility use each industry's eligible months within the {len(common)} common evaluation months.",
             fontsize=8, color=MUTED)
    fig.savefig(OUT / "sample_overview.png", dpi=160, facecolor="white")
    plt.close(fig)


def plot_selection(selected, skipped):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = {w: i for i, w in enumerate(PROPOSAL_GRID)}
    years = pd.Index(range(selected.index.min(), selected.index.max() + 1), name="year")
    position = selected.map(levels).reindex(years)

    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    ax.axvspan(min(skipped), max(skipped) + 1, color=GRID, zorder=0)
    ax.annotate(f"{min(skipped)}-{max(skipped)}: no selection", (min(skipped) + .5, len(levels) - .55), fontsize=8, color=INK, va="top")
    for start_index in np.split(np.arange(len(years)), np.where(position.isna())[0]):
        part = position.iloc[start_index].dropna()
        if len(part):
            ax.step(np.r_[part.index, part.index[-1] + 1], np.r_[part, part.iloc[-1]], where="post", color=BLUE, linewidth=2)
    ax.set_yticks(list(levels.values()), [f"{w:g}" for w in levels], fontsize=8, color=MUTED)
    ax.set_ylim(-.4, len(levels) - .5)
    ax.set_ylabel("Selected w", fontsize=8, color=MUTED)
    ax.set_title("Scalar EPO: w chosen each January from the preceding 60 months", fontsize=10, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=.8)
    style(ax)
    fig.tight_layout(rect=(0, .06, 1, 1))
    changes = int((selected.diff().fillna(0) != 0).sum())
    fig.text(.01, .02, f"Notes: {len(selected)} selection years, 1942-2018; w changes {changes} times and equals 0 in {int((selected == 0).sum())} years. "
                       "Candidate positions are scaled to 10% predicted volatility. Grid spacing is not to scale.",
             fontsize=8, color=MUTED)
    fig.savefig(OUT / "selected_w.png", dpi=160, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scaled, unscaled, zero_filled, members_by_t, excess = candidate_tables()

    proposal = list(PROPOSAL_GRID)
    selection = annual_selection(scaled[proposal])
    oos = apply_selection(scaled[proposal], selection)
    panel = scaled[proposal].assign(oos=oos["return"]).loc[START:END]
    common = panel.dropna().index

    construction, counts, descriptives, summary, skipped, missing_held = sample_tables(scaled, members_by_t, excess, common)
    grid, steps = convention_tables(scaled, unscaled, zero_filled, common)

    published_row = pd.read_csv(cfg.OUTPUT_DIR / "replication_table.csv").set_index("portfolio").loc["OOS EPO"]
    if int(published_row["months"]) != len(common) or abs(published_row["sharpe"] - steps.oos_epo_sharpe.iloc[-1]) > 1e-4:
        raise ValueError("The final step does not match replication_table.csv.")

    selected = selection.loc[int(START[:4]):int(END[:4]), "selected"]

    construction.to_csv(OUT / "sample_construction.csv", index=False)
    counts.to_csv(OUT / "sample_monthly.csv", date_format="%Y-%m-%d")
    descriptives.to_csv(OUT / "industry_descriptives.csv", float_format="%.4f")
    summary.to_csv(OUT / "industry_summary.csv", float_format="%.4f")
    grid.to_csv(OUT / "convention_grid.csv", index=False, float_format="%.4f")
    steps.to_csv(OUT / "convention_steps.csv", index=False, float_format="%.4f")
    selection.loc[selected.index, ["selected", "score", "window_start", "window_end"]].to_csv(
        OUT / "selected_w.csv", date_format="%Y-%m-%d", float_format="%.4f")

    plot_sample(counts, descriptives, skipped, missing_held, common)
    plot_selection(selected, skipped)

    pd.set_option("display.width", 200, "display.max_colwidth", 80)
    for title, frame in (("sample construction", construction), ("industry summary", summary.round(2)),
                         ("convention grid", grid.round(4)), ("convention steps", steps.round(4))):
        print(f"\n{title}\n{frame.to_string()}")
    print("\nselected w:", selected.value_counts().sort_index().to_dict(), "changes", int((selected.diff().fillna(0) != 0).sum()))


if __name__ == "__main__":
    main()
