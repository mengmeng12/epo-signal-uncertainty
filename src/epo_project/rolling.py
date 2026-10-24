"""Common rolling selection: the historical candidate-return interface.

A candidate-return table has one row per realization month m (month-end) and
one column per candidate parameter (e.g. each w). The row for m holds the
return earned over month m by positions formed at month-end m-1 with
information available at m-1.

Each January of year Y a parameter is chosen from the candidate returns
realized in the preceding SELECTION_WINDOW months (Jan Y-5 .. Dec Y-1 for 60
months) and is held for the realizations of Jan Y .. Dec Y, i.e. for the
positions formed at month-ends Dec Y-1 .. Nov Y. The chosen value therefore
never uses a return realized after the formation date it is applied to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def annualized_sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std(ddof=1) * np.sqrt(12))


def certainty_equivalent(r: pd.Series, gamma: float = cfg.GAMMA) -> float:
    """Annualized mean-variance certainty equivalent 12 mean(r) - (gamma/2) 12 Var(r)."""
    return float(12 * r.mean() - gamma / 2 * 12 * r.var(ddof=1))


def annual_selection(candidates: pd.DataFrame, score=annualized_sharpe,
                     window: int = cfg.SELECTION_WINDOW, tie_break: str = cfg.TIE_BREAK) -> pd.DataFrame:
    """Choose one column for every calendar year that has `window` complete
    preceding months. Ties (within 1e-12) go to the largest column label when
    tie_break == "higher_w"."""
    if tie_break != "higher_w":
        raise ValueError("only the 'higher_w' tie-break rule is implemented")
    index = candidates.index
    rows = []
    for year in range(index.min().year + 1, index.max().year + 1):
        start = pd.Timestamp(year, 1, 1)
        past = candidates[index < start].tail(window)
        expected_first = (start - pd.DateOffset(months=window)) + pd.offsets.MonthEnd(0)
        if len(past) < window or past.index[0] != expected_first or past.isna().any().any():
            continue                                    # require a complete, contiguous window
        scores = past.apply(score)
        best = scores.max()
        chosen = max(c for c in scores.index if scores[c] >= best - 1e-12)
        rows.append({"year": year, "selected": chosen, "score": best,
                     "window_start": past.index[0], "window_end": past.index[-1]})
    return pd.DataFrame(rows).set_index("year")


def apply_selection(candidates: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    """Realized returns of the selected candidate, month by month, with the parameter used."""
    rows = [(m, selection.at[m.year, "selected"], candidates.at[m, selection.at[m.year, "selected"]])
            for m in candidates.index if m.year in selection.index]
    return pd.DataFrame(rows, columns=["date", "selected", "return"]).set_index("date")
