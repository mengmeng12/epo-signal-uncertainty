"""One fixed-label placebo for the incremental information in estimated h.

This is a descriptive retrospective diagnostic, not a permutation hypothesis
test. Industry labels need not be exchangeable. No parameter is retuned.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from epo_project import backtest as bt, config as cfg, evaluation as ev

SEED = 20261111
PERMUTATIONS = 100
DEST = ROOT / "report/generated/label_placebo"
INPUTS = [
    ROOT / "data/processed/returns.csv",
    ROOT / "data/interim/forecast_reliability.csv",
    *[
        ROOT / "output" / name
        for name in (
            "annual_selection.csv",
            "matched_returns.csv",
            "deployment_returns.csv",
        )
    ],
    *[
        ROOT / "src/epo_project" / f"{name}.py"
        for name in ("config", "backtest", "evaluation", "data", "reliability")
    ],
    Path(__file__).resolve(),
]


def run(destination: Path = DEST) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    total, rf, forecasts = bt.load_market()
    selection = pd.read_csv(ROOT / "output/annual_selection.csv")
    chosen = selection[selection.selected].set_index("year").selected_w
    assert chosen.index.is_unique
    random = np.random.default_rng(SEED)
    priorities = np.array(
        [random.permutation(total.shape[1]) for _ in range(PERMUTATIONS)]
    )
    labels = ["scalar", "observed_h", "uniform_control"] + [
        f"shuffle_{i:03}" for i in range(PERMUTATIONS)
    ]
    policies = ["fixed_gamma", "targeted_capped"]
    drift = np.zeros((2, len(labels), total.shape[1]))
    alive = np.ones((2, len(labels)), dtype=bool)
    rows, exits = [], []
    max_allocation_difference = 0.0
    for number, month in enumerate(
        bt.formation_inputs(
            total, rf, forecasts, cfg.EXTENSION_PERIOD[0], cfg.SUBSEQUENT_PERIOD[1]
        )
    ):
        w = float(chosen.loc[month.target_date.year])
        # A fixed random ranking yields a fixed mapping whenever eligibility is unchanged.
        mapping = np.argsort(priorities[:, month.ids], axis=1)
        shuffled = month.h[mapping]
        np.testing.assert_allclose(
            np.sort(shuffled, axis=1),
            np.broadcast_to(np.sort(month.h), shuffled.shape),
            atol=0,
            rtol=0,
        )
        h = np.vstack([np.ones_like(month.h), month.h, np.ones_like(month.h), shuffled])
        matrices = np.broadcast_to(month.q, (len(labels), *month.q.shape)).copy()
        diagonal = np.arange(len(month.ids))
        matrices[:, diagonal, diagonal] += w / (1 - w) * np.diag(month.q)[None, :] * h
        signal = np.broadcast_to(month.signal, (len(labels), len(month.ids)))[..., None]
        local = np.linalg.solve(matrices, signal)[..., 0] / cfg.GAMMA
        for index, profile in ((0, None), (1, month.h), (3, shuffled[0])):
            reference = bt.allocate(month.q, month.signal, w, profile)
            max_allocation_difference = max(
                max_allocation_difference,
                float(np.max(np.abs(local[index] - reference))),
            )
            np.testing.assert_allclose(local[index], reference, rtol=1e-11, atol=1e-12)
        np.testing.assert_array_equal(local[0], local[2])
        for policy_index, policy in enumerate(policies):
            for index, label in enumerate(labels):
                if not alive[policy_index, index]:
                    continue
                position = local[index]
                if policy_index:
                    position, _ = bt.deploy_weights(position, month.q)
                target = np.zeros(total.shape[1])
                target[month.ids] = position
                try:
                    next_drift, result = bt.portfolio_month(
                        target,
                        drift[policy_index, index],
                        month.realized_total,
                        month.rf,
                    )
                except bt.PortfolioInsolvency as error:
                    alive[policy_index, index] = False
                    exits.append(
                        {
                            "date": month.target_date,
                            "policy": policy,
                            "strategy": label,
                            "wealth_factor": error.growth,
                        }
                    )
                    continue
                drift[policy_index, index] = next_drift
                rows.append(
                    {
                        "date": month.target_date,
                        "policy": policy,
                        "strategy": label,
                        **result,
                    }
                )
        if (number + 1) % 168 == 0:
            print(f"Replayed {number+1}/672 months", flush=True)
    paths = pd.DataFrame(rows)
    control_errors = {}
    for policy, filename in zip(
        policies, ("matched_returns.csv", "deployment_returns.csv")
    ):
        recorded = pd.read_csv(ROOT / "output" / filename, parse_dates=["date"])
        observed = paths[
            (paths.policy == policy) & paths.strategy.isin(["scalar", "observed_h"])
        ].copy()
        observed["strategy"] = observed.strategy.replace(
            {"observed_h": "heterogeneous"}
        )
        merged = observed.merge(
            recorded,
            on=["date", "strategy"],
            suffixes=("_new", "_old"),
            validate="one_to_one",
        )
        assert len(merged) == len(recorded) == 1344
        for column in (
            "gross_excess",
            "net_excess",
            "turnover",
            "gross",
            "trading_cost",
            "borrow_cost",
        ):
            error = float(
                np.max(np.abs(merged[column + "_new"] - merged[column + "_old"]))
            )
            control_errors[f"{policy}/{column}"] = error
            np.testing.assert_allclose(
                merged[column + "_new"], merged[column + "_old"], rtol=2e-10, atol=2e-10
            )
        uniform = paths[
            (paths.policy == policy)
            & paths.strategy.isin(["scalar", "uniform_control"])
        ].pivot(index="date", columns="strategy", values="net_excess")
        np.testing.assert_array_equal(uniform.scalar, uniform.uniform_control)
    metrics = []
    for period, (start, end) in ev.PERIODS.items():
        count = len(pd.date_range(start, end, freq=pd.offsets.MonthEnd()))
        for policy in policies:
            part = paths[(paths.policy == policy) & paths.date.between(start, end)]
            for label in labels:
                frame = part[part.strategy == label]
                # Do not silently discard failed paths or splice in new capital.
                complete = len(frame) == count
                row = {
                    "period": period,
                    "policy": policy,
                    "strategy": label,
                    "complete": complete,
                    "months": len(frame),
                }
                if complete:
                    row.update(
                        net_ce_bp_year=1e4
                        * float(ev.certainty_equivalent(frame.net_excess.to_numpy())),
                        gross_ce_bp_year=1e4
                        * float(ev.certainty_equivalent(frame.gross_excess.to_numpy())),
                        mean_turnover=frame.turnover.mean(),
                        mean_gross=frame.gross.mean(),
                        cost_bp_year=12e4
                        * (frame.trading_cost + frame.borrow_cost).mean(),
                    )
                metrics.append(row)
    metrics = pd.DataFrame(metrics)
    summary = []
    for (period, policy), part in metrics.groupby(["period", "policy"], sort=False):
        indexed = part.set_index("strategy")
        scalar, observed = indexed.loc["scalar"], indexed.loc["observed_h"]
        shuffled = part[part.strategy.str.startswith("shuffle_")]
        complete = shuffled[shuffled.complete]
        assert scalar.complete and observed.complete
        row = {
            "period": period,
            "policy": policy,
            "permutations": PERMUTATIONS,
            "complete_placebos": len(complete),
            "incomplete_placebos": len(shuffled) - len(complete),
            "scalar_net_ce_bp_year": scalar.net_ce_bp_year,
            "observed_net_ce_bp_year": observed.net_ce_bp_year,
            "observed_minus_scalar_net_bp_year": observed.net_ce_bp_year
            - scalar.net_ce_bp_year,
            "observed_turnover": observed.mean_turnover,
            "placebo_median_turnover": complete.mean_turnover.median(),
        }
        for outcome in ("net", "gross"):
            column = f"{outcome}_ce_bp_year"
            delta = complete[column] - scalar[column]
            row.update(
                {
                    f"observed_minus_scalar_{outcome}_bp_year": observed[column]
                    - scalar[column],
                    f"placebo_median_delta_{outcome}_bp_year": delta.median(),
                    f"placebo_p05_delta_{outcome}_bp_year": delta.quantile(0.05),
                    f"placebo_p95_delta_{outcome}_bp_year": delta.quantile(0.95),
                    f"observed_percentile_{outcome}": 100
                    * float((complete[column] < observed[column]).mean()),
                }
            )
        summary.append(row)
    summary = pd.DataFrame(summary)
    for name, frame in (
        ("label_placebo_summary", summary),
        ("label_placebo_metrics", metrics),
        ("label_placebo_returns", paths),
        (
            "label_placebo_exits",
            pd.DataFrame(
                exits, columns=["date", "policy", "strategy", "wealth_factor"]
            ),
        ),
    ):
        frame.to_csv(
            destination / f"{name}.csv",
            index=False,
            float_format="%.12g",
            date_format="%Y-%m-%d",
        )
    manifest = {
        "seed": SEED,
        "permutations": PERMUTATIONS,
        "industry_order": list(total.columns),
        "fixed_random_priorities": priorities.tolist(),
        "mapping_rule": "argsort priorities on eligible source indices; assigned to canonical eligible recipient order",
        "selection": "frozen primary scalar selection",
        "inference": "descriptive placebo distribution, not a formal permutation p-value",
        "allocation_control_max_difference": max_allocation_difference,
        "recorded_path_max_differences": control_errors,
        "failed_paths": exits,
        "input_code_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in INPUTS
        },
        "output_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in destination.glob("*.csv")
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    print(summary.to_string(index=False))
    return {"summary": summary, "manifest": manifest}


if __name__ == "__main__":
    run()
