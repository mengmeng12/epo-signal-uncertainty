# When Is One Shrinkage Parameter Enough

Signal uncertainty in Enhanced Portfolio Optimization.

We study when compressing heterogeneous signal uncertainty into one scalar preserves portfolio decisions. Simulations will identify the decision loss remaining with the best scalar; a matched industry-portfolio experiment will evaluate an implementable reliability proxy after estimation error and costs.

## Project status

This feature branch contains Steps 0–9: data and benchmark work by Yuzhen, simulation and estimator calibration by Yadan, and forecasts, matched backtesting and evaluation by Shuchen (Isabella). It includes Yadan's Step 6 commit `b41d0fe`. Step 10 (report and presentation integration) is deferred. This status describes `feature/reliability-and-oos`; the work has not been merged into `main`.

## Layout

```text
epo-signal-uncertainty/
├── doc/
│   ├── Pedersen_Babu_Levine_2021_Enhanced_Portfolio_Optimization.pdf
│   ├── EPO_Revised_Proposal.docx
│   ├── README.md
│   └── document_manifest.json
├── src/epo_project/           # Data, benchmark, simulation, reliability, backtest and evaluation
├── data/                      # Raw, processed and intermediate data
├── output/                    # Results, diagnostics and design decisions
├── tests/                     # Estimator, timing, accounting and inference tests
├── end_to_end_pipeline.ipynb   # Calls and results for Steps 1–9
├── workflow.md                # Team tasks, outputs and acceptance criteria
└── README.md
```

## Start here

1. Read the [revised proposal](doc/EPO_Revised_Proposal.docx) and [reference paper](doc/Pedersen_Babu_Levine_2021_Enhanced_Portfolio_Optimization.pdf).
2. Agree on the shared data, timing and parameter conventions in the [workflow table](workflow.md).
3. Create an individual feature branch, implement a small task, and open a pull request for the assigned reviewer.
4. Run [end_to_end_pipeline.ipynb](end_to_end_pipeline.ipynb) from the repository root. Use the existing environment if its packages are available; no new environment is needed for Step 5.

To rebuild only Step 5 from the tracked CSV inputs:

```bash
PYTHONPATH=src python -m epo_project.reliability
PYTHONPATH=src python -m pytest tests/test_reliability.py -q
```

The estimator needs NumPy and pandas; tests additionally need pytest. Step 5 does not require the untracked `risk_models.npz` or a download. Its default output contains 46,662 industry-month observations for 1,014 formation months (1941-06 through 2025-11), with target months ending in 2025-12. These are forecast inputs, not portfolio-performance results.

To rebuild Steps 7–9 from the tracked forecasts and market data:

```bash
PYTHONPATH=src python -m epo_project.backtest
PYTHONPATH=src python -m epo_project.evaluation
PYTHONPATH=src python -m pytest -q
```

Step 7 saves monthly returns, holdings, candidate exits and annual selection. Step 8 saves the performance table and 1,000-draw paired-bootstrap intervals (12-month primary blocks; 6/24-month sensitivities). Step 9 reuses the selected w, targets 10% predicted annual volatility, caps gross exposure at 3 and recomputes costs from its own holdings. The primary test uses fixed gamma=3 without volatility targeting. Both paths run continuously from 1970 through 2025; tables separate 1970–2018 and 2019–2025.

Definitions and artifact schemas are in [data/README.md](data/README.md); accounting and eligibility conventions are in [output/design_decisions.md](output/design_decisions.md). `output/empirical_manifest.json` records the input, code and output hashes for Steps 7–9.

The revised proposal is the design authority. Document provenance, source links, copyright/license details, and checksums are in [doc/README.md](doc/README.md) and [doc/document_manifest.json](doc/document_manifest.json). The reference paper retains its own license; it is not a blanket license for this repository.
