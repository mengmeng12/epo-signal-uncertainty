# When Is One Shrinkage Parameter Enough

Signal uncertainty in Enhanced Portfolio Optimization.

We study when compressing heterogeneous signal uncertainty into one scalar preserves portfolio decisions. Simulations will identify the decision loss remaining with the best scalar; a matched industry-portfolio experiment will evaluate an implementable reliability proxy after estimation error and costs.

## Project status

This feature branch contains Steps 0–5: data and benchmark work by Yuzhen, scalar compression loss by Yadan, and forecasting / relative uncertainty by Shuchen (Isabella). It builds on `feature/theory-and-simulation` at `667e8d6`, which includes the data branch at `f789590`. Steps 6–10 remain to be implemented. This status describes the feature branch; the work has not been merged into `main`.

The [Step 5 handoff](doc/step5_handoff.md) gives Yadan the estimator interface for Step 6. The [alignment review](output/step5_alignment_review.md) documents differences from the local walkthrough, including missing-return treatment and industry admission rules.

## Layout

```text
epo-signal-uncertainty/
├── doc/
│   ├── Pedersen_Babu_Levine_2021_Enhanced_Portfolio_Optimization.pdf
│   ├── EPO_Revised_Proposal.docx
│   ├── README.md
│   └── document_manifest.json
├── src/epo_project/           # Data, benchmark, simulation and reliability modules
├── data/                      # Raw, processed and intermediate data
├── output/                    # Results, diagnostics and design decisions
├── tests/                     # Forecast estimator and information-timing tests
├── end_to_end_pipeline.ipynb   # Calls and results for Steps 1–5
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

The revised proposal is the design authority. Document provenance, source links, copyright/license details, and checksums are in [doc/README.md](doc/README.md) and [doc/document_manifest.json](doc/document_manifest.json). The reference paper retains its own license; it is not a blanket license for this repository.
