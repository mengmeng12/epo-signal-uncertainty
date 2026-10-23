# When Is One Shrinkage Parameter Enough

Signal uncertainty in Enhanced Portfolio Optimization — FRE 6123 Financial Risk Management, Professor Adams.

**Team:** Yuzhen Chen, Yadan Zhan, Shuchen Meng.

We study when compressing heterogeneous signal uncertainty into one scalar preserves portfolio decisions. Simulations will identify the decision loss remaining with the best scalar; a matched industry-portfolio experiment will evaluate an implementable reliability proxy after estimation error and costs.

## Project status

This is the team's starting repository. **The notebook, `src`, and `data` are intentionally empty.** Each folder contains only a `.gitkeep` placeholder because Git cannot track an empty directory. There is no implemented pipeline, downloaded market data, or research result. The owner will add the two collaborators; the team will develop the project incrementally according to the workflow.

## Layout

```text
epo-signal-uncertainty/
├── doc/
│   ├── Pedersen_Babu_Levine_2021_Enhanced_Portfolio_Optimization.pdf
│   ├── EPO_Revised_Proposal.docx
│   ├── README.md
│   └── document_manifest.json
├── src/                       # Empty; future source code
├── data/                      # Empty; future data and results
├── end_to_end_pipeline.ipynb   # Blank notebook
├── workflow.md                # Team tasks, outputs and acceptance criteria
└── README.md
```

## Start here

1. Read the [revised proposal](doc/EPO_Revised_Proposal.docx) and [reference paper](doc/Pedersen_Babu_Levine_2021_Enhanced_Portfolio_Optimization.pdf).
2. Agree on the shared data, timing and parameter conventions in the [workflow table](workflow.md).
3. Create an individual feature branch, implement a small task, and open a pull request for the assigned reviewer.
4. Fill [end_to_end_pipeline.ipynb](end_to_end_pipeline.ipynb) as the modules are completed. Decide and record the Python environment before development.

The revised proposal is the design authority. Document provenance, source links, copyright/license details, and checksums are in [doc/README.md](doc/README.md) and [doc/document_manifest.json](doc/document_manifest.json). The reference paper retains its own license; it is not a blanket license for this repository.
