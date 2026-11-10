# When Is One Shrinkage Parameter Enough?

English working draft of **Signal Uncertainty in Enhanced Portfolio Optimization**. The cover contains the title only. Sources are split by section so collaborators can edit independently.

## Build

From the repository root, in the project's Python environment:

```sh
python report/build.py
```

The command generates 15 vector figures and 9 tables from result CSVs, then runs `latexmk` and BibTeX. The PDF is `report/build/main.pdf`. A TeX distribution with `pdflatex`, `bibtex`, `latexmk`, Latin Modern, `natbib`, `caption`, `microtype` and the standard mathematics packages is required. MacTeX and TeX Live provide these tools. The Python dependencies are in the repository's `requirements.txt`.

For prose-only changes after the first build:

```sh
python report/build.py --skip-assets
```

The build reads tracked benchmark, simulation and empirical outputs. If the additional diagnostics are missing or their manifest does not match current inputs/code, it recomputes them under `report/generated/diagnostic_data/`. No data download or risk-matrix cache is needed for this report build. To rerun the research itself, use the repository pipeline first, then rebuild the report. The report build does not rerun the simulations or primary backtest.

`generated/` and `build/` are ignored by Git. This keeps generated PDF, figure and intermediate-file changes out of collaborative source commits. Do not edit generated tables or figures directly. The report manifest records the exact input and source hashes and Python library versions used in a build; it does not supersede the research pipeline's historical manifests.

## Collaborate

Use the `feature/report-draft` branch as the shared starting point. Each collaborator can create a working branch, change the relevant section and merge through a pull request. Pull the latest shared branch before starting a new revision. A practical division is:

| Work area | Source |
|---|---|
| Motivation, interpretation and conclusion | `sections/introduction.tex`, `sections/discussion.tex`, `sections/conclusion.tex` |
| Theory and reproducible methods | `sections/framework.tex`, `sections/methodology.tex`, corresponding parts of `sections/appendix.tex` |
| Results, figures and tables | `sections/results.tex`, `scripts/generate_assets.py`, corresponding appendix exhibits |
| Abstract, bibliography and document layout | `sections/abstract.tex`, `references.bib`, `main.tex` |

The manuscript has six main sections: Introduction; Theoretical Framework; Data and Methodology; Results; Discussion; Conclusion. The introduction has no subsections. Limitations appear in the conclusion. The original-method replication remains a substantial results subsection; the detailed numerical discrepancy accounting is Appendix C.

Keep the distinction between established methods and project work explicit. The Gaussian experiment, bootstrap uncertainty integration and matched evaluation are the project's design. Mean–variance optimization, enhanced portfolio optimization, ordinary least squares and block resampling are existing methods. Update the abstract and conclusion if the evidence changes. The revised abstract has 97 whitespace-delimited words and contains no abbreviations.

## Exhibit sources

| Main exhibit | Input under `output/` |
|---|---|
| Table 1: sample/design | `benchmark/sample_construction.csv`, `diagnostics/sample_summary.csv`, `src/epo_project/config.py` settings |
| Table 2: replication | `replication_table.csv` |
| Table 3: forecast/uncertainty | `diagnostics/forecast_summary.csv`, `diagnostics/reliability_summary.csv` |
| Table 4: net performance under two exposure policies | `performance_table.csv`, `deployment_sensitivity.csv` |
| Table 5: sensitivity | `diagnostics/robustness_summary.csv`, `deployment_sensitivity.csv` |
| Figure 1: principal components | `problem_pc_diagnostics.csv` |
| Figure 2: compression | `compression_loss.csv` |
| Figure 3: calibration | `estimator_calibration.csv` |
| Figure 4: utility accounting | `diagnostics/ce_decomposition.csv` |

Appendices A–F contain derivations, simulation and calibration tables, data and replication-discrepancy accounting, forecast diagnostics, selection and holdings, and additional performance exhibits. There are 11 appendix figures and 4 appendix tables. Captions specify units, samples and uncertainty conventions. Figures are re-exported from the project's plotting functions as vectors; the report export adjusts labels and layout, without changing plotted numerical values.

## References and checks

The four article references were checked against Crossref DOI metadata and the primary publication/author sources. The source article supplied in `doc/` supports the EPO attribution and published benchmark values. The French data documentation supports the dataset description. The paper cites:

- Markowitz (1952), DOI `10.1111/j.1540-6261.1952.tb01525.x`: the mean–variance objective.
- Ledoit and Wolf (2004), DOI `10.1016/S0047-259X(03)00096-4`: covariance conditioning and shrinkage context.
- Pedersen, Babu and Levine (2021), DOI `10.1080/0015198X.2020.1854543`: EPO, the general uncertainty framework, principal-component interpretation and Table 5 benchmark.
- Künsch (1989), DOI `10.1214/aos/1176347265`: block resampling for dependent observations.
- Kenneth R. French Data Library: industry portfolio data and construction documentation.

`python report/scripts/check_manuscript.py` checks source structure, bibliography keys, the abstract and compiled references; if PyMuPDF is installed, it also verifies the title-only cover and PDF text boundaries. Visual review remains necessary after changing figure layout or prose length. The first draft was compiled and its pages visually inspected. It is a working paper for collaborative revision, with empirical claims restricted to the implemented designs and observed estimates.

## Scalar-sufficiency revision

The interpretation centers on a practical scalar baseline when incremental net utility from estimated heterogeneity has not been established. Statistical nonsignificance is not an equivalence claim. Maintenance and computation costs are discussed qualitatively and are not included in reported net returns. The primary estimand remains the change in the uncertainty profile at a common scalar-selected penalty.

`sections/scalar_sufficiency.tex`, included in Appendix A, derives exact agreement on held exposures, a sufficient proportional-loss bound, a local second-order loss expansion, and an estimation-versus-compression identity. These are consequences of the stated quadratic objective, not new empirical estimates of latent uncertainty. Check the algebra numerically with:

```sh
python report/scripts/check_scalar_sufficiency.py
```

The checks cover 120 positive-definite matrix cases, an exact agreement case with nonuniform uncertainty, the zero-signal case, and the local expansion. They supplement the written proofs. Table 4 presents the existing fixed-risk-aversion and volatility-targeted results together; no new backtest or simulation experiment is introduced by this revision.
