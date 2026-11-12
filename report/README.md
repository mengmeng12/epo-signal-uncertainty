# When Is One Shrinkage Parameter Enough?

English working draft of **Signal Uncertainty in Enhanced Portfolio Optimization**. The cover contains the title only. Sources are split by section so collaborators can edit independently.

## Build

From the repository root, in the project's Python environment:

```sh
python report/build.py
```

The command generates 16 vector figures and 11 tables from result CSVs, then runs `latexmk` and BibTeX. The PDF is `report/build/main.pdf`. A TeX distribution with `pdflatex`, `bibtex`, `latexmk`, Latin Modern, `natbib`, `caption`, `microtype`, `placeins` and the standard mathematics packages is required. MacTeX and TeX Live provide these tools. The Python dependencies are in the repository's `requirements.txt`.

For prose-only changes after the first build:

```sh
python report/build.py --skip-assets
```

The build reads tracked benchmark, simulation and empirical outputs. If the additional diagnostics are missing or their manifest does not match current inputs/code, it recomputes them under `report/generated/diagnostic_data/` and `report/generated/label_placebo/`. No data download or risk-matrix cache is needed for this report build. To rerun the research itself, use the repository pipeline first, then rebuild the report. The report build does not rerun the simulations or primary backtest.

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

Keep the distinction between established methods and project work explicit. The Gaussian experiment, bootstrap uncertainty integration and matched evaluation are the project's design. Mean–variance optimization, enhanced portfolio optimization, ordinary least squares and block resampling are existing methods. Update the abstract and conclusion if the evidence changes. The abstract is approximately 100 words and includes the main quantitative findings. It uses bp as the requested unit abbreviation, while spelling out method names.

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
| Table 6: label-assignment diagnostic | report cache `generated/label_placebo/label_placebo_summary.csv` |
| Figure 4: composition and risk direction | `diagnostics/portfolio_monthly.csv` |
| Figure 5: utility accounting | `diagnostics/ce_decomposition.csv` |

Appendices A–F contain derivations, simulation and calibration tables, data and replication-discrepancy accounting, forecast diagnostics, selection and holdings, and additional performance exhibits. There are 11 appendix figures and 5 appendix tables. Captions specify units, samples and uncertainty conventions. Figures are re-exported from the project's plotting functions as vectors; the report export adjusts labels and layout, without changing research estimates. Simulation utility displays use the annualization convention below.

## References and checks

The four article references were checked against Crossref DOI metadata and the primary publication/author sources. The source article supplied in `doc/` supports the EPO attribution and published benchmark values. The French data documentation supports the dataset description. The paper cites:

- Markowitz (1952), DOI `10.1111/j.1540-6261.1952.tb01525.x`: the mean–variance objective.
- Ledoit and Wolf (2004), DOI `10.1016/S0047-259X(03)00096-4`: covariance conditioning and shrinkage context.
- Pedersen, Babu and Levine (2021), DOI `10.1080/0015198X.2020.1854543`: EPO, the general uncertainty framework, principal-component interpretation and Table 5 benchmark.
- Künsch (1989), DOI `10.1214/aos/1176347265`: block resampling for dependent observations.
- Kenneth R. French Data Library: industry portfolio data and construction documentation.

`python report/scripts/check_manuscript.py` checks source structure, bibliography keys, the abstract and compiled references; if PyMuPDF is installed, it also verifies the title-only cover and PDF text boundaries. Visual review remains necessary after changing figure layout or prose length. The first draft was compiled and its pages visually inspected. It is a working paper for collaborative revision, with empirical claims restricted to the implemented designs and observed estimates.

## Scalar-sufficiency revision (November 10, 2026)

The interpretation centers on a practical scalar baseline when incremental net utility from estimated heterogeneity has not been established. Statistical nonsignificance is not an equivalence claim. Maintenance and computation costs are discussed qualitatively and are not included in reported net returns. The primary estimand remains the change in the uncertainty profile at a common scalar-selected penalty.

`sections/scalar_sufficiency.tex`, included in Appendix A, derives exact agreement on held exposures, a sufficient proportional-loss bound, a local second-order loss expansion, and an estimation-versus-compression identity. These are consequences of the stated quadratic objective, not new empirical estimates of latent uncertainty. Check the algebra numerically with:

```sh
python report/scripts/check_scalar_sufficiency.py
```

The checks cover 120 positive-definite matrix cases, an exact agreement case with nonuniform uncertainty, the zero-signal case, and the local expansion. They supplement the written proofs. Table 4 presents the existing fixed-risk-aversion and volatility-targeted results together; no new backtest or simulation experiment is introduced by this revision.

## Simulation-to-empirical bridge and label diagnostic (November 11, 2026)

Discussion distinguishes informative signals and known uncertainty in the simulation from estimated historical inputs. Negative predictive squared-error performance relative to a rolling mean does not establish that all signal information is absent.

Main Table 6 and Appendix F.1 report one descriptive label-assignment diagnostic, generated by `scripts/label_placebo.py`. It uses 100 random industry rankings with seed 20261111, held fixed and restricted to each month's eligible industries. Each permutation preserves the monthly uncertainty distribution. Forecasts, covariance inputs, scalar-selected penalties and fee rates stay unchanged; each portfolio carries its own holdings and costs through the complete period under both existing exposure policies. No tuning is performed. Run it independently with:

```sh
python report/scripts/label_placebo.py
```

The script checks original-path reproduction, exact uniform-profile equivalence, allocation agreement and preservation of each uncertainty distribution. Failed paths are recorded rather than restarted or silently removed. The manifest stores input/code/output hashes and the sampled rankings; the normal report build reruns the diagnostic when those inputs change.

The original matching beats 99 and 88 of 100 shuffled portfolios in 1970–2018 under fixed risk aversion and targeted exposure, respectively; the counts fall to 21 and 29 in 2019–2025. All paths remain solvent. These are descriptive ranks, not permutation p-values: industries are not assumed exchangeable. They neither prove equivalence to scalar nor isolate forecast quality from uncertainty-estimation error. The matched scalar comparison remains the main estimand.

## Collaborator integration (November 12, 2026)

The report incorporates `yc/fig1-appendix-c` while retaining the revised introduction, standalone abstract, contents, expanded methodology and label diagnostic. Figure 1 now compares raw mean–variance, corrected mean–variance and selected EPO: its first three panels use each rule's own risk matrix; variance shares use the common Q matrix. Appendix C adds fixed-portfolio convention accounting from `output/benchmark/convention_steps_fixed.csv`. Benchmark performance and the scalar/heterogeneous extension results are unchanged.

## Reporting units and evidence placement (November 12, 2026)

The six main tables and five main figures include composition/risk-direction transmission, utility decomposition, cost sensitivity and the label-assignment diagnostic. Appendix E retains exposure levels and selection histories; Appendix F.1 gives the label protocol. The manuscript omits development and replay history; implementation checks remain documented here.

`annual_simulation_report()` reads the unchanged monthly `output/compression_loss.csv` and scales every utility level, loss, Monte Carlo standard error and interval endpoint by 12. Tables and figures share that adapter. This is arithmetic annualization of monthly quadratic utility, not compounding or a simulated twelve-month holding process. Loss shares, profile dispersion and alignment are unchanged. Forecast error and bootstrap standard error still use monthly basis points. The report manifest records these conventions. Annualized model losses and historical net certainty equivalents remain different estimands.

Figure 1 includes a negative Sharpe tick and zero reference. Table 4 groups columns by period before its body rule. Displayed year ranges are normalized without changing input keys. Real `placeins` barriers constrain floats while preserving the separate cover, abstract and contents pages.

The expanding-selection attribution is checked against journal p. 138 (PDF page 17) of Pedersen, Babu and Levine (2021), under “Global Asset Classes: Beating Time-Series Momentum” / “Performance of EPO vs. benchmark portfolios,” including the Figure 2 note. The equity Table 5 note on journal p. 142 (PDF page 21) says only that selection uses past data. Appendix C explicitly identifies applying the monthly expanding rule to Equity 1 as a diagnostic interpretation, rather than an equity-specific implementation detail stated in that table.
