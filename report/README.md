# Does Estimating Asset-Specific Uncertainty Improve Portfolio Decisions?

English working report, subtitled **Evidence from Enhanced Portfolio Optimization**. The cover contains only the title and subtitle. Sources are split by section for collaborative editing.

## Build

From the repository root, with the project dependencies available:

```sh
python report/build.py
```

This generates 17 vector figures and 15 tables, then runs `latexmk` and BibTeX. The PDF is `report/build/main.pdf`. A TeX distribution with `pdflatex`, `bibtex`, `latexmk`, Latin Modern, `natbib`, `caption`, `microtype`, `placeins` and standard mathematics packages is required. MacTeX and TeX Live provide these tools. Python dependencies are listed in the repository's `requirements.txt`.

For prose-only changes after generating assets:

```sh
python report/build.py --skip-assets
```

The build reads the existing benchmark, simulation and empirical outputs. It creates missing or stale report diagnostics in `report/generated/diagnostic_data/`, `report/generated/label_placebo/`, `report/generated/uncertainty_value/` and `report/generated/calibration_bridge/`. These caches contain input/code/output hashes. The primary research results remain unchanged. No data download or risk-matrix cache is required for the report build.

`generated/` and `build/` are ignored by Git. Do not edit generated tables or figures directly. The report manifest records source/input hashes, units, runtime versions and the Git revision used for the build.

## Collaborate

Use `feature/report-draft` as the shared starting point and pull before editing. Contributors can edit individual sections on their own branches and merge through a pull request.

| Work area | Source |
|---|---|
| Motivation, interpretation and conclusion | `sections/introduction.tex`, `sections/discussion.tex`, `sections/conclusion.tex` |
| Theory and methods | `sections/framework.tex`, `sections/methodology.tex`, `sections/scalar_sufficiency.tex` |
| Results, figures and tables | `sections/results.tex`, `scripts/generate_assets.py`, `scripts/uncertainty_value.py`, `scripts/calibration_bridge.py` |
| Technical details | `sections/appendix.tex` |
| Abstract, references and layout | `sections/abstract.tex`, `references.bib`, `main.tex` |

The six main sections are Introduction, Theoretical Framework, Data and Methodology, Results, Discussion and Conclusion. The introduction has no subsections. Discussion interprets the evidence; concise limitations appear in Conclusion. The original-method replication remains a substantial results subsection, with detailed numerical discrepancy accounting in Appendix C.

The paper studies the utility of estimating asset-specific uncertainty within existing allocation methods. It distinguishes the fixed-exposure equivalent scalar, the arithmetic-mean scalar and the utility-optimal scalar. The estimated-uncertainty comparison evaluates restriction cost and estimation loss under the same true conditional objective. Historical returns are evaluated separately with trading costs and two position-sizing rules.

## Additional analysis

`scripts/uncertainty_value.py` supplies three calculations:

1. **Controlled estimation error.** Reconstruct the original Gaussian-model profiles and signal draws with seed 20261023. Apply multiplicative relative-uncertainty errors with log standard deviations 0, the measured calibration value (about 0.414), 0.5 and 1, using independent seed 20261112. Normalize each estimated vector to mean one. At the specified scalar `kappa = lambda/tau = 2.5`, calculate restriction cost, estimation loss and their difference. Original oracle compression loss remains a distinct statistic.
2. **Common position-sizing intervals.** Use existing volatility-targeted/capped returns and paired moving-block bootstrap samples to estimate the interval for heterogeneous-minus-scalar CE.
3. **Assignment-benchmark intervals.** Compute original CE minus the mean of the 100 individual random-assignment CEs. Resample all strategies on identical dates. This does not calculate the CE of an equally weighted mixture of random strategies.

`scripts/calibration_bridge.py` reruns estimator calibration with the original seed, verifies the aggregate calibration results, and saves industry/history estimates alongside normalized analytical relative uncertainty. The root mean within-history sample log-error variance defines the matched error level. Only the zero-feedback reference is exact conditional on predictors. The calibrated marker matches error dispersion, not the full estimator error distribution. At this value, both nonuniform controlled settings retain positive gains; it does not establish that historical estimation loss exceeds restriction cost.

The same script recomputes targeted/capped positions at trading-cost rates of 0 and 25 bp, preserving selected penalties and charging each strategy its own drift-dependent costs. It adds 6- and 24-month-block intervals at the 10 bp baseline. Cost comparisons are available under both position-sizing rules; the 60-ready-month eligibility check is a supporting fixed-gamma exercise.

The empirical analyses use 1,000 bootstrap draws with 12-month blocks and seed 20261105 combined with the period start year and block length, matching the original fixed-gamma evaluation. The fixed-gamma interval is reproduced as a control. The assignment intervals condition on the specified assignments and fitted strategy returns; they are time-series comparisons, not industry-label permutation tests.

The random-assignment returns are generated by `scripts/label_placebo.py`. Its 100 seeded rankings remain fixed and are restricted to each month's eligible industries. Every strategy has its own holdings, turnover and costs under both position-sizing rules. Controls reproduce the original strategies, verify uniform uncertainty and preserve each monthly uncertainty distribution. All generated outputs are local build artifacts.

## Exhibit sources

| Main exhibit | Source |
|---|---|
| Table 1: design | `output/benchmark/sample_construction.csv`, sample diagnostics, configuration |
| Table 2: replication | `output/replication_table.csv` |
| Table 3: forecast and uncertainty diagnostics | forecast/reliability diagnostic summaries |
| Table 4: net performance | `output/performance_table.csv`, `output/deployment_sensitivity.csv`, generated empirical intervals |
| Table 5: uncertainty-to-asset assignment | generated empirical intervals from the fixed random assignments |
| Table 6: cost and eligibility sensitivity | robustness diagnostics, generated targeted cost summary |
| Figure 1: principal components | `output/problem_pc_diagnostics.csv` |
| Figure 2: oracle compression | `output/simulation_draws.csv`, paired utility-share calculations |
| Figure 3: restriction cost and estimation loss | generated error-simulation summary and calibration log-error summary |
| Figure 4: estimator calibration | `output/estimator_calibration.csv` |
| Figure 5: composition and risk direction | portfolio diagnostics |
| Figure 6: fixed-risk-aversion CE accounting | CE-decomposition diagnostics |

Appendices contain 11 figures and nine tables, including the complete controlled-error results and random-assignment distribution summaries. Main exhibit numbering follows manuscript order; cross-references use LaTeX labels.

Main simulation figures and text report shares of mean oracle utility. Error bars use paired delta-method standard errors for ratios of means. Appendix B retains annual utility bp: monthly utilities, losses, standard errors and interval endpoints multiplied by 12. This is arithmetic annualization of quadratic utility. Forecast errors and forecast-bootstrap standard errors remain monthly. Table 3 expresses predictive R² and its intervals in percent. The anchor uses `a`, controlled uncertainty-error dispersion uses `delta`, and regression coefficients use `alpha` and `beta`; data-field names in cached results retain their original definitions. Conditional model utility and historical net CE have separate definitions despite their common reporting unit.

## References and validation

The paper cites 13 references, including the industry-return and risk-free-rate data sources. Article metadata and relevant method attributions were checked against publisher, Crossref and primary sources; see [reference_audit.md](reference_audit.md). The EPO article supplied in `doc/` supports the benchmark comparison. The discussion of monthly expanding-history selection is attributed in Appendix C to journal p. 138, the global-asset section and Figure 2 note; its application to the equity diagnostic is explicitly identified as an implementation interpretation.

Run:

```sh
PYTHONPATH=src python -m pytest -q
python report/scripts/check_scalar_sufficiency.py
python report/scripts/check_manuscript.py
```

The controlled-error script checks each loss identity, zero-error equivalence, uniform-uncertainty behavior, positivity/normalization and reproduction of the original simulated utilities. The empirical script checks matched dates, stored deployment-return reproduction and agreement with the existing paired-bootstrap function. Tests independently check assignment-utility arithmetic, the quadratic loss identity, paired ratio uncertainty and normalization-invariant relative log errors. The manuscript checker verifies structure, references, a roughly 100-word abstract, title-only cover and PDF text bounds. Visual review is required after layout changes.
