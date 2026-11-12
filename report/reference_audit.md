# Literature verification for the report

Checked on November 12, 2026 against publisher records and the Crossref record for each DOI. Journal years below are issue years, not advance-online dates. This document records attribution checks; it is not part of the manuscript.

| Reference | Verified journal, volume(issue), pages | DOI / publisher record | Use in the report |
|---|---|---|---|
| Goldfarb and Iyengar (2003), *Robust Portfolio Selection Problems* | Mathematics of Operations Research **28(1), 1–38** | [INFORMS](https://pubsonline.informs.org/doi/10.1287/moor.28.1.1.14260) | Statistically motivated uncertainty sets and tractable robust portfolio selection. |
| Ceria and Stubbs (2006), *Incorporating estimation errors into portfolio selection: Robust portfolio construction* | Journal of Asset Management **7(2), 109–127** | [Springer](https://link.springer.com/article/10.1057/palgrave.jam.2240207) | Briefly distinguish expected-return uncertainty from return risk and robust optimization from Gaussian conditioning. |
| Garlappi, Uppal and Wang (2007), *Portfolio Selection with Parameter and Model Uncertainty: A Multi-Prior Approach* | The Review of Financial Studies **20(1), 41–81** | [Oxford](https://academic.oup.com/rfs/article-abstract/20/1/41/1588211) | Multiple priors, ambiguity aversion and differing uncertainty across asset groups. |
| Jagannathan and Ma (2003), *Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps* | The Journal of Finance **58(4), 1651–1683** | [Wiley](https://onlinelibrary.wiley.com/doi/10.1111/1540-6261.00580) | Nonnegative weights in minimum-variance portfolios have a covariance-shrinkage interpretation and can reduce estimation risk; this is not an equivalence between every constraint and EPO shrinkage. |
| DeMiguel, Garlappi and Uppal (2009), *Optimal Versus Naive Diversification: How Inefficient Is the 1/N Portfolio Strategy?* | The Review of Financial Studies **22(5), 1915–1953** | [Oxford](https://academic.oup.com/rfs/article-abstract/22/5/1915/1592901?login=false) | No consistently superior strategy relative to equal weighting within their tested models and datasets; not a universal result against optimization. |
| Michaud (1989), *The Markowitz Optimization Enigma: Is ‘Optimized’ Optimal?* | Financial Analysts Journal **45(1), 31–42** | [Publisher DOI](https://doi.org/10.2469/faj.v45.n1.31); [author-hosted article](https://newfrontieradvisors.com/media/t04a3oct/the-markowitz-optimization-enigma.pdf) | Optimization's sensitivity to estimation error. This article is not used as the reference for the later resampled-efficiency algorithm. |

The six DOI records are independently retrievable at `https://api.crossref.org/works/{DOI}`. Garlappi et al. was online in 2006 and DeMiguel et al. in 2007; their journal issue years are 2007 and 2009. Wiley and Crossref agree on Jagannathan–Ma's final page **1683**; a university profile gives 1684, which is not used.

## Resampled efficiency

The method citation is Richard O. Michaud and Robert O. Michaud (2008), *Efficient Asset Management: A Practical Guide to Stock Portfolio Optimization and Asset Allocation*, **second edition**, Oxford University Press, Chapter 6, “The Resampled Efficient Frontier.” Print ISBN **9780195331912**; DOI **10.1093/oso/9780195331912.001.0001**. [Publisher book record](https://academic.oup.com/book/52209) and [chapter record](https://academic.oup.com/book/52209/chapter-abstract/421081650) verify the authors, year, edition and method. It is a book, so journal volume, issue and article page fields do not apply. The method was developed after the 1989 article; the cited second edition provides the method account used here.

Resampled efficiency uses Monte Carlo investment inputs and averages optimized portfolios. The report uses moving-block pairs resampling to estimate the variance of a fitted next-month conditional mean, then forms one portfolio using the original point forecast and its estimated uncertainty profile. The two uses of resampling are distinguished in the introduction and Section 3.4.

## Ceria–Stubbs notation and objective

The published article's “Robust portfolio optimisation” section, **pp. 117–119**, gives the confidence ellipsoid in equation (1) and the maximum-utility formulation in equation (8). Its expected-return estimation-error covariance is **Σ**, while its return covariance is **Q**. The article's [full-text transcription](https://www.academia.edu/80480005/Incorporating_estimation_errors_into_portfolio_selection_Robust_portfolio_construction) was checked against those equation and page markers. These notation details are retained only in this audit; the manuscript does not introduce an additional matrix symbol.

Their estimation-error covariance and the report's Λ both represent uncertainty about expected returns, separately from return covariance. Their statistical conditioning and decision criteria differ. The standard ellipsoid yields a worst-case expected-return deduction proportional to the square root of portfolio error variance. The report uses Gaussian conditioning and an EPO inverse-matrix allocation. Equal numerical matrices therefore do not establish equal allocations. The manuscript contains only two sentences of positioning and does not reproduce the ellipsoid or objective, or claim to implement any of their robust strategies.

## Contribution scope

1. Replication and principal-component evidence establish the scalar baseline.
2. Oracle comparisons and analytical conditions characterize scalar compression independently of scalar-selection error.
3. Calibration and matched historical evaluation assess an implementable uncertainty profile, with cost, exposure and label-assignment diagnostics.

These are analytical and empirical contributions using established methods. No new general uncertainty set, bootstrap algorithm or robust optimizer is claimed. This revision changes literature positioning and attribution; it does not change experiments or numerical results.

The Jagannathan–Ma shrinkage interpretation was also checked in the authors’ [NBER working paper 8922](https://www.nber.org/system/files/working_papers/w8922/w8922.pdf), abstract and Section II. Its 2002 working-paper date is not used in place of the final 2003 journal citation. At the user’s request, the manuscript retains an author–year citation for Ceria–Stubbs without page or original-equation locators; detailed locators above are audit records only.
