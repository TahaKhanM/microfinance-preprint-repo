# Microfinance, borrower growth and the female employment gap

An exploratory study of Indian microfinance institutions (MFIs), combining company operating data with an accounting scenario for the borrower growth associated with a share of the India–world female employment gap. The repository now includes a **chronological forecast audit**: a reviewer can compare polynomial extrapolation with a last-observation baseline, inspect held-out errors and see which future values violate their natural domain.

The [2024 preprint](papers/Microfinancepreprintfinalasof10thoct.pdf) is a collaboration between **Shiv Barua, Koby Reiss Din, Taha Khan and Jack Wickham**, listed alphabetically. The paper and available history do not establish a precise individual division of research or implementation. The original scripts and notebook exports are preserved; the revised audit is later work and does not retroactively validate the paper's conclusions.

**The borrower scenario is not a causal estimate of jobs created by loans.** It combines extrapolated series and explicit assumptions. The paper's 2028 crossover is a historical model output, not a validated prediction or evidence that microfinance closes an employment gap.

## Reproduce the audit

Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m src.analysis --output results/my-run
```

The workbook is included; no credentials, downloads or GUI are required. Paths are resolved relative to the project, so the analysis does not depend on an implicit working-directory spreadsheet lookup.

The [committed results](results/) contain:

| File | What to inspect |
| --- | --- |
| `audit.json` | Model choice, every candidate's cross-validation MAE, holdout errors, missing series, domain flags, library versions and workbook SHA-256 |
| `holdout.csv` | Actual and predicted values for each held-out year, alongside persistence |
| `forecasts.csv` | Raw 2024–2030 extrapolations after refitting the selected model |
| `borrower-scenarios.csv` | Per-company growth and required-borrower accounting or a specific reason it cannot be computed |
| `operations-observed.csv` | Observed operating metrics indexed to each company's 2017 value |

## What the revised evaluation shows

With the default 2020 development cutoff, **46 series have enough data for evaluation; one does not**. The selected model beats the last-value baseline on held-out MAE for **19 of 46 series**. The remaining cases tie or do worse. The extrapolations contain **11 values outside their natural domain**, recorded rather than silently clipped. These results are a reason to be cautious about smooth polynomial projections, especially over a seven-year scenario horizon.

The workbook has 25 annual rows (1999–2023) and 79 data columns, with uneven reporting windows. The audit selects six macro series plus available borrower, branch, loan-size, expense and utilisation columns for ten MFIs. It uses the explicit `Year` column and retains each observation's actual year after dropping missing values. The original code created a continuous year range independently of its dropped values, which would misalign any series with internal gaps.

## Model selection and forecast boundary

[forecasting.py](src/forecasting.py) compares persistence, linear, quadratic and cubic models. Each candidate is tested on successive observations using only earlier observations for fitting, starting with four training cases. The mean absolute error across those expanding windows selects the model; an exact tie favors the simpler candidate. At least five development observations are required, so even the smallest eligible series has only one validation point. Sparse series cannot support a strong model-selection conclusion.

All selection data end by 2020. The chosen model is then fitted to that development period and evaluated against available **2021–2023 observations without refitting during the holdout**. This tests a fixed-origin, multi-year extrapolation. Only after recording that result does the code refit the selected model on all observations to construct the 2024–2030 scenarios. The future scenario fit therefore includes the held-out historical years, while the recorded holdout score does not.

Dates are centered on the first training year before polynomial expansion, reducing numerical conditioning problems from powers of calendar years. There is no future-dependent scaling or interpolation. Missing years make some rolling validations longer than one calendar year; their MAEs are per observation, not horizon-normalised. MAE is reported in each series' workbook units and should not be averaged across unrelated metrics. The separation of selection and assessment follows the principle described in [scikit-learn's cross-validation documentation](https://scikit-learn.org/stable/modules/cross_validation.html).

The original approach selected degree by **in-sample R²**. Nested polynomial models generally improve that score as degree increases; a cap of three does not itself prevent overfitting. The revised audit keeps these simple candidates because their behavior is understandable and directly tests the original modelling choice. Damped trends, structural time-series models or domain-constrained forecasts are credible alternatives, but should earn their complexity against the same historical boundaries and simple baseline.

## The borrower accounting scenario

For future working-age female population $W_t$, world employment rate $e_t$ and observed 2023 baseline population and Indian employment rate $W_0,e_0$, the model uses:

$$J_t = \max(0, W_t e_t - W_0 e_0), \qquad N_{i,t} = \frac{J_t s_t m_t \alpha_i}{p_f p_{ig}}.$$

Here $s_t$ is self-employment share, $m_t$ is microfinance market share and $\alpha_i$ is the company's **observed 2023 branch share**, held fixed. The defaults $p_f=0.99$ and $p_{ig}=0.985$ are explicit scenario assumptions inherited from the historical model, not newly estimated company parameters. Rate columns expressed as percentages are divided by 100; all scenario shares must be within `[0,1]`, with positive denominators. Negative projected borrower counts and invalid rate projections produce unavailable scenarios.

The comparison is projected borrowers minus **observed 2023 borrowers**, against $N_{i,t}$. This replaces the original positional slices and fitted baseline that made year interpretation fragile. It also avoids summing independently scaled company gaps: such a sum is neither total borrowers nor total jobs.

Explore sensitivity explicitly:

```bash
python -m src.analysis --female-share 0.95 --income-share 0.80 --output results/sensitivity
```

Holding everything else fixed, a lower assumed income-generating share raises the required borrower count inversely. Neither “income-generating loan” nor “female borrower” establishes an additional job. The scenario has no counterfactual, borrower-level outcomes, identification strategy, default/loss process or double-borrowing adjustment. Branch share also assumes comparable branches and is only a proxy for responsibility or capacity.

## Operating metrics and source limitations

The revised operating table uses **observations only**, indexed to a fixed observed 2017 baseline per company. It does not group companies by a model-generated outcome and then interpret differences between those groups as effects of management decisions. The original plots normalised by a maximum that included future projections and summed groups of different sizes; that comparison mixed composition, scale and extrapolation.

The workbook is preserved unchanged. Its source reconciliation remains incomplete: annual statements, definitions and currency multipliers need a column-by-column provenance audit before making quantitative cross-company claims. For example, some expense-per-borrower series show abrupt scale changes and some loan-size columns use small decimal units rather than explicit rupee values. The code deliberately retains workbook units; normalising a series does not repair a unit error. Employment-rate construction also needs verification against original source definitions, because unemployment and participation rates can have different denominators.

The audit tests temporal isolation by changing holdout values and checking that model selection is unchanged. Other tests cover gaps in year indices, a manually calculated rolling baseline, polynomial behavior, insufficient data, percentage arithmetic, scenario sensitivity and workbook coverage. CI also runs the complete analysis from the bundled data.

## Original artifacts

- [Preprint](papers/Microfinancepreprintfinalasof10thoct.pdf).
- [Original employment scenario](historical/male_employment_gap_modelling_code.py) and [management comparison](historical/microfinance_management_practices_code.py), unchanged. Their broad exception handling, overwritten rate estimates, spelling errors and positional assumptions remain historical code, not the supported entry points.
- [Notebook PDF exports](notebooks_pdf_exports/), retained as the original calculation record.

The implementation now supports a defensible discussion of model validation, numerical conditioning, baseline comparisons and why the original research cannot establish causality. Stronger economic conclusions still require reconciled source data and an appropriate research design. A chronological split prevents one form of leakage; it does not remove observational confounding, revised historical-data bias or structural breaks.
