# Microfinance and borrower-growth research

An exploratory study of Indian microfinance institutions with a reproducible forecast audit. The analysis compares polynomial forecasts with a last-observation baseline, measures errors on later years and flags projections outside their natural domain.

The audit evaluates **46 series**. Selected models beat persistence on held-out mean absolute error for **19**, while the future extrapolations contain **11 invalid values**. Keeping these results visible is central to the project: a smooth historical fit can still make a poor forecast.

The [2024 preprint](papers/Microfinancepreprintfinalasof10thoct.pdf) is by **Shiv Barua, Koby Reiss Din, Taha Khan and Jack Wickham**. The forecast audit is later work. The original paper's borrower-growth scenario does not establish that loans create jobs or close an employment gap.

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

## Forecast methodology

[forecasting.py](src/forecasting.py) compares persistence with linear, quadratic and cubic models. Expanding-window validation uses only earlier observations to predict each next observation. Mean absolute error selects the candidate, with ties favouring the simpler model.

Selection ends in 2020. The chosen model is fitted to that period and assessed on available 2021–2023 observations without updating it during the holdout. Only then is it refitted on all observations for the 2024–2030 scenario.

The code preserves actual years when dropping missing data and centres dates before polynomial expansion. Series need at least five development observations. Errors retain workbook units, so they should not be averaged across unrelated metrics.

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

## Data and verification

The included workbook has 25 annual rows from 1999 to 2023 and 79 data columns. The audit selects six macro series and available operating series for ten institutions. Operating comparisons use observed data indexed to each company's 2017 baseline.

Tests change holdout values and verify that model selection remains unchanged. They also cover missing years, rolling baselines, percentage arithmetic, scenario sensitivity and workbook coverage. CI runs the complete analysis.

Some source definitions and currency scales still need reconciliation against annual statements. Normalising a series does not repair a unit error. The borrower scenario has no counterfactual or borrower-level outcomes and assumes fixed branch shares. Its outputs are accounting scenarios rather than causal estimates or validated long-term forecasts.

## Original work

- [Preprint](papers/Microfinancepreprintfinalasof10thoct.pdf)
- [Original employment script](historical/male_employment_gap_modelling_code.py) and [management comparison](historical/microfinance_management_practices_code.py)
- [Notebook PDF exports](notebooks_pdf_exports/)

The historical files remain unchanged. Use `python -m src.analysis` for the supported analysis.
