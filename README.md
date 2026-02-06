# Microfinance & the Female Employment Gap (India)

This repository contains the code and data used for the preprint:

- **“A Detailed Study Examining The Current Contributions Of Microfinance Institutions In Closing The Female Employment Gap Between India And The World …”** (Oct 10, 2024)  
  See: `papers/Microfinancepreprintfinalasof10thoct.pdf`

The project combines:
1) **a borrower-growth model** to estimate when large Indian MFIs could *proportionally* meet their share of closing the India–world female employment-rate gap, and  
2) **an operational/management-practices analysis** linking branch utilization, loan sizes, and costs to effectiveness in narrowing the gap.

---

## Repository contents

- `src/male_employment_gap_modelling_code.py`  
  Projections + “borrowers needed” model + trend classification (positive/negative/no-trend).
- `src/microfinance_management_practices_code.py`  
  Operational metrics comparison between positive vs negative companies (loan size, expense per borrower, borrowers per branch).
- `Microfinance Data 21.xlsx`  
  Consolidated dataset used by both scripts.
- `papers/`  
  The preprint PDF + the original notebook-to-PDF code exports.
- `notebooks_pdf_exports/`  
  The two PDF “code exports” these scripts were transcribed from.

> Note: The Python scripts are intended as a direct transcription of the notebook code exported to PDF. They are not “refactored” into a package on purpose, to keep the implementation aligned with what was written during the project.

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

Run the analyses:

```bash
python src/male_employment_gap_modelling_code.py
python src/microfinance_management_practices_code.py
```

Both scripts expect the Excel file to be present at:

```text
./Microfinance Data 21.xlsx
```

---

## What the research is doing

### 1) Estimating “borrowers needed” for proportional contribution

The core idea is to estimate how many **new income-generating female clients** an MFI would need by a future year \(t\) to meet its *proportional* share of closing the female employment-rate gap between India and the world.

The code constructs:

**New jobs needed for parity by year \(t\):**
\[
\Delta J_t
= W_{	ext{IN},t}\,e_{	ext{world},t}
- W_{	ext{IN},2023}\,e_{	ext{IN},2023}
\]
where:
- \(W_{	ext{IN},t}\) is India’s working-age female population (projected),
- \(e_{	ext{world},t}\) is the world female employment rate (projected),
- \(e_{	ext{IN},2023}\) is treated as a static baseline (2023).

**Proportional share for MFI \(i\)** is proxied using branch share:
\[
lpha_i = rac{b_i}{B}
\]
where \(b_i\) is the MFI’s number of branches and \(B\) is total MFI branches.

The final “borrowers needed” estimate used in the scripts is:
\[
N_{i,t}
= rac{\Delta J_t \cdot s_t \cdot m_t \cdot lpha_i}{p^{(f)}_i\,p^{(ig)}_i}
\]
where:
- \(s_t\) is the self-employment rate (projected),
- \(m_t\) is microfinance market-share of borrowers (projected),
- \(p^{(f)}_i\) is proportion of female borrowers (defaults to 0.99 if missing),
- \(p^{(ig)}_i\) is proportion of income-generating loans (defaults to 0.985 if missing).

The code then:
- projects each MFI’s borrower growth, forms a **cumulative change in borrowers**, and
- compares it to \(N_{i,t}\) for 2024–2030 via a (scaled) difference curve.

Summing across the surveyed MFIs, the preprint reports that the **first year the total scaled difference turns positive is 2028**, suggesting that *if trends persist and there are no large shocks*, MFIs could meet their proportional contribution by then.

### 2) Linking management practices to impact

MFIs are grouped by whether their borrower-projection-vs-needed curves show:
- an **upward** trend,
- a **downward** trend, or
- no clear trend.

The management-practices script then compares (normalized and summed across groups):
- **Average loan size**
- **Expense per borrower** (a proxy for LRAC-style operational efficiency)
- **Borrowers per branch** (branch utilization / “crowded vs underutilized”)

Key patterns discussed in the preprint:
- Positive-trend MFIs tend to have **lower expense per borrower**, consistent with leveraging economies of scale more effectively.
- Underutilized branches (lower borrowers/branch) correlate with reduced effectiveness, while MFIs operating **fewer but larger branches** tend to contribute more to narrowing the gap.
- Positive-trend MFIs are associated with **larger average loan sizes**, interpreted as supporting more sustainable micro-enterprise success and longer-run employment effects.

---

## Modelling details

### Polynomial regression for projections

Both scripts use the same approach to project time series:
- Try polynomial degrees up to **3** (to mitigate overfitting),
- Choose the degree with the best \(R^2\),
- In some cases, reduce degree further (e.g., if a “percentage” projection exceeds 100%).

This is implemented via `PolynomialFeatures` + `LinearRegression` and selecting the best model via in-sample \(R^2\).

---

## What was learned building this project (technical)

### Regression & model selection
- Implementing **polynomial regression** in scikit-learn using feature expansion (`PolynomialFeatures`) and fitting linear models on the transformed design matrix.
- Using **\(R^2\)** as a simple fit criterion across candidate model degrees, and imposing a **degree cap** to reduce overfitting risk in short time series.
- Adding practical constraints for “bounded” variables (e.g., keeping rate-like series below 100%) to avoid nonsense extrapolations.

### Data handling in Python
- Reading and managing multi-column time-series data from Excel with **pandas** (`pd.ExcelFile`, `pd.read_excel`) and aligning indices to years.
- Handling missingness using `dropna()`, boolean masks, and `np.isnan` filtering so comparisons across companies don’t break when reporting windows differ.

### Comparative analysis & normalization
- Making cross-company comparisons fair by **normalizing each company’s metric by its own max** over the observed horizon, then aggregating within groups (positive vs negative).
- Designing plots that reveal operational differences in *level* vs *shape* (e.g., borrowers/branch trajectories).

### Practical workflow learnings
- Consolidating disparate public sources into one reproducible workbook makes analysis far easier to share and review.
- Exporting notebooks to PDFs is convenient for publication, but for reproducibility it helps to also maintain runnable scripts (this repo) and pin dependencies.

---

## Caveats / assumptions

The preprint discusses simplifying assumptions, including:
- using industry averages for some missing company-level parameters,
- treating branches as uniform in output/efficiency,
- using polynomial regression for extrapolation (with guardrails), and
- surveying a subset of the full market (large MFIs / ~60% coverage).
