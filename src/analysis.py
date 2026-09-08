"""Run a forecast audit and explicit borrower scenarios on the bundled workbook."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn

from .forecasting import evaluate_series

ROOT = Path(__file__).resolve().parents[1]
COMPANIES = ['annapurna', 'chaitanya', 'muthoot', 'belstar', 'satin', 'creditaccess', 'asirvad',
             'spandanassphoorty', 'fusionfinance', 'bandhan']
MACRO = ['workingAgesFemaleIndia', 'totalBranches', 'selfEmploymentIndia', 'mfiMarketshare',
         'employmentRateWorld', 'employmentRateIndia']
OPERATIONS = ['AverageLoanSize', 'ExpensePerBorrower', 'BorrowersPerBranch']
PERCENTAGES = {'selfEmploymentIndia', 'mfiMarketshare', 'employmentRateWorld', 'employmentRateIndia'}


def load_workbook(path):
    frame = pd.read_excel(path)
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any() or 'Year' not in frame:
        raise ValueError('workbook needs unique columns and an explicit Year column')
    year = pd.to_numeric(frame.pop('Year'), errors='raise')
    if year.isna().any() or not np.isfinite(year).all() or (year % 1 != 0).any() or year.duplicated().any():
        raise ValueError('Year must contain distinct finite integer years')
    frame.index = year.astype(int)
    frame = frame.sort_index()
    for column in frame:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
        if np.isinf(frame[column]).any():
            raise ValueError(f'{column}: infinite observation')
    # A documented naming alias, not a computed or imputed observation.
    frame = frame.rename(columns={'fusionfinanceAverageBorrowersPerBranch': 'fusionfinanceBorrowersPerBranch'})
    return frame


def borrowers_needed(population_future, world_employment_rate, population_base, india_employment_rate,
                     self_employment_rate, market_share, branch_share, female_share, income_share):
    rates = [world_employment_rate, india_employment_rate, self_employment_rate, market_share, branch_share, female_share, income_share]
    if not all(np.isfinite(r) and 0 <= r <= 1 for r in rates) or female_share == 0 or income_share == 0:
        raise ValueError('scenario rates must be in [0, 1], with positive borrower shares')
    if not all(np.isfinite(p) and p > 0 for p in [population_future, population_base]):
        raise ValueError('scenario populations must be finite and positive')
    jobs_gap = population_future * world_employment_rate - population_base * india_employment_rate
    # A negative gap implies no additional jobs needed under this accounting scenario.
    return max(0.0, jobs_gap) * self_employment_rate * market_share * branch_share / female_share / income_share


def run(path, output, cutoff=2020, female_share=.99, income_share=.985):
    if not 0 < female_share <= 1 or not 0 < income_share <= 1:
        raise ValueError('borrower shares must be in (0, 1]')
    frame = load_workbook(path)
    if frame.index[-1] != 2023 or 2017 not in frame.index or cutoff >= 2023:
        raise ValueError('this historical scenario requires data ending in 2023 and a cutoff before it')
    columns = MACRO + [company + suffix for company in COMPANIES for suffix in ['Borrowers', 'Branches', *OPERATIONS] if company + suffix in frame]
    models, forecasts, holdouts, errors = {}, {}, [], {}
    for column in columns:
        try:
            model, forecast, holdout = evaluate_series(frame[column], cutoff)
            models[column] = model
            forecasts[column] = forecast
            holdouts.append(holdout.assign(series=column).rename_axis('year').reset_index())
        except ValueError as error:
            errors[column] = str(error)
    if not forecasts:
        raise ValueError('no series has enough development and holdout observations for this cutoff')
    future = pd.DataFrame(forecasts)
    scenarios = []
    for company in COMPANIES:
        required = MACRO + [company + 'Branches', company + 'Borrowers']
        missing = [name for name in required if name not in forecasts or pd.isna(frame.loc[2023, name])]
        if missing:
            scenarios.append({'company': company, 'status': 'unavailable: ' + ', '.join(missing)})
            continue
        for year in range(2024, 2031):
            try:
                if frame.loc[2023, 'totalBranches'] <= 0 or frame.loc[2023, company + 'Borrowers'] <= 0:
                    raise ValueError('baseline branches and borrowers must be positive')
                needed = borrowers_needed(
                    future.loc[year, 'workingAgesFemaleIndia'], future.loc[year, 'employmentRateWorld'] / 100,
                    frame.loc[2023, 'workingAgesFemaleIndia'], frame.loc[2023, 'employmentRateIndia'] / 100,
                    future.loc[year, 'selfEmploymentIndia'] / 100, future.loc[year, 'mfiMarketshare'] / 100,
                    frame.loc[2023, company + 'Branches'] / frame.loc[2023, 'totalBranches'], female_share, income_share)
                projected = future.loc[year, company + 'Borrowers']
                if not np.isfinite(projected) or projected < 0:
                    raise ValueError('borrower forecast outside nonnegative domain')
                growth = projected - frame.loc[2023, company + 'Borrowers']
                scenarios.append({'company': company, 'year': year, 'status': 'scenario', 'additional_borrowers': growth,
                                  'borrowers_needed': needed, 'difference': growth - needed})
            except ValueError as error:
                scenarios.append({'company': company, 'year': year, 'status': str(error)})
    # Historical operating trends, indexed only to an observed 2017 value.
    operations = []
    for company in COMPANIES:
        for metric in OPERATIONS:
            column = company + metric
            if column not in frame or pd.isna(frame.loc[2017, column]) or frame.loc[2017, column] <= 0:
                continue
            observed = frame.loc[2017:2023, column].dropna()
            for year, value in observed.items():
                operations.append({'company': company, 'metric': metric, 'year': year, 'workbook_value': value,
                                   'index_2017': value / frame.loc[2017, column]})
    flags = []
    for column in future:
        for year, value in future[column].items():
            if not np.isfinite(value) or value < 0 or (column in PERCENTAGES and value > 100):
                flags.append({'series': column, 'year': int(year), 'value': float(value), 'reason': 'outside natural domain'})
    metadata = {'workbook_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'cutoff': cutoff,
                'scenario_baseline': 2023, 'female_share': female_share, 'income_share': income_share,
                'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
                'models': models, 'unavailable': errors, 'forecast_flags': flags}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'audit.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    future.rename_axis('year').to_csv(output / 'forecasts.csv')
    pd.concat(holdouts, ignore_index=True).to_csv(output / 'holdout.csv', index=False)
    pd.DataFrame(scenarios).to_csv(output / 'borrower-scenarios.csv', index=False)
    pd.DataFrame(operations).to_csv(output / 'operations-observed.csv', index=False)
    print(json.dumps({'evaluated_series': len(models), 'unavailable_series': len(errors),
                      'better_than_persistence_on_holdout': sum(m['holdout_mae'] < m['last_value_holdout_mae'] for m in models.values()),
                      'out_of_domain_forecasts': len(flags)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook', type=Path, default=ROOT / 'Microfinance Data 21.xlsx')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--cutoff', type=int, default=2020)
    parser.add_argument('--female-share', type=float, default=.99)
    parser.add_argument('--income-share', type=float, default=.985)
    args = parser.parse_args()
    try:
        run(args.workbook, args.output, args.cutoff, args.female_share, args.income_share)
    except (ValueError, OSError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
