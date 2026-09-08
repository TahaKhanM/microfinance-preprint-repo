import numpy as np
import pandas as pd
import pytest
from src.forecasting import clean_series, evaluate_series, predict_model, rolling_scores
from src.analysis import borrowers_needed, load_workbook, ROOT


def test_holdout_cannot_affect_selection():
    series = pd.Series(np.arange(12.) ** 2, index=np.arange(2012, 2024))
    report, _, _ = evaluate_series(series)
    changed = series.copy()
    changed.loc[2021:] = [-20000, 0, 90000]
    other, _, _ = evaluate_series(changed)
    assert report['model'] == other['model']
    assert report['cv_mae'] == other['cv_mae']
    assert report['holdout_mae'] != other['holdout_mae']


def test_actual_years_survive_internal_missingness():
    series = pd.Series([2., 6., np.nan, 14., 18., 22.], index=[2010, 2012, 2013, 2016, 2018, 2020])
    np.testing.assert_allclose(predict_model(series, [2022], 'linear'), [26.])


def test_rolling_scores_match_manual_past_only_baseline():
    series = pd.Series([1., 3., 7., 15., 31., 63.], index=range(2010, 2016))
    assert rolling_scores(series)['last_value'] == pytest.approx((16 + 32) / 2)


def test_polynomial_centering_and_constant_baseline():
    series = pd.Series([4., 4., 4., 4., 4., 4., 4.], index=range(2017, 2024))
    report, forecast, _ = evaluate_series(series, cutoff=2021)
    assert report['model'] == 'last_value'
    np.testing.assert_allclose(forecast, 4)


def test_bad_years_and_insufficient_samples():
    with pytest.raises(ValueError):
        clean_series(pd.Series([1., 2.], index=[2020, 2020]))
    with pytest.raises(ValueError):
        evaluate_series(pd.Series([1., 2.], index=[2020, 2023]))


def test_scenario_units_and_sensitivity():
    values = (200., .5, 100., .5, .5, .4, .1, 1., 1.)
    assert borrowers_needed(*values) == pytest.approx(1.)
    assert borrowers_needed(*values[:-1], .5) == pytest.approx(2.)
    with pytest.raises(ValueError):
        borrowers_needed(*values[:-1], 0)
    with pytest.raises(ValueError):
        borrowers_needed(200., 1.2, *values[2:])


def test_workbook_years_names_and_coverage():
    frame = load_workbook(ROOT / 'Microfinance Data 21.xlsx')
    assert frame.index.tolist() == list(range(1999, 2024))
    assert 'fusionfinanceBorrowers' in frame
    assert 'fusionfinanceBorrowersPerBranch' in frame
    assert frame.shape == (25, 79)
