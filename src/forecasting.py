"""Time-ordered model selection with an untouched chronological holdout."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

CANDIDATES = ('last_value', 'linear', 'quadratic', 'cubic')


def clean_series(series):
    values = pd.to_numeric(series, errors='raise').dropna().sort_index()
    years = np.asarray(values.index, dtype=float)
    if len(values) < 1 or not np.all(np.isfinite(values)) or not np.all(np.isfinite(years)):
        raise ValueError('series needs finite observed values and years')
    if np.any(years % 1) or len(np.unique(years)) != len(years):
        raise ValueError('years must be distinct integers')
    return values.astype(float)


def predict_model(series, years, name):
    series = clean_series(series)
    target = np.asarray(years, dtype=float)
    if target.ndim != 1 or not np.all(np.isfinite(target)) or name not in CANDIDATES:
        raise ValueError('invalid target years or model')
    if name == 'last_value':
        return np.full(len(target), series.iloc[-1])
    degree = CANDIDATES.index(name)
    if len(series) < degree + 1:
        raise ValueError('not enough observations for the polynomial degree')
    origin = float(series.index[0])
    # Center dates on the first training year; no future-derived scaling.
    x = (np.asarray(series.index, dtype=float) - origin)[:, None]
    model = make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression())
    model.fit(x, series.to_numpy())
    return model.predict((target - origin)[:, None])


def rolling_scores(series, minimum_train=4):
    series = clean_series(series)
    if len(series) <= minimum_train:
        raise ValueError('at least one validation observation is required')
    scores = {}
    for model in CANDIDATES:
        errors = []
        for split in range(minimum_train, len(series)):
            training = series.iloc[:split]
            year = series.index[split]
            prediction = predict_model(training, [year], model)[0]
            errors.append(abs(prediction - series.iloc[split]))
        scores[model] = float(np.mean(errors))
    return scores


def evaluate_series(series, cutoff=2020, forecast_end=2030):
    """Select on <=cutoff only, score later observations, then refit for scenarios."""
    series = clean_series(series)
    development = series.loc[series.index <= cutoff]
    holdout = series.loc[series.index > cutoff]
    if len(development) < 5 or len(holdout) < 1:
        raise ValueError('need >=5 development observations and >=1 holdout observation')
    scores = rolling_scores(development)
    # Stable ordering favors the simpler model on an exact tie.
    selected = min(CANDIDATES, key=lambda name: scores[name])
    predictions = predict_model(development, holdout.index, selected)
    baseline = predict_model(development, holdout.index, 'last_value')
    holdout_mae = float(np.mean(abs(predictions - holdout.to_numpy())))
    baseline_mae = float(np.mean(abs(baseline - holdout.to_numpy())))
    years = np.arange(int(series.index[-1]) + 1, forecast_end + 1)
    forecast = predict_model(series, years, selected)
    report = {'model': selected, 'development_cases': len(development), 'holdout_cases': len(holdout),
              'development_last_year': int(development.index[-1]), 'holdout_first_year': int(holdout.index[0]),
              'cv_mae': scores, 'holdout_mae': holdout_mae, 'last_value_holdout_mae': baseline_mae}
    return report, pd.Series(forecast, index=years, dtype=float), pd.DataFrame({
        'observed': holdout, 'prediction': predictions, 'last_value': baseline})
