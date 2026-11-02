"""One-day-ahead forecasts in a rolling window (PDF, sections 2.2 and 5.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from riskbt import config
from riskbt.estimators import ESTIMATORS, Forecast, fit_garch

ALPHAS = (config.ALPHA_VAR, config.ALPHA_ES)


@dataclass(frozen=True)
class RollingResult:
    forecasts: pd.DataFrame  # estimator, alpha, date, return, var, es, hit
    parameters: pd.DataFrame  # one row per refit, estimator and alpha
    tail_losses: dict[tuple[str, float], np.ndarray]  # days x tail points, for the Z2 simulation


def _fit_all(window: np.ndarray, names: tuple[str, ...], alphas: tuple[float, ...]) -> dict:
    """Fit every estimator on one window, sharing a single GARCH fit among those that use it."""
    needs_garch = any(ESTIMATORS[name].uses_garch for name in names)
    garch = fit_garch(window) if needs_garch else None
    return {
        (name, alpha): ESTIMATORS[name].fit(window, alpha, garch)
        for name in names for alpha in alphas
    }


def latest_forecasts(
    returns: pd.Series,
    names: tuple[str, ...] = tuple(ESTIMATORS),
    alphas: tuple[float, ...] = ALPHAS,
    window: int = config.ESTIMATION_WINDOW,
) -> dict[tuple[str, float], Forecast]:
    """Forecast for the day after the sample, with parameters fitted on the last window."""
    values = returns.to_numpy()[-window:]
    fits = _fit_all(values, names, alphas)
    return {key: ESTIMATORS[key[0]].forecast(values, fit, key[1]) for key, fit in fits.items()}


def _parameter_rows(date: pd.Timestamp, fits: dict) -> list[dict]:
    """Keep the scalar values of each refit, including the convergence flag; arrays are left out."""
    return [
        {
            "date": date, "estimator": name, "alpha": alpha,
            **{k: v for k, v in fit.items() if isinstance(v, (int, float))},
        }
        for (name, alpha), fit in fits.items()
    ]


def _forecast_frame(dates, realized, name, alpha, var, es) -> pd.DataFrame:
    return pd.DataFrame({
        "estimator": name, "alpha": alpha, "date": dates, "return": realized,
        "var": var, "es": es, "hit": realized < -var,
    })


def rolling_forecasts(
    returns: pd.Series,
    names: tuple[str, ...] = tuple(ESTIMATORS),
    alphas: tuple[float, ...] = ALPHAS,
    window: int = config.ESTIMATION_WINDOW,
    refit_every: int = config.REFIT_EVERY,
) -> RollingResult:
    """Forecast day t from the `window` returns ending at t-1, refitting every `refit_every` days.

    Between refits the parameters stay fixed, for all estimators alike, and only the data
    feeding the volatility or quantile recursion moves forward. Raises ValueError when the
    series is too short to produce a forecast.
    """
    values = returns.to_numpy()
    days = len(values) - window
    if days <= 0:
        raise ValueError(f"Need more than {window} returns, got {len(values)}.")
    keys = [(name, alpha) for name in names for alpha in alphas]
    var = {key: np.empty(days) for key in keys}
    es = {key: np.full(days, np.nan) for key in keys}
    tails = {key: np.empty((days, config.TAIL_GRID_POINTS)) for key in keys if key[0] != "caviar"}
    parameter_rows: list[dict] = []
    for day in range(days):
        sample = values[day : day + window]
        if day % refit_every == 0:
            fits = _fit_all(sample, names, alphas)
            parameter_rows += _parameter_rows(returns.index[day + window], fits)
        for key in keys:
            forecast = ESTIMATORS[key[0]].forecast(sample, fits[key], key[1])
            var[key][day] = forecast.var
            if forecast.es is not None:
                es[key][day] = forecast.es
                tails[key][day] = forecast.details["tail_losses"]
    dates, realized = returns.index[window:], values[window:]
    frames = [_forecast_frame(dates, realized, key[0], key[1], var[key], es[key]) for key in keys]
    return RollingResult(pd.concat(frames, ignore_index=True), pd.DataFrame(parameter_rows), tails)
