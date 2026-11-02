"""Daily log return of a constant-weight portfolio."""

from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHT_SUM_TOLERANCE = 1e-6


def validate_weights(weights: dict[str, float]) -> None:
    """Raise ValueError unless the weights are non-negative and sum to 1."""
    if not weights:
        raise ValueError("The portfolio has no assets.")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("Weights must be non-negative.")
    total = sum(weights.values())
    if abs(total - 1) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"Weights must sum to 1, got {total:.6f}.")


def portfolio_log_returns(prices: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Return r_p,t = ln(sum_i w_i * P_i,t / P_i,t-1), rebalancing to `weights` every day.

    Simple returns add up across assets and log returns do not, so the weights are applied
    to simple returns. Days on which any asset has no return are dropped, so the series
    starts when the most recently listed asset starts.
    """
    validate_weights(weights)
    missing = sorted(set(weights) - set(prices.columns))
    if missing:
        raise ValueError(f"No prices for: {', '.join(missing)}.")
    tickers = list(weights)
    gross_returns = prices[tickers] / prices[tickers].shift(1)
    complete = gross_returns.dropna()
    weight_vector = np.array([weights[ticker] for ticker in tickers])
    values = np.log(complete.to_numpy() @ weight_vector)
    return pd.Series(values, index=complete.index, name="portfolio_return")
