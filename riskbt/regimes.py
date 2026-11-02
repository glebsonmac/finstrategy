"""Market regime of each day: calm or crisis, from the Ibovespa's realized volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskbt import config

CALM = "calm"
CRISIS = "crisis"
UNCLASSIFIED = "unclassified"


def realized_volatility(returns: pd.Series, window: int = config.RV_WINDOW) -> pd.Series:
    """Annualized realized volatility: RV_t = sqrt(252 / 21 * sum of r^2 over 21 days) (eq. 2.18)."""
    squared_sum = (returns**2).rolling(window).sum()
    return np.sqrt(config.TRADING_DAYS_PER_YEAR / window * squared_sum)


def _crisis_flags(volatility: pd.Series) -> pd.Series:
    """True when RV_t exceeds the percentile of all RV_s with s <= t; missing during warm-up."""
    threshold = volatility.expanding(min_periods=config.REGIME_MIN_HISTORY).quantile(
        config.REGIME_PERCENTILE / 100
    )
    return (volatility > threshold).where(threshold.notna())


def _labels(flags: pd.Series) -> pd.Series:
    labels = np.where(flags.isna(), UNCLASSIFIED, np.where(flags == True, CRISIS, CALM))  # noqa: E712
    return pd.Series(labels, index=flags.index, name="regime")


def classify_regimes(index_returns: pd.Series) -> pd.Series:
    """Label day t as crisis if RV_t-1 exceeds the 80th percentile of {RV_s : s <= t-1}.

    Only information up to t-1 is used, and at least 250 volatility observations are required
    before classifying (PDF, section 2.5.1).
    """
    flags = _crisis_flags(realized_volatility(index_returns))
    return _labels(flags.astype(object).shift(1))


def next_day_regime(index_returns: pd.Series) -> str:
    """Regime of the first day after the sample, using data up to its last day."""
    flags = _crisis_flags(realized_volatility(index_returns))
    return str(_labels(flags).iloc[-1])


def crisis_episodes(regimes: pd.Series) -> pd.DataFrame:
    """Contiguous crisis periods as a table with columns start, end and days."""
    is_crisis = regimes == CRISIS
    groups = (is_crisis != is_crisis.shift()).cumsum()[is_crisis]
    dates = pd.Series(groups.index, index=groups.index)
    episodes = dates.groupby(groups).agg(start="first", end="last", days="size")
    return episodes.reset_index(drop=True)
