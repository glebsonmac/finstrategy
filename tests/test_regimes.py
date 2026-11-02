import numpy as np
import pandas as pd
import pytest

from riskbt import config
from riskbt.regimes import CALM, CRISIS, UNCLASSIFIED, classify_regimes, crisis_episodes, next_day_regime, realized_volatility


def make_returns(volatilities, seed=1):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=len(volatilities)) * volatilities
    return pd.Series(values, index=pd.bdate_range("2010-01-01", periods=len(values)))


def test_realized_volatility_formula():
    returns = make_returns(np.full(30, 0.01))
    expected = np.sqrt(252 / 21 * np.sum(returns.iloc[:21] ** 2))
    assert realized_volatility(returns).iloc[20] == pytest.approx(expected)
    assert realized_volatility(returns).iloc[:20].isna().all()


def test_constant_returns_annualize_to_daily_vol_times_sqrt_252():
    returns = pd.Series(0.01, index=pd.bdate_range("2020-01-01", periods=30))
    assert realized_volatility(returns).iloc[-1] == pytest.approx(0.01 * np.sqrt(252))


def test_warm_up_stays_unclassified():
    regimes = classify_regimes(make_returns(np.full(600, 0.01)))
    first_valid = config.RV_WINDOW - 1 + config.REGIME_MIN_HISTORY - 1
    assert (regimes.iloc[: first_valid + 1] == UNCLASSIFIED).all()
    assert regimes.iloc[first_valid + 1] != UNCLASSIFIED


def test_volatility_burst_is_detected_with_one_day_lag():
    volatility = np.r_[np.full(800, 0.01), np.full(100, 0.05), np.full(400, 0.01)]
    regimes = classify_regimes(make_returns(volatility))
    assert (regimes.iloc[830:900] == CRISIS).all()
    assert (regimes.iloc[1100:] == CALM).mean() > 0.75  # the 80th percentile keeps some calm days above it


def test_no_look_ahead():
    returns = make_returns(np.full(700, 0.01))
    altered = returns.copy()
    altered.iloc[600:] *= 10
    pd.testing.assert_series_equal(classify_regimes(returns).iloc[:601], classify_regimes(altered).iloc[:601])


def test_crisis_day_depends_on_previous_volatility_only():
    returns = make_returns(np.full(700, 0.01))
    altered = returns.copy()
    altered.iloc[650] = 0.5  # a huge return on day 650 can only affect the label from day 651
    assert classify_regimes(returns).iloc[650] == classify_regimes(altered).iloc[650]
    assert classify_regimes(altered).iloc[651] == CRISIS


def test_next_day_regime_is_the_shifted_label():
    returns = make_returns(np.full(700, 0.01))
    assert next_day_regime(returns.iloc[:-1]) == classify_regimes(returns).iloc[-1]


def test_crisis_episodes():
    regimes = pd.Series(
        [CALM, CRISIS, CRISIS, CALM, CRISIS], index=pd.bdate_range("2020-01-01", periods=5),
    )
    episodes = crisis_episodes(regimes)
    assert list(episodes["days"]) == [2, 1]
    assert episodes["start"].iloc[1] == regimes.index[4]
