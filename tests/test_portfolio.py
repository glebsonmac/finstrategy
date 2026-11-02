import numpy as np
import pandas as pd
import pytest

from riskbt.portfolio import portfolio_log_returns, validate_weights


def make_prices(**columns):
    length = len(next(iter(columns.values())))
    return pd.DataFrame(columns, index=pd.bdate_range("2020-01-01", periods=length), dtype=float)


def test_single_asset_is_its_own_log_return():
    prices = make_prices(A=[100, 110, 99])
    result = portfolio_log_returns(prices, {"A": 1.0})
    np.testing.assert_allclose(result, [np.log(1.1), np.log(0.9)])


def test_weights_apply_to_simple_returns():
    prices = make_prices(A=[100, 120], B=[100, 90])
    result = portfolio_log_returns(prices, {"A": 0.5, "B": 0.5})
    # 0.5 * 20% + 0.5 * (-10%) = 5%, not the average of the two log returns
    assert result.iloc[0] == pytest.approx(np.log(1.05))


def test_weights_are_constant_over_time():
    prices = make_prices(A=[100, 200, 400], B=[100, 100, 100])
    result = portfolio_log_returns(prices, {"A": 0.5, "B": 0.5})
    np.testing.assert_allclose(result, [np.log(1.5), np.log(1.5)])


def test_series_starts_with_most_recent_asset_and_skips_missing_days():
    prices = make_prices(A=[100, 101, 102, 103, 104, 105], B=[np.nan, np.nan, 50, 51, np.nan, 53])
    result = portfolio_log_returns(prices, {"A": 0.5, "B": 0.5})
    assert list(result.index) == [prices.index[3]]


def test_later_prices_do_not_change_earlier_returns():
    prices = make_prices(A=[100, 101, 102, 103], B=[50, 51, 52, 53])
    changed = prices.copy()
    changed.iloc[-1] = [999, 1]
    weights = {"A": 0.3, "B": 0.7}
    original = portfolio_log_returns(prices, weights)
    pd.testing.assert_series_equal(original.iloc[:-1], portfolio_log_returns(changed, weights).iloc[:-1])


@pytest.mark.parametrize("weights", [{}, {"A": 0.5}, {"A": 1.5, "B": -0.5}, {"A": 0.6, "B": 0.6}])
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        validate_weights(weights)


def test_unknown_ticker_is_rejected():
    with pytest.raises(ValueError, match="No prices for: B"):
        portfolio_log_returns(make_prices(A=[100, 101]), {"B": 1.0})
