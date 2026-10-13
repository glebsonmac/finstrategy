import numpy as np
import pytest

from finstrategy.pricing.black_scholes import implied_volatility, price_and_greeks


def test_put_call_parity():
    spot, strike, time_years, rate, vol, div = 100.0, 95.0, 0.5, 0.05, 0.25, 0.01
    call = price_and_greeks("call", spot, strike, time_years, rate, vol, div)
    put = price_and_greeks("put", spot, strike, time_years, rate, vol, div)
    lhs = call.price - put.price
    rhs = spot * np.exp(-div * time_years) - strike * np.exp(-rate * time_years)
    assert lhs == pytest.approx(rhs, abs=1e-8)


def test_call_delta_bounds():
    deep_itm = price_and_greeks("call", 200.0, 50.0, 0.5, 0.05, 0.2)
    deep_otm = price_and_greeks("call", 50.0, 200.0, 0.5, 0.05, 0.2)
    assert deep_itm.delta > 0.95
    assert deep_otm.delta < 0.05


def test_put_delta_bounds():
    deep_itm_put = price_and_greeks("put", 50.0, 200.0, 0.5, 0.05, 0.2)
    deep_otm_put = price_and_greeks("put", 200.0, 50.0, 0.5, 0.05, 0.2)
    assert deep_itm_put.delta < -0.95
    assert deep_otm_put.delta > -0.05


def test_gamma_and_vega_are_positive_and_shared_between_call_and_put():
    call = price_and_greeks("call", 100.0, 100.0, 0.5, 0.05, 0.3)
    put = price_and_greeks("put", 100.0, 100.0, 0.5, 0.05, 0.3)
    assert call.gamma > 0
    assert call.vega > 0
    assert call.gamma == pytest.approx(put.gamma, rel=1e-9)
    assert call.vega == pytest.approx(put.vega, rel=1e-9)


def test_theta_decay_is_negative_for_atm_options_without_dividends():
    call = price_and_greeks("call", 100.0, 100.0, 0.5, 0.0, 0.3)
    assert call.theta < 0


def test_price_increases_with_volatility():
    low_vol = price_and_greeks("call", 100.0, 100.0, 0.5, 0.05, 0.1)
    high_vol = price_and_greeks("call", 100.0, 100.0, 0.5, 0.05, 0.5)
    assert high_vol.price > low_vol.price


def test_implied_volatility_recovers_input_volatility():
    true_vol = 0.35
    spot, strike, time_years, rate, div = 80.0, 85.0, 0.75, 0.04, 0.02
    market_price = price_and_greeks("put", spot, strike, time_years, rate, true_vol, div).price
    recovered = implied_volatility("put", market_price, spot, strike, time_years, rate, div)
    assert recovered == pytest.approx(true_vol, abs=1e-4)


def test_implied_volatility_returns_nan_for_impossible_price():
    # preço de mercado abaixo do valor intrínseco não tem solução de IV
    impossible_price = 0.0001
    result = implied_volatility("put", impossible_price, 100.0, 150.0, 0.5, 0.05)
    assert np.isnan(result)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        price_and_greeks("straddle", 100.0, 100.0, 0.5, 0.05, 0.2)
