import numpy as np
import pandas as pd
import pytest

from finstrategy.config_model import AnalysisSettings
from finstrategy.strategy.risk_analysis import build_risk_analysis


def _chosen_row(stock_price=100.0, strike=90.0, put_premium=3.0, implied_volatility=0.3, days_to_expiration=45) -> dict:
    return {
        "ticker": "FAKE", "contract_symbol": "FAKE_PUT", "expiration": pd.Timestamp("2026-12-31"),
        "days_to_expiration": days_to_expiration, "stock_price": stock_price, "strike": strike,
        "put_premium": put_premium, "quote_source": "ask ao vivo", "shares": 100, "contracts": 1,
        "cost_per_share": stock_price + put_premium, "total_cost": (stock_price + put_premium) * 100,
        "guaranteed_sale_price": strike, "guaranteed_sale_value": strike * 100,
        "max_loss_per_share": stock_price + put_premium - strike, "max_loss_total": (stock_price + put_premium - strike) * 100,
        "max_loss_pct": (stock_price + put_premium - strike) / (stock_price + put_premium),
        "breakeven_price": stock_price + put_premium, "breakeven_change_pct": put_premium / stock_price,
        "protection_cost_pct": put_premium / stock_price, "annualized_protection_cost_pct": put_premium / stock_price * 365 / days_to_expiration,
        "implied_volatility": implied_volatility, "open_interest": 500,
    }


def _result(stock_price=100.0, strike=90.0, put_premium=3.0, seed=0, num_paths=5000) -> dict:
    rng = np.random.default_rng(seed)
    simulated_prices = rng.normal(stock_price, stock_price * 0.1, num_paths).clip(min=1.0)
    return {
        "recommended": _chosen_row(stock_price, strike, put_premium),
        "dividend_yield": 0.01,
        "simulated_prices": simulated_prices,
    }


def _settings(**overrides) -> AnalysisSettings:
    return AnalysisSettings(exit_time_value_floor_pct=0.25, random_seed=1, **overrides)


def test_build_risk_analysis_returns_all_sections():
    result = _result()
    analysis = build_risk_analysis("FAKE", result, risk_free_rate=0.05, settings=_settings())

    assert analysis.ticker == "FAKE"
    assert analysis.greeks.price > 0
    assert analysis.exit_recommendation["exit_day"] >= 0
    assert isinstance(analysis.exit_date_estimate, pd.Timestamp)
    assert 0 <= analysis.prob_price_up <= 1
    assert 0 <= analysis.prob_price_down <= 1
    assert analysis.price_range_90[0] < analysis.expected_price < analysis.price_range_90[1]


def test_price_diff_pct_positive_when_market_premium_above_theoretical():
    # prêmio de mercado bem acima do que Black-Scholes acharia razoável
    result = _result(put_premium=50.0)
    analysis = build_risk_analysis("FAKE", result, risk_free_rate=0.05, settings=_settings())
    assert analysis.price_diff_pct > 0


def test_price_diff_pct_negative_when_market_premium_below_theoretical():
    result = _result(put_premium=0.05)
    analysis = build_risk_analysis("FAKE", result, risk_free_rate=0.05, settings=_settings())
    assert analysis.price_diff_pct < 0


def test_prob_price_up_and_down_sum_near_one():
    result = _result()
    analysis = build_risk_analysis("FAKE", result, risk_free_rate=0.05, settings=_settings())
    assert analysis.prob_price_up + analysis.prob_price_down == pytest.approx(1.0, abs=0.01)


def test_expected_put_payoff_matches_manual_intrinsic_calc():
    result = _result(strike=90.0)
    analysis = build_risk_analysis("FAKE", result, risk_free_rate=0.05, settings=_settings())
    manual_payoff = np.maximum(90.0 - result["simulated_prices"], 0.0)
    assert analysis.expected_put_payoff_at_expiration == pytest.approx(manual_payoff.mean())
    assert analysis.put_payoff_range_90[0] <= analysis.put_payoff_range_90[1]


def test_deep_otm_put_has_near_zero_expected_payoff():
    # strike bem abaixo do preço simulado -> quase nunca exercida
    result = _result(stock_price=100.0, strike=50.0)
    analysis = build_risk_analysis("FAKE", result, risk_free_rate=0.05, settings=_settings())
    assert analysis.expected_put_payoff_at_expiration < 1.0


def test_mc_precision_pct_is_small_fraction_for_large_sample():
    result = _result(num_paths=20_000)
    analysis = build_risk_analysis("FAKE", result, risk_free_rate=0.05, settings=_settings())
    assert 0 <= analysis.mc_precision_pct < 0.1  # amostra grande -> IC estreito em relação à média
    assert analysis.price_mean_ci_95[0] < analysis.expected_price < analysis.price_mean_ci_95[1]


def test_exit_date_estimate_is_after_today():
    today = pd.Timestamp("2026-10-14")
    result = _result()
    analysis = build_risk_analysis("FAKE", result, risk_free_rate=0.05, settings=_settings(), today=today)
    assert analysis.exit_date_estimate >= today


def test_build_risk_analysis_is_deterministic_with_fixed_seed():
    result = _result()
    settings = _settings()
    first = build_risk_analysis("FAKE", result, risk_free_rate=0.05, settings=settings, seed=42)
    second = build_risk_analysis("FAKE", result, risk_free_rate=0.05, settings=settings, seed=42)
    assert first.price_mean_ci_95 == second.price_mean_ci_95
    assert first.mc_precision_pct == second.mc_precision_pct
