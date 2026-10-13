import numpy as np
import pandas as pd
import pytest

from finstrategy.pricing.garch_montecarlo import GarchParameters
from finstrategy.strategy.exit_timing_bs import (
    expected_monte_carlo_value_curve,
    recommend_exit_day,
    time_value_curve,
)
from finstrategy.strategy.protective_put import ProtectivePutPlan


def _plan(stock_price=100.0, strike=95.0, implied_volatility=0.3) -> ProtectivePutPlan:
    return ProtectivePutPlan(
        ticker="FAKE", contract_symbol="FAKE_PUT", expiration=pd.Timestamp("2026-12-31"),
        days_to_expiration=45, stock_price=stock_price, strike=strike, put_premium=3.0,
        quote_source="ask ao vivo", shares=100, contracts=1, cost_per_share=103.0, total_cost=10300.0,
        guaranteed_sale_price=strike, guaranteed_sale_value=strike * 100, max_loss_per_share=8.0,
        max_loss_total=800.0, max_loss_pct=0.08, breakeven_price=103.0, breakeven_change_pct=0.03,
        protection_cost_pct=0.03, annualized_protection_cost_pct=0.24, implied_volatility=implied_volatility,
        open_interest=500,
    )


def test_time_value_curve_decays_to_zero_at_expiration_for_otm_put():
    plan = _plan(stock_price=100.0, strike=90.0)  # put fora do dinheiro (sem valor intrínseco)
    curve = time_value_curve(plan, risk_free_rate=0.05, dividend_yield=0.0, total_trading_days=30)
    assert curve.iloc[0]["time_value"] > 0
    assert curve.iloc[-1]["time_value"] == pytest.approx(0.0, abs=1e-6)
    assert curve.iloc[-1]["days_remaining"] == 0


def test_time_value_curve_is_monotonically_non_increasing_holding_spot_fixed():
    plan = _plan(stock_price=100.0, strike=98.0)
    curve = time_value_curve(plan, risk_free_rate=0.03, dividend_yield=0.0, total_trading_days=40)
    diffs = curve["time_value"].diff().dropna()
    assert (diffs <= 1e-9).all()  # valor-tempo só cai (ou fica igual por arredondamento) com o tempo


def test_time_value_curve_intrinsic_value_matches_itm_put():
    plan = _plan(stock_price=90.0, strike=100.0)  # put dentro do dinheiro, 10 de valor intrínseco
    curve = time_value_curve(plan, risk_free_rate=0.05, dividend_yield=0.0, total_trading_days=10)
    assert curve.iloc[0]["intrinsic_value"] == pytest.approx(10.0)
    # put europeia profunda ITM pode precificar abaixo do valor intrínseco (sem exercício
    # antecipado, o strike só é recebido no vencimento, então é descontado) — por isso
    # time_value é sempre clipado em >= 0 no código, nunca negativo.
    assert curve.iloc[0]["time_value"] >= 0.0


def test_recommend_exit_day_finds_threshold_crossing():
    curve = pd.DataFrame({
        "trading_day": [0, 1, 2, 3],
        "time_value": [1.0, 0.6, 0.2, 0.0],
    })
    result = recommend_exit_day(curve, time_value_floor_pct=0.25)
    assert result["exit_day"] == 2  # primeiro dia com time_value (0.2) <= 25% de 1.0 (0.25)
    assert result["initial_time_value"] == pytest.approx(1.0)


def test_recommend_exit_day_defaults_to_expiration_when_never_crosses():
    curve = pd.DataFrame({"trading_day": [0, 1, 2], "time_value": [1.0, 0.9, 0.8]})
    result = recommend_exit_day(curve, time_value_floor_pct=0.1)
    assert result["exit_day"] == 2


def test_recommend_exit_day_handles_zero_initial_time_value():
    curve = pd.DataFrame({"trading_day": [0, 1], "time_value": [0.0, 0.0]})
    result = recommend_exit_day(curve, time_value_floor_pct=0.25)
    assert result["exit_day"] == 0
    assert result["initial_time_value"] == 0.0


def test_expected_monte_carlo_value_curve_runs_and_decays_toward_intrinsic():
    plan = _plan(stock_price=100.0, strike=95.0)
    garch = GarchParameters(
        mean_daily_return=0.0, omega=1e-6, alpha=0.05, beta=0.9, degrees_of_freedom=6, next_day_variance=1e-4
    )
    curve = expected_monte_carlo_value_curve(
        plan, garch, risk_free_rate=0.05, dividend_yield=0.0, total_trading_days=20,
        monte_carlo_paths=2_000, seed=0, step_days=5,
    )
    assert (curve["expected_mc_value"] >= 0).all()
    assert curve.iloc[-1]["days_remaining"] == 0
    assert curve.iloc[-1]["expected_mc_value"] == pytest.approx(max(plan.strike - plan.stock_price, 0.0))
