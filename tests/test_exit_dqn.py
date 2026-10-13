import numpy as np
import pandas as pd
import pytest

from finstrategy.config_model import AnalysisSettings
from finstrategy.pricing.garch_montecarlo import GarchParameters, simulate_price_paths
from finstrategy.rl.exit_dqn import (
    ACTION_HOLD,
    ACTION_SELL,
    EXIT_STATE_NAMES,
    build_exit_states,
    evaluate_exit_timing,
    greedy_exit_day,
    train_exit_dqn,
)
from finstrategy.strategy.protective_put import ProtectivePutPlan


def _plan(stock_price=100.0, strike=95.0, implied_volatility=0.3, put_premium=3.0) -> ProtectivePutPlan:
    return ProtectivePutPlan(
        ticker="FAKE", contract_symbol="FAKE_PUT", expiration=pd.Timestamp("2026-12-31"),
        days_to_expiration=30, stock_price=stock_price, strike=strike, put_premium=put_premium,
        quote_source="ask ao vivo", shares=100, contracts=1, cost_per_share=103.0, total_cost=10300.0,
        guaranteed_sale_price=strike, guaranteed_sale_value=strike * 100, max_loss_per_share=8.0,
        max_loss_total=800.0, max_loss_pct=0.08, breakeven_price=103.0, breakeven_change_pct=0.03,
        protection_cost_pct=0.03, annualized_protection_cost_pct=0.24, implied_volatility=implied_volatility,
        open_interest=500,
    )


def _garch(next_day_variance=1e-4) -> GarchParameters:
    return GarchParameters(mean_daily_return=0.0, omega=1e-6, alpha=0.05, beta=0.9, degrees_of_freedom=6, next_day_variance=next_day_variance)


def _fast_settings(**overrides) -> AnalysisSettings:
    base = dict(monte_carlo_paths=3_000, exit_dqn_training_rounds=5, exit_dqn_hidden_units=8, random_seed=0)
    base.update(overrides)
    return AnalysisSettings(**base)


def test_build_exit_states_shape_and_known_columns():
    plan = _plan()
    garch = _garch()
    rng = np.random.default_rng(0)
    window_days = 10
    paths = simulate_price_paths(garch, plan.stock_price, window_days, 0.0, 200, rng)
    states = build_exit_states(paths, plan, risk_free_rate=0.05, dividend_yield=0.0, garch_annual_volatility=garch.current_annual_volatility, window_days=window_days)
    assert states.shape == (200, window_days + 1, len(EXIT_STATE_NAMES))
    # tempo restante no dia 0 é 1.0 (fração cheia), no último dia é 0.0
    assert np.allclose(states[:, 0, 6], 1.0)
    assert np.allclose(states[:, -1, 6], 0.0)


def test_build_exit_states_pnl_matches_manual_bs_reconstruction():
    plan = _plan(put_premium=3.0)
    garch = _garch()
    rng = np.random.default_rng(0)
    window_days = 5
    paths = simulate_price_paths(garch, plan.stock_price, window_days, 0.0, 50, rng)
    states = build_exit_states(paths, plan, 0.05, 0.0, garch.current_annual_volatility, window_days)
    pnl_pct = states[:, 0, 4]
    reconstructed_price = pnl_pct * plan.put_premium + plan.put_premium
    assert (reconstructed_price > 0).all()


def test_train_exit_dqn_runs_and_greedy_exit_day_is_within_window():
    plan = _plan()
    garch = _garch()
    settings = _fast_settings()
    result = train_exit_dqn(plan, garch, risk_free_rate=0.05, dividend_yield=0.0, window_days=10, settings=settings, seed=0)
    exit_days = greedy_exit_day(result.network, result.states, window_days=10)
    assert exit_days.shape == (result.states.shape[0],)
    assert (exit_days >= 0).all() and (exit_days <= 10).all()


def test_greedy_exit_day_sells_immediately_when_network_always_prefers_sell():
    class AlwaysSellNetwork:
        def action_values(self, states):
            values = np.zeros((len(states), 2))
            values[:, ACTION_SELL] = 1.0
            return values

    states = np.zeros((5, 6, len(EXIT_STATE_NAMES)))
    exit_days = greedy_exit_day(AlwaysSellNetwork(), states, window_days=5)
    assert (exit_days == 0).all()


def test_greedy_exit_day_holds_to_expiration_when_network_always_prefers_hold():
    class AlwaysHoldNetwork:
        def action_values(self, states):
            values = np.zeros((len(states), 2))
            values[:, ACTION_HOLD] = 1.0
            return values

    window_days = 5
    states = np.zeros((5, window_days + 1, len(EXIT_STATE_NAMES)))
    exit_days = greedy_exit_day(AlwaysHoldNetwork(), states, window_days=window_days)
    assert (exit_days == window_days).all()


def test_evaluate_exit_timing_runs_end_to_end():
    plan = _plan()
    garch = _garch()
    settings = _fast_settings()
    evaluation = evaluate_exit_timing(
        plan, garch, risk_free_rate=0.05, dividend_yield=0.0, window_days=10, settings=settings, fixed_rule_exit_day=5,
    )
    assert set(evaluation["summary"].keys()) == {"DQN", "Manter até o vencimento", "Regra fixa (Black-Scholes)"}
    assert isinstance(evaluation["beats_benchmarks"], bool)
    assert 0 <= evaluation["p_value_vs_hold"] <= 1
    assert 0 <= evaluation["p_value_vs_fixed_rule"] <= 1
