import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from finstrategy.config_model import AnalysisSettings
from finstrategy.reporting.charts import (
    plot_entry_evaluation,
    plot_exit_evaluation,
    plot_protection_payoff,
    plot_strike_menu,
    plot_time_value_curve,
)
from finstrategy.strategy.exit_timing_bs import recommend_exit_day, time_value_curve
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


def _menu_and_result():
    plan = _plan()
    menu = pd.DataFrame({
        "strike": [90.0, 95.0, 100.0],
        "protection_cost_pct": [0.02, 0.03, 0.05],
        "max_loss_pct": [0.12, 0.08, 0.04],
    })
    recommended = {**plan.__dict__, "strike": 95.0, "protection_cost_pct": 0.03, "max_loss_pct": 0.08}
    rng = np.random.default_rng(0)
    simulated_prices = rng.normal(100, 10, 2000).clip(min=1)
    result = {"menu": menu, "recommended": recommended, "expiration": plan.expiration, "simulated_prices": simulated_prices}
    return plan, result


def test_plot_strike_menu_returns_figure_with_one_axis():
    _, result = _menu_and_result()
    fig = plot_strike_menu("FAKE", result, max_acceptable_loss_pct=0.08)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_plot_protection_payoff_returns_figure_with_two_axes():
    _, result = _menu_and_result()
    settings = AnalysisSettings()
    fig = plot_protection_payoff("FAKE", result, settings)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2


def test_plot_entry_evaluation_returns_figure():
    summary = pd.DataFrame({
        "economia média (%)": [1.2, 0.0, 0.5, -0.3],
        "IC 95% inferior": [-0.5, 0.0, -0.2, -1.0],
        "IC 95% superior": [2.9, 0.0, 1.2, 0.4],
        "% janelas melhores que o 1º dia": [0.6, 0.0, 0.5, 0.4],
    }, index=["DQN", "Comprar no 1º dia", "Preço médio da janela", "Dia aleatório"])
    evaluation = {"summary": summary, "test_windows": 12, "test_period_start": pd.Timestamp("2026-01-01").date()}
    fig = plot_entry_evaluation("FAKE", evaluation)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_plot_exit_evaluation_handles_dict_of_dicts_summary():
    evaluation = {
        "summary": {
            "DQN": {"valor médio realizado": 2.1, "IC 95% inferior": 1.0, "IC 95% superior": 3.2},
            "Manter até o vencimento": {"valor médio realizado": 1.5, "IC 95% inferior": 0.2, "IC 95% superior": 2.8},
            "Regra fixa (Black-Scholes)": {"valor médio realizado": 1.9, "IC 95% inferior": 0.8, "IC 95% superior": 3.0},
        },
    }
    fig = plot_exit_evaluation("FAKE", evaluation)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_plot_time_value_curve_returns_figure_and_marks_exit_day():
    plan = _plan(stock_price=100.0, strike=90.0)
    curve = time_value_curve(plan, risk_free_rate=0.05, dividend_yield=0.0, total_trading_days=20)
    recommendation = recommend_exit_day(curve, time_value_floor_pct=0.25)
    fig = plot_time_value_curve(curve, recommendation)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_plot_time_value_curve_without_recommendation_still_works():
    plan = _plan(stock_price=100.0, strike=90.0)
    curve = time_value_curve(plan, risk_free_rate=0.05, dividend_yield=0.0, total_trading_days=10)
    fig = plot_time_value_curve(curve)
    assert isinstance(fig, Figure)
