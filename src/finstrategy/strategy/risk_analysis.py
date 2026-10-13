"""Pacote de "Análise de Risco" combinando Black-Scholes, Monte Carlo e os sinais já
calculados — para qualquer ticker elegível a derivativo (plan.md, seção 11).

Reúne numa função só o que hoje está espalhado: preço teórico da opção
(`pricing.black_scholes`), timing de saída (`strategy.exit_timing_bs` — já existia,
mas só era chamado dentro do treino manual da DQN de saída, nunca no relatório
principal) e leitura da distribuição Monte Carlo que `analyze_ticker` já calculou.
**Não faz nenhuma chamada de rede nova** — tudo vem do `result` que `analyze_ticker`
já devolveu, e da taxa livre de risco já buscada para chegar até ali.

Três noções de "confiança"/"risco" ficam deliberadamente separadas (não existe um
número único de confiança — seria matematicamente enganoso):
1. confiança estatística do sinal da DQN (calculada em `rl/entry_dqn.py`/`rl/exit_dqn.py`,
   fora deste módulo — é sobre o SINAL, não sobre o preço);
2. precisão da simulação Monte Carlo (`mc_precision_pct` abaixo — a largura do
   intervalo de confiança da MÉDIA simulada, nunca "confiança de alta/queda": o
   Monte Carlo aqui é neutro ao risco por construção, não aposta em direção);
3. risco de entrada "naquele momento" (P(perda), ES 97,5% — já calculados em
   `build_strike_menu`, só reunidos aqui pra leitura junto do resto).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config_model import AnalysisSettings
from ..pricing.black_scholes import OptionGreeks, price_and_greeks
from ..pricing.garch_montecarlo import TRADING_DAYS_PER_YEAR
from ..rl.entry_dqn import bootstrap_mean_interval
from .exit_timing_bs import recommend_exit_day, time_value_curve
from .protective_put import ProtectivePutPlan


@dataclass
class RiskAnalysis:
    ticker: str
    plan: ProtectivePutPlan

    # 11.1 — preço teórico (Black-Scholes) vs. prêmio de mercado
    greeks: OptionGreeks
    price_diff_pct: float  # (prêmio pago - preço teórico) / preço teórico; >0 = opção "cara"

    # 11.2 — timing de saída
    time_value_curve_df: pd.DataFrame
    exit_recommendation: dict
    exit_date_estimate: pd.Timestamp

    # 11.3 — tendência do preço e do valor da put no vencimento
    prob_price_up: float
    prob_price_down: float
    expected_price: float
    price_range_90: tuple
    expected_put_payoff_at_expiration: float
    put_payoff_range_90: tuple

    # 11.4 — precisão da simulação Monte Carlo (não é confiança direcional)
    price_mean_ci_95: tuple
    mc_precision_pct: float  # largura do IC da média, em % da média — menor = mais preciso


def build_risk_analysis(
    ticker: str,
    result: dict,
    risk_free_rate: float,
    settings: AnalysisSettings,
    today: pd.Timestamp | None = None,
    seed: int | None = None,
) -> RiskAnalysis:
    """Monta a análise de risco completa a partir do `result` de `analyze_ticker`.

    Levanta `KeyError`/`AttributeError` se `result` não tiver a forma esperada (não
    deveria acontecer vindo de `analyze_ticker`, que já garante os campos) — não
    captura exceções aqui, quem chama decide como lidar com falha (mesmo padrão do
    resto do pipeline, ver `run_full_pipeline` em app.py).
    """
    today = today or pd.Timestamp.today().normalize()
    chosen = result["recommended"]
    plan = ProtectivePutPlan(**{key: chosen[key] for key in ProtectivePutPlan.__dataclass_fields__})
    dividend_yield = result["dividend_yield"]
    simulated_prices = np.asarray(result["simulated_prices"])

    # --- 11.1: preço teórico (Black-Scholes) vs. prêmio de mercado ---
    time_years = plan.days_to_expiration / 365
    volatility = max(plan.implied_volatility, 1e-4)
    greeks = price_and_greeks("put", plan.stock_price, plan.strike, time_years, risk_free_rate, volatility, dividend_yield)
    price_diff_pct = (plan.put_premium - greeks.price) / max(greeks.price, 1e-6)

    # --- 11.2: timing de saída (reaproveita exit_timing_bs, hoje só usado na DQN de saída) ---
    window_days = max(int(round(plan.days_to_expiration * TRADING_DAYS_PER_YEAR / 365)), 1)
    curve = time_value_curve(plan, risk_free_rate, dividend_yield, window_days)
    exit_recommendation = recommend_exit_day(curve, settings.exit_time_value_floor_pct)
    exit_calendar_days = int(round(exit_recommendation["exit_day"] * 365 / TRADING_DAYS_PER_YEAR))
    exit_date_estimate = today + pd.Timedelta(days=exit_calendar_days)

    # --- 11.3: tendência do preço (já simulado) e do valor da put no vencimento ---
    prob_price_up = float((simulated_prices > plan.stock_price).mean())
    prob_price_down = float((simulated_prices < plan.stock_price).mean())
    expected_price = float(simulated_prices.mean())
    price_range_90 = tuple(np.percentile(simulated_prices, [5, 95]))
    put_payoff_at_expiration = np.maximum(plan.strike - simulated_prices, 0.0)
    expected_put_payoff = float(put_payoff_at_expiration.mean())
    put_payoff_range_90 = tuple(np.percentile(put_payoff_at_expiration, [5, 95]))

    # --- 11.4: precisão da simulação Monte Carlo (bootstrap da média) ---
    rng = np.random.default_rng(seed if seed is not None else settings.random_seed)
    price_mean_ci = bootstrap_mean_interval(simulated_prices, rng)
    mc_precision_pct = (price_mean_ci[1] - price_mean_ci[0]) / expected_price if expected_price else float("nan")

    return RiskAnalysis(
        ticker=ticker, plan=plan, greeks=greeks, price_diff_pct=price_diff_pct,
        time_value_curve_df=curve, exit_recommendation=exit_recommendation, exit_date_estimate=exit_date_estimate,
        prob_price_up=prob_price_up, prob_price_down=prob_price_down, expected_price=expected_price,
        price_range_90=price_range_90, expected_put_payoff_at_expiration=expected_put_payoff,
        put_payoff_range_90=put_payoff_range_90, price_mean_ci_95=price_mean_ci, mc_precision_pct=mc_precision_pct,
    )
