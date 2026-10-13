"""Janela de saída do derivativo via Black-Scholes (plan.md, seção 3).

Depois de comprada a put de proteção, duas perguntas diferentes, respondidas por dois
sinais complementares:

1. **"Quanto da proteção que eu paguei já foi consumida pelo tempo?"** — resolvida
   repricando a put dia a dia com Black-Scholes, **mantendo o preço da ação fixo no
   preço atual** (isola o efeito puro do tempo: `time_value_curve`). É o sinal usado
   pela regra fixa (`recommend_exit_day`): vender quando o valor-tempo restante cair
   abaixo de `exit_time_value_floor_pct` do prêmio original.
2. **"Dado o que o GARCH Monte Carlo (já calibrado no notebook) projeta para o preço,
   quanto vale, em expectativa, continuar segurando a put até o vencimento?"** —
   resolvida repreçando o payoff esperado da put a cada horizonte remanescente
   (`expected_monte_carlo_value_curve`). É um sinal de validação/estado para a DQN de
   saída (rl/exit_dqn.py), não a regra fixa — porque assume que a volatilidade/drift
   calibrados hoje continuam válidos até o vencimento, o que é uma aproximação (ver
   limitações, seção 11 do notebook original).

As duas curvas são deliberadamente mantidas separadas: a primeira é *qual é o valor
contratual restante da put*, a segunda é *o que o modelo acha que esse valor vale*.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..pricing.black_scholes import price_and_greeks
from ..pricing.garch_montecarlo import TRADING_DAYS_PER_YEAR, GarchParameters, simulate_prices_at_expiration
from .protective_put import ProtectivePutPlan


@dataclass
class ExitTimingPoint:
    trading_day: int
    days_remaining: int
    bs_price: float
    intrinsic_value: float
    time_value: float
    delta: float
    theta_per_day: float
    vega: float
    expected_mc_value: float


def time_value_curve(
    plan: ProtectivePutPlan,
    risk_free_rate: float,
    dividend_yield: float,
    total_trading_days: int,
) -> pd.DataFrame:
    """Curva de valor-tempo da put, repreçando BS dia a dia com o preço da ação FIXO
    no preço de compra (isola o efeito do tempo passando, sem ruído de preço)."""
    volatility = max(plan.implied_volatility, 1e-4)
    rows = []
    for day in range(total_trading_days + 1):
        days_remaining = total_trading_days - day
        time_years = days_remaining / TRADING_DAYS_PER_YEAR
        greeks = price_and_greeks(
            "put", plan.stock_price, plan.strike, time_years, risk_free_rate, volatility, dividend_yield
        )
        intrinsic_value = max(plan.strike - plan.stock_price, 0.0)
        rows.append({
            "trading_day": day,
            "days_remaining": days_remaining,
            "bs_price": greeks.price,
            "intrinsic_value": intrinsic_value,
            "time_value": max(greeks.price - intrinsic_value, 0.0),
            "delta": greeks.delta,
            "theta_per_day": greeks.theta / TRADING_DAYS_PER_YEAR,
            "vega": greeks.vega,
        })
    return pd.DataFrame(rows)


def expected_monte_carlo_value_curve(
    plan: ProtectivePutPlan,
    garch: GarchParameters,
    risk_free_rate: float,
    dividend_yield: float,
    total_trading_days: int,
    monte_carlo_paths: int,
    seed: int,
    step_days: int = 5,
) -> pd.DataFrame:
    """Valor esperado (descontado) de continuar segurando a put, a cada horizonte
    remanescente, via GARCH Monte Carlo — reaproveita `simulate_prices_at_expiration`
    já validada no notebook. `step_days` evita recalcular Monte Carlo para cada dia
    (caro); interpolar entre pontos é suficiente para a curva ser lida pela DQN."""
    rng = np.random.default_rng(seed)
    rows = []
    for days_remaining in range(total_trading_days, -1, -step_days):
        if days_remaining == 0:
            expected_value = max(plan.strike - plan.stock_price, 0.0)
        else:
            simulated_prices = simulate_prices_at_expiration(
                garch, plan.stock_price, days_remaining, risk_free_rate - dividend_yield, monte_carlo_paths, rng
            )
            expected_payoff = np.maximum(plan.strike - simulated_prices, 0.0).mean()
            expected_value = float(expected_payoff * np.exp(-risk_free_rate * days_remaining / TRADING_DAYS_PER_YEAR))
        rows.append({
            "trading_day": total_trading_days - days_remaining,
            "days_remaining": days_remaining,
            "expected_mc_value": expected_value,
        })
    return pd.DataFrame(rows).sort_values("trading_day").reset_index(drop=True)


def recommend_exit_day(curve: pd.DataFrame, time_value_floor_pct: float) -> dict:
    """Regra fixa: sair no primeiro dia em que o valor-tempo restante cai abaixo de
    `time_value_floor_pct` do valor-tempo inicial (dia 0). Serve de benchmark para a
    DQN de saída (rl/exit_dqn.py) — a DQN precisa bater esta regra, não só "não perder
    dinheiro", para ser considerada útil (mesmo padrão de avaliação do DQN de entrada)."""
    initial_time_value = float(curve.iloc[0]["time_value"])
    if initial_time_value <= 0:
        return {"exit_day": int(curve.iloc[0]["trading_day"]), "reason": "put já sem valor-tempo no dia 0", "initial_time_value": 0.0}

    threshold = initial_time_value * time_value_floor_pct
    below_threshold = curve[curve["time_value"] <= threshold]
    if below_threshold.empty:
        last_row = curve.iloc[-1]
        return {
            "exit_day": int(last_row["trading_day"]),
            "reason": "valor-tempo nunca caiu abaixo do piso antes do vencimento; sair no vencimento",
            "initial_time_value": initial_time_value,
        }
    first_hit = below_threshold.iloc[0]
    return {
        "exit_day": int(first_hit["trading_day"]),
        "reason": f"valor-tempo caiu para {first_hit['time_value']:.4f} (piso: {threshold:.4f})",
        "initial_time_value": initial_time_value,
    }
