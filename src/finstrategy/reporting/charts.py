"""Gráficos de ação e derivativo para o frontend (plan.md, seção 10.1).

Portado do notebook `put_de_protecao_DQN_montecarlos.ipynb` (`plot_strike_menu`,
`plot_protection_payoff`, `plot_entry_evaluation`), com duas mudanças: cada função
devolve uma `matplotlib.figure.Figure` em vez de chamar `plt.show()` (o chamador decide
onde desenhar — `st.pyplot(fig)` no Streamlit), e os parâmetros que o notebook lia de
uma variável global `SETTINGS` agora são argumentos explícitos. A lógica de cálculo
(payoff, perda máxima, etc.) não mudou — só a forma de entregar o gráfico.

Duas visualizações novas, que não existiam no notebook: `plot_time_value_curve`
(decaimento do valor-tempo da put, strategy/exit_timing_bs.py) e
`plot_exit_evaluation` (mesmo estilo de `plot_entry_evaluation`, para a DQN de saída).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from ..config_model import AnalysisSettings
from ..strategy.protective_put import ProtectivePutPlan, profit_per_share_at_expiration, stock_only_profit_per_share

COLORS = {
    "stock": "#2a78d6", "protected": "#eb6834", "accent": "#1baf7a", "warning": "#eda100",
    "ink": "#52514e", "ink_strong": "#0b0b0b", "grid": "#e6e5e1", "surface": "#fcfcfb",
    "loss": "#d03b3b", "muted": "#a3a29c",
}


def _new_figure(figsize: tuple) -> tuple[Figure, "Axes"]:
    figure = Figure(figsize=figsize, facecolor=COLORS["surface"])
    axis = figure.add_subplot(111)
    axis.set_facecolor(COLORS["surface"])
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.grid(True, color=COLORS["grid"])
    axis.tick_params(colors=COLORS["ink"])
    axis.xaxis.label.set_color(COLORS["ink"])
    axis.yaxis.label.set_color(COLORS["ink"])
    axis.title.set_color(COLORS["ink_strong"])
    return figure, axis


def plot_strike_menu(ticker: str, result: dict, max_acceptable_loss_pct: float) -> Figure:
    """Cada ponto é uma strike real da cadeia: quanto custa (x) vs. quanto você pode
    perder no máximo (y). A recomendada vem destacada."""
    menu = result["menu"]
    figure, axis = _new_figure((9, 4.2))
    axis.plot(menu["protection_cost_pct"], menu["max_loss_pct"], "-", color=COLORS["muted"], lw=1)
    axis.plot(menu["protection_cost_pct"], menu["max_loss_pct"], "o", color=COLORS["stock"], ms=7, mec=COLORS["surface"], mew=1.5)
    chosen = result["recommended"]
    axis.plot(chosen["protection_cost_pct"], chosen["max_loss_pct"], "o", color=COLORS["protected"], ms=12, mec=COLORS["surface"], mew=2)
    axis.annotate(
        f"recomendada: strike {chosen['strike']:,.2f}", (chosen["protection_cost_pct"], chosen["max_loss_pct"]),
        xytext=(-10, 10), textcoords="offset points", ha="right", color=COLORS["ink_strong"],
    )
    axis.axhline(max_acceptable_loss_pct, color=COLORS["loss"], lw=1, ls="--")
    axis.annotate(
        f"perda máxima aceitável ({max_acceptable_loss_pct:.0%})", (axis.get_xlim()[1], max_acceptable_loss_pct),
        xytext=(-4, 4), textcoords="offset points", ha="right", color=COLORS["loss"],
    )
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("custo da proteção (prêmio ÷ preço da ação)")
    axis.set_ylabel("perda máxima da posição protegida")
    axis.set_title(f"{ticker}: quanto custa cada nível de proteção (vencimento {result['expiration'].date()})")
    figure.tight_layout()
    return figure


def plot_protection_payoff(ticker: str, result: dict, settings: AnalysisSettings) -> Figure:
    """Resultado da posição no vencimento, com e sem put, e a distribuição de preços
    simulada pelo Monte Carlo — painel de cima é o payoff, o de baixo é a distribuição."""
    chosen = result["recommended"]
    plan = ProtectivePutPlan(**{key: chosen[key] for key in ProtectivePutPlan.__dataclass_fields__})
    prices = result["simulated_prices"]
    low, high = np.percentile(prices, [0.5, 99.5])
    price_grid = np.linspace(min(low, plan.strike * 0.92), high, 500)
    shares = plan.shares
    protected = profit_per_share_at_expiration(plan, price_grid, settings) * shares
    unprotected = stock_only_profit_per_share(plan.stock_price, price_grid, settings) * shares

    figure = Figure(figsize=(9, 6.0), facecolor=COLORS["surface"])
    payoff_axis = figure.add_subplot(2, 1, 1)
    distribution_axis = figure.add_subplot(2, 1, 2, sharex=payoff_axis)
    for axis in (payoff_axis, distribution_axis):
        axis.set_facecolor(COLORS["surface"])
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.tick_params(colors=COLORS["ink"])

    payoff_axis.plot(price_grid, unprotected, color=COLORS["stock"], label="só a ação")
    payoff_axis.plot(price_grid, protected, color=COLORS["protected"], label=f"ação + put {plan.strike:,.2f}")
    payoff_axis.axhline(0, color=COLORS["ink"], lw=1)
    payoff_axis.axhline(-plan.max_loss_total, color=COLORS["loss"], lw=1, ls="--")
    payoff_axis.annotate(
        f"perda máxima {plan.max_loss_total:,.0f}", (price_grid[0], -plan.max_loss_total),
        xytext=(2, -14), textcoords="offset points", color=COLORS["loss"],
    )
    payoff_axis.axvline(plan.strike, color=COLORS["ink"], lw=1, ls="--")
    payoff_axis.annotate(
        f"venda garantida {plan.strike:,.2f}", (plan.strike, payoff_axis.get_ylim()[1]),
        xytext=(4, -14), textcoords="offset points", color=COLORS["ink"],
    )
    payoff_axis.plot(plan.breakeven_price, 0, "o", color=COLORS["protected"], ms=8, mec=COLORS["surface"], mew=2)
    payoff_axis.annotate(
        f"break-even {plan.breakeven_price:,.2f}", (plan.breakeven_price, 0),
        xytext=(6, -16), textcoords="offset points", color=COLORS["ink"],
    )
    payoff_axis.set_ylabel("resultado da posição no vencimento")
    payoff_axis.set_title(f"{ticker}: {shares} ações + {plan.contracts} put(s), vencimento {plan.expiration.date()}")
    payoff_axis.legend(loc="upper left")
    payoff_axis.grid(True, color=COLORS["grid"])

    distribution_axis.hist(prices, bins=120, range=(price_grid[0], price_grid[-1]), color=COLORS["stock"], alpha=0.5)
    distribution_axis.axvline(plan.stock_price, color=COLORS["ink"], lw=1, ls=":")
    distribution_axis.set_yticks([])
    distribution_axis.grid(axis="y", visible=False)
    distribution_axis.set_xlabel("preço da ação no vencimento; abaixo, distribuição simulada (Monte Carlo GARCH-t)")

    figure.tight_layout()
    return figure


def _plot_strategy_comparison_forest(summary: pd.DataFrame, title: str, xlabel: str, highlight_name: str = "DQN") -> Figure:
    """Base compartilhada de `plot_entry_evaluation`/`plot_exit_evaluation`: um ponto +
    intervalo de confiança de 95% por estratégia, ordenadas, com a estratégia destacada
    (DQN) em outra cor — estilo *forest plot*, comum em comparação de métodos."""
    summary = summary.iloc[::-1]
    figure, axis = _new_figure((9, 3.2 + 0.4 * len(summary)))
    for position, (name, row) in enumerate(summary.iterrows()):
        color = COLORS["protected"] if name == highlight_name else COLORS["stock"]
        axis.plot([row["IC 95% inferior"], row["IC 95% superior"]], [position, position], color=color, lw=2)
        axis.plot(row["valor"], position, "o", color=color, ms=8, mec=COLORS["surface"], mew=2)
        axis.annotate(f"{row['valor']:+.2f}", (row["IC 95% superior"], position), xytext=(6, -4), textcoords="offset points", color=COLORS["ink"])
    axis.axvline(0, color=COLORS["ink"], lw=1)
    axis.set_yticks(range(len(summary)), summary.index)
    axis.grid(axis="y", visible=False)
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    figure.tight_layout()
    return figure


def plot_entry_evaluation(ticker: str, evaluation: dict) -> Figure:
    """Economia média de cada estratégia de entrada (DQN vs. 3 referências), com
    intervalo de confiança de 95% — a DQN só é útil se estiver visivelmente à frente."""
    summary = evaluation["summary"].rename(columns={"economia média (%)": "valor"})
    title = f"{ticker}: ponto de entrada no período de teste ({evaluation['test_windows']} janelas desde {evaluation['test_period_start']})"
    xlabel = "economia média por compra vs. comprar no 1º dia (%), com intervalo de confiança de 95%"
    return _plot_strategy_comparison_forest(summary, title, xlabel)


def plot_exit_evaluation(ticker: str, evaluation: dict) -> Figure:
    """Mesmo estilo de `plot_entry_evaluation`, para a DQN de saída: valor médio
    realizado (manter até o vencimento vs. regra fixa de Black-Scholes vs. DQN)."""
    summary = pd.DataFrame(evaluation["summary"]).T.rename(columns={"valor médio realizado": "valor"})
    title = f"{ticker}: timing de saída da put — DQN vs. manter até o vencimento vs. regra fixa"
    xlabel = "valor médio realizado ao sair (por ação), com intervalo de confiança de 95%"
    return _plot_strategy_comparison_forest(summary, title, xlabel)


def plot_time_value_curve(curve: pd.DataFrame, exit_recommendation: dict | None = None) -> Figure:
    """Decaimento do valor-tempo da put ao longo dos dias (Black-Scholes, preço da ação
    fixo) — mostra visualmente por que a regra de saída recomenda sair naquele dia."""
    figure, axis = _new_figure((9, 3.6))
    axis.plot(curve["trading_day"], curve["time_value"], color=COLORS["stock"], lw=2)
    axis.fill_between(curve["trading_day"], curve["time_value"], color=COLORS["stock"], alpha=0.12)
    if exit_recommendation is not None:
        exit_day = exit_recommendation["exit_day"]
        axis.axvline(exit_day, color=COLORS["loss"], lw=1.5, ls="--")
        axis.annotate(
            f"sair no dia {exit_day}", (exit_day, axis.get_ylim()[1]), xytext=(6, -14),
            textcoords="offset points", color=COLORS["loss"],
        )
    axis.set_xlabel("dias de pregão desde a compra da put")
    axis.set_ylabel("valor-tempo restante (por ação)")
    axis.set_title("Quanto da proteção que você pagou ainda resta (valor-tempo)")
    figure.tight_layout()
    return figure
