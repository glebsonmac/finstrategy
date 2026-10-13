"""Painel final: uma linha por ativo, com o que o usuário precisa para decidir.

Extraído/generalizado da seção 10 do notebook `put_de_protecao_DQN_montecarlos.ipynb`.
Separado em duas camadas: `build_panel_dataframe` (tabela técnica, para a tela
"Histórico" do frontend) e `build_plain_language_summary` (texto direto, sem jargão,
para o PDF do usuário não técnico — seção 7 do plano).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def describe_entry_signal(evaluation: dict | None) -> str:
    if evaluation is None:
        return "sem avaliação de ponto de entrada"
    if not evaluation["beats_benchmarks"]:
        return f"{evaluation['today_action']} (sinal fraco: a DQN não bateu as referências no teste)"
    if evaluation["today_action"] == "esperar":
        return f"esperar (ganho estimado ~{evaluation['today_expected_gain_from_waiting_pct']:.2f}%)"
    return "comprar hoje"


def build_panel_dataframe(results: dict, not_eligible: dict, entry_evaluations: dict) -> pd.DataFrame:
    """`results`: {ticker: dict devolvido por strategy.protective_put.analyze_ticker}.
    `not_eligible`: {ticker: motivo} — tickers sem derivativo elegível (ou qualquer
    outra falha de dados). `entry_evaluations`: {ticker: dict de rl.entry_dqn.evaluate_entry_timing}.
    """
    rows = []
    for ticker, result in results.items():
        chosen = result["recommended"]
        evaluation = entry_evaluations.get(ticker)
        rows.append({
            "ativo": ticker,
            "entrada (DQN)": describe_entry_signal(evaluation),
            "comprar": f"{chosen['shares']} ações a ~{chosen['stock_price']:,.2f} + {chosen['contracts']} put(s) {chosen['contract_symbol']}",
            "strike / vencimento": f"{chosen['strike']:,.2f} / {chosen['expiration'].date()}",
            "prêmio (preço usado)": f"{chosen['put_premium']:,.2f} ({chosen['quote_source']})",
            "custo total": f"{chosen['total_cost']:,.0f}",
            "venda garantida": f"{chosen['guaranteed_sale_value']:,.0f}",
            "perda máxima": f"{chosen['max_loss_total']:,.0f} ({chosen['max_loss_pct']:.1%})",
            "lucro acima de": f"{chosen['breakeven_price']:,.2f} ({chosen['breakeven_change_pct']:+.1%})",
            "ES 97,5% sem → com put": f"{chosen['ES 97,5% (sem put)']:.1%} → {chosen['ES 97,5% (com put)']:.1%}",
            "P(perda) com put": f"{chosen['P(perda) (com put)']:.0%}",
        })
    for ticker, reason in not_eligible.items():
        evaluation = entry_evaluations.get(ticker)
        rows.append({
            "ativo": ticker, "entrada (DQN)": describe_entry_signal(evaluation),
            "comprar": f"não elegível a derivativo: {reason}" if evaluation is None else f"ação apenas (sem derivativo): {reason}",
        })
    return pd.DataFrame(rows).set_index("ativo").fillna("") if rows else pd.DataFrame()


@dataclass
class PlainLanguageSummary:
    ticker: str
    headline: str
    body_lines: list
    is_eligible_for_protection: bool


def build_plain_language_summary(ticker: str, result: dict | None, evaluation: dict | None, not_eligible_reason: str | None = None) -> PlainLanguageSummary:
    """Texto direto, sem jargão, para quem não é técnico — base do PDF (seção 7)."""
    entry_text = describe_entry_signal(evaluation)
    if result is None:
        headline = f"{ticker}: sem proteção com opção disponível agora"
        body = [
            f"Esta ação não tem opção de venda (put) disponível no Yahoo Finance no momento "
            f"({not_eligible_reason or 'motivo não informado'}).",
            f"Sinal de entrada (apenas da ação, sem proteção): {entry_text}.",
            "Sem a opção de proteção, o risco de queda não tem um limite garantido — "
            "avalie com mais cautela antes de investir.",
        ]
        return PlainLanguageSummary(ticker=ticker, headline=headline, body_lines=body, is_eligible_for_protection=False)

    chosen = result["recommended"]
    headline = f"{ticker}: comprar a ação + proteção (put) com strike {chosen['strike']:,.2f}"
    body = [
        f"Sinal de entrada: {entry_text}.",
        f"Comprando {chosen['shares']} ações e {chosen['contracts']} opção(ões) de proteção, "
        f"você garante vender a posição por pelo menos {chosen['guaranteed_sale_value']:,.0f} "
        f"até {chosen['expiration'].date()}.",
        f"A perda máxima possível é {chosen['max_loss_total']:,.0f} ({chosen['max_loss_pct']:.1%} do que foi "
        f"investido), mesmo que a ação despenque.",
        f"Para começar a ter lucro, a ação precisa estar acima de {chosen['breakeven_price']:,.2f} "
        f"no vencimento ({chosen['breakeven_change_pct']:+.1%} em relação a hoje).",
    ]
    if not result.get("recommended_respects_limit", True):
        body.append("Atenção: nenhuma opção de proteção disponível ficou dentro do limite de perda desejado; "
                     "esta é a de menor perda possível entre as disponíveis.")
    return PlainLanguageSummary(ticker=ticker, headline=headline, body_lines=body, is_eligible_for_protection=True)
