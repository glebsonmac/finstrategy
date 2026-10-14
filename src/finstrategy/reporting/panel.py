"""Painel final: uma linha por ativo, com o que o usuário precisa para decidir.

Extraído/generalizado da seção 10 do notebook `put_de_protecao_DQN_montecarlos.ipynb`.
Separado em camadas: `build_panel_dataframe` (tabela técnica, para a tela "Histórico"
do frontend), `build_plain_language_summary` (texto direto, sem jargão, para o PDF do
usuário não técnico — seção 7 do plano), e o restante do conteúdo das seções 8 e 9 do
notebook (menu completo de strikes, avaliação da DQN de entrada) — que o relatório do
Streamlit/PDF só mostrava parcialmente (só o gráfico, sem a tabela/texto por trás) e
agora replica integralmente. A seção 11 do notebook (limitações/referências) fica só
no notebook, por decisão do usuário — não é conteúdo do relatório do app/PDF.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Seção 8 do notebook: tabela com todas as strikes disponíveis, não só a recomendada.
MENU_COLUMNS = {
    "strike": "strike (venda garantida/ação)", "put_premium": "prêmio da put", "quote_source": "preço usado",
    "guaranteed_sale_value": "venda garantida (posição)", "total_cost": "custo total", "max_loss_total": "perda máx.",
    "max_loss_pct": "perda máx. %", "breakeven_price": "break-even", "breakeven_change_pct": "break-even vs. hoje",
    "protection_cost_pct": "custo da proteção", "annualized_protection_cost_pct": "custo anualizado",
    "P(exercer a put)": "P(exercer)", "P(perda) (com put)": "P(perda) com put",
    "ES 97,5% (sem put)": "ES sem put", "ES 97,5% (com put)": "ES com put",
    "custo líquido esperado %": "custo líq. esperado",
}
MENU_FORMATS = {
    "strike (venda garantida/ação)": "{:,.2f}", "prêmio da put": "{:,.2f}", "venda garantida (posição)": "{:,.0f}",
    "custo total": "{:,.0f}", "perda máx.": "{:,.0f}", "perda máx. %": "{:.1%}", "break-even": "{:,.2f}",
    "break-even vs. hoje": "{:+.1%}", "custo da proteção": "{:.2%}", "custo anualizado": "{:.1%}",
    "P(exercer)": "{:.0%}", "P(perda) com put": "{:.0%}", "ES sem put": "{:.1%}", "ES com put": "{:.1%}",
    "custo líq. esperado": "{:+.2%}",
}

# Seção 9/10 do notebook: tabela de avaliação da DQN de entrada (4 estratégias comparadas).
ENTRY_SUMMARY_FORMATS = {
    "economia média (%)": "{:+.2f}", "IC 95% inferior": "{:+.2f}",
    "IC 95% superior": "{:+.2f}", "% janelas melhores que o 1º dia": "{:.0%}",
}


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


def build_strike_menu_header(ticker: str, result: dict) -> str:
    """Linha de contexto acima da tabela de strikes — notebook, seção 8."""
    chosen = result["recommended"]
    contracts = chosen["contracts"]
    return (
        f"{ticker} · preço {result['stock_price']:,.2f} · {result['shares']} ações "
        f"({contracts} {'contrato' if contracts == 1 else 'contratos'}) · vencimento "
        f"{result['expiration'].date()} · vol. atual (GARCH) {result['garch'].current_annual_volatility:.1%}"
    )


def build_strike_menu_table(result: dict) -> pd.DataFrame:
    """Tabela com todas as strikes disponíveis (não só a recomendada) — notebook, seção 8."""
    menu = result["menu"]
    recommended_strike = result["recommended"]["strike"]
    table = menu[[column for column in MENU_COLUMNS if column in menu.columns]].rename(columns=MENU_COLUMNS)
    table.index = ["► recomendada" if strike == recommended_strike else "" for strike in menu["strike"]]
    for column, fmt in MENU_FORMATS.items():
        if column in table.columns:
            table[column] = table[column].map(fmt.format)
    return table


def build_entry_evaluation_table(evaluation: dict) -> pd.DataFrame:
    """Tabela com as 4 estratégias de entrada comparadas (DQN vs. benchmarks) — notebook, seção 9."""
    table = evaluation["summary"].copy()
    for column, fmt in ENTRY_SUMMARY_FORMATS.items():
        if column in table.columns:
            table[column] = table[column].map(fmt.format)
    return table


def build_entry_evaluation_verdict(evaluation: dict) -> str:
    """Parágrafo com o veredito estatístico da DQN de entrada — notebook, seção 9."""
    verdict = "**bateu** as referências com significância" if evaluation["beats_benchmarks"] else "**não bateu** as referências com significância"
    return (
        f"A DQN {verdict} (p-valor vs. 1º dia = {evaluation['p_value_vs_first_day']:.2f}; vs. preço médio = "
        f"{evaluation['p_value_vs_average']:.2f}). Em média comprou no dia {evaluation['average_purchase_day']:.1f} "
        f"da janela. Sinal de hoje: **{evaluation['today_action']}**."
    )


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
