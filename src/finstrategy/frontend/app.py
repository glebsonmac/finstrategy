"""Frontend simples em Streamlit (plan.md, seção 8).

Rodar com: `streamlit run src/finstrategy/frontend/app.py` (de dentro do venv do
projeto). Todas as variáveis do pipeline (watchlist, `AnalysisSettings`, coletor) são
editadas aqui, sem precisar tocar em código — é o requisito central da seção 8.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st

import sys

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from finstrategy.collector import csv_store
from finstrategy.collector.market_session import any_market_open, exchange_calendar_for_ticker
from finstrategy.collector.overnight import US_TOP20_SEED_PATH
from finstrategy.collector.scheduler import CollectorConfig, read_status, run_forever
from finstrategy.config_model import AnalysisSettings, DEFAULT_SETTINGS_PATH, load_settings, save_settings
from finstrategy.data import local_stock_list, symbol_search, watchlist as watchlist_module
from finstrategy.data.errors import MarketDataError
from finstrategy.data.watchlist import DEFAULT_DATA_STORE_ROOT, DEFAULT_WATCHLIST_PATH
from finstrategy.data.yahoo_provider import YahooProvider
from finstrategy.pricing.garch_montecarlo import TRADING_DAYS_PER_YEAR
from finstrategy.reporting import charts, model_store
from finstrategy.reporting.panel import build_panel_dataframe, build_plain_language_summary
from finstrategy.reporting.pdf_report import generate_pdf_report
from finstrategy.rl.entry_dqn import evaluate_entry_timing
from finstrategy.rl.exit_dqn import evaluate_exit_timing
from finstrategy.strategy.exit_timing_bs import recommend_exit_day, time_value_curve
from finstrategy.strategy.protective_put import ProtectivePutPlan, analyze_ticker
from finstrategy.strategy.risk_analysis import RiskAnalysis, build_risk_analysis

RISK_METRIC_HELP = {
    "perda máxima": "Quanto você pode perder no pior caso, já incluindo taxas — é o valor final, contratual (não depende de modelo).",
    "ES 97,5% sem → com put": "Perda média nos 2,5% piores cenários simulados: sem proteção → com a put. Quanto menor o segundo número, melhor a proteção.",
    "P(perda) com put": "Em quantos % dos cenários simulados (Monte Carlo) você ainda termina no prejuízo, mesmo com a proteção.",
    "entrada (DQN)": "O que o modelo de IA recomenda hoje: comprar agora ou esperar por um preço melhor nos próximos dias úteis.",
}

st.set_page_config(page_title="finstrategy", layout="wide")

PROVIDER = YahooProvider()

FUNDAMENTALS_LABELS = {
    "ticker": "Ticker", "name": "Nome", "sector": "Setor", "industry": "Indústria",
    "pe_ratio": "P/L", "forward_pe": "P/L projetado", "price_to_book": "P/VP", "roe": "ROE",
    "debt_to_equity": "Dívida/Patrimônio", "dividend_yield_ttm": "Dividend yield (12m)",
    "revenue_growth_yoy": "Crescimento de receita (a/a)", "free_cash_flow": "Fluxo de caixa livre",
    "market_cap": "Valor de mercado", "profit_margin": "Margem líquida", "collected_at": "Coletado em",
}
FUNDAMENTALS_PERCENT_FIELDS = {"roe", "dividend_yield_ttm", "revenue_growth_yoy", "profit_margin"}
FUNDAMENTALS_BIG_NUMBER_FIELDS = {"market_cap", "free_cash_flow"}


def _init_state():
    st.session_state.setdefault("collector_thread", None)
    st.session_state.setdefault("collector_stop_event", None)
    st.session_state.setdefault("collector_started_at", None)


_init_state()

PAGE = st.sidebar.radio(
    "Navegação",
    ["Ações", "Configuração", "Histórico & Relatórios", "Coletor", "Modelos"],
)


def _load_watchlist_df() -> pd.DataFrame:
    entries = watchlist_module.load_watchlist()
    if not entries:
        return pd.DataFrame(columns=["symbol", "name", "tags", "elegivel_derivativo", "last_price", "last_updated"])
    return pd.DataFrame([entry.to_dict() for entry in entries])


def _format_watchlist_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    table = df.rename(columns={
        "symbol": "Ticker", "name": "Nome", "tags": "Tags", "currency": "Moeda", "exchange": "Bolsa",
        "elegivel_derivativo": "Tem opção (put)?", "last_price": "Último preço", "added_at": "Adicionado em",
        "last_updated": "Atualizado em",
    })
    if "Tem opção (put)?" in table:
        table["Tem opção (put)?"] = table["Tem opção (put)?"].map({True: "Sim", False: "Não"})
    if "Último preço" in table:
        table["Último preço"] = table.apply(
            lambda row: f"{row['Último preço']:,.2f} {row.get('Moeda', '')}".strip() if pd.notna(row["Último preço"]) else "—", axis=1
        )
    return table


def _format_big_number(value: float) -> str:
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    for threshold, suffix in ((1e12, "tri"), (1e9, "bi"), (1e6, "mi"), (1e3, "mil")):
        if abs(value) >= threshold:
            return f"{value / threshold:,.2f} {suffix}"
    return f"{value:,.2f}"


def _format_fundamentals_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    table = df.copy()
    for field in FUNDAMENTALS_PERCENT_FIELDS:
        if field in table:
            table[field] = table[field].apply(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    for field in FUNDAMENTALS_BIG_NUMBER_FIELDS:
        if field in table:
            table[field] = table[field].apply(_format_big_number)
    for field in ("pe_ratio", "forward_pe", "price_to_book", "debt_to_equity"):
        if field in table:
            table[field] = table[field].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    return table.rename(columns=FUNDAMENTALS_LABELS)


def _format_quotes_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    table = df.rename(columns={"ticker": "Ticker", "collected_at": "Coletado em", "price": "Preço", "status": "Status"})
    if "Preço" in table:
        table["Preço"] = table["Preço"].apply(lambda v: f"{v:,.2f}" if pd.notna(v) else "—")
    return table.sort_values("Coletado em", ascending=False) if "Coletado em" in table else table


def run_full_pipeline(tickers: list[str], settings: AnalysisSettings, progress_label: str = "") -> tuple[dict, dict, dict, float]:
    """Roda DQN de entrada + put de proteção (quando elegível) para uma lista de
    tickers. Chamado pela tela "Histórico & Relatórios" (botão "Gerar novo relatório").
    Devolve `risk_free_rate` junto (já buscada aqui) pra `render_risk_analysis_section`
    não precisar buscar de novo."""
    results, not_eligible, entry_evaluations = {}, {}, {}
    try:
        risk_free_rate = PROVIDER.get_risk_free_rate()
    except MarketDataError as reason:
        st.error(f"Não foi possível obter a taxa livre de risco: {reason}")
        risk_free_rate = 0.05
    if not tickers:
        return results, not_eligible, entry_evaluations, risk_free_rate

    progress = st.progress(0.0, text=progress_label or "Iniciando...")
    for index, ticker in enumerate(tickers):
        progress.progress(index / len(tickers), text=f"Analisando {ticker}...")
        try:
            closes = PROVIDER.download_price_history(ticker, settings.history_start_date)
            entry_evaluations[ticker] = evaluate_entry_timing(ticker, closes, settings)
        except Exception as unexpected:
            st.warning(f"{ticker}: DQN de entrada falhou ({unexpected})")

        try:
            results[ticker] = analyze_ticker(ticker, PROVIDER, risk_free_rate, settings)
        except MarketDataError as reason:
            not_eligible[ticker] = str(reason)
        except Exception as unexpected:
            not_eligible[ticker] = f"erro inesperado: {unexpected}"
        progress.progress((index + 1) / len(tickers), text=f"{ticker} concluído")
    progress.empty()
    return results, not_eligible, entry_evaluations, risk_free_rate


def render_risk_analysis_section(ticker: str, result: dict, entry_evaluation: dict | None, risk_free_rate: float, settings: AnalysisSettings) -> None:
    """Seção "📊 Análise de Risco" (plan.md, seção 11): Black-Scholes explica o preço
    da opção e diz quantos dias até considerar vender; a distribuição Monte Carlo já
    calculada vira leitura de tendência; e as três noções de "confiança" ficam
    deliberadamente separadas (nunca um número único genérico — ver `risk_analysis.py`)."""
    analysis = build_risk_analysis(ticker, result, risk_free_rate, settings)
    plan = analysis.plan

    st.markdown("##### 📊 Análise de Risco")

    expensive_or_cheap = "mais cara" if analysis.price_diff_pct > 0 else "mais barata"
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Perda máxima com proteção", f"{plan.max_loss_pct:.1%}",
        help="O pior resultado possível, mesmo se a ação for a zero — já inclui taxas; é contratual, não depende de modelo.",
    )
    col2.metric(
        "Prêmio pago vs. teórico (Black-Scholes)", f"{analysis.price_diff_pct:+.1%}",
        help=(
            f"Preço teórico (Black-Scholes, com a volatilidade implícita do mercado): {analysis.greeks.price:,.2f}. "
            f"Prêmio realmente pago: {plan.put_premium:,.2f}. Positivo = você pagou mais do que o modelo acha que "
            f"a opção vale ({expensive_or_cheap})."
        ),
    )
    col3.metric(
        "Considere vender até", analysis.exit_date_estimate.strftime("%d/%m/%Y"),
        help=f"{analysis.exit_recommendation['reason']} — regra fixa de Black-Scholes, dia {analysis.exit_recommendation['exit_day']} de pregão a partir de hoje.",
    )
    if entry_evaluation is not None:
        confidence_text = "Bateu benchmarks" if entry_evaluation.get("beats_benchmarks") else "Sinal fraco"
        p_value = entry_evaluation.get("p_value_vs_first_day", float("nan"))
        col4.metric(
            "Confiança do sinal de entrada", confidence_text,
            help=(
                f"p-valor vs. comprar no 1º dia: {p_value:.3f} (abaixo de 0,05 = significativo). "
                "É a confiança ESTATÍSTICA do sinal da DQN — não é garantia de preço."
            ),
        )
    else:
        col4.metric("Confiança do sinal de entrada", "—", help="DQN de entrada não avaliada para este ativo nesta rodada.")

    trend_col1, trend_col2 = st.columns(2)
    with trend_col1:
        st.write("**Tendência simulada do preço da ação**")
        st.write(
            f"P(acima de hoje): {analysis.prob_price_up:.0%} · P(abaixo de hoje): {analysis.prob_price_down:.0%}  \n"
            f"Preço médio esperado no vencimento: {analysis.expected_price:,.2f} "
            f"(faixa de 90%: {analysis.price_range_90[0]:,.2f} a {analysis.price_range_90[1]:,.2f})"
        )
    with trend_col2:
        st.write("**Tendência simulada do valor da put (se mantida até o vencimento)**")
        st.write(
            f"Valor esperado: {analysis.expected_put_payoff_at_expiration:,.2f}  \n"
            f"Faixa de 90%: {analysis.put_payoff_range_90[0]:,.2f} a {analysis.put_payoff_range_90[1]:,.2f}"
        )
    st.caption(
        "Simulação Monte Carlo calibrada no histórico real (GARCH-t) — não é garantia, é uma distribuição de "
        "cenários possíveis. O Monte Carlo é neutro ao risco por construção, não aposta em direção."
    )

    st.pyplot(charts.plot_time_value_curve(analysis.time_value_curve_df, analysis.exit_recommendation), clear_figure=True)
    st.caption(
        "Quanto do valor da proteção que você pagou ainda resta, dia a dia (Black-Scholes, preço da ação fixo no "
        "atual, isola o efeito do tempo passando). A linha vermelha marca o dia recomendado para considerar vender."
    )

    with st.expander("🔧 Detalhes técnicos da análise de risco"):
        st.write("**Greeks da strike recomendada (Black-Scholes)**")
        st.json({
            "preço teórico": round(analysis.greeks.price, 4), "delta": round(analysis.greeks.delta, 4),
            "theta (por dia)": round(analysis.greeks.theta / 365, 4), "vega (por 1 p.p. de vol.)": round(analysis.greeks.vega / 100, 4),
            "gamma": round(analysis.greeks.gamma, 6), "volatilidade implícita usada": round(plan.implied_volatility, 4),
        })
        st.caption(
            "delta: quanto a put ganha de valor pra cada R$1 que a ação cai. theta: quanto a put perde de valor "
            "por dia só de o tempo passar. vega: quanto a put ganha/perde se a volatilidade do mercado mudar 1 p.p."
        )
        st.write("**Precisão da simulação Monte Carlo** (não é confiança direcional)")
        st.write(
            f"Intervalo de confiança de 95% do preço médio simulado: {analysis.price_mean_ci_95[0]:,.2f} a "
            f"{analysis.price_mean_ci_95[1]:,.2f} (largura = {analysis.mc_precision_pct:.1%} da média)."
        )
        st.caption(
            "Isso mede o quão preciso é o CÁLCULO da simulação (mais caminhos simulados = intervalo mais estreito), "
            "não se o preço vai subir ou descer."
        )


def render_pipeline_results(tickers: list[str], results: dict, not_eligible: dict, entry_evaluations: dict, settings: AnalysisSettings, risk_free_rate: float = 0.05) -> list:
    st.subheader("Painel técnico")
    panel = build_panel_dataframe(results, not_eligible, entry_evaluations)
    column_config = {col: st.column_config.Column(help=text) for col, text in RISK_METRIC_HELP.items() if col in panel.columns}
    st.dataframe(panel, width="stretch", column_config=column_config)
    if column_config:
        st.caption("💡 Passe o mouse sobre o nome de uma coluna técnica para ver o que ela significa.")

    summaries = [
        build_plain_language_summary(ticker, results.get(ticker), entry_evaluations.get(ticker), not_eligible.get(ticker))
        for ticker in tickers
    ]
    st.subheader("Resumo em linguagem simples")
    for summary in summaries:
        st.markdown(f"**{summary.headline}**")
        for line in summary.body_lines:
            st.write(f"- {line}")

    chartable_tickers = [t for t in tickers if results.get(t) is not None or entry_evaluations.get(t) is not None]
    if chartable_tickers:
        st.subheader("Gráficos")
        for ticker in chartable_tickers:
            result = results.get(ticker)
            evaluation = entry_evaluations.get(ticker)
            with st.expander(f"📊 {ticker}", expanded=(len(chartable_tickers) == 1)):
                if result is not None:
                    try:
                        render_risk_analysis_section(ticker, result, evaluation, risk_free_rate, settings)
                        st.divider()
                    except Exception as unexpected:
                        st.warning(f"Análise de risco indisponível: {unexpected}")
                    try:
                        st.pyplot(charts.plot_strike_menu(ticker, result, settings.max_acceptable_loss_pct), clear_figure=True)
                        st.caption(
                            "Cada ponto é uma strike real disponível no Yahoo. Mais à direita = proteção mais cara; "
                            "mais pra cima = mais risco mesmo com a put. A bolinha laranja é a recomendada."
                        )
                        st.pyplot(charts.plot_protection_payoff(ticker, result, settings), clear_figure=True)
                        st.caption(
                            "Em cima: quanto você ganha/perde no vencimento, com e sem a put, para cada preço possível da ação. "
                            "Embaixo: a distribuição de preços simulada (de onde vêm as métricas de risco da tabela acima)."
                        )
                    except Exception as unexpected:
                        st.warning(f"Gráfico de proteção indisponível: {unexpected}")
                if evaluation is not None:
                    try:
                        st.pyplot(charts.plot_entry_evaluation(ticker, evaluation), clear_figure=True)
                        st.caption(
                            "Economia média de cada estratégia de entrada, testada num pedaço dos dados que o modelo nunca viu. "
                            "Se a barra laranja (DQN) não estiver claramente à direita das outras, trate o sinal com cautela."
                        )
                    except Exception as unexpected:
                        st.warning(f"Gráfico da DQN indisponível: {unexpected}")
    return summaries


def _model_card_summary_row(summary, name: str) -> dict:
    if isinstance(summary, pd.DataFrame):
        return summary.loc[name].to_dict()
    return dict(summary[name])


def render_model_card(ticker: str, model_type: str, evaluation: dict, hyperparameters: dict, trained_at: str) -> None:
    """Explica o modelo pra dois públicos ao mesmo tempo (plan.md, seção 10.3):
    parágrafo simples visível de cara, números crus atrás de um expander técnico."""
    is_entry = model_type == "entry"
    dqn_row = _model_card_summary_row(evaluation["summary"], "DQN")
    beats_benchmarks = bool(evaluation.get("beats_benchmarks", False))

    if is_entry:
        value = dqn_row["economia média (%)"]
        value_label, value_str = "Economia média vs. comprar no 1º dia", f"{value:+.2f}%"
        ci_str = f"{dqn_row['IC 95% inferior']:+.2f}% a {dqn_row['IC 95% superior']:+.2f}%"
        sinal = evaluation.get("today_action", "—")
    else:
        value = dqn_row["valor médio realizado"]
        value_label, value_str = "Valor médio realizado ao sair", f"{value:,.2f}"
        ci_str = f"{dqn_row['IC 95% inferior']:,.2f} a {dqn_row['IC 95% superior']:,.2f}"
        sinal = "—"

    verdict = "bateu as referências com significância estatística" if beats_benchmarks else "**não** bateu as referências com significância — trate o sinal com cautela"
    plain_text = (
        f"Esse modelo foi avaliado num pedaço dos dados que ele nunca viu durante o treino (treino e teste "
        f"separados, do jeito certo). Em média, {value_label.lower()} foi de **{value_str}** "
        f"(intervalo de confiança de 95%: {ci_str}). Isso significa que o modelo {verdict}."
    )
    if is_entry:
        plain_text += f" Sinal de hoje: **{sinal}**."

    st.markdown(f"#### {ticker} — DQN de {'entrada (quando comprar a ação)' if is_entry else 'saída (quando vender a put)'}")
    st.caption(f"Treinado em {str(trained_at)[:19].replace('T', ' ')}")
    st.info(plain_text)

    metric_cols = st.columns(3) if is_entry else st.columns(2)
    metric_cols[0].metric(value_label, value_str)
    metric_cols[1].metric("Bateu os benchmarks?", "Sim ✅" if beats_benchmarks else "Não ⚠️")
    if is_entry:
        metric_cols[2].metric("Sinal de hoje", sinal)

    with st.expander("🔧 Detalhes técnicos"):
        st.write("**Métricas completas** (todas as estratégias comparadas, com intervalo de confiança de 95%)")
        summary_df = evaluation["summary"] if isinstance(evaluation["summary"], pd.DataFrame) else pd.DataFrame(evaluation["summary"]).T
        st.dataframe(summary_df, width="stretch")
        p_value_keys = [k for k in evaluation if k.startswith("p_value")]
        if p_value_keys:
            st.write("**P-valores** (abaixo de 0,05 = diferença estatisticamente significativa)")
            st.json({k: float(evaluation[k]) for k in p_value_keys})
        st.write("**Hiperparâmetros usados no treino**")
        relevant_keys = (
            ["dqn_training_rounds", "dqn_hidden_units", "entry_window_days", "test_period_fraction", "monte_carlo_paths", "random_seed"]
            if is_entry else
            ["exit_dqn_training_rounds", "exit_dqn_hidden_units", "exit_time_value_floor_pct", "monte_carlo_paths", "random_seed"]
        )
        st.json({key: hyperparameters.get(key) for key in relevant_keys})
        st.caption(
            "Explicação técnica: a rede é um MLP de 2 camadas (Double DQN), treinada só com dados antigos e "
            "avaliada num período que nunca viu. 'Bateu os benchmarks' exige economia/valor médio positivo E "
            "p-valor < 0,05 contra as referências — não é só 'deu lucro uma vez'."
        )


def generate_and_offer_pdf(summaries: list, key_suffix: str = "") -> None:
    pdf_path = csv_store.reports_csv_path(DEFAULT_DATA_STORE_ROOT, pd.Timestamp.today()).with_suffix(".pdf")
    generate_pdf_report(summaries, pdf_path)
    with open(pdf_path, "rb") as handle:
        st.download_button(
            "Baixar relatório em PDF", handle.read(), file_name=pdf_path.name, mime="application/pdf",
            key=f"pdf_download_{key_suffix}",
        )
    st.caption(f"Salvo em `{pdf_path}` — também listado na aba Histórico & Relatórios.")


# --------------------------------------------------------------------------------------
# Tela "Ações" — watchlist dinâmica (busca + checkbox, texto livre, tabela atual)
# --------------------------------------------------------------------------------------
if PAGE == "Ações":
    st.title("Ações monitoradas")
    st.caption(
        "A watchlist não é uma lista fixa no código — qualquer ação do Yahoo Finance pode "
        "entrar aqui, uma ou várias de uma vez. O setor elétrico é só o ponto de partida."
    )

    if not DEFAULT_WATCHLIST_PATH.exists():
        st.caption("Watchlist vazia — popule com uma semente pronta ou use as abas abaixo para escolher você mesmo.")
        seed_col1, seed_col2 = st.columns(2)
        if seed_col1.button("Popular com as 20 maiores ações dos EUA"):
            with st.spinner("Buscando dados das 20 maiores ações dos EUA..."):
                added = watchlist_module.ensure_seeded(seed_path=US_TOP20_SEED_PATH)
            st.success(f"{len(added)} tickers adicionados.")
            st.rerun()
        if seed_col2.button("Popular com a semente do setor elétrico"):
            with st.spinner("Buscando dados dos tickers semente..."):
                added = watchlist_module.ensure_seeded()
            st.success(f"{len(added)} tickers adicionados.")
            st.rerun()

    tab_list, tab_search, tab_paste = st.tabs(["📋 Escolher da lista", "🔎 Buscar no Yahoo", "✍️ Colar tickers"])

    with tab_list:
        st.caption(
            "Lista local com as ~500 ações do S&P 500 (ticker, nome, setor) — filtra na hora, "
            "sem bater na internet. Para uma ação que não está aqui, use a aba 'Buscar no Yahoo'."
        )
        list_query = st.text_input("Filtrar por ticker ou nome (ex.: 'duke', 'tech', 'AAPL')", key="local_list_query")
        local_matches = local_stock_list.search_local_stocks(list_query, max_results=60)
        if not local_matches:
            st.info("Nenhuma ação encontrada nesse filtro.")
        else:
            st.caption(f"{len(local_matches)} resultado(s) — marque as que quiser adicionar.")
            selected_local = []
            for match in local_matches:
                label = f"{match.symbol} — {match.name} ({match.exchange})"
                if st.checkbox(label, key=f"local_{match.symbol}"):
                    selected_local.append(match.symbol)
            if selected_local and st.button(f"Adicionar {len(selected_local)} selecionada(s) à watchlist", key="add_from_list"):
                with st.spinner("Confirmando preço, fundamentalista e elegibilidade de derivativo no Yahoo..."):
                    added, invalid = watchlist_module.add_tickers(selected_local)
                st.success(f"Adicionados/atualizados: {', '.join(e.symbol for e in added)}")
                if invalid:
                    st.warning(f"Falharam: {invalid}")
                st.rerun()

    with tab_search:
        st.caption("Busca ao vivo no Yahoo Finance — mais lenta, mas encontra qualquer ação (B3, ADR, ETF, não só S&P 500).")
        query = st.text_input("Nome da empresa ou código (ex.: 'eletrica', 'duke energy', 'PETR')")
        if query:
            with st.spinner("Buscando no Yahoo Finance..."):
                matches = symbol_search.search_symbols(query)
            if not matches:
                st.info("Nenhum resultado para essa busca.")
            else:
                selected = []
                for match in matches:
                    label = f"{match.symbol} — {match.name} ({match.exchange}, {match.quote_type})"
                    if st.checkbox(label, key=f"match_{match.symbol}"):
                        selected.append(match.symbol)
                if selected and st.button(f"Adicionar {len(selected)} selecionada(s) à watchlist"):
                    with st.spinner("Buscando preço, fundamentalista e elegibilidade de derivativo..."):
                        added, invalid = watchlist_module.add_tickers(selected)
                    st.success(f"Adicionados/atualizados: {', '.join(e.symbol for e in added)}")
                    if invalid:
                        st.warning(f"Falharam: {invalid}")
                    st.rerun()

    with tab_paste:
        raw_tickers = st.text_area("Um ticker por linha ou separados por vírgula (ex.: AAPL, PETR4.SA, TAEE11.SA)")
        if st.button("Adicionar tickers digitados"):
            tickers = [t.strip() for chunk in raw_tickers.splitlines() for t in chunk.split(",") if t.strip()]
            if not tickers:
                st.warning("Nenhum ticker digitado.")
            else:
                with st.spinner(f"Validando e buscando dados de {len(tickers)} ticker(s)..."):
                    added, invalid = watchlist_module.add_tickers(tickers)
                if added:
                    st.success(f"Adicionados/atualizados: {', '.join(e.symbol for e in added)}")
                if invalid:
                    st.error("Falharam:\n" + "\n".join(f"- {ticker}: {reason}" for ticker, reason in invalid.items()))
                st.rerun()

    st.subheader("Watchlist atual")
    watchlist_df = _load_watchlist_df()
    if watchlist_df.empty:
        st.info("Watchlist vazia — busque ou cole um ticker acima.")
    else:
        st.dataframe(_format_watchlist_table(watchlist_df), width="stretch")
        remove_symbol = st.selectbox("Remover ticker", [""] + list(watchlist_df["symbol"]))
        if remove_symbol and st.button(f"Remover {remove_symbol}"):
            watchlist_module.remove_ticker(remove_symbol)
            st.success(f"{remove_symbol} removido.")
            st.rerun()


# --------------------------------------------------------------------------------------
# Tela "Configuração" — AnalysisSettings
# --------------------------------------------------------------------------------------
elif PAGE == "Configuração":
    st.title("Configuração do pipeline")
    settings = load_settings()

    with st.form("settings_form"):
        col1, col2 = st.columns(2)
        with col1:
            target_protection_days = st.number_input("Prazo desejado da put (dias)", value=settings.target_protection_days, min_value=1)
            minimum_protection_days = st.number_input("Prazo mínimo da put (dias)", value=settings.minimum_protection_days, min_value=1)
            max_acceptable_loss_pct = st.slider("Perda máxima aceitável (%)", 0.0, 1.0, settings.max_acceptable_loss_pct)
            minimum_open_interest = st.number_input("Contratos em aberto mínimo (liquidez)", value=settings.minimum_open_interest, min_value=0)
            maximum_bid_ask_spread_pct = st.slider("Spread bid-ask máximo (%)", 0.0, 1.0, settings.maximum_bid_ask_spread_pct)
            monte_carlo_paths = st.number_input("Caminhos do Monte Carlo", value=settings.monte_carlo_paths, min_value=500, step=500)
        with col2:
            entry_window_days = st.number_input("Janela de entrada (dias úteis)", value=settings.entry_window_days, min_value=1)
            dqn_training_rounds = st.number_input("Rodadas de treino (DQN entrada)", value=settings.dqn_training_rounds, min_value=1)
            exit_dqn_training_rounds = st.number_input("Rodadas de treino (DQN saída)", value=settings.exit_dqn_training_rounds, min_value=1)
            exit_time_value_floor_pct = st.slider("Piso de valor-tempo p/ sair (%)", 0.0, 1.0, settings.exit_time_value_floor_pct)
            stock_commission_pct = st.number_input("Corretagem da ação (%)", value=settings.stock_commission_pct, format="%.5f")
            option_commission_per_contract = st.number_input("Taxa por contrato de opção", value=settings.option_commission_per_contract)
        submitted = st.form_submit_button("Salvar configuração")

    if submitted:
        new_settings = AnalysisSettings(
            **{**settings.to_dict(), **{
                "target_protection_days": int(target_protection_days),
                "minimum_protection_days": int(minimum_protection_days),
                "max_acceptable_loss_pct": float(max_acceptable_loss_pct),
                "minimum_open_interest": int(minimum_open_interest),
                "maximum_bid_ask_spread_pct": float(maximum_bid_ask_spread_pct),
                "monte_carlo_paths": int(monte_carlo_paths),
                "entry_window_days": int(entry_window_days),
                "dqn_training_rounds": int(dqn_training_rounds),
                "exit_dqn_training_rounds": int(exit_dqn_training_rounds),
                "exit_time_value_floor_pct": float(exit_time_value_floor_pct),
                "stock_commission_pct": float(stock_commission_pct),
                "option_commission_per_contract": float(option_commission_per_contract),
            }}
        )
        save_settings(new_settings)
        st.success(f"Configuração salva em {DEFAULT_SETTINGS_PATH}")

    st.caption(f"Arquivo: `{DEFAULT_SETTINGS_PATH}` (se não existir, usa a semente `config/settings.example.yaml`)")


# --------------------------------------------------------------------------------------
# Tela "Histórico & Relatórios" — cotação e fundamentalista claros + gerar relatório
# --------------------------------------------------------------------------------------
elif PAGE == "Histórico & Relatórios":
    st.title("Histórico & Relatórios")
    watchlist_df = _load_watchlist_df()
    tickers = list(watchlist_df["symbol"]) if not watchlist_df.empty else []
    ticker = st.selectbox("Ticker", tickers) if tickers else None

    if ticker:
        st.subheader(f"Cotação — {ticker}")
        market_dir = DEFAULT_DATA_STORE_ROOT / "market" / ticker.replace("/", "_")
        quotes_frame = pd.DataFrame()
        if market_dir.exists():
            files = sorted(market_dir.glob("*.csv"), reverse=True)[:30]
            frames = [csv_store.read_csv(f) for f in files]
            frames = [f for f in frames if not f.empty]
            if frames:
                quotes_frame = pd.concat(frames, ignore_index=True)

        if quotes_frame.empty:
            st.info(
                "Nenhuma cotação coletada ainda para este ticker — isso só acontece via o "
                "Coletor contínuo (tela 'Coletor'). O botão 'Gerar novo relatório agora', mais "
                "abaixo, busca o preço na hora, mas não grava no histórico de cotações."
            )
        else:
            quotes_frame = quotes_frame.sort_values("collected_at", ascending=False)
            latest = quotes_frame.iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric("Último preço coletado", f"{latest['price']:,.2f}" if pd.notna(latest["price"]) else "—")
            col2.metric("Coletado em", str(latest["collected_at"])[:19].replace("T", " "))
            col3.metric("Status", latest["status"])
            with st.expander(f"Ver últimas {min(len(quotes_frame), 50)} cotações coletadas"):
                st.dataframe(_format_quotes_table(quotes_frame.head(50)), width="stretch")

        st.subheader(f"Fundamentalista — {ticker}")
        fundamentals_frame = csv_store.read_csv(csv_store.fundamentals_csv_path(DEFAULT_DATA_STORE_ROOT, ticker))
        if fundamentals_frame.empty:
            st.info("Sem snapshot fundamentalista ainda — adicionado automaticamente quando o ticker entra na watchlist.")
        else:
            latest_fundamentals = fundamentals_frame.sort_values("collected_at").iloc[-1]
            metric_fields = [
                ("P/L", "pe_ratio", "{:.2f}"), ("P/VP", "price_to_book", "{:.2f}"), ("ROE", "roe", "{:.1%}"),
                ("Dividend yield (12m)", "dividend_yield_ttm", "{:.1%}"), ("Margem líquida", "profit_margin", "{:.1%}"),
                ("Valor de mercado", "market_cap", None),
            ]
            cols = st.columns(3)
            for index, (label, field, fmt) in enumerate(metric_fields):
                value = latest_fundamentals.get(field)
                if pd.isna(value):
                    display_value = "—"
                elif fmt is None:
                    display_value = _format_big_number(value)
                else:
                    display_value = fmt.format(value)
                cols[index % 3].metric(label, display_value)
            st.caption(f"Setor: {latest_fundamentals.get('sector', '—')} · Indústria: {latest_fundamentals.get('industry', '—')} · Coletado em {str(latest_fundamentals.get('collected_at', ''))[:19].replace('T', ' ')}")
            with st.expander("📄 Ver todos os indicadores (dado bruto, como veio do Yahoo) e histórico de coletas"):
                st.dataframe(_format_fundamentals_table(fundamentals_frame.sort_values("collected_at", ascending=False)), width="stretch")

        st.subheader(f"Dados brutos — {ticker}")
        st.caption(
            "Histórico de preço completo (o que os modelos realmente usam pra treinar) e a cadeia de opções "
            "mais recente, direto do Yahoo — pra quem quer auditar/conferir de onde vêm os números do sistema."
        )
        if st.button("🔍 Carregar dados brutos", key="load_raw_data"):
            raw_settings = load_settings()
            with st.spinner("Baixando histórico completo de preço..."):
                try:
                    full_history = PROVIDER.download_price_history(ticker, raw_settings.history_start_date)
                    history_table = full_history.rename("Preço de fechamento").reset_index()
                    history_table.columns = ["Data", "Preço de fechamento"]
                    st.line_chart(full_history)
                    st.caption(f"{len(history_table)} pontos, de {history_table['Data'].min().date()} até {history_table['Data'].max().date()}.")
                    with st.expander("Ver tabela completa"):
                        st.dataframe(history_table, width="stretch")
                    st.download_button(
                        "Baixar histórico de preço em CSV", history_table.to_csv(index=False), file_name=f"{ticker}_historico_precos.csv",
                        mime="text/csv", key="dl_raw_history",
                    )
                except Exception as unexpected:
                    st.error(f"Não consegui baixar o histórico completo agora: {unexpected}")

            entry = next((e for e in watchlist_module.load_watchlist() if e.symbol == ticker), None)
            if entry and entry.elegivel_derivativo:
                with st.spinner("Buscando cadeia de opções (puts líquidas)..."):
                    try:
                        risk_free_rate = PROVIDER.get_risk_free_rate()
                        raw_result = analyze_ticker(ticker, PROVIDER, risk_free_rate, raw_settings)
                        st.write(f"**Cadeia de opções líquidas** — vencimento {raw_result['expiration'].date()}")
                        st.dataframe(raw_result["liquid_puts"], width="stretch")
                        st.download_button(
                            "Baixar cadeia de opções em CSV", raw_result["liquid_puts"].to_csv(index=False),
                            file_name=f"{ticker}_opcoes.csv", mime="text/csv", key="dl_raw_options",
                        )
                    except MarketDataError as reason:
                        st.info(f"Sem cadeia de opções elegível agora: {reason}")
                    except Exception as unexpected:
                        st.warning(f"Não consegui buscar a cadeia de opções agora: {unexpected}")
            else:
                st.caption("Este ticker não tem opções no Yahoo — sem cadeia pra mostrar.")

    st.divider()
    st.subheader("Relatórios")
    st.caption("Gera a análise completa (ação + put de proteção + DQN de entrada), com gráficos e PDF para download.")
    report_tickers = st.multiselect("Ativos para o novo relatório", tickers, default=tickers, key="report_tickers")
    if st.button("Gerar novo relatório agora", key="generate_report_button") and report_tickers:
        report_settings = load_settings()
        results, not_eligible, entry_evaluations, risk_free_rate = run_full_pipeline(report_tickers, report_settings)
        summaries = render_pipeline_results(report_tickers, results, not_eligible, entry_evaluations, report_settings, risk_free_rate)
        generate_and_offer_pdf(summaries, key_suffix="historico")

    st.subheader("Relatórios já gerados")
    reports_dir = DEFAULT_DATA_STORE_ROOT / "reports"
    pdfs = sorted(reports_dir.glob("*.pdf"), reverse=True) if reports_dir.exists() else []
    if pdfs:
        for pdf_path in pdfs[:10]:
            with open(pdf_path, "rb") as handle:
                st.download_button(f"Baixar {pdf_path.name}", handle.read(), file_name=pdf_path.name, mime="application/pdf", key=f"existing_{pdf_path}")
    else:
        st.info("Nenhum relatório PDF gerado ainda.")


# --------------------------------------------------------------------------------------
# Tela "Coletor" — status/start/stop do loop de coleta contínua
# --------------------------------------------------------------------------------------
elif PAGE == "Coletor":
    st.title("Coletor contínuo (pregão)")

    refresh_col, _ = st.columns([1, 5])
    if refresh_col.button("🔄 Atualizar status"):
        st.rerun()

    now_utc = pd.Timestamp.now(tz="UTC")
    watchlist_df = _load_watchlist_df()
    calendars = sorted({exchange_calendar_for_ticker(symbol) for symbol in watchlist_df.get("symbol", [])}) or ["NYSE"]
    market_open_now = any_market_open(now_utc, calendars)

    thread_alive = st.session_state["collector_thread"] is not None and st.session_state["collector_thread"].is_alive()
    status = read_status(DEFAULT_DATA_STORE_ROOT)

    # --- Indicador principal: coletor realmente vivo e ativo (não só "o botão foi clicado") ---
    last_check_at = status.get("last_check_at")
    seconds_since_check = None
    if last_check_at:
        seconds_since_check = (pd.Timestamp.now() - pd.Timestamp(last_check_at)).total_seconds()

    if thread_alive and seconds_since_check is not None and seconds_since_check < 180:
        st.success(f"🟢 Coletor ATIVO — última verificação há {int(seconds_since_check)}s (às {last_check_at[11:19]}).")
    elif thread_alive:
        st.warning("🟡 Coletor marcado como rodando, mas sem verificação recente — pode estar preso numa chamada de rede. Clique em 'Atualizar status'.")
    else:
        st.info("⚪ Coletor parado.")

    if status.get("last_error"):
        st.error(f"Último erro registrado pelo coletor: {status['last_error']}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mercado aberto agora?", "Sim" if market_open_now else "Não")
    col2.metric("Verificações desde o início", status.get("iterations", 0))
    col3.metric("Última coleta de preço bem-sucedida", (status.get("last_collection_at") or "—")[11:19] if status.get("last_collection_at") else "—")
    col4.metric("Última rodada completa de modelo", (status.get("last_model_run_at") or "—")[11:19] if status.get("last_model_run_at") else "—")
    st.caption(f"Calendários considerados: {', '.join(calendars)}. Fora do pregão o coletor só verifica e dorme, não insiste no Yahoo.")

    last_cycle = status.get("last_cycle_tickers") or []
    if last_cycle:
        st.subheader("O que foi coletado na última rodada")
        st.dataframe(_format_quotes_table(pd.DataFrame(last_cycle)), width="stretch")
    elif thread_alive:
        st.info("Coletor rodando, mas ainda não completou uma rodada com o mercado aberto (ou a watchlist está vazia).")

    st.divider()
    if thread_alive:
        st.caption(f"Iniciado pela sessão atual às {st.session_state['collector_started_at']}.")
        if st.button("Parar coletor"):
            st.session_state["collector_stop_event"].set()
            st.session_state["collector_thread"].join(timeout=5)
            st.session_state["collector_thread"] = None
            st.rerun()
    else:
        poll_interval = st.number_input("Intervalo de coleta de preço (segundos)", value=900, min_value=60)
        model_interval = st.number_input("Intervalo de rodada completa de modelo (segundos)", value=1800, min_value=300)
        if st.button("Iniciar coletor"):
            stop_event = threading.Event()
            config = CollectorConfig(poll_interval_seconds=int(poll_interval), model_run_interval_seconds=int(model_interval))
            thread = threading.Thread(target=run_forever, kwargs={"config": config, "stop_event": stop_event}, daemon=True)
            thread.start()
            st.session_state["collector_thread"] = thread
            st.session_state["collector_stop_event"] = stop_event
            st.session_state["collector_started_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
            st.rerun()


# --------------------------------------------------------------------------------------
# Tela "Modelos" — treinar/carregar DQN de entrada e saída, com "model card" explicado
# --------------------------------------------------------------------------------------
elif PAGE == "Modelos":
    st.title("Modelos (DQN)")
    st.caption(
        "Cada modelo responde a uma pergunta: a DQN de **entrada** diz se vale comprar a ação hoje ou esperar; "
        "a de **saída** diz se vale manter ou vender a put de proteção antes do vencimento. "
        "Pra quem não é técnico: o parágrafo em destaque já basta. Pra quem é técnico: abra "
        "'Detalhes técnicos' em cada modelo pra ver pesos, hiperparâmetros e as métricas completas."
    )

    tab_saved, tab_train = st.tabs(["📂 Modelos salvos", "🧠 Treinar novo"])

    with tab_saved:
        index = model_store.list_models()
        if index.empty:
            st.info("Nenhum modelo salvo ainda — treine um na aba 'Treinar novo' (fica salvo em `.pkl`, não precisa retreinar depois).")
        else:
            display_index = index.rename(columns={
                "ticker": "Ticker", "model_type": "Tipo", "trained_at": "Treinado em", "metrica_principal": "Métrica principal",
                "valor_principal": "Valor", "bateu_benchmark": "Bateu benchmark?", "sinal_hoje": "Sinal de hoje",
            })
            display_index["Tipo"] = display_index["Tipo"].map({"entry": "Entrada", "exit": "Saída"})
            display_index["Bateu benchmark?"] = display_index["Bateu benchmark?"].astype(str).map({"True": "Sim", "False": "Não"}).fillna(display_index["Bateu benchmark?"])
            st.dataframe(display_index.drop(columns=["filename"], errors="ignore"), width="stretch")

            chosen_filename = st.selectbox("Carregar um modelo salvo (sem retreinar)", index["filename"])
            if st.button("Carregar modelo"):
                payload = model_store.load_model(chosen_filename)
                render_model_card(payload["ticker"], payload["model_type"], payload["evaluation"], payload["hyperparameters"], payload["trained_at"])
                st.download_button(
                    "Baixar este modelo (.pkl)", (model_store.DEFAULT_MODELS_DIR / chosen_filename).read_bytes(),
                    file_name=chosen_filename, mime="application/octet-stream",
                )

    with tab_train:
        watchlist_df = _load_watchlist_df()
        if watchlist_df.empty:
            st.warning("Watchlist vazia. Adicione ações na tela 'Ações' primeiro.")
        else:
            ticker = st.selectbox("Ticker", list(watchlist_df["symbol"]), key="train_ticker")
            entries = watchlist_module.load_watchlist()
            entry = next((e for e in entries if e.symbol == ticker), None)
            type_options = ["Entrada (quando comprar a ação)"]
            if entry and entry.elegivel_derivativo:
                type_options.append("Saída (quando vender a put)")
            else:
                st.caption("Este ticker não tem opções no Yahoo — só a DQN de entrada está disponível para ele.")
            type_label = st.radio("Qual modelo treinar?", type_options)
            model_type = "exit" if type_label.startswith("Saída") else "entry"

            if st.button("Treinar e salvar"):
                settings = load_settings()
                evaluation = None
                try:
                    if model_type == "entry":
                        with st.spinner(f"Baixando histórico e treinando DQN de entrada para {ticker}..."):
                            closes = PROVIDER.download_price_history(ticker, settings.history_start_date)
                            evaluation = evaluate_entry_timing(ticker, closes, settings)
                    else:
                        with st.spinner(f"Montando a put de proteção e treinando DQN de saída para {ticker}..."):
                            risk_free_rate = PROVIDER.get_risk_free_rate()
                            result = analyze_ticker(ticker, PROVIDER, risk_free_rate, settings)
                            chosen = result["recommended"]
                            plan = ProtectivePutPlan(**{key: chosen[key] for key in ProtectivePutPlan.__dataclass_fields__})
                            window_days = max(int(round(plan.days_to_expiration * TRADING_DAYS_PER_YEAR / 365)), 1)
                            curve = time_value_curve(plan, risk_free_rate, result["dividend_yield"], window_days)
                            fixed_rule = recommend_exit_day(curve, settings.exit_time_value_floor_pct)
                            evaluation = evaluate_exit_timing(
                                plan, result["garch"], risk_free_rate, result["dividend_yield"], window_days, settings, fixed_rule["exit_day"],
                            )
                except MarketDataError as reason:
                    st.error(f"Não foi possível treinar: {reason}")
                except Exception as unexpected:
                    st.error(f"Falhou: {unexpected}")

                if evaluation is not None:
                    saved_path = model_store.save_model(ticker, model_type, evaluation, settings)
                    # st.tabs roda o conteúdo das DUAS abas na mesma execução, em ordem de código —
                    # "Modelos salvos" vem antes de "Treinar novo", então leria o índice antes do
                    # save_model acima rodar nesta mesma rodada. Guarda em session_state + st.rerun()
                    # pra "Modelos salvos" já aparecer atualizado e o card continuar visível aqui.
                    st.session_state["last_trained_model"] = {
                        "ticker": ticker, "model_type": model_type, "evaluation": evaluation,
                        "hyperparameters": settings.to_dict(), "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                        "saved_filename": saved_path.name,
                    }
                    st.rerun()

            last_trained = st.session_state.get("last_trained_model")
            if last_trained:
                st.success(f"Modelo treinado e salvo em `{last_trained['saved_filename']}` — disponível na aba 'Modelos salvos' sem precisar retreinar.")
                render_model_card(
                    last_trained["ticker"], last_trained["model_type"], last_trained["evaluation"],
                    last_trained["hyperparameters"], last_trained["trained_at"],
                )
