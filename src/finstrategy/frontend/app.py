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
from finstrategy.collector.scheduler import CollectorConfig, run_forever
from finstrategy.config_model import AnalysisSettings, DEFAULT_SETTINGS_PATH, load_settings, save_settings
from finstrategy.data import symbol_search, watchlist as watchlist_module
from finstrategy.data.errors import MarketDataError
from finstrategy.data.watchlist import DEFAULT_DATA_STORE_ROOT, DEFAULT_WATCHLIST_PATH
from finstrategy.data.yahoo_provider import YahooProvider
from finstrategy.reporting.panel import build_panel_dataframe, build_plain_language_summary
from finstrategy.reporting.pdf_report import generate_pdf_report
from finstrategy.rl.entry_dqn import evaluate_entry_timing
from finstrategy.strategy.protective_put import analyze_ticker

st.set_page_config(page_title="finstrategy", layout="wide")

PROVIDER = YahooProvider()


def _init_state():
    st.session_state.setdefault("collector_thread", None)
    st.session_state.setdefault("collector_stop_event", None)
    st.session_state.setdefault("collector_started_at", None)


_init_state()

PAGE = st.sidebar.radio(
    "Navegação",
    ["Ações", "Configuração", "Rodar agora", "Histórico", "Coletor", "Modelos"],
)


def _load_watchlist_df() -> pd.DataFrame:
    entries = watchlist_module.load_watchlist()
    if not entries:
        return pd.DataFrame(columns=["symbol", "name", "tags", "elegivel_derivativo", "last_price", "last_updated"])
    return pd.DataFrame([entry.to_dict() for entry in entries])


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
        if st.button("Popular com a semente do setor elétrico (primeira vez)"):
            with st.spinner("Buscando dados dos tickers semente..."):
                added = watchlist_module.ensure_seeded()
            st.success(f"{len(added)} tickers adicionados.")
            st.rerun()

    st.subheader("Buscar e marcar ações")
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

    st.subheader("Ou cole um ou mais tickers direto")
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
        st.dataframe(watchlist_df, use_container_width=True)
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
# Tela "Rodar agora" — pipeline completo manual + PDF
# --------------------------------------------------------------------------------------
elif PAGE == "Rodar agora":
    st.title("Rodar o pipeline agora")
    watchlist_df = _load_watchlist_df()
    if watchlist_df.empty:
        st.warning("Watchlist vazia. Adicione ações na tela 'Ações' primeiro.")
    else:
        chosen_tickers = st.multiselect("Ativos para analisar", list(watchlist_df["symbol"]), default=list(watchlist_df["symbol"]))
        if st.button("Rodar análise completa") and chosen_tickers:
            settings = load_settings()
            results, not_eligible, entry_evaluations = {}, {}, {}
            progress = st.progress(0.0)
            with st.spinner("Buscando taxa livre de risco..."):
                try:
                    risk_free_rate = PROVIDER.get_risk_free_rate()
                except MarketDataError as reason:
                    st.error(f"Não foi possível obter a taxa livre de risco: {reason}")
                    risk_free_rate = 0.05

            for index, ticker in enumerate(chosen_tickers):
                try:
                    closes = PROVIDER.download_price_history(ticker, settings.history_start_date)
                    entry_evaluations[ticker] = evaluate_entry_timing(ticker, closes, settings)
                except Exception as unexpected:
                    st.warning(f"{ticker}: DQN de entrada falhou ({unexpected})")

                try:
                    results[ticker] = analyze_ticker(ticker, PROVIDER, risk_free_rate, settings)
                except MarketDataError as reason:
                    not_eligible[ticker] = str(reason)
                progress.progress((index + 1) / len(chosen_tickers))

            st.subheader("Painel técnico")
            panel = build_panel_dataframe(results, not_eligible, entry_evaluations)
            st.dataframe(panel, use_container_width=True)

            summaries = [
                build_plain_language_summary(ticker, results.get(ticker), entry_evaluations.get(ticker), not_eligible.get(ticker))
                for ticker in chosen_tickers
            ]
            st.subheader("Resumo em linguagem simples")
            for summary in summaries:
                st.markdown(f"**{summary.headline}**")
                for line in summary.body_lines:
                    st.write(f"- {line}")

            pdf_path = csv_store.reports_csv_path(DEFAULT_DATA_STORE_ROOT, pd.Timestamp.today()).with_suffix(".pdf")
            generate_pdf_report(summaries, pdf_path)
            with open(pdf_path, "rb") as handle:
                st.download_button("Baixar relatório em PDF", handle.read(), file_name=pdf_path.name, mime="application/pdf")


# --------------------------------------------------------------------------------------
# Tela "Histórico" — ler CSVs já coletados
# --------------------------------------------------------------------------------------
elif PAGE == "Histórico":
    st.title("Histórico coletado")
    watchlist_df = _load_watchlist_df()
    tickers = list(watchlist_df["symbol"]) if not watchlist_df.empty else []
    ticker = st.selectbox("Ticker", tickers) if tickers else None

    if ticker:
        st.subheader(f"Preço coletado — {ticker}")
        market_dir = DEFAULT_DATA_STORE_ROOT / "market" / ticker.replace("/", "_")
        if market_dir.exists():
            files = sorted(market_dir.glob("*.csv"), reverse=True)[:30]
            frames = [csv_store.read_csv(f) for f in files]
            frames = [f for f in frames if not f.empty]
            if frames:
                st.dataframe(pd.concat(frames, ignore_index=True), use_container_width=True)
            else:
                st.info("Nenhum dado coletado ainda para este ticker.")
        else:
            st.info("Nenhum dado coletado ainda para este ticker.")

        st.subheader(f"Fundamentalista — {ticker}")
        fundamentals_frame = csv_store.read_csv(csv_store.fundamentals_csv_path(DEFAULT_DATA_STORE_ROOT, ticker))
        if not fundamentals_frame.empty:
            st.dataframe(fundamentals_frame, use_container_width=True)
        else:
            st.info("Sem snapshot fundamentalista ainda.")

    st.subheader("Relatórios gerados")
    reports_dir = DEFAULT_DATA_STORE_ROOT / "reports"
    pdfs = sorted(reports_dir.glob("*.pdf"), reverse=True) if reports_dir.exists() else []
    if pdfs:
        for pdf_path in pdfs[:10]:
            with open(pdf_path, "rb") as handle:
                st.download_button(f"Baixar {pdf_path.name}", handle.read(), file_name=pdf_path.name, mime="application/pdf", key=str(pdf_path))
    else:
        st.info("Nenhum relatório PDF gerado ainda.")


# --------------------------------------------------------------------------------------
# Tela "Coletor" — status/start/stop do loop de coleta contínua
# --------------------------------------------------------------------------------------
elif PAGE == "Coletor":
    st.title("Coletor contínuo (pregão)")

    now = pd.Timestamp.now(tz="UTC")
    watchlist_df = _load_watchlist_df()
    calendars = sorted({exchange_calendar_for_ticker(symbol) for symbol in watchlist_df.get("symbol", [])}) or ["NYSE"]
    market_open_now = any_market_open(now, calendars)
    st.metric("Algum mercado da watchlist está aberto agora?", "Sim" if market_open_now else "Não")
    st.caption(f"Calendários considerados: {', '.join(calendars)}. Fora do pregão o coletor só verifica e dorme, não insiste no Yahoo.")

    thread_alive = st.session_state["collector_thread"] is not None and st.session_state["collector_thread"].is_alive()
    if thread_alive:
        st.success(f"Coletor rodando desde {st.session_state['collector_started_at']}.")
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
# Tela "Modelos" — reavaliar DQN de entrada (e saída, quando elegível)
# --------------------------------------------------------------------------------------
elif PAGE == "Modelos":
    st.title("Modelos (DQN)")
    watchlist_df = _load_watchlist_df()
    if watchlist_df.empty:
        st.warning("Watchlist vazia.")
    else:
        ticker = st.selectbox("Ticker", list(watchlist_df["symbol"]))
        if st.button("Treinar e avaliar"):
            settings = load_settings()
            with st.spinner(f"Baixando histórico e treinando DQN de entrada para {ticker}..."):
                try:
                    closes = PROVIDER.download_price_history(ticker, settings.history_start_date)
                    evaluation = evaluate_entry_timing(ticker, closes, settings)
                except Exception as unexpected:
                    st.error(f"Falhou: {unexpected}")
                    evaluation = None

            if evaluation:
                st.subheader("DQN de entrada — avaliação honesta (treino/teste separados)")
                st.dataframe(evaluation["summary"], use_container_width=True)
                verdict = "bateu" if evaluation["beats_benchmarks"] else "não bateu"
                st.write(
                    f"A DQN **{verdict}** as referências com significância "
                    f"(p-valor vs. 1º dia = {evaluation['p_value_vs_first_day']:.2f}). "
                    f"Sinal de hoje: **{evaluation['today_action']}**."
                )
