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
from finstrategy.reporting.panel import build_panel_dataframe, build_plain_language_summary
from finstrategy.reporting.pdf_report import generate_pdf_report
from finstrategy.rl.entry_dqn import evaluate_entry_timing
from finstrategy.strategy.protective_put import analyze_ticker

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
    ["Ações", "Configuração", "Rodar agora", "Histórico & Relatórios", "Coletor", "Modelos"],
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


def run_full_pipeline(tickers: list[str], settings: AnalysisSettings, progress_label: str = "") -> tuple[dict, dict, dict]:
    """Roda DQN de entrada + put de proteção (quando elegível) para uma lista de
    tickers. Compartilhado pelas telas "Rodar agora" e "Histórico & Relatórios" (botão
    "Gerar novo relatório") para não duplicar a lógica."""
    results, not_eligible, entry_evaluations = {}, {}, {}
    if not tickers:
        return results, not_eligible, entry_evaluations

    progress = st.progress(0.0, text=progress_label or "Iniciando...")
    try:
        risk_free_rate = PROVIDER.get_risk_free_rate()
    except MarketDataError as reason:
        st.error(f"Não foi possível obter a taxa livre de risco: {reason}")
        risk_free_rate = 0.05

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
    return results, not_eligible, entry_evaluations


def render_pipeline_results(tickers: list[str], results: dict, not_eligible: dict, entry_evaluations: dict) -> list:
    st.subheader("Painel técnico")
    panel = build_panel_dataframe(results, not_eligible, entry_evaluations)
    st.dataframe(panel, width="stretch")

    summaries = [
        build_plain_language_summary(ticker, results.get(ticker), entry_evaluations.get(ticker), not_eligible.get(ticker))
        for ticker in tickers
    ]
    st.subheader("Resumo em linguagem simples")
    for summary in summaries:
        st.markdown(f"**{summary.headline}**")
        for line in summary.body_lines:
            st.write(f"- {line}")
    return summaries


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
            results, not_eligible, entry_evaluations = run_full_pipeline(chosen_tickers, settings)
            summaries = render_pipeline_results(chosen_tickers, results, not_eligible, entry_evaluations)
            generate_and_offer_pdf(summaries, key_suffix="rodar_agora")


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
                "Coletor contínuo (tela 'Coletor'). 'Rodar agora' busca o preço na hora, mas "
                "não grava no histórico de cotações."
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
            with st.expander("Ver todos os indicadores e histórico de coletas"):
                st.dataframe(_format_fundamentals_table(fundamentals_frame.sort_values("collected_at", ascending=False)), width="stretch")

    st.divider()
    st.subheader("Relatórios")
    st.caption("Gera a análise completa (ação + put de proteção + DQN de entrada) para os tickers escolhidos e um PDF para download, sem precisar ir até 'Rodar agora'.")
    report_tickers = st.multiselect("Ativos para o novo relatório", tickers, default=tickers, key="report_tickers")
    if st.button("Gerar novo relatório agora", key="generate_report_button") and report_tickers:
        settings = load_settings()
        results, not_eligible, entry_evaluations = run_full_pipeline(report_tickers, settings)
        summaries = render_pipeline_results(report_tickers, results, not_eligible, entry_evaluations)
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
                st.dataframe(evaluation["summary"], width="stretch")
                verdict = "bateu" if evaluation["beats_benchmarks"] else "não bateu"
                st.write(
                    f"A DQN **{verdict}** as referências com significância "
                    f"(p-valor vs. 1º dia = {evaluation['p_value_vs_first_day']:.2f}). "
                    f"Sinal de hoje: **{evaluation['today_action']}**."
                )
