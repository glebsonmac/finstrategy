"""Loop de coleta contínua durante o pregão (plan.md, seção 6).

`run_collection_cycle` faz UMA rodada (preço de cada ticker da watchlist + full
pipeline de modelo nos tickers cuja vez chegou) e é a unidade testável; `run_forever`
só embrulha isso num `while True` com `time.sleep`, respeitando o calendário de pregão
(seção 7/`market_session.py`) e aceitando um `stop_event`/`max_iterations` para poder
ser testado e para a tela "Coletor" do frontend conseguir pará-lo (seção 8 do plano).

Fora do pregão, `collect_price_snapshot` ainda funciona (o Yahoo devolve o último
preço negociado) — só não atualiza em tempo real; isso é esperado, não um erro do
coletor (ver aviso do notebook original sobre bid/ask zerado fora do pregão).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config_model import AnalysisSettings, load_settings
from ..data.errors import MarketDataError
from ..data.watchlist import DEFAULT_DATA_STORE_ROOT, DEFAULT_WATCHLIST_PATH, WatchlistEntry, load_watchlist
from ..data.yahoo_provider import YahooProvider
from ..rl.entry_dqn import evaluate_entry_timing
from ..strategy.protective_put import analyze_ticker
from .csv_store import append_row, market_csv_path, model_runs_csv_path
from .market_session import any_market_open, exchange_calendar_for_ticker


def _now_iso() -> str:
    return pd.Timestamp.now().isoformat(timespec="seconds")


def collect_price_snapshot(entry: WatchlistEntry, provider, today: pd.Timestamp) -> dict:
    """Uma linha de preço para `data_store/market/<ticker>/<data>.csv`."""
    try:
        closes = provider.download_price_history(entry.symbol, (today - pd.Timedelta(days=5)).strftime("%Y-%m-%d"))
        fallback = float(closes.iloc[-1]) if len(closes) else float("nan")
        price = provider.get_current_stock_price(entry.symbol, fallback)
        return {"ticker": entry.symbol, "collected_at": _now_iso(), "price": price, "status": "ok"}
    except MarketDataError as reason:
        return {"ticker": entry.symbol, "collected_at": _now_iso(), "price": None, "status": f"erro: {reason}"}
    except Exception as unexpected:
        return {"ticker": entry.symbol, "collected_at": _now_iso(), "price": None, "status": f"erro inesperado: {unexpected}"}


def run_model_pipeline_for_ticker(entry: WatchlistEntry, provider, settings: AnalysisSettings, risk_free_rate: float, today: pd.Timestamp) -> dict:
    """Roda o pipeline de modelo (ação + put de proteção, se elegível; DQN de entrada
    sempre) para um ticker e devolve um resumo pronto para `model_runs/<data>.csv`."""
    base_row = {"ticker": entry.symbol, "evaluated_at": _now_iso()}
    try:
        closes = provider.download_price_history(entry.symbol, settings.history_start_date)
    except MarketDataError as reason:
        return {**base_row, "status": f"sem histórico: {reason}"}

    entry_evaluation = None
    try:
        entry_evaluation = evaluate_entry_timing(entry.symbol, closes, settings)
    except Exception as unexpected:
        base_row["entry_dqn_erro"] = str(unexpected)

    if entry.elegivel_derivativo:
        try:
            put_result = analyze_ticker(entry.symbol, provider, risk_free_rate, settings, today)
        except MarketDataError as reason:
            base_row["status"] = f"sem derivativo elegível agora: {reason}"
            put_result = None
    else:
        base_row["status"] = "não elegível: sem opções no Yahoo"
        put_result = None

    if entry_evaluation is not None:
        base_row.update({
            "entry_dqn_sinal": entry_evaluation["today_action"],
            "entry_dqn_bateu_benchmarks": entry_evaluation["beats_benchmarks"],
        })
    if put_result is not None:
        recommended = put_result["recommended"]
        base_row.update({
            "status": "ok",
            "strike_recomendada": float(recommended["strike"]),
            "vencimento": str(put_result["expiration"].date()),
            "perda_maxima_pct": float(recommended["max_loss_pct"]),
            "breakeven": float(recommended["breakeven_price"]),
        })
    return base_row


@dataclass
class CollectorConfig:
    watchlist_path: Path = DEFAULT_WATCHLIST_PATH
    data_store_root: Path = DEFAULT_DATA_STORE_ROOT
    poll_interval_seconds: int = 900
    model_run_interval_seconds: int = 1800
    provider: object = field(default_factory=YahooProvider)


def run_collection_cycle(config: CollectorConfig, settings: AnalysisSettings, risk_free_rate: float, run_model: bool, today: pd.Timestamp | None = None) -> list[dict]:
    """Uma rodada de coleta sobre toda a watchlist. Devolve as linhas gravadas (para log/teste)."""
    today = today or pd.Timestamp.today().normalize()
    entries = load_watchlist(config.watchlist_path)
    rows_written = []
    for entry in entries:
        price_row = collect_price_snapshot(entry, config.provider, today)
        append_row(market_csv_path(config.data_store_root, entry.symbol, today), price_row)
        rows_written.append(price_row)

        if run_model:
            model_row = run_model_pipeline_for_ticker(entry, config.provider, settings, risk_free_rate, today)
            append_row(model_runs_csv_path(config.data_store_root, today), model_row)
            rows_written.append(model_row)
    return rows_written


def run_forever(
    config: CollectorConfig | None = None,
    settings: AnalysisSettings | None = None,
    stop_event: threading.Event | None = None,
    max_iterations: int | None = None,
) -> None:
    """Loop principal do coletor. Só roda a coleta de preço/modelo enquanto algum
    mercado da watchlist estiver aberto (`any_market_open`); fora do pregão, dorme até
    o próximo horário útil para não bater no Yahoo sem necessidade. Pare com
    `stop_event.set()` (usado pela tela "Coletor" do frontend) ou `max_iterations`
    (usado pelos testes, para não rodar para sempre)."""
    config = config or CollectorConfig()
    settings = settings or load_settings()
    stop_event = stop_event or threading.Event()

    provider = config.provider
    risk_free_rate = provider.get_risk_free_rate()
    last_model_run = 0.0
    iterations = 0

    while not stop_event.is_set():
        now = pd.Timestamp.now(tz="UTC")
        entries = load_watchlist(config.watchlist_path)
        calendars = {exchange_calendar_for_ticker(entry.symbol) for entry in entries} or {"NYSE"}
        if any_market_open(now, list(calendars)):
            run_model = (time.monotonic() - last_model_run) >= config.model_run_interval_seconds
            run_collection_cycle(config, settings, risk_free_rate, run_model=run_model, today=now.normalize())
            if run_model:
                last_model_run = time.monotonic()
            sleep_seconds = config.poll_interval_seconds
        else:
            sleep_seconds = min(config.poll_interval_seconds, 300)  # checa de novo em até 5 min

        iterations += 1
        if max_iterations is not None and iterations >= max_iterations:
            break
        stop_event.wait(sleep_seconds)


if __name__ == "__main__":
    run_forever()
