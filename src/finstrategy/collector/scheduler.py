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

import json
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


def status_path(data_store_root: Path | str = DEFAULT_DATA_STORE_ROOT) -> Path:
    return Path(data_store_root) / "collector_status.json"


def write_status(data_store_root: Path | str, **fields) -> None:
    """Grava o estado do coletor para a tela "Coletor" do frontend conseguir mostrar
    feedback ao vivo (checado em quê, coletou o quê, algum erro) sem precisar inspecionar
    a thread em background diretamente — um arquivo pequeno, lido a cada rerun da UI."""
    path = status_path(data_store_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    current = read_status(data_store_root)
    current.update(fields)
    current["updated_at"] = _now_iso()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(current, handle, ensure_ascii=False, indent=2, default=str)


def read_status(data_store_root: Path | str = DEFAULT_DATA_STORE_ROOT) -> dict:
    path = status_path(data_store_root)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}


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
        except Exception as unexpected:
            # qualquer erro inesperado aqui NÃO pode derrubar a thread do coletor
            # (já aconteceu: bug de timezone matou o loop em background sem avisar
            # a UI) — vira status visível em vez de exceção não tratada.
            base_row["status"] = f"erro inesperado no derivativo: {unexpected}"
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
    """Uma rodada de coleta sobre toda a watchlist. Devolve as linhas gravadas (para log/teste).

    `today` deve ser um Timestamp NAIVE (sem timezone) — vencimentos de opção que vêm
    do Yahoo são naive, e comparar naive com tz-aware (`now.normalize()` com tz="UTC",
    por exemplo) levanta `TypeError` dentro de `analyze_ticker`. `run_forever` cuida
    de passar a data certa; outros chamadores devem fazer o mesmo."""
    today = today or pd.Timestamp.today().normalize()
    entries = load_watchlist(config.watchlist_path)
    rows_written = []
    for entry in entries:
        try:
            price_row = collect_price_snapshot(entry, config.provider, today)
            append_row(market_csv_path(config.data_store_root, entry.symbol, today), price_row)
            rows_written.append(price_row)

            if run_model:
                model_row = run_model_pipeline_for_ticker(entry, config.provider, settings, risk_free_rate, today)
                append_row(model_runs_csv_path(config.data_store_root, today), model_row)
                rows_written.append(model_row)
        except Exception as unexpected:
            # um ticker com problema (rede, dado inesperado) não pode derrubar a
            # rodada inteira nem, pior, a thread do coletor — registra e segue.
            rows_written.append({"ticker": entry.symbol, "collected_at": _now_iso(), "status": f"erro inesperado: {unexpected}"})
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
    write_status(config.data_store_root, started_at=_now_iso(), iterations=0, last_error=None)

    while not stop_event.is_set():
        try:
            now_utc = pd.Timestamp.now(tz="UTC")  # só para checar calendário de pregão (schedule vem em UTC)
            today = pd.Timestamp.today().normalize()  # NAIVE — é o que analyze_ticker/choose_expiration esperam
            entries = load_watchlist(config.watchlist_path)
            calendars = {exchange_calendar_for_ticker(entry.symbol) for entry in entries} or {"NYSE"}
            market_open = any_market_open(now_utc, list(calendars))

            if market_open:
                run_model = (time.monotonic() - last_model_run) >= config.model_run_interval_seconds
                rows = run_collection_cycle(config, settings, risk_free_rate, run_model=run_model, today=today)
                if run_model:
                    last_model_run = time.monotonic()
                sleep_seconds = config.poll_interval_seconds
                price_rows = [r for r in rows if "price" in r]
                write_status(
                    config.data_store_root, last_check_at=_now_iso(), market_open=True,
                    last_collection_at=_now_iso(), last_model_run_at=_now_iso() if run_model else None,
                    last_cycle_tickers=price_rows, last_error=None,
                )
            else:
                sleep_seconds = min(config.poll_interval_seconds, 300)  # checa de novo em até 5 min
                write_status(config.data_store_root, last_check_at=_now_iso(), market_open=False)

            iterations += 1
            write_status(config.data_store_root, iterations=iterations)
        except Exception as unexpected:
            # rede fora do ar um instante, erro inesperado de um provider, etc. — a
            # thread do coletor NUNCA pode morrer em silêncio (já aconteceu uma vez
            # por um bug de timezone); registra o erro no status e segue vivo.
            sleep_seconds = min(config.poll_interval_seconds, 300)
            write_status(config.data_store_root, last_check_at=_now_iso(), last_error=str(unexpected))

        if max_iterations is not None and iterations >= max_iterations:
            break
        stop_event.wait(sleep_seconds)

    write_status(config.data_store_root, stopped_at=_now_iso())


if __name__ == "__main__":
    run_forever()
