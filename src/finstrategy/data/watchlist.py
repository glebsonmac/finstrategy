"""Watchlist dinâmica — a lista de ações que o pipeline analisa.

Requisito do usuário: a seleção de ativos não pode ser uma lista fixa no código.
`config/watchlist.yaml` é o dado; este módulo é o único lugar que lê/escreve esse
arquivo. Qualquer ticker reconhecido pelo Yahoo Finance pode entrar, um ou vários de
uma vez (`add_tickers`), e ao entrar o sistema busca preço, fundamentalista e
elegibilidade de derivativo sozinho — o usuário só informa o ticker.

O setor elétrico (`config/tickers_setor_eletrico.yaml`) é só a semente usada para
popular a watchlist na primeira execução; depois disso o arquivo semente não é mais
tocado por este módulo.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from ..collector.csv_store import append_row, fundamentals_csv_path
from . import fundamentals_provider
from .base import SymbolInfo
from .symbol_search import validate_symbols

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WATCHLIST_PATH = PROJECT_ROOT / "config" / "watchlist.yaml"
SEED_TICKERS_PATH = PROJECT_ROOT / "config" / "tickers_setor_eletrico.yaml"
DEFAULT_DATA_STORE_ROOT = PROJECT_ROOT / "data_store"


@dataclass
class WatchlistEntry:
    symbol: str
    name: str = ""
    tags: list = field(default_factory=list)
    currency: str = "?"
    exchange: str = "?"
    elegivel_derivativo: bool = False
    last_price: Optional[float] = None
    added_at: str = ""
    last_updated: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _now_iso() -> str:
    return __import__("pandas").Timestamp.now().isoformat(timespec="seconds")


def load_watchlist(path: Path | str = DEFAULT_WATCHLIST_PATH) -> list[WatchlistEntry]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    entries = []
    for item in raw.get("tickers", []):
        entries.append(WatchlistEntry(**{**{"tags": []}, **item}))
    return entries


def save_watchlist(entries: list[WatchlistEntry], path: Path | str = DEFAULT_WATCHLIST_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tickers": [entry.to_dict() for entry in entries]}
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def seed_tickers(seed_path: Path | str = SEED_TICKERS_PATH) -> list[dict]:
    seed_path = Path(seed_path)
    if not seed_path.exists():
        return []
    with open(seed_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return raw.get("tickers", [])


def ensure_seeded(
    watchlist_path: Path | str = DEFAULT_WATCHLIST_PATH,
    seed_path: Path | str = SEED_TICKERS_PATH,
    data_store_root: Path | str = DEFAULT_DATA_STORE_ROOT,
) -> list[WatchlistEntry]:
    """Se a watchlist ainda não existir, popula a partir da semente do setor elétrico."""
    existing = load_watchlist(watchlist_path)
    if existing:
        return existing
    seed = seed_tickers(seed_path)
    if not seed:
        return []
    symbols = [item["symbol"] for item in seed]
    tags_by_symbol = {item["symbol"]: item.get("tags", []) for item in seed}
    added, _failed = add_tickers(
        symbols, tags_by_symbol=tags_by_symbol, watchlist_path=watchlist_path, data_store_root=data_store_root
    )
    return added


def _collect_fundamentals(ticker: str, data_store_root: Path | str) -> None:
    """Busca fundamentalista e grava uma linha no CSV do ticker. Falha em silêncio
    (loga mas não bloqueia) — fundamentalista ausente não deve impedir a entrada do
    ticker na watchlist, já que a análise de ação/derivativo não depende dela."""
    try:
        snapshot = fundamentals_provider.get_fundamental_snapshot(ticker)
    except Exception:
        return
    row = snapshot.__dict__ | {"collected_at": _now_iso()}
    append_row(fundamentals_csv_path(data_store_root, ticker), row)


def add_tickers(
    tickers: list[str],
    tags: Optional[list] = None,
    tags_by_symbol: Optional[dict] = None,
    watchlist_path: Path | str = DEFAULT_WATCHLIST_PATH,
    data_store_root: Path | str = DEFAULT_DATA_STORE_ROOT,
    collect_fundamentals: bool = True,
) -> tuple[list[WatchlistEntry], dict[str, str]]:
    """Adiciona um ou mais tickers à watchlist, buscando os dados sozinho.

    `tags` aplica a mesma lista de tags a todos os tickers passados; `tags_by_symbol`
    permite tags diferentes por ticker (usado por `ensure_seeded`). Tickers já
    presentes na watchlist são atualizados (preço/elegibilidade), não duplicados.
    Devolve (entradas adicionadas/atualizadas, {ticker_inválido: motivo}).
    """
    valid_infos, invalid = validate_symbols(tickers)
    existing = load_watchlist(watchlist_path)
    by_symbol = {entry.symbol: entry for entry in existing}

    added_or_updated: list[WatchlistEntry] = []
    now = _now_iso()
    for info in valid_infos:
        info: SymbolInfo
        entry_tags = (tags_by_symbol or {}).get(info.symbol, tags or [])
        previous = by_symbol.get(info.symbol)
        entry = WatchlistEntry(
            symbol=info.symbol,
            name=info.name,
            tags=list(entry_tags) if entry_tags else (previous.tags if previous else []),
            currency=info.currency,
            exchange=info.exchange,
            elegivel_derivativo=info.has_options,
            last_price=info.current_price,
            added_at=previous.added_at if previous else now,
            last_updated=now,
        )
        by_symbol[info.symbol] = entry
        added_or_updated.append(entry)
        if collect_fundamentals:
            _collect_fundamentals(info.symbol, data_store_root)

    save_watchlist(list(by_symbol.values()), watchlist_path)
    return added_or_updated, invalid


def remove_ticker(symbol: str, watchlist_path: Path | str = DEFAULT_WATCHLIST_PATH) -> bool:
    existing = load_watchlist(watchlist_path)
    remaining = [entry for entry in existing if entry.symbol != symbol]
    removed = len(remaining) != len(existing)
    if removed:
        save_watchlist(remaining, watchlist_path)
    return removed
