"""Busca e validação de ticker — qualquer ação do Yahoo Finance, não só uma lista fixa.

`search_symbols` alimenta a tela "Ações" do frontend (busca por nome/código, resultado
em tabela com checkbox). `validate_symbol` é usado quando o usuário digita um ticker
direto (ou cola uma lista), para confirmar que o Yahoo reconhece o código antes de
adicioná-lo à watchlist.
"""
from __future__ import annotations

import logging

import yfinance as yf

from .base import SymbolInfo, SymbolMatch
from .errors import MarketDataError

logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def search_symbols(query: str, max_results: int = 10) -> list[SymbolMatch]:
    """Busca por nome da empresa ou código parcial (ex. 'eletrica', 'PETR', 'duke energy').

    Usa o mesmo serviço de busca/autocomplete que alimenta a busca do site do Yahoo
    Finance. Devolve uma lista vazia (não lança erro) se nada for encontrado ou se a
    busca falhar — quem chama decide como mostrar "nenhum resultado".
    """
    query = (query or "").strip()
    if not query:
        return []
    try:
        results = yf.Search(query, max_results=max_results, news_count=0, include_cb=False).quotes
    except Exception:
        return []
    matches = []
    for item in results:
        symbol = item.get("symbol")
        if not symbol:
            continue
        matches.append(
            SymbolMatch(
                symbol=symbol,
                name=item.get("longname") or item.get("shortname") or symbol,
                exchange=item.get("exchDisp") or item.get("exchange") or "?",
                quote_type=item.get("typeDisp") or item.get("quoteType") or "?",
            )
        )
    return matches


def validate_symbol(ticker: str) -> SymbolInfo:
    """Confirma que o Yahoo reconhece `ticker` e devolve metadados mínimos.

    Levanta MarketDataError com mensagem clara se o ticker não existir ou não tiver
    preço disponível — o chamador (watchlist.add_tickers) usa isso para reportar quais
    tickers digitados pelo usuário falharam, sem travar os que são válidos.
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        raise MarketDataError("ticker vazio")
    handle = yf.Ticker(ticker)
    price = 0.0
    try:
        price = float(handle.fast_info["lastPrice"])
    except Exception:
        price = 0.0
    if not price or price <= 0:
        try:
            history = handle.history(period="5d", auto_adjust=True)
            if history is not None and not history.empty:
                price = float(history["Close"].dropna().iloc[-1])
        except Exception:
            pass
    if not price or price <= 0:
        raise MarketDataError(f"ticker '{ticker}' não encontrado (ou sem preço) no Yahoo Finance")

    info = {}
    try:
        info = handle.get_info() or {}
    except Exception:
        info = {}

    try:
        has_options = bool(handle.options)
    except Exception:
        has_options = False

    return SymbolInfo(
        symbol=ticker,
        name=info.get("longName") or info.get("shortName") or ticker,
        current_price=price,
        currency=info.get("currency") or "?",
        exchange=info.get("exchange") or "?",
        has_options=has_options,
    )


def validate_symbols(tickers: list[str]) -> tuple[list[SymbolInfo], dict[str, str]]:
    """Valida vários tickers de uma vez. Devolve (válidos, {ticker_inválido: motivo})."""
    valid: list[SymbolInfo] = []
    invalid: dict[str, str] = {}
    for raw in tickers:
        ticker = (raw or "").strip()
        if not ticker:
            continue
        try:
            valid.append(validate_symbol(ticker))
        except MarketDataError as reason:
            invalid[ticker] = str(reason)
        except Exception as unexpected:  # qualquer outra falha de rede/parsing
            invalid[ticker] = f"erro ao validar: {unexpected}"
    return valid, invalid
