from types import SimpleNamespace

import pytest

from finstrategy.data import symbol_search
from finstrategy.data.errors import MarketDataError


class _FakeFastInfo(dict):
    pass


class _FakeTicker:
    def __init__(self, symbol, price=None, info=None, has_options=True, history_rows=1):
        self._symbol = symbol
        self._price = price
        self._info = info or {}
        self._has_options = has_options
        self._history_rows = history_rows

    @property
    def fast_info(self):
        if self._price is None:
            raise KeyError("lastPrice")
        return _FakeFastInfo(lastPrice=self._price)

    def history(self, period="5d", auto_adjust=True):
        import pandas as pd

        if self._history_rows == 0 or self._price is None:
            return pd.DataFrame()
        return pd.DataFrame({"Close": [self._price] * self._history_rows})

    def get_info(self):
        return self._info

    @property
    def options(self):
        return ("2026-12-18",) if self._has_options else ()


def test_search_symbols_empty_query_returns_empty_list():
    assert symbol_search.search_symbols("") == []
    assert symbol_search.search_symbols("   ") == []


def test_search_symbols_parses_yfinance_results(monkeypatch):
    class FakeSearch:
        def __init__(self, query, max_results=10, news_count=0, include_cb=False):
            self.quotes = [
                {"symbol": "DUK", "longname": "Duke Energy Corporation", "exchDisp": "NYSE", "typeDisp": "Equity"},
                {"symbol": "NEE", "shortname": "NextEra Energy", "exchange": "NMS", "quoteType": "EQUITY"},
                {"shortname": "sem symbol, deve ser ignorado"},
            ]

    monkeypatch.setattr(symbol_search.yf, "Search", FakeSearch)
    matches = symbol_search.search_symbols("energy")
    assert [m.symbol for m in matches] == ["DUK", "NEE"]
    assert matches[0].name == "Duke Energy Corporation"
    assert matches[1].name == "NextEra Energy"  # cai para shortname quando não tem longname


def test_search_symbols_returns_empty_list_on_failure(monkeypatch):
    class BrokenSearch:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Yahoo fora do ar")

    monkeypatch.setattr(symbol_search.yf, "Search", BrokenSearch)
    assert symbol_search.search_symbols("qualquer coisa") == []


def test_validate_symbol_ok_with_options(monkeypatch):
    monkeypatch.setattr(symbol_search.yf, "Ticker", lambda t: _FakeTicker(t, price=117.5, info={"longName": "Duke Energy Corporation", "currency": "USD", "exchange": "NYQ"}, has_options=True))
    info = symbol_search.validate_symbol("duk")
    assert info.symbol == "DUK"  # normaliza para maiúsculas
    assert info.name == "Duke Energy Corporation"
    assert info.current_price == 117.5
    assert info.has_options is True


def test_validate_symbol_falls_back_to_history_when_fast_info_fails(monkeypatch):
    monkeypatch.setattr(symbol_search.yf, "Ticker", lambda t: _FakeTicker(t, price=None, has_options=False))

    class TickerWithHistoryOnly(_FakeTicker):
        @property
        def fast_info(self):
            raise KeyError("sem fast_info")

        def history(self, period="5d", auto_adjust=True):
            import pandas as pd
            return pd.DataFrame({"Close": [46.45]})

    monkeypatch.setattr(symbol_search.yf, "Ticker", lambda t: TickerWithHistoryOnly(t, price=46.45, has_options=False))
    info = symbol_search.validate_symbol("TAEE11.SA")
    assert info.current_price == 46.45
    assert info.has_options is False


def test_validate_symbol_raises_for_empty_ticker():
    with pytest.raises(MarketDataError):
        symbol_search.validate_symbol("")


def test_validate_symbol_raises_when_no_price_found(monkeypatch):
    monkeypatch.setattr(symbol_search.yf, "Ticker", lambda t: _FakeTicker(t, price=None, history_rows=0))
    with pytest.raises(MarketDataError, match="não encontrado"):
        symbol_search.validate_symbol("NOPE123XYZ")


def test_validate_symbols_splits_valid_and_invalid(monkeypatch):
    def fake_ticker(symbol):
        if symbol == "BROKEN":
            return _FakeTicker(symbol, price=None, history_rows=0)
        return _FakeTicker(symbol, price=10.0, info={"longName": symbol}, has_options=True)

    monkeypatch.setattr(symbol_search.yf, "Ticker", fake_ticker)
    valid, invalid = symbol_search.validate_symbols(["AAPL", "BROKEN", "", "DUK"])
    assert sorted(v.symbol for v in valid) == ["AAPL", "DUK"]
    assert "BROKEN" in invalid
