import pandas as pd
import pytest
import yaml

from finstrategy.data import watchlist as wl
from finstrategy.data.base import SymbolInfo
from finstrategy.data.fundamentals_provider import FundamentalSnapshot


def _fake_symbol_info(symbol, has_options=False, price=10.0, name=None):
    return SymbolInfo(symbol=symbol, name=name or symbol, current_price=price, currency="USD", exchange="NYQ", has_options=has_options)


def _patch_validate_symbols(monkeypatch, responses: dict):
    """`responses`: {ticker: SymbolInfo | "fail"}."""

    def fake_validate_symbols(tickers):
        valid, invalid = [], {}
        for ticker in tickers:
            response = responses.get(ticker, "fail")
            if response == "fail":
                invalid[ticker] = "ticker não encontrado (mock)"
            else:
                valid.append(response)
        return valid, invalid

    monkeypatch.setattr(wl, "validate_symbols", fake_validate_symbols)


def _patch_fundamentals(monkeypatch, should_fail=False):
    def fake_snapshot(ticker):
        if should_fail:
            raise RuntimeError("sem fundamentalista (mock)")
        return FundamentalSnapshot(
            ticker=ticker, name=ticker, sector="Utilities", industry="Regulated Electric", pe_ratio=15.0,
            forward_pe=14.0, price_to_book=1.5, roe=0.1, debt_to_equity=100.0, dividend_yield_ttm=0.03,
            revenue_growth_yoy=0.02, free_cash_flow=1_000_000.0, market_cap=5_000_000_000.0, profit_margin=0.15,
        )

    monkeypatch.setattr(wl.fundamentals_provider, "get_fundamental_snapshot", fake_snapshot)


def test_add_tickers_writes_watchlist_and_fundamentals(tmp_path, monkeypatch):
    watchlist_path = tmp_path / "watchlist.yaml"
    data_store_root = tmp_path / "data_store"
    _patch_validate_symbols(monkeypatch, {
        "AAPL": _fake_symbol_info("AAPL", has_options=True, price=250.0, name="Apple Inc."),
        "TAEE11.SA": _fake_symbol_info("TAEE11.SA", has_options=False, price=46.0),
    })
    _patch_fundamentals(monkeypatch)

    added, invalid = wl.add_tickers(["AAPL", "TAEE11.SA", "NOPE"], tags=["teste"], watchlist_path=watchlist_path, data_store_root=data_store_root)

    assert {e.symbol for e in added} == {"AAPL", "TAEE11.SA"}
    assert "NOPE" in invalid
    aapl = next(e for e in added if e.symbol == "AAPL")
    assert aapl.elegivel_derivativo is True
    assert aapl.last_price == 250.0

    loaded = wl.load_watchlist(watchlist_path)
    assert {e.symbol for e in loaded} == {"AAPL", "TAEE11.SA"}

    fundamentals_frame = pd.read_csv(data_store_root / "fundamentals" / "AAPL.csv")
    assert fundamentals_frame.iloc[0]["sector"] == "Utilities"


def test_add_tickers_does_not_fail_when_fundamentals_unavailable(tmp_path, monkeypatch):
    watchlist_path = tmp_path / "watchlist.yaml"
    data_store_root = tmp_path / "data_store"
    _patch_validate_symbols(monkeypatch, {"AAPL": _fake_symbol_info("AAPL")})
    _patch_fundamentals(monkeypatch, should_fail=True)

    added, invalid = wl.add_tickers(["AAPL"], watchlist_path=watchlist_path, data_store_root=data_store_root)

    assert [e.symbol for e in added] == ["AAPL"]
    assert not invalid
    assert not (data_store_root / "fundamentals" / "AAPL.csv").exists()


def test_add_tickers_updates_existing_entry_preserving_added_at(tmp_path, monkeypatch):
    watchlist_path = tmp_path / "watchlist.yaml"
    data_store_root = tmp_path / "data_store"
    _patch_validate_symbols(monkeypatch, {"AAPL": _fake_symbol_info("AAPL", price=100.0)})
    _patch_fundamentals(monkeypatch)
    first, _ = wl.add_tickers(["AAPL"], watchlist_path=watchlist_path, data_store_root=data_store_root)
    first_added_at = first[0].added_at

    _patch_validate_symbols(monkeypatch, {"AAPL": _fake_symbol_info("AAPL", price=105.0)})
    second, _ = wl.add_tickers(["AAPL"], watchlist_path=watchlist_path, data_store_root=data_store_root)

    assert len(wl.load_watchlist(watchlist_path)) == 1  # não duplicou
    assert second[0].last_price == 105.0
    assert second[0].added_at == first_added_at  # preserva data de adição original


def test_remove_ticker(tmp_path, monkeypatch):
    watchlist_path = tmp_path / "watchlist.yaml"
    data_store_root = tmp_path / "data_store"
    _patch_validate_symbols(monkeypatch, {"AAPL": _fake_symbol_info("AAPL"), "DUK": _fake_symbol_info("DUK")})
    _patch_fundamentals(monkeypatch)
    wl.add_tickers(["AAPL", "DUK"], watchlist_path=watchlist_path, data_store_root=data_store_root)

    assert wl.remove_ticker("AAPL", watchlist_path) is True
    assert wl.remove_ticker("AAPL", watchlist_path) is False  # já removido
    remaining = wl.load_watchlist(watchlist_path)
    assert [e.symbol for e in remaining] == ["DUK"]


def test_ensure_seeded_populates_only_when_watchlist_missing(tmp_path, monkeypatch):
    watchlist_path = tmp_path / "watchlist.yaml"
    data_store_root = tmp_path / "data_store"
    seed_path = tmp_path / "seed.yaml"
    seed_path.write_text(yaml.safe_dump({"tickers": [{"symbol": "AAPL", "tags": ["teste"]}]}), encoding="utf-8")

    _patch_validate_symbols(monkeypatch, {"AAPL": _fake_symbol_info("AAPL", has_options=True)})
    _patch_fundamentals(monkeypatch)

    first = wl.ensure_seeded(watchlist_path=watchlist_path, seed_path=seed_path, data_store_root=data_store_root)
    assert [e.symbol for e in first] == ["AAPL"]
    assert first[0].tags == ["teste"]

    # na segunda chamada, a watchlist já existe: não deve rodar a semente de novo
    def fail_if_called(tickers):
        raise AssertionError("ensure_seeded não deveria chamar validate_symbols de novo")

    monkeypatch.setattr(wl, "validate_symbols", fail_if_called)
    second = wl.ensure_seeded(watchlist_path=watchlist_path, seed_path=seed_path, data_store_root=data_store_root)
    assert [e.symbol for e in second] == ["AAPL"]


def test_ensure_seeded_returns_empty_when_no_seed_file(tmp_path):
    watchlist_path = tmp_path / "watchlist.yaml"
    missing_seed = tmp_path / "does_not_exist.yaml"
    assert wl.ensure_seeded(watchlist_path=watchlist_path, seed_path=missing_seed, data_store_root=tmp_path / "data_store") == []


def test_load_watchlist_missing_file_returns_empty_list(tmp_path):
    assert wl.load_watchlist(tmp_path / "nope.yaml") == []
