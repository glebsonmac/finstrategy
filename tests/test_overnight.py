import pandas as pd
import yaml

from finstrategy.collector import overnight
from finstrategy.data.base import SymbolInfo
from finstrategy.data.errors import MarketDataError
from finstrategy.data.fundamentals_provider import FundamentalSnapshot
from finstrategy.data.watchlist import WatchlistEntry, save_watchlist


def _fake_symbol_info(ticker: str) -> SymbolInfo:
    return SymbolInfo(symbol=ticker, name=f"{ticker} Inc.", current_price=100.0, currency="USD", exchange="NMS", has_options=True)


def _fake_snapshot(ticker: str) -> FundamentalSnapshot:
    return FundamentalSnapshot(
        ticker=ticker, name=f"{ticker} Inc.", sector="Technology", industry="Software", pe_ratio=20.0,
        forward_pe=18.0, price_to_book=5.0, roe=0.2, debt_to_equity=50.0, dividend_yield_ttm=0.01,
        revenue_growth_yoy=0.1, free_cash_flow=1_000_000.0, market_cap=1_000_000_000.0, profit_margin=0.2,
    )


# --- resolve_tickers: prioridade explícita > watchlist > semente top-20 EUA ---

def test_resolve_tickers_prefers_explicit_list(tmp_path):
    watchlist_path = tmp_path / "watchlist.yaml"
    save_watchlist([WatchlistEntry(symbol="DUK")], watchlist_path)
    result = overnight.resolve_tickers(["AAPL", "MSFT"], watchlist_path=watchlist_path)
    assert result == ["AAPL", "MSFT"]


def test_resolve_tickers_falls_back_to_watchlist_when_no_explicit_list(tmp_path):
    watchlist_path = tmp_path / "watchlist.yaml"
    save_watchlist([WatchlistEntry(symbol="DUK"), WatchlistEntry(symbol="SPY")], watchlist_path)
    result = overnight.resolve_tickers(None, watchlist_path=watchlist_path)
    assert result == ["DUK", "SPY"]


def test_resolve_tickers_falls_back_to_seed_when_watchlist_empty(tmp_path):
    watchlist_path = tmp_path / "watchlist.yaml"  # não existe
    seed_path = tmp_path / "seed.yaml"
    seed_path.write_text(yaml.safe_dump({"tickers": [{"symbol": "NVDA"}, {"symbol": "AAPL"}]}), encoding="utf-8")
    result = overnight.resolve_tickers(None, watchlist_path=watchlist_path, seed_path=seed_path)
    assert result == ["NVDA", "AAPL"]


def test_resolve_tickers_empty_when_nothing_available(tmp_path):
    result = overnight.resolve_tickers(None, watchlist_path=tmp_path / "nope.yaml", seed_path=tmp_path / "nope2.yaml")
    assert result == []


# --- collect_essential_row ---

def test_collect_essential_row_ok_with_fundamentals(monkeypatch):
    monkeypatch.setattr(overnight.symbol_search, "validate_symbol", lambda t: _fake_symbol_info(t))
    monkeypatch.setattr(overnight.fundamentals_provider, "get_fundamental_snapshot", lambda t: _fake_snapshot(t))
    row = overnight.collect_essential_row("AAPL")
    assert row["status"] == "ok"
    assert row["price"] == 100.0
    assert row["pe_ratio"] == 20.0
    assert row["elegivel_derivativo"] is True


def test_collect_essential_row_invalid_ticker_reports_status_without_raising(monkeypatch):
    def fail_validate(ticker):
        raise MarketDataError(f"ticker '{ticker}' não encontrado")

    monkeypatch.setattr(overnight.symbol_search, "validate_symbol", fail_validate)
    row = overnight.collect_essential_row("NOPE")
    assert row["status"].startswith("sem cotação")
    assert "price" not in row


def test_collect_essential_row_missing_fundamentals_keeps_quote_data(monkeypatch):
    monkeypatch.setattr(overnight.symbol_search, "validate_symbol", lambda t: _fake_symbol_info(t))

    def fail_fundamentals(_ticker):
        raise RuntimeError("sem fundamentalista")

    monkeypatch.setattr(overnight.fundamentals_provider, "get_fundamental_snapshot", fail_fundamentals)
    row = overnight.collect_essential_row("AAPL")
    assert row["price"] == 100.0  # cotação não se perde
    assert "ok" in row["status"]
    assert "sem fundamentalista" in row["status"]


# --- run_overnight_collection ---

def test_run_overnight_collection_writes_dated_and_latest_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(overnight.symbol_search, "validate_symbol", lambda t: _fake_symbol_info(t))
    monkeypatch.setattr(overnight.fundamentals_provider, "get_fundamental_snapshot", lambda t: _fake_snapshot(t))

    logs = []
    dated_path = overnight.run_overnight_collection(["AAPL", "MSFT"], output_dir=tmp_path, log=logs.append)

    today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
    assert dated_path == tmp_path / f"{today_str}_essencial.csv"
    assert dated_path.exists()
    latest_path = tmp_path / "latest.csv"
    assert latest_path.exists()

    dated_frame = pd.read_csv(dated_path)
    latest_frame = pd.read_csv(latest_path)
    assert list(dated_frame["ticker"]) == ["AAPL", "MSFT"]
    assert dated_frame.equals(latest_frame)
    assert any("2/2" in line for line in logs)


def test_run_overnight_collection_one_bad_ticker_does_not_stop_the_batch(tmp_path, monkeypatch):
    def validate(ticker):
        if ticker == "BAD":
            raise MarketDataError("não encontrado")
        return _fake_symbol_info(ticker)

    monkeypatch.setattr(overnight.symbol_search, "validate_symbol", validate)
    monkeypatch.setattr(overnight.fundamentals_provider, "get_fundamental_snapshot", lambda t: _fake_snapshot(t))

    dated_path = overnight.run_overnight_collection(["AAPL", "BAD"], output_dir=tmp_path, log=lambda *_: None)
    frame = pd.read_csv(dated_path)
    statuses = dict(zip(frame["ticker"], frame["status"]))
    assert statuses["AAPL"] == "ok"
    assert statuses["BAD"].startswith("sem cotação")


def test_run_overnight_collection_latest_csv_gets_overwritten_each_run(tmp_path, monkeypatch):
    monkeypatch.setattr(overnight.symbol_search, "validate_symbol", lambda t: _fake_symbol_info(t))
    monkeypatch.setattr(overnight.fundamentals_provider, "get_fundamental_snapshot", lambda t: _fake_snapshot(t))

    overnight.run_overnight_collection(["AAPL"], output_dir=tmp_path, log=lambda *_: None)
    overnight.run_overnight_collection(["MSFT"], output_dir=tmp_path, log=lambda *_: None)

    latest_frame = pd.read_csv(tmp_path / "latest.csv")
    assert list(latest_frame["ticker"]) == ["MSFT"]  # última rodada venceu


# --- main (CLI) ---

def test_main_returns_nonzero_when_no_tickers_resolved(tmp_path, monkeypatch):
    monkeypatch.setattr(overnight, "DEFAULT_WATCHLIST_PATH", tmp_path / "nope.yaml")
    monkeypatch.setattr(overnight, "US_TOP20_SEED_PATH", tmp_path / "nope_seed.yaml")
    exit_code = overnight.main([])
    assert exit_code == 1


def test_main_with_explicit_tickers_runs_successfully(tmp_path, monkeypatch):
    monkeypatch.setattr(overnight.symbol_search, "validate_symbol", lambda t: _fake_symbol_info(t))
    monkeypatch.setattr(overnight.fundamentals_provider, "get_fundamental_snapshot", lambda t: _fake_snapshot(t))
    exit_code = overnight.main(["--tickers", "AAPL,MSFT", "--output-dir", str(tmp_path)])
    assert exit_code == 0
    assert (tmp_path / "latest.csv").exists()
