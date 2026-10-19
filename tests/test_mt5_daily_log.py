import pandas as pd
import pytest

from finstrategy.collector import mt5_daily_log
from finstrategy.collector.csv_store import read_csv
from finstrategy.data.errors import MarketDataError
from finstrategy.data.fundamentals_provider import FundamentalSnapshot


def _fake_snapshot(ticker: str) -> FundamentalSnapshot:
    return FundamentalSnapshot(
        ticker=ticker, name=f"{ticker} Inc.", sector="Utilities", industry="Regulated Electric", pe_ratio=18.0,
        forward_pe=17.0, price_to_book=2.0, roe=0.11, debt_to_equity=130.0, dividend_yield_ttm=0.035,
        revenue_growth_yoy=0.04, free_cash_flow=-500_000.0, market_cap=90_000_000_000.0, profit_margin=0.15,
    )


class FakeProvider:
    def __init__(self, close=118.0, with_options=True, expiration=None, raise_price=False):
        self.close = close
        self.with_options = with_options
        self.expiration = expiration or pd.Timestamp.today().normalize() + pd.Timedelta(days=45)
        self.raise_price = raise_price

    def download_price_history(self, ticker, start_date):
        if self.raise_price:
            raise MarketDataError("sem histórico no MT5")
        index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=3)
        return pd.Series([self.close - 1, self.close - 0.5, self.close], index=index)

    def list_option_expirations(self, ticker):
        if not self.with_options:
            raise MarketDataError("sem opções no MT5")
        return [self.expiration]

    def download_call_chain(self, ticker, expiration):
        return pd.DataFrame({
            "contractSymbol": [f"{ticker}_C"], "strike": [self.close * 1.05], "bid": [2.0], "ask": [2.2],
            "lastPrice": [2.1], "openInterest": [40], "volume": [3], "impliedVolatility": [float("nan")],
        })

    def download_put_chain(self, ticker, expiration):
        return pd.DataFrame({
            "contractSymbol": [f"{ticker}_P"], "strike": [self.close * 0.9], "bid": [1.0], "ask": [1.2],
            "lastPrice": [1.1], "openInterest": [60], "volume": [5], "impliedVolatility": [float("nan")],
        })


# --- collect_stock_row ---

def test_collect_stock_row_ok_with_fundamentals(monkeypatch):
    monkeypatch.setattr(mt5_daily_log.fundamentals_provider, "get_fundamental_snapshot", lambda t: _fake_snapshot(t))
    row = mt5_daily_log.collect_stock_row("DUK", FakeProvider(close=118.0))
    assert row["status"] == "ok"
    assert row["close"] == 118.0
    assert row["sector"] == "Utilities"
    assert row["date"] == pd.Timestamp.today().normalize().strftime("%Y-%m-%d")


def test_collect_stock_row_reports_status_without_raising_when_price_fails():
    row = mt5_daily_log.collect_stock_row("NOPE", FakeProvider(raise_price=True))
    assert row["status"].startswith("sem preço")
    assert "close" not in row or row.get("close") is None


def test_collect_stock_row_ok_even_when_fundamentals_fail(monkeypatch):
    def fail(ticker):
        raise RuntimeError("Yahoo fora do ar")

    monkeypatch.setattr(mt5_daily_log.fundamentals_provider, "get_fundamental_snapshot", fail)
    row = mt5_daily_log.collect_stock_row("DUK", FakeProvider(close=118.0))
    assert row["status"].startswith("ok (sem fundamentalista")
    assert row["close"] == 118.0


# --- collect_stock_history_rows (backfill de N anos) ---

def test_collect_stock_history_rows_returns_one_row_per_day(monkeypatch):
    monkeypatch.setattr(mt5_daily_log.fundamentals_provider, "get_fundamental_snapshot", lambda t: _fake_snapshot(t))
    rows = mt5_daily_log.collect_stock_history_rows("DUK", FakeProvider(close=118.0), years_back=2)
    assert len(rows) == 3  # o FakeProvider sempre devolve 3 dias, independente do start_date pedido
    assert all(row["status"] == "ok" for row in rows)
    assert [row["close"] for row in rows] == [117.0, 117.5, 118.0]


def test_collect_stock_history_rows_only_attaches_fundamentals_to_latest_row(monkeypatch):
    monkeypatch.setattr(mt5_daily_log.fundamentals_provider, "get_fundamental_snapshot", lambda t: _fake_snapshot(t))
    rows = mt5_daily_log.collect_stock_history_rows("DUK", FakeProvider(close=118.0), years_back=1)
    assert "sector" not in rows[0] or rows[0].get("sector") is None
    assert rows[-1]["sector"] == "Utilities"


def test_collect_stock_history_rows_reports_status_without_raising_when_price_fails():
    rows = mt5_daily_log.collect_stock_history_rows("NOPE", FakeProvider(raise_price=True), years_back=2)
    assert len(rows) == 1
    assert rows[0]["status"].startswith("sem preço")


def test_collect_stock_history_rows_ok_even_when_fundamentals_fail(monkeypatch):
    def fail(ticker):
        raise RuntimeError("Yahoo fora do ar")

    monkeypatch.setattr(mt5_daily_log.fundamentals_provider, "get_fundamental_snapshot", fail)
    rows = mt5_daily_log.collect_stock_history_rows("DUK", FakeProvider(close=118.0), years_back=1)
    assert rows[-1]["status"].startswith("ok (sem fundamentalista")
    assert rows[0]["status"] == "ok"


# --- collect_option_rows ---

def test_collect_option_rows_includes_calls_and_puts():
    provider = FakeProvider(close=100.0)
    rows = mt5_daily_log.collect_option_rows("DUK", provider, underlying_price=100.0, date_str="2026-10-19")
    types = sorted(r["option_type"] for r in rows)
    assert types == ["CALL", "PUT"]
    put_row = next(r for r in rows if r["option_type"] == "PUT")
    assert put_row["premium"] == pytest.approx(1.2)  # ask ao vivo
    assert put_row["quote_source"] == "ask ao vivo"
    assert put_row["underlying_price"] == 100.0
    assert put_row["expiration"] == provider.expiration.strftime("%Y-%m-%d")


def test_collect_option_rows_empty_when_no_options():
    provider = FakeProvider(with_options=False)
    rows = mt5_daily_log.collect_option_rows("TAEE11.SA", provider, underlying_price=35.0, date_str="2026-10-19")
    assert rows == []


def test_collect_option_rows_logs_the_reason_when_no_options():
    # Sem isso, "sem opções" vira uma lista vazia silenciosa — impossível saber se é
    # porque o corretor não tem opções desse ativo ou se o padrão de símbolo está
    # errado. Foi exatamente o que faltou pra diagnosticar o caso real (MT5 trouxe
    # ações mas não opções) — o motivo agora vai pro log.
    messages = []
    provider = FakeProvider(with_options=False)
    mt5_daily_log.collect_option_rows("TAEE11.SA", provider, underlying_price=35.0, date_str="2026-10-19", log=messages.append)
    assert len(messages) == 1
    assert "sem opções" in messages[0]
    assert "sem opções no MT5" in messages[0]  # a MarketDataError original (ver FakeProvider) aparece no log


# --- run_daily_collection: append-only + sort by date ---

def test_run_daily_collection_appends_without_overwriting(tmp_path, monkeypatch):
    monkeypatch.setattr(mt5_daily_log.fundamentals_provider, "get_fundamental_snapshot", lambda t: _fake_snapshot(t))
    provider = FakeProvider(close=118.0)

    mt5_daily_log.run_daily_collection(["DUK"], provider, data_store_root=tmp_path, log=lambda *_: None)
    mt5_daily_log.run_daily_collection(["DUK"], provider, data_store_root=tmp_path, log=lambda *_: None)

    stocks = read_csv(mt5_daily_log.stock_log_path(tmp_path))
    options = read_csv(mt5_daily_log.option_log_path(tmp_path))
    assert len(stocks) == 2  # duas rodadas -> duas linhas (append puro, sem deduplicar)
    assert len(options) == 4  # 2 rodadas x (1 call + 1 put)


def test_run_daily_collection_with_years_back_appends_history_and_options_once(tmp_path, monkeypatch):
    monkeypatch.setattr(mt5_daily_log.fundamentals_provider, "get_fundamental_snapshot", lambda t: _fake_snapshot(t))
    provider = FakeProvider(close=118.0)

    mt5_daily_log.run_daily_collection(["DUK"], provider, data_store_root=tmp_path, log=lambda *_: None, years_back=2)

    stocks = read_csv(mt5_daily_log.stock_log_path(tmp_path))
    options = read_csv(mt5_daily_log.option_log_path(tmp_path))
    assert len(stocks) == 3  # backfill: 1 linha por dia devolvido pelo provider
    assert len(options) == 2  # opções só pra linha mais recente (1 call + 1 put), não por dia histórico
    assert stocks.iloc[-1]["sector"] == "Utilities"
    assert pd.isna(stocks.iloc[0]["sector"])


def test_run_daily_collection_sorts_by_date_ascending(tmp_path, monkeypatch):
    monkeypatch.setattr(mt5_daily_log.fundamentals_provider, "get_fundamental_snapshot", lambda t: _fake_snapshot(t))

    class DatedProvider(FakeProvider):
        def __init__(self, close, date):
            super().__init__(close=close)
            self._date = date

        def download_price_history(self, ticker, start_date):
            return pd.Series([self.close], index=[self._date])

    stocks_path = mt5_daily_log.stock_log_path(tmp_path)
    mt5_daily_log.run_daily_collection(["DUK"], DatedProvider(118.0, pd.Timestamp("2026-10-15")), data_store_root=tmp_path, log=lambda *_: None)
    mt5_daily_log.run_daily_collection(["DUK"], DatedProvider(119.0, pd.Timestamp("2026-10-13")), data_store_root=tmp_path, log=lambda *_: None)
    mt5_daily_log.run_daily_collection(["DUK"], DatedProvider(117.0, pd.Timestamp("2026-10-14")), data_store_root=tmp_path, log=lambda *_: None)

    frame = read_csv(stocks_path)
    assert list(frame["date"]) == ["2026-10-13", "2026-10-14", "2026-10-15"]
    assert list(frame["close"]) == [119.0, 117.0, 118.0]


def test_run_daily_collection_skips_options_for_tickers_without_price(tmp_path):
    stocks_path = mt5_daily_log.stock_log_path(tmp_path)
    options_path = mt5_daily_log.option_log_path(tmp_path)
    mt5_daily_log.run_daily_collection(["BROKEN"], FakeProvider(raise_price=True), data_store_root=tmp_path, log=lambda *_: None)

    stocks = read_csv(stocks_path)
    assert stocks.iloc[0]["status"].startswith("sem preço")
    assert not options_path.exists()


def test_stock_and_option_log_paths(tmp_path):
    assert mt5_daily_log.stock_log_path(tmp_path) == tmp_path / "mt5" / "acoes_diario.csv"
    assert mt5_daily_log.option_log_path(tmp_path) == tmp_path / "mt5" / "opcoes_diario.csv"
