import threading

import numpy as np
import pandas as pd
import pytest

from finstrategy.collector import scheduler as scheduler_module
from finstrategy.collector.csv_store import market_csv_path, model_runs_csv_path, read_csv
from finstrategy.collector.scheduler import (
    CollectorConfig,
    collect_price_snapshot,
    run_collection_cycle,
    run_forever,
    run_model_pipeline_for_ticker,
)
from finstrategy.config_model import AnalysisSettings
from finstrategy.data.errors import MarketDataError
from finstrategy.data.watchlist import WatchlistEntry, save_watchlist


def _synthetic_closes(days: int = 320, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0003, 0.015, days)
    prices = 60.0 * np.exp(np.cumsum(log_returns))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=days)
    return pd.Series(prices, index=dates)


class FakeProvider:
    def __init__(self, closes: pd.Series, current_price: float, with_options: bool, expiration: pd.Timestamp, risk_free_rate: float = 0.05):
        self._closes = closes
        self._current_price = current_price
        self._with_options = with_options
        self._expiration = expiration
        self._risk_free_rate = risk_free_rate

    def download_price_history(self, ticker, start_date):
        return self._closes

    def get_current_stock_price(self, ticker, fallback_price):
        return self._current_price

    def get_trailing_dividend_yield(self, ticker, current_price):
        return 0.0

    def get_risk_free_rate(self):
        return self._risk_free_rate

    def list_option_expirations(self, ticker):
        if not self._with_options:
            raise MarketDataError("sem opções")
        return [self._expiration]

    def download_put_chain(self, ticker, expiration):
        strike = self._current_price * 0.9
        return pd.DataFrame({
            "contractSymbol": ["PUT1"], "strike": [strike], "bid": [1.0], "ask": [1.2],
            "lastPrice": [1.1], "openInterest": [500], "volume": [10], "impliedVolatility": [0.3],
        })


class FailingProvider:
    def download_price_history(self, ticker, start_date):
        raise MarketDataError("sem histórico de preços no Yahoo")

    def get_current_stock_price(self, ticker, fallback_price):
        return fallback_price

    def get_risk_free_rate(self):
        return 0.05


def _fast_settings(**overrides) -> AnalysisSettings:
    base = dict(entry_window_days=5, dqn_training_rounds=3, dqn_hidden_units=8, monte_carlo_paths=1_000, minimum_open_interest=100)
    base.update(overrides)
    return AnalysisSettings(**base)


def test_collect_price_snapshot_ok():
    closes = _synthetic_closes()
    provider = FakeProvider(closes, float(closes.iloc[-1]), with_options=False, expiration=pd.Timestamp.today() + pd.Timedelta(days=45))
    entry = WatchlistEntry(symbol="FAKE")
    row = collect_price_snapshot(entry, provider, pd.Timestamp.today().normalize())
    assert row["status"] == "ok"
    assert row["price"] == pytest.approx(float(closes.iloc[-1]))


def test_collect_price_snapshot_reports_market_data_error():
    entry = WatchlistEntry(symbol="BROKEN")
    row = collect_price_snapshot(entry, FailingProvider(), pd.Timestamp.today().normalize())
    assert row["status"].startswith("erro:")
    assert row["price"] is None


def test_run_model_pipeline_without_derivative_eligibility():
    closes = _synthetic_closes()
    provider = FakeProvider(closes, float(closes.iloc[-1]), with_options=False, expiration=pd.Timestamp.today() + pd.Timedelta(days=45))
    entry = WatchlistEntry(symbol="FAKE", elegivel_derivativo=False)
    settings = _fast_settings()
    row = run_model_pipeline_for_ticker(entry, provider, settings, risk_free_rate=0.05, today=pd.Timestamp.today().normalize())
    assert row["status"] == "não elegível: sem opções no Yahoo"
    assert "entry_dqn_sinal" in row


def test_run_model_pipeline_with_derivative_eligibility():
    closes = _synthetic_closes()
    current_price = float(closes.iloc[-1])
    expiration = pd.Timestamp.today().normalize() + pd.Timedelta(days=45)
    provider = FakeProvider(closes, current_price, with_options=True, expiration=expiration)
    entry = WatchlistEntry(symbol="FAKE", elegivel_derivativo=True)
    settings = _fast_settings()
    row = run_model_pipeline_for_ticker(entry, provider, settings, risk_free_rate=0.05, today=pd.Timestamp.today().normalize())
    assert row["status"] == "ok"
    assert "strike_recomendada" in row
    assert "entry_dqn_sinal" in row


def test_run_collection_cycle_writes_market_and_model_csv(tmp_path):
    watchlist_path = tmp_path / "watchlist.yaml"
    data_store_root = tmp_path / "data_store"
    closes = _synthetic_closes()
    current_price = float(closes.iloc[-1])
    expiration = pd.Timestamp.today().normalize() + pd.Timedelta(days=45)
    provider = FakeProvider(closes, current_price, with_options=True, expiration=expiration)

    save_watchlist([WatchlistEntry(symbol="FAKE", elegivel_derivativo=True)], watchlist_path)
    config = CollectorConfig(watchlist_path=watchlist_path, data_store_root=data_store_root, provider=provider)
    settings = _fast_settings()
    today = pd.Timestamp.today().normalize()

    run_collection_cycle(config, settings, risk_free_rate=0.05, run_model=True, today=today)

    market_frame = read_csv(market_csv_path(data_store_root, "FAKE", today))
    model_frame = read_csv(model_runs_csv_path(data_store_root, today))
    assert len(market_frame) == 1
    assert market_frame.iloc[0]["status"] == "ok"
    assert len(model_frame) == 1
    assert model_frame.iloc[0]["status"] == "ok"


def test_run_forever_stops_after_max_iterations_without_sleeping(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler_module, "any_market_open", lambda now, calendars: False)
    watchlist_path = tmp_path / "watchlist.yaml"
    save_watchlist([], watchlist_path)
    config = CollectorConfig(watchlist_path=watchlist_path, data_store_root=tmp_path / "data_store", provider=FailingProvider())
    settings = _fast_settings()

    # com any_market_open sempre False e max_iterations=1, o loop deve voltar rápido
    # (o break acontece antes do sleep, então não precisa de timeout nem thread)
    run_forever(config=config, settings=settings, stop_event=threading.Event(), max_iterations=1)
