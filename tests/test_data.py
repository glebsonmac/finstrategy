"""Data module tests. They use a fake source, so they need no network."""

import json

import numpy as np
import pandas as pd
import pytest

from riskbt import config, data


def make_bars(ticker, dates, adj_close, **overrides):
    adj_close = np.asarray(adj_close, dtype=float)
    bars = pd.DataFrame({
        "date": pd.to_datetime(dates), "ticker": ticker, "open": adj_close, "high": adj_close,
        "low": adj_close, "close": adj_close, "adj_close": adj_close, "volume": 1000.0,
    })
    return bars.assign(**overrides)


def business_days(count, start="2020-01-01"):
    return pd.bdate_range(start, periods=count)


def yahoo_frame(closes):
    """Build a frame shaped like yf.download's output: (field, ticker) columns."""
    fields = {"Open": 1.0, "High": 1.01, "Low": 0.99, "Close": 1.0, "Adj Close": 0.9, "Volume": None}
    columns = {}
    for ticker, series in closes.items():
        for field, factor in fields.items():
            columns[(field, ticker)] = 500.0 if factor is None else series * factor
    frame = pd.DataFrame(columns)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["Price", "Ticker"])
    return frame.rename_axis("Date")


def fake_fetch(calls=None, unknown=()):
    def fetch(tickers, start, end):
        if calls is not None:
            calls.append(list(tickers))
        dates = business_days(30)
        closes = {t: pd.Series(np.linspace(10, 12, 30), index=dates) for t in tickers}
        for ticker in unknown:
            closes[ticker] = closes[ticker] * np.nan
        return yahoo_frame(closes)
    return fetch


def test_to_long_format_keeps_only_days_with_data():
    dates = business_days(3)
    raw = yahoo_frame({
        "A.SA": pd.Series([10.0, 11.0, 12.0], index=dates),
        "B.SA": pd.Series([np.nan, np.nan, 5.0], index=dates),
    })
    long = data.to_long_format(raw)
    assert list(long.columns) == data.PRICE_COLUMNS
    assert len(long[long.ticker == "A.SA"]) == 3
    assert len(long[long.ticker == "B.SA"]) == 1
    assert long.loc[long.ticker == "A.SA", "adj_close"].iloc[0] == pytest.approx(9.0)


def test_short_gap_is_filled_and_long_gap_is_kept():
    dates = business_days(20)
    prices = np.arange(100.0, 120.0)
    keep = np.ones(20, dtype=bool)
    keep[2:4] = False  # short gap
    keep[8:15] = False  # 7 days: too long to fill
    bars = make_bars("A", dates[keep], prices[keep])
    clean, report = data.validate_prices(bars, ["A"], dates)
    assert clean["A"].iloc[2] == clean["A"].iloc[3] == prices[1]
    assert clean["A"].iloc[8:15].isna().all()
    assert report.gaps_filled == {"A": 2}
    assert report.gaps_unfilled == {"A": 7}


def test_days_before_listing_and_after_delisting_are_not_gaps():
    dates = business_days(10)
    bars = make_bars("A", dates[3:7], [10.0, 11.0, 12.0, 13.0])
    clean, report = data.validate_prices(bars, ["A"], dates)
    assert clean["A"].iloc[:3].isna().all() and clean["A"].iloc[7:].isna().all()
    assert report.gaps_filled == {} and report.gaps_unfilled == {}


def test_inconsistent_prices_become_gaps():
    dates = business_days(6)
    prices = [10.0, 11.0, -5.0, 12.0, 13.0, 14.0]
    bars = make_bars("A", dates, prices)
    clean, report = data.validate_prices(bars, ["A"], dates)
    assert report.inconsistent_prices == {"A": 1}
    assert clean["A"].iloc[2] == 11.0


def test_close_outside_daily_range_is_inconsistent():
    dates = business_days(3)
    bars = make_bars("A", dates, [10.0, 11.0, 12.0])
    bars.loc[1, "high"] = 10.5
    _, report = data.validate_prices(bars, ["A"], dates)
    assert report.inconsistent_prices == {"A": 1}


def test_duplicates_and_off_calendar_rows_are_removed():
    dates = business_days(5)
    bars = make_bars("A", dates, [10.0, 11.0, 12.0, 13.0, 14.0])
    bars = pd.concat([bars, bars.iloc[[1]]], ignore_index=True)
    clean, report = data.validate_prices(bars, ["A"], dates[:4])
    assert report.duplicates_removed == 1
    assert report.off_calendar_rows_removed == 1
    assert len(clean) == 4


def test_extreme_returns_are_reported_but_not_altered():
    dates = business_days(4)
    bars = make_bars("A", dates, [10.0, 10.0, 20.0, 20.0])
    clean, report = data.validate_prices(bars, ["A"], dates)
    assert clean["A"].iloc[2] == 20.0
    assert report.extreme_returns == [
        {"ticker": "A", "date": dates[2].strftime("%Y-%m-%d"), "log_return": pytest.approx(np.log(2))}
    ]


def test_adjustment_factor_checks():
    dates = business_days(5)
    closes = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    good = make_bars("GOOD", dates, closes * [0.9, 0.9, 0.95, 0.95, 1.0], close=closes, high=10.0, low=10.0)
    above_one = make_bars("HIGH", dates, closes * 1.1, close=closes, high=10.0, low=10.0)
    falling = make_bars("FALL", dates, closes * [1.0, 1.0, 0.9, 0.9, 0.9], close=closes, high=10.0, low=10.0)
    bars = pd.concat([good, above_one, falling], ignore_index=True)
    _, report = data.validate_prices(bars, ["GOOD", "HIGH", "FALL"], dates)
    assert report.adjustment_issues == {"HIGH": 5, "FALL": 1}


def test_missing_ticker_is_reported():
    dates = business_days(3)
    _, report = data.validate_prices(make_bars("A", dates, [1.0, 2.0, 3.0]), ["A", "NOPE"], dates)
    assert report.missing_tickers == ["NOPE"]


def test_extraction_is_cached_per_date(tmp_path):
    calls = []
    args = (["A.SA", "B.SA"], "2020-01-01", "2021-01-01")
    first = data.download_prices(*args, "2026-01-01", tmp_path, fake_fetch(calls))
    second = data.download_prices(*args, "2026-01-01", tmp_path, fake_fetch(calls))
    assert calls == [["A.SA", "B.SA"]]
    pd.testing.assert_frame_equal(first, second)
    data.download_prices(*args, "2026-02-01", tmp_path, fake_fetch(calls))
    assert len(calls) == 2  # a new extraction date is a new extraction


def test_load_prices_uses_ibovespa_calendar_and_requires_it(tmp_path):
    prices, report = data.load_prices(
        ["A.SA", "NOPE.SA"], "2020-01-01", "2021-01-01", "2026-01-01", tmp_path,
        fake_fetch(unknown=["NOPE.SA"]),
    )
    assert config.IBOVESPA_TICKER in prices.columns
    assert report.missing_tickers == ["NOPE.SA"]
    with pytest.raises(data.DataError):
        data.load_prices(
            ["A.SA"], "2020-01-01", "2021-01-01", "2026-01-02", tmp_path,
            fake_fetch(unknown=[config.IBOVESPA_TICKER]),
        )


def test_freeze_round_trip_and_reproducibility(tmp_path):
    args = (["A.SA"], "v1", "2020-01-01", "2021-01-01", "2026-01-01", tmp_path, fake_fetch())
    folder = data.freeze_version(*args)
    frozen = data.load_frozen("v1", tmp_path)
    again = data.load_frozen("v1", tmp_path)
    pd.testing.assert_frame_equal(frozen.prices, again.prices)
    assert frozen.sha256 == again.sha256
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["sha256"] == frozen.sha256
    assert manifest["package_versions"]["arch"]
    assert frozen.report.missing_tickers == []


def test_frozen_version_is_never_replaced(tmp_path):
    args = (["A.SA"], "v1", "2020-01-01", "2021-01-01", "2026-01-01", tmp_path, fake_fetch())
    data.freeze_version(*args)
    with pytest.raises(FileExistsError):
        data.freeze_version(*args)


def test_altered_frozen_file_is_refused(tmp_path):
    folder = data.freeze_version(
        ["A.SA"], "v1", "2020-01-01", "2021-01-01", "2026-01-01", tmp_path, fake_fetch(),
    )
    prices = pd.read_parquet(folder / "prices.parquet")
    (prices * 1.01).to_parquet(folder / "prices.parquet")
    with pytest.raises(data.DataError, match="altered"):
        data.load_frozen("v1", tmp_path)


def test_unknown_frozen_version_is_refused(tmp_path):
    with pytest.raises(data.DataError, match="not found"):
        data.load_frozen("nope", tmp_path)


def test_zero_volume_quotes_are_gaps_for_stocks_only():
    dates = business_days(5)
    stock = make_bars("A", dates, [10.0, 11.0, 50.0, 12.0, 13.0])
    stock.loc[2, "volume"] = 0.0
    index = make_bars("^BVSP", dates, [10.0, 11.0, 12.0, 13.0, 14.0], volume=0.0)
    clean, report = data.validate_prices(pd.concat([stock, index]), ["A", "^BVSP"], dates)
    assert report.zero_volume_days == {"A": 1}
    assert clean["A"].iloc[2] == 11.0  # the stale 50.0 is gone
    assert clean["^BVSP"].iloc[2] == 12.0


def test_known_break_becomes_a_zero_return(monkeypatch):
    dates = business_days(4)
    monkeypatch.setitem(config.KNOWN_PRICE_BREAKS, ("A", dates[2].strftime("%Y-%m-%d")), "test")
    clean, report = data.validate_prices(make_bars("A", dates, [10.0, 11.0, 22.0, 23.0]), ["A"], dates)
    np.testing.assert_allclose(clean["A"], [20.0, 22.0, 22.0, 23.0])
    assert report.breaks_removed[0]["log_return"] == pytest.approx(np.log(2))
