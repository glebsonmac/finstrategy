"""Download, validation, Parquet cache and frozen versions of the price data.

This module and `app.py` are the only ones with side effects (network and disk).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yfinance as yf

from riskbt import config

PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
RECORDED_PACKAGES = ["numpy", "pandas", "scipy", "statsmodels", "arch", "yfinance", "pyarrow"]
YAHOO_COLUMNS = {
    "Open": "open", "High": "high", "Low": "low", "Close": "close",
    "Adj Close": "adj_close", "Volume": "volume",
}


class DataError(Exception):
    """The requested price data is unavailable or does not match its recorded hash."""


@dataclass(frozen=True)
class DataReport:
    """What validation found and corrected. Plain data, so it can be saved as JSON."""

    missing_tickers: list[str]  # no data returned by the source
    duplicates_removed: int
    off_calendar_rows_removed: int  # rows on days the reference market did not trade
    inconsistent_prices: dict[str, int]  # non-positive prices or close outside [low, high]
    gaps_filled: dict[str, int]  # short gaps carried forward (zero return on those days)
    gaps_unfilled: dict[str, int]  # long gaps left missing
    zero_volume_days: dict[str, int]  # stale quotes of stocks, treated as gaps
    breaks_removed: list[dict]  # known artificial price jumps taken out of the series
    adjustment_issues: dict[str, int]  # days with an impossible adj_close / close factor
    extreme_returns: list[dict]  # ticker, date and log return above the review threshold


@dataclass(frozen=True)
class PriceData:
    """Clean adjusted prices (dates in rows, tickers in columns) and their provenance."""

    prices: pd.DataFrame
    report: DataReport
    version: str
    sha256: str


def fetch_yahoo(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download daily bars from Yahoo Finance; `end` is exclusive."""
    return yf.download(
        tickers, start=start, end=end, auto_adjust=False, actions=False,
        threads=False, progress=False, multi_level_index=True,
    )


def to_long_format(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn yfinance's (field, ticker) columns into one row per ticker and date."""
    frames = []
    for ticker in raw.columns.get_level_values("Ticker").unique():
        bars = raw.xs(ticker, axis=1, level="Ticker").rename(columns=YAHOO_COLUMNS)
        bars = bars.dropna(subset=["adj_close"]).rename_axis("date").reset_index()
        if not bars.empty:
            frames.append(bars.assign(ticker=ticker)[PRICE_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    long = pd.concat(frames, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None).dt.normalize()
    return long


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    as_of: str,
    cache_dir: Path = config.DATA_DIR / "cache",
    fetch: Callable[[list[str], str, str], pd.DataFrame] = fetch_yahoo,
) -> pd.DataFrame:
    """Return raw bars, keeping each extraction on disk under its `as_of` date.

    Only tickers absent from that extraction are downloaded. Tickers the source does not
    know are simply absent from the result; `validate_prices` reports them.
    """
    folder = Path(cache_dir) / f"{as_of}_{start}_{end}"
    folder.mkdir(parents=True, exist_ok=True)
    paths = {ticker: folder / f"{ticker}.parquet" for ticker in tickers}
    to_download = [ticker for ticker, path in paths.items() if not path.exists()]
    if to_download:
        fresh = to_long_format(fetch(to_download, start, end))
        for ticker, bars in fresh.groupby("ticker"):
            bars.to_parquet(paths[ticker], index=False)
    frames = [pd.read_parquet(path) for path in paths.values() if path.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PRICE_COLUMNS)


def _inconsistent_mask(bars: pd.DataFrame) -> pd.Series:
    """Rows with non-positive prices, or a close outside the day's [low, high] range."""
    non_positive = (bars["close"] <= 0) | (bars["adj_close"] <= 0)
    slack = 1 + config.ADJUSTMENT_TOLERANCE
    out_of_range = (bars["close"] > bars["high"] * slack) | (bars["close"] < bars["low"] / slack)
    return non_positive | out_of_range | (bars["high"] < bars["low"])


def _stale_mask(bars: pd.DataFrame) -> pd.Series:
    """Stock rows with zero volume: no trade happened, so the quote carries no information.

    Yahoo repeats old quotes on these days, often on an unadjusted scale. The index is exempt
    because its volume field is not reliable.
    """
    return (bars["volume"] == 0) & ~bars["ticker"].str.startswith("^")


def _remove_known_breaks(prices: pd.DataFrame, breaks: dict[tuple[str, str], str]) -> list[dict]:
    """Rescale prices before each listed break, in place, so that day's return becomes zero."""
    removed = []
    for (ticker, date), reason in breaks.items():
        day = pd.Timestamp(date)
        if ticker not in prices.columns or day not in prices.index:
            continue
        position = prices.index.get_loc(day)
        before, after = prices[ticker].iloc[position - 1], prices[ticker].iloc[position]
        if np.isnan(before) or np.isnan(after):
            continue
        prices.loc[prices.index < day, ticker] *= after / before
        removed.append({"ticker": ticker, "date": date, "log_return": float(np.log(after / before)), "reason": reason})
    return removed


def _fill_short_gaps(prices: pd.Series, max_days: int) -> tuple[pd.Series, int, int]:
    """Carry prices forward over interior gaps of up to `max_days` rows.

    Longer gaps stay missing as a whole, and so do the days before listing and after the
    last quote. Returns the series, the number of days filled and the days left missing.
    """
    first, last = prices.first_valid_index(), prices.last_valid_index()
    if first is None:
        return prices, 0, 0
    span = prices.loc[first:last]
    missing = span.isna()
    gap_sizes = missing.groupby((~missing).cumsum()).transform("sum")
    fillable = missing & (gap_sizes <= max_days)
    filled = prices.copy()
    filled.loc[fillable[fillable].index] = span.ffill().loc[fillable]
    return filled, int(fillable.sum()), int((missing & ~fillable).sum())


def _adjustment_issue_counts(bars: pd.DataFrame) -> dict[str, int]:
    """Count days whose adj_close / close factor is above 1 or falls over time.

    Dividend adjustment scales past prices down, so the factor must stay at or below 1
    and never decrease from one day to the next.
    """
    wide = bars.assign(factor=bars["adj_close"] / bars["close"])
    factor = wide.pivot(index="date", columns="ticker", values="factor").sort_index()
    tolerance = config.ADJUSTMENT_TOLERANCE
    issues = (factor > 1 + tolerance) | (factor.ffill().diff() < -tolerance)
    return _positive_counts(issues.sum())


def _extreme_returns(prices: pd.DataFrame) -> list[dict]:
    """List daily log returns above the review threshold. They are reported, not altered."""
    returns = np.log(prices).diff().stack()
    extreme = returns[returns.abs() > config.EXTREME_RETURN]
    return [
        {"ticker": ticker, "date": date.strftime("%Y-%m-%d"), "log_return": float(value)}
        for (date, ticker), value in extreme.items()
    ]


def _positive_counts(counts: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in counts.items() if value > 0}


def validate_prices(
    bars: pd.DataFrame, tickers: list[str], calendar: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, DataReport]:
    """Clean raw bars into a wide table of adjusted prices and report every change.

    `calendar` holds the reference trading days (the Ibovespa's); it defaults to all dates
    found in `bars`. Inconsistent and stale prices are treated as gaps; short gaps are carried
    forward; breaks listed in `config.KNOWN_PRICE_BREAKS` are removed.
    """
    unique = bars.drop_duplicates(["date", "ticker"], keep="last")
    calendar = pd.DatetimeIndex(sorted(unique["date"].unique())) if calendar is None else calendar
    on_calendar = unique[unique["date"].isin(calendar)]
    inconsistent = _inconsistent_mask(on_calendar)
    stale = _stale_mask(on_calendar)
    usable = on_calendar[~inconsistent & ~stale]
    wide = usable.pivot(index="date", columns="ticker", values="adj_close").reindex(calendar)
    wide = wide.rename_axis("date").rename_axis(None, axis=1)

    filled, gaps_filled, gaps_unfilled = {}, {}, {}
    for ticker in wide.columns:
        series, n_filled, n_unfilled = _fill_short_gaps(wide[ticker], config.MAX_FILLED_GAP_DAYS)
        filled[ticker] = series
        gaps_filled[ticker], gaps_unfilled[ticker] = n_filled, n_unfilled
    clean = pd.DataFrame(filled, index=wide.index)
    breaks_removed = _remove_known_breaks(clean, config.KNOWN_PRICE_BREAKS)

    report = DataReport(
        missing_tickers=sorted(set(tickers) - set(clean.columns)),
        duplicates_removed=len(bars) - len(unique),
        off_calendar_rows_removed=len(unique) - len(on_calendar),
        inconsistent_prices=_positive_counts(on_calendar[inconsistent].groupby("ticker").size()),
        gaps_filled=_positive_counts(pd.Series(gaps_filled, dtype=int)),
        gaps_unfilled=_positive_counts(pd.Series(gaps_unfilled, dtype=int)),
        zero_volume_days=_positive_counts(on_calendar[stale].groupby("ticker").size()),
        breaks_removed=breaks_removed,
        adjustment_issues=_adjustment_issue_counts(usable),
        extreme_returns=_extreme_returns(clean),
    )
    return clean, report


def load_prices(
    tickers: list[str],
    start: str,
    end: str,
    as_of: str,
    cache_dir: Path = config.DATA_DIR / "cache",
    fetch: Callable[[list[str], str, str], pd.DataFrame] = fetch_yahoo,
) -> tuple[pd.DataFrame, DataReport]:
    """Download (or reuse) and validate `tickers` on the Ibovespa's trading calendar."""
    requested = list(dict.fromkeys([*tickers, config.IBOVESPA_TICKER]))
    bars = download_prices(requested, start, end, as_of, cache_dir, fetch)
    is_index = bars["ticker"] == config.IBOVESPA_TICKER
    if not is_index.any():
        raise DataError(f"No data for {config.IBOVESPA_TICKER}, the reference calendar.")
    calendar = pd.DatetimeIndex(sorted(bars.loc[is_index, "date"].unique()))
    return validate_prices(bars, requested, calendar)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_version(
    tickers: list[str],
    version: str,
    start: str,
    end: str,
    as_of: str,
    data_dir: Path = config.DATA_DIR,
    fetch: Callable[[list[str], str, str], pd.DataFrame] = fetch_yahoo,
) -> Path:
    """Save prices and a manifest under `frozen/<version>`. An existing version is never replaced."""
    folder = Path(data_dir) / "frozen" / version
    if folder.exists():
        raise FileExistsError(f"Data version {version!r} is already frozen at {folder}.")
    prices, report = load_prices(tickers, start, end, as_of, Path(data_dir) / "cache", fetch)
    folder.mkdir(parents=True)
    prices_path = folder / "prices.parquet"
    prices.to_parquet(prices_path)
    manifest = {
        "version": version,
        "source": "Yahoo Finance (yfinance)",
        "start": start,
        "end_exclusive": end,
        "extraction_date": as_of,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tickers": list(prices.columns),
        "first_date": prices.index[0].strftime("%Y-%m-%d"),
        "last_date": prices.index[-1].strftime("%Y-%m-%d"),
        "sha256": _file_sha256(prices_path),
        "package_versions": {name: package_version(name) for name in RECORDED_PACKAGES},
        "report": asdict(report),
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return folder


def latest_frozen_version(data_dir: Path = config.DATA_DIR) -> str:
    """Name of the most recent frozen version; names are dates, so they sort in time."""
    folder = Path(data_dir) / "frozen"
    versions = sorted(path.name for path in folder.glob("*") if (path / "manifest.json").exists())
    if not versions:
        raise DataError(f"No frozen data at {folder}. Run: python scripts/freeze_data.py")
    return versions[-1]


def load_frozen(version: str, data_dir: Path = config.DATA_DIR) -> PriceData:
    """Read a frozen version, refusing files that no longer match their recorded hash."""
    folder = Path(data_dir) / "frozen" / version
    manifest_path, prices_path = folder / "manifest.json", folder / "prices.parquet"
    if not manifest_path.exists() or not prices_path.exists():
        raise DataError(f"Data version {version!r} not found at {folder}.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sha256 = _file_sha256(prices_path)
    if sha256 != manifest["sha256"]:
        raise DataError(f"Data version {version!r} was altered: hash does not match the manifest.")
    return PriceData(pd.read_parquet(prices_path), DataReport(**manifest["report"]), version, sha256)
