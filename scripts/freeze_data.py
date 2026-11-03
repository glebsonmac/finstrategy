"""Download the experiment universe from Yahoo Finance and freeze one data version.

Usage: python scripts/freeze_data.py [version] [end_exclusive YYYY-MM-DD]

Both default to today, so the data ends on the last finished session. An existing version is
never replaced: to recreate the data, use another version name.
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riskbt import config, data  # noqa: E402


def main(version: str, end: str) -> None:
    folder = data.freeze_version(
        list(config.UNIVERSE), version, config.START_DATE, end, date.today().isoformat(),
    )
    frozen = data.load_frozen(version, folder.parents[1])
    prices = frozen.prices
    print(f"Frozen at {folder}: {prices.shape[1]} series, {prices.index[0].date()} to {prices.index[-1].date()}")
    if frozen.report.missing_tickers:
        print(f"No data for: {', '.join(frozen.report.missing_tickers)}")


if __name__ == "__main__":
    today = date.today().isoformat()
    main(sys.argv[1] if len(sys.argv) > 1 else today, sys.argv[2] if len(sys.argv) > 2 else today)
