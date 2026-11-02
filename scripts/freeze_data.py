"""Download the experiment universe and freeze one data version.

Usage: python scripts/freeze_data.py <version> <end_exclusive YYYY-MM-DD>
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
    print(f"Frozen at {folder}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
