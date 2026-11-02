"""Write a check chart of the crisis periods and list them.

Usage: python scripts/check_regimes.py <data_version>
"""

import sys
from pathlib import Path

import altair as alt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riskbt import config, data, experiment, regimes  # noqa: E402


def main(version: str) -> None:
    market = experiment.index_returns(data.load_frozen(version).prices)
    labels = regimes.classify_regimes(market)
    episodes = regimes.crisis_episodes(labels)
    frame = pd.DataFrame({"date": market.index, "volatility": regimes.realized_volatility(market)}).dropna()
    bands = alt.Chart(episodes).mark_rect(color="#d9d9d9").encode(x="start:T", x2="end:T")
    line = alt.Chart(frame).mark_line(strokeWidth=1).encode(
        x=alt.X("date:T", title="Date"), y=alt.Y("volatility:Q", title="21-day realized volatility, annualized"),
    )
    config.RESULTS_DIR.mkdir(exist_ok=True)
    path = config.RESULTS_DIR / "regimes_check.html"
    (bands + line).properties(width=1000, height=350, title="Ibovespa volatility, crisis days shaded").save(str(path))
    print(labels.value_counts().to_string())
    print(episodes[episodes["days"] >= 20].to_string())
    print(f"Chart at {path}")


if __name__ == "__main__":
    main(sys.argv[1])
