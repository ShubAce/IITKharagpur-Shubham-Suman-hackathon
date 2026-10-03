"""Daily prices for the index constituents (adjusted open and close) -> data/prices/constituents_daily.csv

Covers the backtest year (Oct 2021 - Sep 2022, with a lead-in) and the Ukraine replay window.

    python scripts/fetch_prices.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from tremor.config import load_universe  # noqa: E402

OUT = ROOT / "data" / "prices" / "constituents_daily.csv"


def main() -> int:
    import yfinance as yf

    tickers = list(load_universe().index.constituents) + ["^GSPC"]
    raw = yf.download(tickers, start="2021-08-01", end="2022-11-01", auto_adjust=True, progress=False)
    frames = []
    for field in ("Open", "Close"):
        part = raw[field].stack().rename(field.lower())
        frames.append(part)
    df = pd.concat(frames, axis=1).reset_index().rename(columns={"Date": "date", "Ticker": "ticker"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values(["date", "ticker"]).round(4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"saved {OUT} ({df['ticker'].nunique()} tickers x {df['date'].nunique()} days)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
