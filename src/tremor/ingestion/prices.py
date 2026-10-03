"""Live intraday prices for the index constituents (yfinance, 5-minute bars), refreshed in place."""

from __future__ import annotations

import logging

import pandas as pd

from tremor.modules.rebalancer.index import PriceBook

log = logging.getLogger(__name__)


class LivePriceSource:
    """A pseudo-source: each poll refreshes the shared ``PriceBook`` and returns no documents."""

    name = "prices"
    interval_seconds = 300.0

    def __init__(self, tickers: list[str]):
        self.tickers = tickers
        self.book = PriceBook(pd.DataFrame(columns=tickers, index=pd.DatetimeIndex([], tz="UTC")))
        self.poll()

    def poll(self) -> list:
        try:
            import yfinance as yf

            bars = yf.download(self.tickers, period="5d", interval="5m", auto_adjust=True, progress=False)["Close"]
            if len(bars):
                self.book.update(bars)
        except Exception as exc:  # no network: the index simply keeps its last level
            log.warning("price refresh failed: %s", exc)
        return []
