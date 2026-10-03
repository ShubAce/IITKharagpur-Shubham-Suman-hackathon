"""RSS connector: company news (Yahoo Finance per ticker), topic searches (Google News) and
central-bank press releases (Federal Reserve). No API key needed."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import feedparser
import requests

from tremor.ingestion.base import USER_AGENT, Gate
from tremor.nlp.preprocess import clean_text
from tremor.schemas import RawDocument, SourceKind

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Feed:
    url: str
    label: str
    symbol: str | None = None  # the feed is about this ticker
    finance_feed: bool = False


def default_feeds(tickers: list[str]) -> list[Feed]:
    feeds = [Feed(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US", f"yahoo:{t}", t, True) for t in tickers]
    for label, query in [("geo", "sanctions OR invasion OR tariffs OR missile OR coup"),
                         ("macro", "inflation OR \"interest rates\" OR \"central bank\" OR recession"),
                         ("credit", "default OR bankruptcy OR downgrade OR \"credit rating\"")]:
        feeds.append(Feed(f"https://news.google.com/rss/search?q={quote_plus(query + ' when:1d')}&hl=en-US&gl=US&ceid=US:en", f"gnews:{label}"))
    feeds.append(Feed("https://www.federalreserve.gov/feeds/press_all.xml", "fed", None, True))
    return feeds


class RssSource:
    name = "rss"

    def __init__(self, feeds: list[Feed], gate: Gate, interval_seconds: float = 180.0, feeds_per_poll: int = 8):
        self.feeds, self.interval_seconds = feeds, interval_seconds
        self._gate = gate
        self._seen: set[str] = set()
        self._next = 0
        self._per_poll = feeds_per_poll

    def poll(self) -> list[RawDocument]:
        docs: list[RawDocument] = []
        for _ in range(min(self._per_poll, len(self.feeds))):
            feed = self.feeds[self._next % len(self.feeds)]
            self._next += 1
            try:
                resp = requests.get(feed.url, headers=USER_AGENT, timeout=20)
                entries = feedparser.parse(resp.text).entries if resp.status_code == 200 else []
            except requests.RequestException as exc:
                log.warning("feed %s failed: %s", feed.label, exc)
                continue
            for entry in entries:
                key = entry.get("id") or entry.get("link") or entry.get("title")
                if not key or key in self._seen:
                    continue
                self._seen.add(key)
                source_name = (entry.get("source") or {}).get("title")
                title = clean_text(entry.get("title", ""), strip_publisher_suffix=source_name is not None)
                if len(title) < 15 or not self._gate.accept(title, finance_feed=feed.finance_feed):
                    continue
                try:
                    published = parsedate_to_datetime(entry.published)
                except (AttributeError, TypeError, ValueError):
                    published = datetime.now(timezone.utc)
                docs.append(RawDocument(
                    doc_id=RawDocument.make_id("rss", key), source=f"rss:{feed.label.split(':')[0]}", kind=SourceKind.NEWS,
                    published_at=published, text=title, url=entry.get("link"), publisher=source_name or feed.label,
                    finance_feed=feed.finance_feed, meta={"symbol": feed.symbol} if feed.symbol else {},
                ))
        return docs
