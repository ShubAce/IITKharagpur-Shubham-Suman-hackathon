"""Social-media connectors. The X/Twitter API is paid, so the live social leg uses StockTwits
(the cashtag-native "finance Twitter") and Bluesky's public search, both keyless."""

from __future__ import annotations

import logging
from datetime import datetime

import requests

from tremor.ingestion.base import USER_AGENT
from tremor.nlp.preprocess import clean_text
from tremor.schemas import RawDocument, SourceKind

log = logging.getLogger(__name__)


class StockTwitsSource:
    """Latest posts per ticker, one ticker per poll (the public API allows ~200 calls an hour)."""

    name = "stocktwits"

    def __init__(self, symbols: list[str], interval_seconds: float = 20.0):
        self.symbols, self.interval_seconds = symbols, interval_seconds
        self._seen: set[int] = set()
        self._next = 0
        self._session = requests.Session()  # its TLS fingerprint passes the CDN check; httpx's does not
        self._session.headers.update(USER_AGENT)

    def poll(self) -> list[RawDocument]:
        symbol = self.symbols[self._next % len(self.symbols)]
        self._next += 1
        try:
            resp = self._session.get(f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json", timeout=20)
            messages = resp.json().get("messages", []) if resp.status_code == 200 else []
        except (requests.RequestException, ValueError) as exc:
            log.warning("stocktwits %s failed: %s", symbol, exc)
            return []
        docs = []
        for msg in messages:
            if msg["id"] in self._seen:
                continue
            self._seen.add(msg["id"])
            text = clean_text(msg.get("body", ""))
            if len(text) < 10:
                continue
            tag = ((msg.get("entities") or {}).get("sentiment") or {}).get("basic")
            docs.append(RawDocument(
                doc_id=RawDocument.make_id("stocktwits", str(msg["id"])), source="stocktwits", kind=SourceKind.SOCIAL,
                published_at=datetime.fromisoformat(msg["created_at"].replace("Z", "+00:00")), text=text,
                url=f"https://stocktwits.com/message/{msg['id']}", publisher=(msg.get("user") or {}).get("username"),
                finance_feed=True, meta={"symbol": symbol, "author_label": tag},
            ))
        return docs


class BlueskySource:
    """Public post search for each ticker's cashtag."""

    name = "bluesky"

    def __init__(self, symbols: list[str], interval_seconds: float = 30.0):
        self.symbols, self.interval_seconds = symbols, interval_seconds
        self._seen: set[str] = set()
        self._next = 0

    def poll(self) -> list[RawDocument]:
        symbol = self.symbols[self._next % len(self.symbols)]
        self._next += 1
        try:
            resp = requests.get("https://api.bsky.app/xrpc/app.bsky.feed.searchPosts", headers=USER_AGENT, timeout=20,
                                params={"q": f"${symbol}", "limit": 25, "sort": "latest", "lang": "en"})
            posts = resp.json().get("posts", []) if resp.status_code == 200 else []
        except (requests.RequestException, ValueError) as exc:
            log.warning("bluesky %s failed: %s", symbol, exc)
            return []
        docs = []
        for post in posts:
            uri = post.get("uri", "")
            record = post.get("record") or {}
            text = clean_text(record.get("text", ""))
            if not uri or uri in self._seen or len(text) < 10:
                continue
            self._seen.add(uri)
            try:
                published = datetime.fromisoformat(record.get("createdAt", "").replace("Z", "+00:00"))
            except ValueError:
                continue
            handle = (post.get("author") or {}).get("handle", "")
            docs.append(RawDocument(
                doc_id=RawDocument.make_id("bluesky", uri), source="bluesky", kind=SourceKind.SOCIAL, published_at=published,
                text=text, url=f"https://bsky.app/profile/{handle}/post/{uri.rsplit('/', 1)[-1]}", publisher=handle,
                finance_feed=True, meta={"symbol": symbol},
            ))
        return docs
