"""Collect one year of historical headlines for the index constituents (the news leg of the backtest).

Source: Google News RSS search with ``after:`` / ``before:`` date operators, one query per
company per week. The window (Oct 2021 - Sep 2022) is chosen to line up with the timestamped
stock-tweet archive, so the backtest runs on two sources, as the live engine does.

Publication *dates* are reliable; times of day are not (the feed reports a placeholder hour),
so the backtest treats a headline as known only from the next trading day's open.

    python scripts/collect_backtest_news.py     # -> data/raw/backtest_news/headlines.jsonl (resumable)
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

import feedparser
import requests

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "backtest_news" / "headlines.jsonl"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
START, END = date(2021, 9, 27), date(2022, 10, 3)

QUERIES = {
    "AAPL": '"Apple" (AAPL OR stock OR shares OR iPhone OR earnings)',
    "MSFT": '"Microsoft" (MSFT OR stock OR shares OR Azure OR earnings)',
    "NVDA": '"Nvidia" (NVDA OR stock OR shares OR chips OR earnings)',
    "AMZN": '"Amazon" (AMZN OR stock OR shares OR AWS OR earnings)',
    "GOOGL": '("Alphabet" OR "Google") (GOOGL OR stock OR shares OR earnings OR antitrust)',
    "META": '("Meta Platforms" OR "Facebook") (stock OR shares OR earnings OR Zuckerberg)',
    "TSLA": '"Tesla" (TSLA OR stock OR shares OR deliveries OR Musk)',
    "NFLX": '"Netflix" (NFLX OR stock OR shares OR subscribers OR earnings)',
    "DIS": '"Disney" (stock OR shares OR earnings OR streaming OR parks)',
    "INTC": '"Intel" (INTC OR stock OR shares OR chips OR earnings)',
    "AMD": '"AMD" (stock OR shares OR chips OR earnings OR Ryzen)',
    "PG": '"Procter & Gamble" OR "P&G"',
    "KO": '"Coca-Cola" (stock OR shares OR earnings OR sales)',
    "COST": '"Costco" (stock OR shares OR earnings OR sales)',
    "BA": '"Boeing" (stock OR shares OR 737 OR deliveries OR earnings)',
    "LMT": '"Lockheed Martin"',
    "VZ": '"Verizon" (stock OR shares OR earnings OR 5G)',
    "JPM": '"JPMorgan" (stock OR shares OR earnings OR Dimon)',
    "XOM": '"Exxon" (stock OR shares OR earnings OR oil)',
    "JNJ": '"Johnson & Johnson" (stock OR shares OR earnings OR talc OR vaccine)',
    # Market-wide context, so macro / geopolitical / credit events exist in the backtest too.
    "_MACRO": '"Federal Reserve" OR inflation OR "interest rates" OR recession',
    "_GEO": 'sanctions OR invasion OR tariffs OR "trade war" OR missile',
    "_CREDIT": 'default OR bankruptcy OR "credit rating" OR downgrade bonds',
}


def weeks() -> list[tuple[date, date]]:
    out, day = [], START
    while day < END:
        out.append((day, day + timedelta(days=7)))
        day += timedelta(days=7)
    return out


def search(query: str, after: date, before: date) -> list[dict]:
    resp = requests.get("https://news.google.com/rss/search", headers=UA, timeout=30,
                        params={"q": f"{query} after:{after} before:{before}", "hl": "en-US", "gl": "US", "ceid": "US:en"})
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    rows = []
    for entry in feedparser.parse(resp.text).entries:
        try:
            published = parsedate_to_datetime(entry.published)
        except (AttributeError, TypeError, ValueError):
            continue
        source = (entry.get("source") or {}).get("title", "")
        title = entry.title[: -len(source) - 3] if source and entry.title.endswith(f" - {source}") else entry.title
        rows.append({"title": title, "publisher": source, "published_at": published.isoformat(), "url": entry.get("link")})
    return rows


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str]] = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                done.add((row["ticker"], row["week"]))
    jobs = [(ticker, query, a, b) for a, b in weeks() for ticker, query in QUERIES.items() if (ticker, str(a)) not in done]
    print(f"{len(jobs)} requests to make ({len(done)} ticker-weeks already collected)", flush=True)
    failures = 0
    with OUT.open("a", encoding="utf-8") as fh:
        for n, (ticker, query, after, before) in enumerate(jobs):
            try:
                rows = search(query, after, before)
                failures = 0
            except Exception as exc:
                failures += 1
                print(f"  ! {ticker} {after}: {exc} (pause {60 * failures}s)", flush=True)
                time.sleep(60 * failures)
                if failures >= 6:
                    print("giving up: the feed keeps refusing requests", flush=True)
                    return 1
                continue
            for row in rows or [{"title": None}]:  # an empty week is still recorded, so it is not retried
                fh.write(json.dumps({"ticker": ticker, "week": str(after), **row}, ensure_ascii=False) + "\n")
            fh.flush()
            if n % 25 == 0:
                print(f"[{n}/{len(jobs)}] {ticker} {after}: {len(rows)} headlines", flush=True)
            time.sleep(1.6)
    print("done ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
