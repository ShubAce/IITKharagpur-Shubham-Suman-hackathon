"""Collect real headlines for each event type via date-restricted news search (Google News RSS).

The human-labelled topic dataset barely covers credit events, operational incidents or armed
conflict - exactly the classes a risk engine cares most about. This script harvests real
headlines for every event type by running class-specific queries over several past years.
The query class is a *weak* label: it is used for training only, never for reported accuracy.

Replay windows are excluded, so demo-day text never reaches the training set.

    python scripts/collect_event_headlines.py        # -> data/raw/event_headlines/headlines.jsonl
"""

from __future__ import annotations

import json
import random
import sys
import time
from email.utils import parsedate_to_datetime
from pathlib import Path

import feedparser
import requests

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "event_headlines" / "headlines.jsonl"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
YEARS = (2016, 2017, 2018, 2019, 2020, 2021, 2024, 2025)  # 2022 and 2023 hold the replay windows

QUERIES: dict[str, list[str]] = {
    "CREDIT_EVENT": [
        "files for bankruptcy", "Chapter 11 filing company", "defaults on bonds", "missed bond payment",
        "missed coupon payment", "debt restructuring talks creditors", "Moody's downgrades", "S&P cuts rating to junk",
        "Fitch downgrades outlook negative", "credit rating cut", "bank run deposits", "liquidity crisis lender",
        "seeks creditor protection", "insolvency administration", "bailout rescue package bank",
        "going concern doubt", "covenant breach lenders", "distressed debt exchange", "sovereign default",
        "credit default swaps widen", "rating upgraded to investment grade", "emergency funding lender",
    ],
    "OPERATIONAL_INCIDENT": [
        "cyberattack disrupts operations", "ransomware attack company", "data breach customers exposed",
        "recalls vehicles", "product recall safety", "plant explosion", "factory fire halts production",
        "outage disrupts services", "workers strike walkout", "grounded fleet", "oil spill", "pipeline shutdown",
        "supply chain disruption shortage", "earthquake halts production", "hurricane shuts refineries", "train derailment",
    ],
    "GEOPOLITICAL": [
        "imposes sanctions", "new tariffs on imports", "trade war escalates", "military strike", "troops border invasion",
        "missile attack", "military coup seizes power", "export controls chips", "oil embargo", "ceasefire talks",
        "nuclear test", "border clash soldiers", "retaliatory tariffs", "state of emergency declared", "terror attack",
    ],
    "MACROECONOMIC": [
        "inflation rises CPI", "Fed raises interest rates", "central bank cuts rates", "GDP contracts recession",
        "jobs report payrolls", "unemployment rate falls", "oil prices jump OPEC", "bond yields surge",
        "currency plunges record low", "manufacturing PMI", "retail sales data", "consumer confidence index",
    ],
    "MERGER_ACQUISITION": [
        "to acquire in deal", "agrees to buy for billion", "takeover bid rejected", "merger approved regulators",
        "to spin off unit", "IPO priced", "private equity buyout", "sells stake",
    ],
    "PRODUCT_LAUNCH": [
        "launches new product", "unveils new", "introduces new model", "rolls out new service", "FDA approves drug",
        "wins contract", "announces partnership with", "debuts new chip",
    ],
    "EARNINGS_GUIDANCE": [
        "beats estimates quarterly earnings", "misses revenue estimates", "cuts full-year guidance",
        "raises forecast", "profit warning", "raises dividend buyback", "quarterly profit falls", "reports record revenue",
    ],
    "REGULATORY_LEGAL": [
        "fined by regulators", "antitrust lawsuit", "SEC charges", "settles lawsuit", "under investigation probe",
        "court rules against", "regulator bans", "class action lawsuit shareholders",
    ],
    "MANAGEMENT_GOVERNANCE": [
        "CEO steps down", "names new CEO", "CFO resigns", "board ousts chief executive", "accounting scandal",
        "activist investor board seats", "chairman to retire",
    ],
    "MARKET_COMMENTARY": [
        "price target raised analyst", "downgrades to neutral", "upgrades to buy", "shares jump", "stock falls",
        "initiates coverage", "stock hits record high", "why shares are moving",
    ],
}


def search(query: str, year: int, half: int) -> list[dict]:
    after, before = (f"{year}-01-01", f"{year}-07-01") if half == 0 else (f"{year}-07-01", f"{year + 1}-01-01")
    resp = requests.get("https://news.google.com/rss/search", headers=UA, timeout=30,
                        params={"q": f"{query} after:{after} before:{before}", "hl": "en-US", "gl": "US", "ceid": "US:en"})
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    rows = []
    for entry in feedparser.parse(resp.text).entries:
        try:
            published = parsedate_to_datetime(entry.published).isoformat()
        except (AttributeError, TypeError, ValueError):
            continue
        source = (entry.get("source") or {}).get("title", "")
        title = entry.title[: -len(source) - 3] if source and entry.title.endswith(f" - {source}") else entry.title
        rows.append({"title": title, "publisher": source, "published_at": published})
    return rows


def main() -> int:
    rng = random.Random(11)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    total = 0
    with OUT.open("w", encoding="utf-8") as fh:
        for event_type, queries in QUERIES.items():
            added = 0
            for query in queries:
                for year in rng.sample(YEARS, 2):
                    try:
                        rows = search(query, year, rng.randrange(2))
                    except Exception as exc:  # one failed query must not lose the rest
                        print(f"  ! {query!r} {year}: {exc}", flush=True)
                        time.sleep(20)
                        continue
                    for row in rows:
                        key = row["title"].lower()
                        if key in seen:
                            continue
                        seen.add(key)
                        fh.write(json.dumps({**row, "query_class": event_type, "query": query}, ensure_ascii=False) + "\n")
                        added += 1
                    time.sleep(1.5)
            total += added
            print(f"{event_type:22s} +{added}", flush=True)
            fh.flush()
    print(f"{total} headlines -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
