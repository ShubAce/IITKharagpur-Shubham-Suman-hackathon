"""Collect public StockTwits posts, many of which carry the author's own Bullish/Bearish tag.

Those self-labels are free ground truth for *retail-investor* language (slang, emojis,
cashtags) - a register that news-trained sentiment models handle poorly. Output goes to
``data/raw/stocktwits/messages.jsonl`` (git-ignored); only the trained head weights and
the evaluation numbers derived from it are committed.

The public API allows ~200 unauthenticated requests per hour, so the collector is paced
at one request every 19 seconds and is safe to leave running in the background.

    python scripts/collect_stocktwits.py --requests 400
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests  # StockTwits' CDN challenges httpx's TLS fingerprint but accepts requests

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "stocktwits" / "messages.jsonl"
API = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "AMD", "NFLX", "DIS", "INTC", "CRM", "PG", "KO",
    "COST", "BA", "VZ", "JPM", "XOM", "JNJ", "SPY", "QQQ", "GS", "BAC", "PLTR", "COIN", "GME", "F", "WMT", "PFE",
]
UA = {"User-Agent": "Mozilla/5.0 (research prototype; sentiment evaluation set)"}


def load_seen() -> tuple[set[int], dict[str, int]]:
    """Resume support: ids already stored and the oldest id seen per symbol."""
    seen: set[int] = set()
    oldest: dict[str, int] = {}
    if OUT.exists():
        with OUT.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                seen.add(row["id"])
                sym = row["query_symbol"]
                oldest[sym] = min(oldest.get(sym, row["id"]), row["id"])
    return seen, oldest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requests", type=int, default=400, help="total API calls to make")
    parser.add_argument("--pause", type=float, default=19.0, help="seconds between calls")
    args = parser.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen, oldest = load_seen()
    labelled = 0
    with requests.Session() as client, OUT.open("a", encoding="utf-8") as fh:
        client.headers.update(UA)
        for n in range(args.requests):
            symbol = SYMBOLS[n % len(SYMBOLS)]
            params = {"max": oldest[symbol]} if symbol in oldest else {}
            try:
                resp = client.get(API.format(symbol=symbol), params=params, timeout=30)
            except requests.RequestException as exc:
                print(f"[{n}] {symbol}: {type(exc).__name__}", flush=True)
                time.sleep(args.pause)
                continue
            if resp.status_code == 429:
                print(f"[{n}] rate limited - sleeping 10 min", flush=True)
                time.sleep(600)
                continue
            if resp.status_code != 200:
                print(f"[{n}] {symbol}: HTTP {resp.status_code}", flush=True)
                time.sleep(args.pause)
                continue
            new = 0
            for msg in resp.json().get("messages", []):
                oldest[symbol] = min(oldest.get(symbol, msg["id"]), msg["id"])
                if msg["id"] in seen:
                    continue
                seen.add(msg["id"])
                tag = ((msg.get("entities") or {}).get("sentiment") or {}).get("basic")
                labelled += tag is not None
                new += 1
                fh.write(json.dumps({
                    "id": msg["id"],
                    "created_at": msg["created_at"],
                    "query_symbol": symbol,
                    "symbols": [s["symbol"] for s in msg.get("symbols", [])],
                    "body": msg["body"],
                    "user_sentiment": tag,
                    "followers": (msg.get("user") or {}).get("followers"),
                    "likes": (msg.get("likes") or {}).get("total", 0),
                }, ensure_ascii=False) + "\n")
            fh.flush()
            if n % 10 == 0:
                print(f"[{n}] {symbol}: +{new} | total {len(seen)} ({labelled} self-labelled this run)", flush=True)
            time.sleep(args.pause)
    print(f"done: {len(seen)} messages in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
