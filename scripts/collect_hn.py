"""Social leg of a replay pack from Hacker News (the Algolia HN search API, keyless, timestamped to the second).

The tweet archive used for the 2022 pack ends in September 2022, the X API is paid, and StockTwits has
removed the streams of delisted tickers such as $SIVB. For the March 2023 bank run Hacker News is the
natural social source anyway: Silicon Valley Bank was the startup world's bank, and founders discussed
the run on it there in real time.

A comment is a paragraph, not a post, so each one is cut to the sentence that names one of the crisis
names its query is about (plus the next sentence when it is short) - about the length of a tweet. Search
typo-tolerance is switched off ("UBS" would otherwise match "USB" and pull in hardware chatter), every
text then goes through the same gate as the news, and only distinct texts are kept.

    python scripts/collect_hn.py --name svb_2023 --start 2023-03-08T18:00 --end 2023-03-16T00:00
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import requests  # noqa: E402

from tremor.config import load_taxonomy, load_universe  # noqa: E402
from tremor.ingestion.base import USER_AGENT, Gate  # noqa: E402
from tremor.nlp.cues import CueMatcher  # noqa: E402
from tremor.nlp.entities import EntityLinker  # noqa: E402
from tremor.nlp.preprocess import clean_text, fingerprint  # noqa: E402
from tremor.schemas import RawDocument, SourceKind  # noqa: E402

API = "https://hn.algolia.com/api/v1/search_by_date"
# The banks of the run, the deposit insurer, the Fed and the big banks the contagion was measured against.
CRISIS = {"SIVB", "SBNY", "FRC", "SI", "PACW", "WAL", "SCHW", "CS", "UBS", "FDIC", "FED", "JPM", "BAC", "C", "WFC", "GS", "MS", "HSBC", "DB"}
# query -> the entities a kept sentence must name
QUERIES = {"SVB": {"SIVB"}, "Silicon Valley Bank": {"SIVB"}, "Signature Bank": {"SBNY"}, "First Republic": {"FRC"},
           "Silvergate": {"SI"}, "Credit Suisse": {"CS"}, "UBS": {"UBS", "CS"}, "PacWest": {"PACW"}, "Western Alliance": {"WAL"},
           "FDIC": CRISIS, "bank run": CRISIS, "deposits": CRISIS}
_TAGS = re.compile(r"<[^>]+>")
_SENTENCES = re.compile(r"(?<=[.!?])\s+")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def excerpt(text: str, linker: EntityLinker, targets: set[str], max_chars: int = 280) -> str | None:
    """The first sentence naming one of ``targets`` (plus the next one when short), about a tweet long."""
    sentences = [s.strip() for s in _SENTENCES.split(text) if s.strip()]
    for i, sentence in enumerate(sentences):
        if targets.intersection(linker.entity_ids(sentence)):
            out = sentence
            if i + 1 < len(sentences) and len(out) + len(sentences[i + 1]) < max_chars:
                out += " " + sentences[i + 1]
            return out[:max_chars].rsplit(" ", 1)[0] if len(out) > max_chars else out
    return None


def fetch(query: str, start: datetime, end: datetime, session: requests.Session) -> list[dict]:
    """All hits for one query in [start, end), in 6-hour windows (the API returns at most 1000 per call)."""
    hits, t = [], start
    while t < end:
        t1 = min(end, t + timedelta(hours=6))
        params = {"query": query, "tags": "(story,comment)", "hitsPerPage": 1000, "typoTolerance": "false",
                  "numericFilters": f"created_at_i>={int(t.timestamp())},created_at_i<{int(t1.timestamp())}"}
        for attempt in range(3):
            try:
                resp = session.get(API, params=params, timeout=30)
                resp.raise_for_status()
                hits += resp.json().get("hits", [])
                break
            except (requests.RequestException, ValueError):
                time.sleep(2 + 3 * attempt)
        t = t1
        time.sleep(0.3)
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--start", required=True, type=parse_time)
    parser.add_argument("--end", required=True, type=parse_time)
    args = parser.parse_args()

    universe = load_universe()
    linker = EntityLinker(universe)
    gate = Gate(linker, CueMatcher(load_taxonomy()), [e.id for e in universe.entities if e.type == "company"])
    out_dir = ROOT / "data" / "replay" / args.name
    seen_ids, seen_text, docs = set(), set(), []
    with requests.Session() as session:
        session.headers.update(USER_AGENT)
        for query, targets in QUERIES.items():
            hits = fetch(query, args.start, args.end, session)
            kept = 0
            for hit in hits:
                if hit["objectID"] in seen_ids:
                    continue
                seen_ids.add(hit["objectID"])
                raw = hit.get("title") or _TAGS.sub(" ", hit.get("comment_text") or "")
                text = clean_text(raw).replace("�", "'")  # a few items carry a mangled apostrophe ("SVB�s")
                if hit.get("title"):
                    text = text if targets.intersection(linker.entity_ids(text)) else ""
                else:
                    text = excerpt(text, linker, targets) or ""
                if len(text) < 15 or not gate.accept(text) or fingerprint(text) in seen_text:
                    continue
                seen_text.add(fingerprint(text))
                docs.append(RawDocument(
                    doc_id=RawDocument.make_id("hackernews", hit["objectID"]), source="hackernews", kind=SourceKind.SOCIAL,
                    published_at=datetime.fromtimestamp(hit["created_at_i"], tz=timezone.utc), text=text,
                    url=f"https://news.ycombinator.com/item?id={hit['objectID']}", publisher=hit.get("author"),
                    meta={"hn_type": "story" if hit.get("title") else "comment"}))
                kept += 1
            print(f"  {query:22s} {len(hits):5d} hits, kept {kept:4d}", flush=True)
    docs.sort(key=lambda d: d.published_at)
    with (out_dir / "social.jsonl").open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(doc.model_dump_json() + "\n")
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"name": args.name}
    manifest.setdefault("sources", {})["social"] = "Hacker News via the Algolia HN Search API (stories and comment excerpts)"
    manifest["social"] = {"posts": len(docs), "stories": sum(d.meta["hn_type"] == "story" for d in docs),
                          "comment_excerpts": sum(d.meta["hn_type"] == "comment" for d in docs)}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["social"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
