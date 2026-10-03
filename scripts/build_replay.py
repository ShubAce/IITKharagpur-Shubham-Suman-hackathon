"""Build a replay pack: real news and social posts from a historical window.

A replay pack lets the engine be demonstrated - and regression-tested - on a real market
crisis, deterministically and offline. News comes from GDELT's raw 15-minute archive; social
posts come from the timestamped stock-tweet archive (``scripts/fetch_data.py --only stock_tweets``).

    python scripts/build_replay.py --name ukraine_2022 \
        --start 2022-02-23T18:00 --end 2022-02-25T00:00 \
        --title "Russia invades Ukraine (24 Feb 2022)"

Output: ``data/replay/<name>/{news.jsonl, social.jsonl, manifest.json}``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from tremor.config import load_taxonomy, load_universe  # noqa: E402
from tremor.ingestion.base import Gate  # noqa: E402
from tremor.ingestion.gdelt import iter_range  # noqa: E402
from tremor.nlp.cues import CueMatcher  # noqa: E402
from tremor.nlp.entities import EntityLinker  # noqa: E402
from tremor.nlp.preprocess import clean_text, fingerprint  # noqa: E402
from tremor.paths import DATA_DIR, RAW_DIR  # noqa: E402
from tremor.schemas import RawDocument, SourceKind  # noqa: E402


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def build_news(start: datetime, end: datetime, gate: Gate, out: Path, max_copies: int) -> dict:
    """Stream GDELT slots to ``out``. Syndicated copies of one headline are kept up to
    ``max_copies`` times: enough for the engine to measure corroboration, without bloating the repo."""
    copies: dict[str, int] = {}
    kept = seen = 0
    t0 = time.time()
    with out.open("w", encoding="utf-8") as fh:
        for slot, docs in iter_range(start, end, gate):
            for doc in docs:
                seen += 1
                fp = fingerprint(doc.text)
                copies[fp] = copies.get(fp, 0) + 1
                if copies[fp] > max_copies:
                    continue
                fh.write(doc.model_dump_json() + "\n")
                kept += 1
            print(f"  {slot:%Y-%m-%d %H:%M}  +{len(docs):4d} gated | kept {kept:6d} | {time.time() - t0:5.0f}s", flush=True)
    return {"gated_articles": seen, "kept_articles": kept, "distinct_headlines": len(copies)}


def build_social(start: datetime, end: datetime, out: Path) -> dict:
    path = RAW_DIR / "stock_tweets" / "stock_tweets.csv"
    if not path.exists():
        print(f"  (no tweet archive at {path}; skipping the social leg)")
        return {"posts": 0}
    tweets = pd.read_csv(path)
    tweets["ts"] = pd.to_datetime(tweets["Date"], utc=True, errors="coerce")
    window = tweets[(tweets["ts"] >= start) & (tweets["ts"] < end)].sort_values("ts")
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for row in window.itertuples(index=False):
            text = clean_text(row.Tweet)
            if len(text) < 15:
                continue
            doc = RawDocument(
                doc_id=RawDocument.make_id("tweet_archive", f"{row.Date}|{text[:80]}"),
                source="tweet_archive",
                kind=SourceKind.SOCIAL,
                published_at=row.ts.to_pydatetime(),
                text=text,
                finance_feed=True,
                meta={"symbol": row._2},  # "Stock Name" column
            )
            fh.write(doc.model_dump_json() + "\n")
            n += 1
    return {"posts": n}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--start", required=True, type=parse_time, help="UTC, e.g. 2022-02-23T18:00")
    parser.add_argument("--end", required=True, type=parse_time)
    parser.add_argument("--title", default="")
    parser.add_argument("--max-copies", type=int, default=4, help="syndicated copies kept per distinct headline")
    parser.add_argument("--skip-news", action="store_true")
    args = parser.parse_args()

    universe = load_universe()
    gate = Gate(EntityLinker(universe), CueMatcher(load_taxonomy()), [e.id for e in universe.entities if e.type == "company"])
    out_dir = DATA_DIR / "replay" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"name": args.name, "title": args.title or args.name, "start": args.start.isoformat(), "end": args.end.isoformat(),
                "sources": {"news": "GDELT 2.0 Global Knowledge Graph, raw 15-minute files (gdeltproject.org)",
                            "social": "Kaggle: equinxx/stock-tweets-for-sentiment-analysis-and-prediction"}}
    if not args.skip_news:
        print("news  <- GDELT")
        manifest["news"] = build_news(args.start, args.end, gate, out_dir / "news.jsonl", args.max_copies)
    print("social <- tweet archive")
    manifest["social"] = build_social(args.start, args.end, out_dir / "social.jsonl")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
