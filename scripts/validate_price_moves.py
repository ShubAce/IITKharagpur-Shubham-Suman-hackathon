"""Check the price-direction extractor against what prices actually did.

The stress scenario trusts headlines like "oil surges" for the direction of a price, so the rules in
``tremor.nlp.price_moves`` are tested where the truth is known: company headlines from the Module A
backtest corpus (Google News archive, Oct 2021 - Sep 2022) that report a move in the company's own
share price ("Tesla shares slump", "Apple jumps 4%"), against the stock's actual return.

News archives give the day of a headline, not the minute: a headline dated D reports either that
day's session or an after-hours move that trades on D+1 (earnings come out after the close). So the
reference move is the close-to-close return from D-1 to D+1 - the two sessions around the headline -
and the single-session agreements are reported alongside, for transparency.

    python scripts/validate_price_moves.py        (needs data/raw/backtest_news from scripts/collect_backtest_news.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from tremor.config import load_universe  # noqa: E402
from tremor.nlp.entities import EntityLinker  # noqa: E402
from tremor.nlp.preprocess import clean_text  # noqa: E402
from tremor.nlp.price_moves import share_direction  # noqa: E402


def main() -> int:
    universe = load_universe()
    linker = EntityLinker(universe)
    prices = pd.read_csv(ROOT / "data" / "prices" / "constituents_daily.csv")
    closes = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    closes.index = pd.to_datetime(closes.index)
    sessions = closes.index

    rows, seen = [], set()
    with (ROOT / "data" / "raw" / "backtest_news" / "headlines.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            ticker, title = item["ticker"], clean_text(item["title"], strip_publisher_suffix=True)
            if ticker not in closes or (ticker, title) in seen:
                continue
            seen.add((ticker, title))
            surfaces = [m.surface.lstrip("$") for m in linker.link(title, finance_context=True) if m.entity_id == ticker]
            direction = next((d for d in (share_direction(title, s) for s in surfaces) if d), 0)
            if not direction:
                continue
            day = pd.Timestamp(item["published_at"][:10])
            i = sessions.searchsorted(day, side="right") - 1  # last session on or before the headline's day
            if i < 2 or i + 1 >= len(sessions):
                continue
            c = closes[ticker]
            rows.append({"ticker": ticker, "title": title, "day": str(day.date()), "direction": direction,
                         "ret_prev": c.iloc[i - 1] / c.iloc[i - 2] - 1, "ret_day": c.iloc[i] / c.iloc[i - 1] - 1,
                         "ret_next": c.iloc[i + 1] / c.iloc[i] - 1, "ret_around": c.iloc[i + 1] / c.iloc[i - 1] - 1})
    df = pd.DataFrame(rows)

    def agree(col: str, sub: pd.DataFrame = df) -> float:
        return round(float(((sub[col] > 0) == (sub["direction"] > 0)).mean()), 3)

    big = df[df["ret_around"].abs() >= 0.03]
    majority = max((df["ret_around"] > 0).mean(), (df["ret_around"] <= 0).mean())
    result = {
        "description": "Company headlines that report a move in the company's own share price, from the Module A backtest corpus; "
                       "extracted direction vs the stock's close-to-close return over the two sessions around the headline (D-1 to D+1).",
        "headlines": len(df), "up": int((df["direction"] > 0).sum()), "down": int((df["direction"] < 0).sum()),
        "agreement": agree("ret_around"),
        "agreement_by_session": {"previous": agree("ret_prev"), "headline_day": agree("ret_day"), "next": agree("ret_next")},
        "moves_over_3pct": {"headlines": len(big), "agreement": agree("ret_around", big)},
        "majority_class_baseline": round(float(majority), 3),  # always guessing the more common direction
        "examples": df.sample(min(12, len(df)), random_state=7)[["day", "ticker", "title", "direction", "ret_around"]].round(4).to_dict("records"),
    }
    out = ROOT / "docs" / "results" / "price_moves_validation.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "examples"}, indent=2))
    print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
