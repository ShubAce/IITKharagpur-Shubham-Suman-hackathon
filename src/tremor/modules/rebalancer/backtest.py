"""One-year backtest of the sentiment-tilted index (Oct 2021 - Sep 2022), two sources.

1. **Signals.** The full engine is run over a year of real text, in time order: ~49k dated company
   headlines (Google News archive) and ~63k timestamped tweets. At every market open the filtered
   sentiment state of each constituent is recorded -> ``data/backtest/signals_daily.csv``.
2. **Trading.** Each day the methodology rebalances at the open using only that snapshot, and the
   portfolio earns open-to-open returns. Headlines carry a date but no reliable time, so they are
   stamped at the *end* of their day: nothing can be traded before it could have been read.
3. **Comparison.** Against the equal-weight parent (same stocks, same daily rebalancing, same
   5 bp cost) and against a *naive* version whose sentiment comes from keyword counting.

Reported honestly: a single year, twenty stocks and a bear market is a small sample - the
information coefficient's t-statistic is shown so significance can be judged, not assumed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from tremor.config import load_settings, load_taxonomy, load_universe
from tremor.engine.pipeline import RiskEngine
from tremor.modules.rebalancer.methodology import TiltConfig, rebalance
from tremor.nlp.models import RuleTextModel, load_text_model
from tremor.nlp.preprocess import clean_text
from tremor.paths import DATA_DIR, DOCS_DIR, RAW_DIR
from tremor.schemas import RawDocument, SourceKind

SIGNALS_PATH = DATA_DIR / "backtest" / "signals_daily.csv"
RESULTS_PATH = DOCS_DIR / "results" / "backtest.json"
START, END = "2021-10-01", "2022-09-30"
TWEET_SYMBOL = {"GOOG": "GOOGL"}
TRADING_DAYS = 252


def build_corpus() -> list[RawDocument]:
    tickers = set(load_universe().index.constituents)
    docs: list[RawDocument] = []
    news = pd.read_json(RAW_DIR / "backtest_news" / "headlines.jsonl", lines=True).dropna(subset=["title"])
    news["day"] = pd.to_datetime(news["published_at"], utc=True).dt.normalize()
    news = news[(news["day"] >= START) & (news["day"] <= END)]
    for row in news.itertuples(index=False):
        text = clean_text(row.title)
        if len(text) < 15:
            continue
        symbol = row.ticker if row.ticker in tickers else None
        docs.append(RawDocument(
            doc_id=RawDocument.make_id("gnews_archive", f"{row.ticker}|{row.title}"), source="gnews_archive", kind=SourceKind.NEWS,
            # Only the date is reliable: stamp at the end of the day so it is first usable at the next open.
            published_at=(row.day + pd.Timedelta(hours=23, minutes=59)).to_pydatetime(), text=text, url=row.url,
            publisher=row.publisher or None, meta={"symbol": symbol} if symbol else {}))
    tweets = pd.read_csv(RAW_DIR / "stock_tweets" / "stock_tweets.csv")
    tweets["symbol"] = tweets["Stock Name"].map(lambda s: TWEET_SYMBOL.get(s, s))
    tweets = tweets[tweets["symbol"].isin(tickers)]
    tweets["ts"] = pd.to_datetime(tweets["Date"], utc=True)
    for row in tweets.itertuples(index=False):
        text = clean_text(row.Tweet)
        if len(text) < 15:
            continue
        docs.append(RawDocument(
            doc_id=RawDocument.make_id("tweet_archive", f"{row.Date}|{text[:60]}"), source="tweet_archive", kind=SourceKind.SOCIAL,
            published_at=row.ts.to_pydatetime(), text=text, finance_feed=True, meta={"symbol": row.symbol}))
    return sorted(docs, key=lambda d: d.published_at)


def trading_days() -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(DATA_DIR / "prices" / "constituents_daily.csv")
    opens = prices.pivot(index="date", columns="ticker", values="open")
    closes = prices.pivot(index="date", columns="ticker", values="close")
    opens.index = pd.to_datetime(opens.index)
    closes.index = pd.to_datetime(closes.index)
    window = (opens.index >= START) & (opens.index <= END)
    return opens[window], closes[window]


def compute_signals(model_name: str = "auto") -> pd.DataFrame:
    """Run the engine over the corpus and snapshot every constituent at each market open (14:30 UTC)."""
    settings, universe, taxonomy = load_settings(), load_universe(), load_taxonomy()
    model = RuleTextModel(taxonomy) if model_name == "rules" else load_text_model(settings, taxonomy)
    engine = RiskEngine(settings, universe, taxonomy, model)
    docs = build_corpus()
    opens, _ = trading_days()
    tickers = list(universe.index.constituents)
    times = np.array([d.published_at.timestamp() for d in docs])
    rows, cursor = [], 0
    print(f"  {model.name}: {len(docs):,} documents over {len(opens)} trading days", flush=True)
    for day in opens.index:
        open_time = datetime(day.year, day.month, day.day, 14, 30, tzinfo=timezone.utc)
        upto = int(np.searchsorted(times, open_time.timestamp(), side="left"))
        for start in range(cursor, upto, 500):
            engine.process(docs[start:min(start + 500, upto)])
        cursor = upto
        for t in tickers:
            sig = engine.entity_signal(t, now=open_time)
            rows.append({"date": day.strftime("%Y-%m-%d"), "ticker": t, "sentiment": sig.sentiment_score,
                         "uncertainty": sig.sentiment_uncertainty, "evidence": sig.evidence_weight})
    return pd.DataFrame(rows)


def simulate(signals: pd.DataFrame, opens: pd.DataFrame, cfg: TiltConfig, sectors: np.ndarray, tilt: bool = True) -> dict:
    """Daily rebalance at the open on that morning's signal; earn open-to-open returns; pay costs."""
    tickers = list(opens.columns)
    panel = signals.pivot(index="date", columns="ticker", values="sentiment").reindex(columns=tickers).fillna(0.0)
    panel.index = pd.to_datetime(panel.index)
    rets = opens.shift(-1) / opens - 1.0
    parent = np.full(len(tickers), 1.0 / len(tickers))
    weights = parent.copy()
    level, levels, daily, turnovers = 1000.0, [], [], []
    for day in opens.index[:-1]:
        sentiment = panel.loc[day].to_numpy() if (tilt and day in panel.index) else np.zeros(len(tickers))
        result = rebalance(parent, weights, sentiment, sectors, cfg)
        r = rets.loc[day].to_numpy()
        gross = float(result.weights @ r)
        net = (1.0 - result.cost) * (1.0 + gross) - 1.0
        level *= 1.0 + net
        drifted = result.weights * (1.0 + r)
        weights = drifted / drifted.sum()
        levels.append((day.strftime("%Y-%m-%d"), round(level, 3)))
        daily.append(net)
        turnovers.append(result.turnover)
    return {"levels": levels, "returns": np.asarray(daily), "turnover": np.asarray(turnovers)}


def metrics(strategy: dict, benchmark: dict) -> dict:
    r, b = strategy["returns"], benchmark["returns"]
    active = r - b
    level = np.cumprod(1 + r)
    drawdown = float((level / np.maximum.accumulate(level) - 1).min())
    ann = lambda x: float((np.prod(1 + x)) ** (TRADING_DAYS / len(x)) - 1)  # noqa: E731
    te = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return {"annual_return": ann(r), "annual_volatility": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "sharpe_rf0": float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)), "max_drawdown": drawdown,
            "excess_return_vs_benchmark": ann(r) - ann(b), "tracking_error": te,
            "information_ratio": float(active.mean() * TRADING_DAYS / te) if te > 0 else 0.0,
            "avg_daily_turnover": float(strategy["turnover"].mean()), "days_outperforming": float((active > 0).mean())}


def information_coefficient(signals: pd.DataFrame, opens: pd.DataFrame) -> dict:
    """Daily cross-sectional rank correlation between the morning signal and the next open-to-open return."""
    panel = signals.pivot(index="date", columns="ticker", values="sentiment").reindex(columns=opens.columns)
    panel.index = pd.to_datetime(panel.index)
    rets = (opens.shift(-1) / opens - 1.0)
    ics = []
    for day in opens.index[:-1]:
        s, r = panel.loc[day], rets.loc[day]
        if s.abs().sum() == 0 or s.nunique() < 5:
            continue
        ics.append(s.rank().corr(r.rank()))
    ics = np.asarray([x for x in ics if np.isfinite(x)])
    return {"mean_ic": float(ics.mean()), "ic_std": float(ics.std(ddof=1)), "t_stat": float(ics.mean() / ics.std(ddof=1) * np.sqrt(len(ics))),
            "days": int(len(ics)), "share_positive": float((ics > 0).mean())}


def run_backtest(rebuild_signals: bool = False) -> dict:
    universe = load_universe()
    ent = universe.by_id()
    opens, _ = trading_days()
    opens = opens[list(universe.index.constituents)]
    sectors = np.array([ent[t].sector for t in opens.columns])
    naive_path = SIGNALS_PATH.with_name("signals_daily_naive.csv")
    if rebuild_signals or not SIGNALS_PATH.exists():
        print("computing daily signals with the TREMOR engine ...", flush=True)
        SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        compute_signals("auto").to_csv(SIGNALS_PATH, index=False)
    if rebuild_signals or not naive_path.exists():
        print("computing daily signals with the naive keyword model ...", flush=True)
        compute_signals("rules").to_csv(naive_path, index=False)
    signals, naive = pd.read_csv(SIGNALS_PATH), pd.read_csv(naive_path)

    cfg = TiltConfig()
    tremor = simulate(signals, opens, cfg, sectors)
    keyword = simulate(naive, opens, cfg, sectors)
    bench = simulate(signals, opens, cfg, sectors, tilt=False)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": [START, END], "constituents": list(opens.columns), "methodology": cfg.model_dump(),
        "description": "Daily open-to-open rebalancing on the engine's filtered sentiment, 5 bp per unit of turnover; "
                       "benchmark = the same 20 stocks equally weighted with the same costs. Headlines are only usable from the "
                       "next day's open. One year, 20 stocks: a small sample, so read the IC t-statistic before reading the return.",
        "strategies": {
            "TREMOR-20 (fine-tuned engine)": {**metrics(tremor, bench), "ic": information_coefficient(signals, opens)},
            "Naive keyword sentiment": {**metrics(keyword, bench), "ic": information_coefficient(naive, opens)},
            "Equal-weight benchmark": metrics(bench, bench),
        },
        "series": [{"name": "TREMOR-20", "points": tremor["levels"]}, {"name": "Equal weight", "points": bench["levels"]},
                   {"name": "Naive keyword tilt", "points": keyword["levels"]}],
    }
    t, k = report["strategies"]["TREMOR-20 (fine-tuned engine)"], report["strategies"]["Naive keyword sentiment"]
    report["metrics"] = {
        "Excess return vs benchmark (%)": t["excess_return_vs_benchmark"] * 100,
        "Information ratio": t["information_ratio"],
        "Mean daily IC": t["ic"]["mean_ic"],
        "IC t-statistic": t["ic"]["t_stat"],
        "Naive tilt: excess return (%)": k["excess_return_vs_benchmark"] * 100,
        "Avg daily turnover (%)": t["avg_daily_turnover"] * 100,
    }
    report["metric_notes"] = {"Mean daily IC": "rank corr. of signal with next-day return", "IC t-statistic": "|t| > 2 ~ significant",
                              "Information ratio": "annual excess / tracking error"}
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, m in report["strategies"].items():
        ic = m.get("ic")
        print(f"{name:32s} return {m['annual_return']:+.2%}  vol {m['annual_volatility']:.1%}  maxDD {m['max_drawdown']:+.1%}  "
              f"excess {m['excess_return_vs_benchmark']:+.2%}  IR {m['information_ratio']:+.2f}"
              + (f"  IC {ic['mean_ic']:+.4f} (t {ic['t_stat']:+.2f}, {ic['days']} days)" if ic else ""))
    print("saved", RESULTS_PATH)
    return report
