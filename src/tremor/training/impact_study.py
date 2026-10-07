"""Does the impact score measure market impact? An event study on a year of company news.

The brief defines the impact score as "a predicted severity score indicating the potential market
impact of the event". This tests that claim where the truth is known. The full engine is replayed over
the Module A corpus (Oct 2021 - Sep 2022: ~49k dated company headlines and ~63k timestamped tweets about
the 20 index stocks), and every company-session gets the highest impact of an event *about* that company
(the event's epicentre - the same rule the stress test notches by). The market's answer is the company's
abnormal move around the news: its close-to-close return from the session before to the session after,
minus the S&P 500's (the [-1, +1] window of a standard daily event study; headlines carry a date, not a
time, so the move can fall on either side of the headline's day).

Three simpler yardsticks get the same test:
    headline volume    how many headlines and posts about the company appeared (attention)
    tone intensity     the average |sentiment| of those documents
    keyword engine     the same impact scorecard driven by the keyword model instead of the fine-tuned one

The company-day table is cached in ``data/backtest/impact_daily.csv`` (the raw corpus is not committed),
so ``python main.py impact`` re-runs the statistics from committed data in seconds; ``--rebuild`` replays
the corpus through both engines (about 15 minutes on a laptop CPU).
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

from tremor.config import load_settings, load_taxonomy, load_universe
from tremor.engine.pipeline import RiskEngine
from tremor.modules.rebalancer.backtest import END, START, build_corpus
from tremor.modules.stress.scenarios import epicentre_of
from tremor.nlp.models import RuleTextModel, load_text_model
from tremor.paths import DATA_DIR, DOCS_DIR

DAILY_PATH = DATA_DIR / "backtest" / "impact_daily.csv"
RESULTS_PATH = DOCS_DIR / "results" / "impact_study.json"
BUCKETS = (("no event", 0.0, 0.0), ("below 4", 0.01, 4.0), ("4 to 5", 4.0, 5.0), ("5 to 6", 5.0, 6.0), ("6 to 7", 6.0, 7.0),
           ("7 and above", 7.0, 10.01))
MEASURES = {"impact": "TREMOR impact score", "impact_rules": "Impact score from the keyword engine",
            "volume": "Headline volume (documents about the company)", "tone": "Tone intensity (mean absolute sentiment)"}


# --------------------------------------------------------------------------- the replay
def replay(model_name: str) -> pd.DataFrame:
    """One engine pass over the corpus: per (date, ticker) the highest impact of an event about the company,
    its event type, and the mean |sentiment| of the documents naming it."""
    settings, universe, taxonomy = load_settings(), load_universe(), load_taxonomy()
    model = RuleTextModel(taxonomy) if model_name == "rules" else load_text_model(settings, taxonomy)
    engine = RiskEngine(settings, universe, taxonomy, model)
    tickers = frozenset(universe.index.constituents)
    by_day: dict = defaultdict(list)
    for doc in build_corpus():
        by_day[doc.published_at.date()].append(doc)
    best: dict = {}
    tone_sum: dict = defaultdict(float)
    tone_n: dict = defaultdict(int)
    print(f"  {model.name}: {sum(map(len, by_day.values())):,} documents over {len(by_day)} days", flush=True)
    for day in sorted(by_day):
        docs, states = by_day[day], {}
        for start in range(0, len(docs), 500):
            result = engine.process(docs[start:start + 500])
            for sig in result.docs:
                for t in tickers.intersection(sig.entities):
                    tone_sum[(day, t)] += abs(sig.entity_sentiment.get(t, sig.sentiment_score))
                    tone_n[(day, t)] += 1
            states.update({ev.event_id: ev for ev in result.events})
        for ev in states.values():  # each event as it stood at the end of the day
            if ev.market_wide:
                continue
            for t in tickers.intersection(epicentre_of(ev)):
                if ev.impact_score > best.get((day, t), (0.0, ""))[0]:
                    best[(day, t)] = (ev.impact_score, ev.event_type)
    keys = sorted(set(best) | set(tone_n))
    return pd.DataFrame({"date": [k[0].isoformat() for k in keys], "ticker": [k[1] for k in keys],
                         "impact": [best.get(k, (0.0, ""))[0] for k in keys], "event_type": [best.get(k, (0.0, ""))[1] for k in keys],
                         "tone": [round(tone_sum[k] / tone_n[k], 4) if tone_n[k] else np.nan for k in keys]})


def build_daily() -> pd.DataFrame:
    """The company-day table: both engines' impact, tone, and headline / post counts straight from the corpus."""
    tremor = replay("auto")
    rules = replay("rules")[["date", "ticker", "impact"]].rename(columns={"impact": "impact_rules"})
    tickers = set(load_universe().index.constituents)
    counts: dict = defaultdict(lambda: [0, 0])
    for doc in build_corpus():
        symbol = doc.meta.get("symbol")
        if symbol in tickers:
            counts[(doc.published_at.date().isoformat(), symbol)][0 if doc.kind.value == "news" else 1] += 1
    volume = pd.DataFrame([{"date": d, "ticker": t, "n_news": n, "n_social": s} for (d, t), (n, s) in counts.items()])
    daily = tremor.merge(rules, on=["date", "ticker"], how="outer").merge(volume, on=["date", "ticker"], how="outer")
    daily[["impact", "impact_rules", "n_news", "n_social"]] = daily[["impact", "impact_rules", "n_news", "n_social"]].fillna(0)
    daily = daily.sort_values(["date", "ticker"]).reset_index(drop=True)
    DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(DAILY_PATH, index=False)
    return daily


# --------------------------------------------------------------------------- the statistics
def company_sessions(daily: pd.DataFrame) -> pd.DataFrame:
    """Fold calendar days onto trading sessions (weekend news meets Monday's market) and attach |CAR[-1,+1]|."""
    prices = pd.read_csv(DATA_DIR / "prices" / "constituents_daily.csv")
    closes = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    closes.index = pd.to_datetime(closes.index)
    sessions = closes.index
    window = closes.shift(-1) / closes.shift(1) - 1.0  # close of the session before -> close of the session after
    abnormal = window.drop(columns="^GSPC").sub(window["^GSPC"], axis=0)
    in_sample = sessions[(sessions >= START) & (sessions <= END)]

    d = daily.copy()
    d["session"] = sessions[np.searchsorted(sessions, pd.to_datetime(d["date"]), side="right") - 1]
    d["volume"] = d["n_news"] + d["n_social"]
    d["tone_w"] = d["tone"].fillna(0) * d["volume"]
    agg = d.groupby(["session", "ticker"]).agg(impact=("impact", "max"), impact_rules=("impact_rules", "max"),
                                               volume=("volume", "sum"), tone_w=("tone_w", "sum"))
    grid = pd.MultiIndex.from_product([in_sample, sorted(abnormal.columns)], names=["session", "ticker"])
    out = agg.reindex(grid).fillna(0.0).reset_index()
    out["tone"] = np.where(out["volume"] > 0, out["tone_w"] / out["volume"].where(out["volume"] > 0, 1), 0.0)
    out["car"] = [abnormal.at[s, t] for s, t in zip(out["session"], out["ticker"])]
    return out.dropna(subset=["car"]).assign(abs_car=lambda x: x["car"].abs()).drop(columns="tone_w")


def _ols_robust(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and heteroskedasticity-robust (HC1) t-statistics."""
    X = np.column_stack([np.ones(len(y)), X])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    n, k = X.shape
    cov = xtx_inv @ (X.T * resid ** 2) @ X @ xtx_inv * n / (n - k)
    return beta, beta / np.sqrt(np.diag(cov))


def analyse(sessions: pd.DataFrame) -> dict:
    news = sessions[sessions["volume"] > 0]
    spearman = {}
    for key, label in MEASURES.items():
        rho, p = stats.spearmanr(news[key], news["abs_car"])
        top = news[key] >= news[key].quantile(0.9)
        spearman[key] = {"label": label, "rho": round(float(rho), 3), "p_value": float(f"{p:.2g}"), "n": int(len(news)),
                         "top_decile_abs_car_pct": round(float(news.loc[top, "abs_car"].mean()) * 100, 2),
                         "top_decile_lift": round(float(news.loc[top, "abs_car"].mean() / news["abs_car"].mean()), 2)}
    buckets = []
    for label, lo, hi in BUCKETS:
        sel = sessions[(sessions["impact"] == 0)] if hi == 0 else sessions[(sessions["impact"] >= lo) & (sessions["impact"] < hi)]
        buckets.append({"bucket": label, "n": int(len(sel)), "mean_abs_car_pct": round(float(sel["abs_car"].mean()) * 100, 2),
                        "median_abs_car_pct": round(float(sel["abs_car"].median()) * 100, 2)})
    # Does the impact score carry information beyond attention and tone? Standardised predictors, robust t.
    z = lambda s: (s - s.mean()) / (s.std() or 1.0)  # noqa: E731
    X = np.column_stack([z(news["impact"]), z(np.log1p(news["volume"])), z(news["tone"])])
    beta, t = _ols_robust(news["abs_car"].to_numpy() * 100, X)
    regression = {"dependent": "|abnormal return| over sessions -1..+1, percent", "n": int(len(news)),
                  "coefficients": {name: {"per_sd": round(float(b), 3), "t_robust": round(float(tt), 2)}
                                   for name, b, tt in zip(("intercept", "impact", "log_volume", "tone"), beta, t)}}
    return {"description": __doc__.strip().split("\n\n")[0].replace("\n", " "), "window": [START, END],
            "company_sessions": int(len(sessions)), "with_news": int(len(news)),
            "with_firm_event": int((sessions["impact"] > 0).sum()), "spearman": spearman, "buckets": buckets,
            "regression": regression}


def run(rebuild: bool = False) -> dict:
    daily = build_daily() if rebuild or not DAILY_PATH.exists() else pd.read_csv(DAILY_PATH)
    result = analyse(company_sessions(daily))
    RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
