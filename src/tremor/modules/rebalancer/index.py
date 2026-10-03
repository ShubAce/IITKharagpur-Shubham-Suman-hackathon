"""The live sentiment-tilted index (Module A).

Subscribes to entity sentiment signals and rebalances a 20-stock index:

* on a schedule (every ``rebalance_minutes`` of event time), and
* tactically, the moment a constituent's sentiment jumps or a high-impact event hits it.

Index arithmetic is done the way index providers do it - with units (shares) fixed between
rebalances - so the level is correct however irregularly prices arrive. A benchmark rebalanced
to the parent weights at the same moments, paying the same trading cost, isolates the effect
of the sentiment tilt. Every rebalance records *why* each weight moved (sentiment + the headline
behind it), so the dashboard can answer "why did Boeing's weight drop?".
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pydantic import Field

from tremor.config import Universe
from tremor.modules.rebalancer.methodology import TiltConfig, rebalance
from tremor.schemas import EntitySignal


class IndexConfig(TiltConfig):
    rebalance_minutes: float = 60.0  # scheduled rebalance interval (event time)
    tactical_sentiment_jump: float = 0.20  # off-cycle rebalance if a constituent's sentiment moves this much ...
    tactical_impact: float = 7.0  # ... or a constituent is hit by an event at least this severe
    tactical_min_evidence: float = 1.5  # ... and the move rests on at least this much evidence (not one stray post)
    min_tactical_minutes: float = 15.0  # tactical rebalances at most this often
    base_value: float = 1000.0
    history_limit: int = Field(default=5000)


class PriceBook:
    """Last known price of each ticker at any timestamp (prices are a step function)."""

    def __init__(self, frame: pd.DataFrame):
        self.update(frame)

    def update(self, frame: pd.DataFrame) -> None:
        """Replace the price history in place (the live feed refreshes the object the index holds)."""
        frame = frame.sort_index()
        frame.index = pd.to_datetime(frame.index, utc=True)
        self.frame = frame.ffill()

    def at(self, ts: datetime) -> pd.Series | None:
        if self.frame.empty:
            return None
        idx = self.frame.index.searchsorted(pd.Timestamp(ts), side="right") - 1
        return None if idx < 0 else self.frame.iloc[idx]


class SentimentIndex:
    def __init__(self, universe: Universe, cfg: IndexConfig | None = None, prices: PriceBook | None = None):
        self.cfg = cfg or IndexConfig()
        ent = universe.by_id()
        self.tickers = list(universe.index.constituents)
        self.names = {t: ent[t].name for t in self.tickers}
        self.sectors = np.array([ent[t].sector for t in self.tickers])
        n = len(self.tickers)
        self.parent = np.full(n, 1.0 / n)
        self.weights = self.parent.copy()
        self.sentiment = np.zeros(n)
        self.uncertainty = np.full(n, 0.35)
        self.impact = np.ones(n)
        self.drivers: dict[str, dict] = {}  # ticker -> latest event behind its signal
        self.prices = prices
        self._units: np.ndarray | None = None
        self._bench_units: np.ndarray | None = None
        self.level = self.bench_level = self.cfg.base_value
        self.last_rebalance: datetime | None = None
        self._sentiment_at_rebalance = np.zeros(n)
        self._pending: list[str] = []
        self._acted_on: set[str | None] = set()  # high-impact events that already caused a tactical rebalance
        self.history: list[dict] = []
        self.rebalances: list[dict] = []
        self.total_turnover = 0.0
        self.total_cost = 0.0

    # ------------------------------------------------------------------ inputs
    def on_signal(self, sig: EntitySignal, driver: dict | None = None) -> None:
        """``driver``: the document that best explains the signal (shown next to each weight change)."""
        if sig.entity_id not in self.tickers:
            return
        i = self.tickers.index(sig.entity_id)
        self.sentiment[i], self.uncertainty[i], self.impact[i] = sig.sentiment_score, sig.sentiment_uncertainty, sig.impact_score
        if driver:
            self.drivers[sig.entity_id] = driver
        if sig.evidence_weight < self.cfg.tactical_min_evidence:
            return
        if abs(sig.sentiment_score - self._sentiment_at_rebalance[i]) >= self.cfg.tactical_sentiment_jump:
            self._pending.append(f"{sig.entity_id} sentiment moved to {sig.sentiment_score:+.2f}")
        elif (sig.impact_score >= self.cfg.tactical_impact and sig.top_event_id not in self._acted_on
              and sig.entity_id not in " ".join(self._pending)):
            self._acted_on.add(sig.top_event_id)
            self._pending.append(f"{sig.entity_id} hit by impact-{sig.impact_score:.1f} event")

    # ------------------------------------------------------------------ actions
    def maybe_rebalance(self, now: datetime) -> dict | None:
        if self.last_rebalance is None:
            return self.rebalance(now, "initial allocation")
        if self._pending and now - self.last_rebalance >= timedelta(minutes=self.cfg.min_tactical_minutes):
            reason = "tactical: " + "; ".join(dict.fromkeys(self._pending))
            return self.rebalance(now, reason[:240])
        if now - self.last_rebalance >= timedelta(minutes=self.cfg.rebalance_minutes):
            return self.rebalance(now, "scheduled")
        return None

    def rebalance(self, now: datetime, reason: str) -> dict:
        prices = self.prices.at(now) if self.prices else None
        self._mark_to(prices)
        before = self.weights.copy()
        result = rebalance(self.parent, before, self.sentiment, self.sectors, self.cfg)
        self.weights = result.weights
        self.level *= 1.0 - result.cost
        bench_turnover = 0.5 * float(np.abs(self.parent - self._bench_weights(prices)).sum()) if prices is not None else 0.0
        self.bench_level *= 1.0 - bench_turnover * self.cfg.cost_bps / 1e4
        if prices is not None:
            px = prices.reindex(self.tickers).to_numpy(float)
            self._units = self.weights * self.level / px
            self._bench_units = self.parent * self.bench_level / px
        self.total_turnover += result.turnover
        self.total_cost += result.cost
        self.last_rebalance = now
        self._sentiment_at_rebalance = self.sentiment.copy()
        self._pending.clear()

        changes = []
        for i in np.argsort(-np.abs(self.weights - before)):
            delta = self.weights[i] - before[i]
            if abs(delta) < 0.0025:
                break
            t = self.tickers[i]
            changes.append({"ticker": t, "name": self.names[t], "from": round(float(before[i]), 4), "to": round(float(self.weights[i]), 4),
                            "sentiment": round(float(self.sentiment[i]), 3), "driver": self.drivers.get(t)})
        record = {"time": now.isoformat(), "reason": reason, "turnover": round(result.turnover, 4), "cost_bps": round(result.cost * 1e4, 2),
                  "level": round(self.level, 3), "changes": changes[:8]}
        self.rebalances.append(record)
        self._record(now)
        return record

    def mark(self, now: datetime) -> None:
        """Update the level from prices without trading (for the chart)."""
        if self.prices is None:
            return
        self._mark_to(self.prices.at(now))
        self._record(now)

    # ------------------------------------------------------------------ views
    def snapshot(self) -> dict:
        rows = [{"ticker": t, "name": self.names[t], "sector": str(self.sectors[i]), "weight": round(float(self.weights[i]), 5),
                 "parent": round(float(self.parent[i]), 5), "active": round(float(self.weights[i] - self.parent[i]), 5),
                 "sentiment": round(float(self.sentiment[i]), 4), "uncertainty": round(float(self.uncertainty[i]), 4),
                 "impact": round(float(self.impact[i]), 2), "driver": self.drivers.get(t)}
                for i, t in enumerate(self.tickers)]
        return {"name": "TREMOR-20", "level": round(self.level, 3), "benchmark": round(self.bench_level, 3),
                "excess_return_pct": round((self.level / self.bench_level - 1) * 100, 3),
                "last_rebalance": self.last_rebalance.isoformat() if self.last_rebalance else None,
                "rebalances": len(self.rebalances), "total_turnover": round(self.total_turnover, 4),
                "total_cost_bps": round(self.total_cost * 1e4, 2), "constituents": rows, "methodology": self.cfg.model_dump()}

    # ------------------------------------------------------------------ internals
    def _bench_weights(self, prices: pd.Series | None) -> np.ndarray:
        if prices is None or self._bench_units is None:
            return self.parent
        values = self._bench_units * prices.reindex(self.tickers).to_numpy(float)
        return values / values.sum()

    def _mark_to(self, prices: pd.Series | None) -> None:
        if prices is None or self._units is None:
            return
        px = prices.reindex(self.tickers).to_numpy(float)
        self.level = float((self._units * px).sum())
        self.bench_level = float((self._bench_units * px).sum())
        values = self._units * px
        self.weights = values / values.sum()  # weights drift with prices between rebalances

    def _record(self, now: datetime) -> None:
        self.history.append({"time": now.isoformat(), "level": round(self.level, 3), "benchmark": round(self.bench_level, 3),
                             "weights": {t: round(float(w), 5) for t, w in zip(self.tickers, self.weights)},
                             "sentiment": {t: round(float(s), 4) for t, s in zip(self.tickers, self.sentiment)}})
        if len(self.history) > self.cfg.history_limit:
            self.history = self.history[-self.cfg.history_limit:]
