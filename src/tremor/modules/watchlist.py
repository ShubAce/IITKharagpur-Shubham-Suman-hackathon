"""Credit early-warning watchlist: which names need a credit analyst's attention now, and why.

Ratings are deliberately stable - through-the-cycle, decided by committee - so banks and rating
agencies run early-warning systems beside them: a fast signal that says which obligors to review
first. This module turns the engine's signals into that list for every company the engine tracks,
with the book's exposure to each, through a transparent points scorecard (the same shape as the
impact score, so every flag can be explained line by line):

    Credit event       named in a live credit event (default, downgrade, bank run, bailout ...):
                       its impact above 5.5, up to 4 points
    High-impact news   named, in a negative tone, in a live event of another kind scoring 7 or
                       more: 0.5 - 1.5
    Epicentre          the company, or its country, is the epicentre of a situation that has been
                       stress-tested: 0.5 - 1.5, rising with the situation's impact
    Sentiment          filtered news / social sentiment below -0.15 on enough evidence: up to 2,
                       plus 0.5 when it fell by 0.15 or more over the last 24 hours
    Stress migration   downgraded by a live situation's stress test: 0.5 per notch, 2 for a default

    3 points or more: Watch Negative (review now)        1.5 or more: Monitor

Event points count in full only when the company is a *focus* of the event: for a firm-specific event
its epicentre (the rule the stress test notches by) or a company a substantial part of its coverage
speaks of negatively (entity-level sentiment of -0.3 or worse, in 50 reports or a tenth of them), for a
market-wide event one of its three most named entities or one named in a fifth of its reports. A
bystander named in passing gets half, and at most one point from someone else's credit event (the big
banks quoted in coverage of SVB's run). A name
leaves a status only when its score falls 0.5 below the bar (hysteresis), so a score hovering at
the threshold does not flap in and out. A name stays listed while its evidence is live (the engine's
48-hour event window and the decaying sentiment state). The first time each name was flagged is
kept: lead time over formal rating actions is what an early-warning system is for.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta

import pandas as pd

from tremor.config import Universe
from tremor.modules.stress.credit import RATING_ORDER
from tremor.modules.stress.scenarios import epicentre_of
from tremor.schemas import EventSignal

WATCH_NEGATIVE, MONITOR = 3.0, 1.5
HYSTERESIS = 0.5  # a flagged name keeps its status until its score is this far below the threshold
MIN_EVENT_IMPACT = 6.0  # weaker events never move the watchlist, so they are not even indexed
FOCUS_SHARE = 0.2  # share of an event's reports naming a company that makes it a focus of the event
BYSTANDER_CAP = 1.0  # most points a company named in passing in someone else's credit event can get from it
# Named this negatively in this many of an event's reports (or this share of them), a company is one of its subjects.
NEGATIVE_FOCUS_TONE, NEGATIVE_FOCUS_MIN, NEGATIVE_FOCUS_SHARE = -0.3, 50, 0.1
SENTIMENT_LEVEL, SENTIMENT_DROP, MIN_EVIDENCE = -0.15, -0.15, 2.0
TREND_HOURS = 24
RANK = {None: 0, "Monitor": 1, "Watch Negative": 2}


def _status(score: float, previous: str | None) -> str | None:
    """Status for a score, with hysteresis: a flagged name keeps its status until the score is clearly lower."""
    if score >= WATCH_NEGATIVE or (previous == "Watch Negative" and score >= WATCH_NEGATIVE - HYSTERESIS):
        return "Watch Negative"
    if score >= MONITOR or (previous is not None and score >= MONITOR - HYSTERESIS):
        return "Monitor"
    return None


def exposures(book: pd.DataFrame) -> pd.DataFrame:
    """Per obligor: exposure at default (loans and drawn-down revolvers, bonds, credit protection sold),
    the credit protection bought against it, and its rating."""
    lending = book["asset_class"].isin(["loan", "revolver"])
    cds = book["asset_class"] == "cds"
    frame = pd.DataFrame({
        "obligor_id": book["obligor_id"],
        "exposure": (book["drawn"] + book["ccf"] * book["undrawn"]).where(lending, 0.0)
        + book["notional"].where((book["asset_class"] == "bond") | (cds & (book["direction"] < 0)), 0.0),
        "protection": book["notional"].where(cds & (book["direction"] > 0), 0.0),
        "rating": book["rating"],
    })
    return frame.groupby("obligor_id").agg(exposure=("exposure", "sum"), protection=("protection", "sum"), rating=("rating", "first"))


class CreditWatch:
    """Subscribes to event signals; re-scores every tracked company after each batch."""

    def __init__(self, universe: Universe, book: pd.DataFrame, window_hours: float = 48.0):
        self._companies = {e.id: e for e in universe.entities if e.type == "company"}
        self._book = exposures(book)
        self._window = timedelta(hours=window_hours)
        self._events: dict[str, EventSignal] = {}
        self._by_entity: dict[str, set[str]] = defaultdict(set)
        self._sentiment: dict[str, deque] = defaultdict(deque)  # entity -> (time, sentiment) samples, for the 24h trend
        self._status: dict[str, str | None] = {}
        self.first_flagged: dict[str, dict] = {}  # entity -> {"Monitor": iso time, "Watch Negative": iso time}
        self.timeline: list[dict] = []  # every change of status, oldest first
        self.entries: list[dict] = []
        self.as_of: datetime | None = None

    def on_event(self, event: EventSignal) -> None:
        if event.impact_score < MIN_EVENT_IMPACT and event.event_id not in self._events:
            return
        self._events[event.event_id] = event
        for entity_id in event.entities:
            if entity_id in self._companies:
                self._by_entity[entity_id].add(event.event_id)

    # ------------------------------------------------------------------ scoring
    def _trend(self, entity_id: str, now: datetime, sentiment: float) -> float:
        """Change in sentiment over the last 24 hours (from the samples taken at earlier updates)."""
        samples = self._sentiment[entity_id]
        if not samples or now - samples[-1][0] >= timedelta(minutes=10):  # live feeds update every few seconds
            samples.append((now, sentiment))
        while len(samples) > 1 and samples[1][0] <= now - timedelta(hours=TREND_HOURS):
            samples.popleft()
        then_time, then_value = samples[0]
        return sentiment - then_value if then_time <= now - timedelta(hours=TREND_HOURS) else sentiment

    @staticmethod
    def _migrations(runs: list) -> dict[str, tuple[str, str]]:
        """Obligor -> (rating before, worst rating after) across the given stress tests."""
        worst: dict[str, tuple[str, str]] = {}
        for run in runs:
            pos = run.positions
            for obligor, g in pos[pos["rating_after"] != pos["rating"]].groupby("obligor_id"):
                after = max(g["rating_after"], key=RATING_ORDER.index)
                if obligor not in worst or RATING_ORDER.index(after) > RATING_ORDER.index(worst[obligor][1]):
                    worst[obligor] = (g["rating"].iloc[0], after)
        return worst

    @staticmethod
    def _focus(event: EventSignal, entity_id: str) -> bool:
        """Is the company what the event is about, rather than a name in a market wrap?

        A firm-specific event is about its epicentre - the same rule the stress test notches by - and about
        any company a substantial part of its coverage speaks of negatively: Signature Bank's closure was
        reported inside the larger SVB story (213 reports, tone -0.49), as was First Republic's slide (59),
        while the big banks quoted in it (about 30 reports each, 1% of the story) were not its subject."""
        mentions = event.entity_mentions.get(entity_id, 0)
        if not event.market_wide:
            named_negatively = (event.entity_sentiment.get(entity_id, 0.0) <= NEGATIVE_FOCUS_TONE
                                and (mentions >= NEGATIVE_FOCUS_MIN or mentions >= NEGATIVE_FOCUS_SHARE * event.n_docs))
            return entity_id in epicentre_of(event) or named_negatively
        return entity_id in event.entities[:3] or mentions >= FOCUS_SHARE * max(1, event.n_docs)

    def _score(self, entity_id: str, engine, now: datetime, epicentres: dict, migrations: dict) -> list[dict]:
        ent = self._companies[entity_id]
        horizon = now - self._window
        factors = []
        live = [self._events[i] for i in sorted(self._by_entity.get(entity_id, ())) if self._events[i].last_updated >= horizon]
        credit = [e for e in live if e.event_type == "CREDIT_EVENT" and e.impact_score > 5.5]
        if credit:
            top = max(credit, key=lambda e: (self._focus(e, entity_id), e.impact_score))
            focus = self._focus(top, entity_id)
            full = min(4.0, top.impact_score - 5.5)
            factors.append({"name": "Credit event", "points": full if focus else min(BYSTANDER_CAP, 0.5 * full),
                            "detail": f"impact {top.impact_score:.1f}{'' if focus else ', named in passing'}: {top.headline}",
                            "event_id": top.event_id})
        other = [e for e in live if e.event_type != "CREDIT_EVENT" and e.impact_score >= 7.0
                 and e.entity_sentiment.get(entity_id, e.sentiment_score) <= -0.1]
        if other:
            top = max(other, key=lambda e: (self._focus(e, entity_id), e.impact_score))
            focus = self._focus(top, entity_id)
            factors.append({"name": "High-impact news", "points": min(1.5, 0.5 + 0.5 * (top.impact_score - 7.0)) * (1.0 if focus else 0.5),
                            "detail": f"{top.event_type_label}, impact {top.impact_score:.1f}{'' if focus else ', named in passing'}: "
                                      f"{top.headline}", "event_id": top.event_id})
        hit = epicentres.get(entity_id) or epicentres.get(ent.country)
        if hit:
            who = "the company" if entity_id in epicentres else f"its country ({ent.country})"
            factors.append({"name": "Epicentre", "points": min(1.5, max(0.5, 1.5 * (hit["impact_score"] - 6.0) / 3.0)),
                            "detail": f"{who} is at the centre of a stress-tested situation: {hit['headline']} (impact {hit['impact_score']:.1f})"})
        sig = engine.entity_signal(entity_id, now)
        trend = self._trend(entity_id, now, sig.sentiment_score)
        if sig.sentiment_score <= SENTIMENT_LEVEL and sig.evidence_weight >= MIN_EVIDENCE:
            factors.append({"name": "Sentiment", "points": min(2.0, (-sig.sentiment_score - 0.1) * 5),
                            "detail": f"filtered sentiment {sig.sentiment_score:+.2f} on an evidence weight of {sig.evidence_weight:.1f}"})
            if trend <= SENTIMENT_DROP:
                factors.append({"name": "Sentiment trend", "points": 0.5, "detail": f"down {abs(trend):.2f} in the last {TREND_HOURS} hours"})
        if entity_id in migrations:
            before, after = migrations[entity_id]
            notches = RATING_ORDER.index(after) - RATING_ORDER.index(before)
            factors.append({"name": "Stress migration", "points": 2.0 if after == "D" else min(2.0, 0.5 * notches),
                            "detail": f"{before} -> {after} in the latest stress test"})
        for f in factors:
            f["points"] = round(f["points"], 2)
        return factors

    def update(self, engine, store, now: datetime, monitor=None) -> list[dict]:
        """Re-score every tracked company as of ``now``; returns the status changes since the last update.

        ``monitor`` (the stress monitor) says which stress-tested situations are still in the news; without
        it, a situation counts as live for the window after its latest stress test."""
        self.as_of = now
        horizon = now - self._window
        if monitor is not None:
            live = monitor.live_situations(now)
        else:
            latest = {}
            for run in store.stress_runs:
                if run.trigger and "situation" in run.trigger and datetime.fromisoformat(run.trigger["detected_at"]) >= horizon:
                    latest[(run.trigger["event_type"], tuple(run.trigger["situation"]))] = run
            live = [(set(run.trigger["situation"]), run) for run in latest.values()]
        # The epicentre (countries or companies) of each live situation, with the trigger of its most severe run.
        epicentres = {}
        for keys, run in live:
            for key in keys:
                if key not in epicentres or run.trigger["impact_score"] > epicentres[key]["impact_score"]:
                    epicentres[key] = run.trigger
        migrations = self._migrations([run for _, run in live])
        changes, entries = [], []
        for entity_id, ent in self._companies.items():
            factors = self._score(entity_id, engine, now, epicentres, migrations)
            score = round(sum(f["points"] for f in factors), 2)
            previous = self._status.get(entity_id)
            status = _status(score, previous)
            if status != previous:
                self._status[entity_id] = status
                if status and status not in self.first_flagged.get(entity_id, {}):
                    self.first_flagged.setdefault(entity_id, {})[status] = now.isoformat()
                change = {"at": now.isoformat(), "entity_id": entity_id, "name": ent.name, "from": previous, "to": status, "score": score,
                          "reason": max(factors, key=lambda f: f["points"])["detail"] if factors else "evidence has gone stale"}
                self.timeline.append(change)
                changes.append(change)
            if status:
                held = self._book.loc[entity_id] if entity_id in self._book.index else None
                entries.append({
                    "entity_id": entity_id, "name": ent.name, "sector": ent.sector, "country": ent.country, "status": status, "score": score,
                    "rating": None if held is None else held["rating"],
                    "exposure_musd": 0.0 if held is None else round(float(held["exposure"]) / 1e6, 1),
                    "protection_musd": 0.0 if held is None else round(float(held["protection"]) / 1e6, 1),
                    "factors": factors, "first_flagged": self.first_flagged.get(entity_id, {}),
                })
        entries.sort(key=lambda e: (-RANK[e["status"]], -e["score"], e["entity_id"]))
        self.entries = entries
        return changes

    def snapshot(self) -> dict:
        listed = [e for e in self.entries if e["exposure_musd"] > 0]
        return {"as_of": self.as_of.isoformat() if self.as_of else None, "thresholds": {"Watch Negative": WATCH_NEGATIVE, "Monitor": MONITOR},
                "entries": self.entries, "exposure_flagged_musd": round(sum(e["exposure_musd"] for e in listed), 1),
                "timeline": self.timeline[-200:], "first_flagged": self.first_flagged}
