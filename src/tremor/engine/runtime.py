"""Runtime: sources -> engine -> downstream modules -> store, with a pub/sub bus in between.

    sources (poll loops) --docs--> RiskEngine --signals--> Bus
                                                          |-- "entity" --> Module A (SentimentIndex)
                                                          |-- "event"  --> Module B (StressMonitor)
                                                          '-- every topic --> SignalStore (API, JSONL file) and SSE clients

Downstream modules *subscribe* to signal topics exactly as the brief describes; the in-process
bus is the integration point where Kafka or Redis Streams would sit in production.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from tremor.config import Settings, Taxonomy, Universe
from tremor.engine.pipeline import BatchResult, RiskEngine
from tremor.ingestion.base import Gate, Source
from tremor.modules.rebalancer.index import IndexConfig, PriceBook, SentimentIndex
from tremor.modules.stress.engine import StressMonitor, StressResult
from tremor.modules.stress.portfolio import load_portfolio
from tremor.modules.stress.scenarios import ScenarioLibrary
from tremor.nlp.cues import CueMatcher
from tremor.nlp.entities import EntityLinker
from tremor.nlp.models import TextModel
from tremor.paths import DATA_DIR, OUTPUT_DIR
from tremor.schemas import DocSignal, EntitySignal, EventSignal

log = logging.getLogger(__name__)
MAX_BATCH = 300


class Bus:
    """Synchronous in-process pub/sub plus asyncio queues for streaming clients (SSE)."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[[Any], None]]] = defaultdict(list)
        self._streams: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        self._subscribers[topic].append(callback)

    def publish(self, topic: str, payload: Any) -> None:
        """Deliver a signal to in-process subscribers (the downstream modules)."""
        for callback in self._subscribers.get(topic, []):
            callback(payload)

    def broadcast(self, topic: str, data: Any) -> None:
        """Push a JSON-able summary to every connected dashboard (server-sent events)."""
        if self._streams and self._loop is not None:
            message = json.dumps({"topic": topic, "data": data}, default=str)
            for queue in list(self._streams):
                self._loop.call_soon_threadsafe(_offer, queue, message)

    def open_stream(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._streams.add(queue)
        return queue

    def close_stream(self, queue: asyncio.Queue) -> None:
        self._streams.discard(queue)


def _offer(queue: asyncio.Queue, message: str) -> None:
    if queue.full():  # a slow browser tab must never block the engine: drop its oldest message
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    queue.put_nowait(message)


class SignalStore:
    """Latest state for the API, plus an append-only JSONL file of every published risk signal."""

    def __init__(self, sink_path=OUTPUT_DIR / "signals.jsonl", max_docs: int = 3000):
        self.events: dict[str, EventSignal] = {}
        self.event_history: dict[str, list[dict]] = defaultdict(list)
        self.entities: dict[str, EntitySignal] = {}
        self.entity_history: dict[str, list[dict]] = defaultdict(list)
        self.docs: deque[DocSignal] = deque(maxlen=max_docs)
        self.stress_runs: list[StressResult] = []
        self.counters = {"documents": 0, "duplicates": 0, "noise": 0}
        sink_path.parent.mkdir(parents=True, exist_ok=True)
        self._sink = sink_path.open("a", encoding="utf-8")

    def apply(self, result: BatchResult) -> None:
        self.counters["documents"] += len(result.docs) + result.duplicates
        self.counters["duplicates"] += result.duplicates
        self.counters["noise"] += result.noise
        self.docs.extend(result.docs)
        for event in result.events:
            self.events[event.event_id] = event
            history = self.event_history[event.event_id]
            point = {"time": event.last_updated.isoformat(), "impact": event.impact_score, "docs": event.n_docs, "stories": event.n_stories}
            if not history or history[-1]["impact"] != point["impact"] or history[-1]["docs"] != point["docs"]:
                history.append(point)
            self._write({"scope": "event", "event_id": event.event_id, "as_of": event.last_updated, "headline": event.headline,
                         "event_type": event.event_type, "sentiment_score": event.sentiment_score, "impact_score": event.impact_score,
                         "n_docs": event.n_docs, "n_publishers": event.n_publishers, "entities": event.entities[:6]})
        for ent in result.entities:
            self.entities[ent.entity_id] = ent
            self.entity_history[ent.entity_id].append({"time": ent.as_of.isoformat(), "sentiment": ent.sentiment_score,
                                                       "uncertainty": ent.sentiment_uncertainty, "impact": ent.impact_score})
            self._write({"scope": "entity", "entity_id": ent.entity_id, "as_of": ent.as_of, "sentiment_score": ent.sentiment_score,
                         "sentiment_uncertainty": ent.sentiment_uncertainty, "event_type": ent.event_type, "impact_score": ent.impact_score})
        self._sink.flush()

    def top_evidence(self, entity_id: str, lookback: int = 600) -> dict | None:
        """The most opinionated recent document about an entity: the 'why' behind its sentiment."""
        best, best_score = None, 0.0
        for i, doc in enumerate(reversed(self.docs)):
            if i >= lookback:
                break
            score = doc.entity_sentiment.get(entity_id)
            if score is not None and abs(score) * doc.relevance > best_score:
                best, best_score = doc, abs(score) * doc.relevance
        if best is None:
            return None
        return {"headline": best.text, "sentiment": best.entity_sentiment[entity_id], "source": best.publisher or best.source,
                "published_at": best.published_at.isoformat(), "event_id": best.event_id, "url": best.url}

    def _write(self, row: dict) -> None:
        self._sink.write(json.dumps(row, default=str) + "\n")

    def close(self) -> None:
        self._sink.close()


def load_replay_prices() -> PriceBook | None:
    """Daily constituent prices as a step function: open at 14:30 UTC, close at 21:00 UTC."""
    path = DATA_DIR / "prices" / "constituents_daily.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    opens = df.pivot(index="date", columns="ticker", values="open")
    closes = df.pivot(index="date", columns="ticker", values="close")
    opens.index = pd.to_datetime(opens.index.astype(str) + " 14:30")
    closes.index = pd.to_datetime(closes.index.astype(str) + " 21:00")
    return PriceBook(pd.concat([opens, closes]).sort_index())


class Runtime:
    """Owns the engine, the modules and the poll loops. One instance per API process."""

    def __init__(self, settings: Settings, universe: Universe, taxonomy: Taxonomy, model: TextModel,
                 sources: list[Source], prices: PriceBook | None = None, mode: str = "replay"):
        self.settings, self.universe, self.taxonomy, self.model, self.mode = settings, universe, taxonomy, model, mode
        self.sources = sources
        self.bus = Bus()
        self.store: SignalStore | None = None
        self._lock = threading.Lock()
        self.busy = False  # True while a batch is being analysed
        self._tasks: list[asyncio.Task] = []
        self.started_at = datetime.now(timezone.utc)
        self.build(prices)

    def build(self, prices: PriceBook | None = None) -> None:
        """(Re)create the engine and both modules - used at start and when a replay restarts."""
        self.engine = RiskEngine(self.settings, self.universe, self.taxonomy, self.model)
        self.index = SentimentIndex(self.universe, IndexConfig(), prices)
        self.library = ScenarioLibrary(encode=lambda texts: self.model.predict(texts).embeddings)
        self.stress = StressMonitor(load_portfolio(), self.library, threshold=self.settings.impact.trigger_threshold,
                                    retrigger_delta=self.settings.impact.retrigger_delta)
        if getattr(self, "store", None) is not None:
            self.store.close()
        self.store = SignalStore()
        self.bus._subscribers.clear()
        self.bus.subscribe("entity", self._on_entity)  # Module A subscribes to sentiment
        self.bus.subscribe("event", self._on_event)  # Module B subscribes to event type + impact

    # ------------------------------------------------------------------ subscriptions
    def _on_entity(self, sig: EntitySignal) -> None:
        driver = self.store.top_evidence(sig.entity_id)
        self.index.on_signal(sig, driver)

    def _on_event(self, event: EventSignal) -> None:
        run = self.stress.on_event(event)
        if run is not None:
            self.store.stress_runs.append(run)
            self.bus.broadcast("stress", stress_summary(run))

    # ------------------------------------------------------------------ processing
    def ingest(self, docs: list) -> BatchResult | None:
        """Run documents through the engine and publish everything that changed (thread-safe)."""
        if not docs:
            return None
        with self._lock:
            self.busy = True
            try:
                return self._ingest(docs)
            finally:
                self.busy = False

    def _ingest(self, docs: list) -> BatchResult:
        merged = BatchResult()
        for start in range(0, len(docs), MAX_BATCH):
            result = self.engine.process(docs[start:start + MAX_BATCH])
            self.store.apply(result)
            for event in result.events:
                self.bus.publish("event", event)
            for ent in result.entities:
                self.bus.publish("entity", ent)
            merged.docs += result.docs
            merged.events += result.events
            merged.entities += result.entities
        now = self.engine.now
        if now is not None:
            record = self.index.maybe_rebalance(now)
            if record is not None:
                self.bus.broadcast("index", record)
            else:
                self.index.mark(now)
        top = sorted(merged.events, key=lambda e: -e.impact_score)[:20]
        self.bus.broadcast("batch", {
            "clock": now.isoformat() if now else None, "counters": self.store.counters,
            "docs": [d.model_dump(mode="json", include=DOC_FIELDS) for d in merged.docs if d.event_id][-30:],
            "events": [event_summary(e) for e in top],
            "entities": [e.model_dump(mode="json") for e in merged.entities if e.entity_id in self.index.tickers],
        })
        return merged

    async def _poll_loop(self, source: Source) -> None:
        while True:
            try:
                docs = await asyncio.to_thread(source.poll)
                if docs:
                    await asyncio.to_thread(self.ingest, docs)
            except Exception:  # a failing feed is logged and retried, never fatal
                log.exception("source %s failed", source.name)
            await asyncio.sleep(source.interval_seconds)

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        loop = loop or asyncio.get_event_loop()
        self.bus.bind_loop(loop)
        self._tasks = [loop.create_task(self._poll_loop(s)) for s in self.sources]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        self._tasks = []


DOC_FIELDS = {"doc_id", "published_at", "source", "kind", "publisher", "text", "url", "entities", "sentiment_score",
              "event_type", "relevance", "novelty", "impact_score", "event_id"}


def event_summary(e: EventSignal) -> dict:
    return {"event_id": e.event_id, "headline": e.headline, "event_type": e.event_type, "event_type_label": e.event_type_label,
            "impact_score": e.impact_score, "sentiment_score": e.sentiment_score, "n_docs": e.n_docs, "n_stories": e.n_stories,
            "n_publishers": e.n_publishers, "first_seen": e.first_seen.isoformat(), "last_updated": e.last_updated.isoformat(),
            "entities": e.entities[:6], "regions": e.regions[:4], "market_wide": e.market_wide}


def stress_summary(run: StressResult) -> dict:
    return {"run_id": run.run_id, "created_at": run.created_at.isoformat(), "trigger": run.trigger, "totals": run.totals,
            "capital": run.capital, "credit": run.credit,
            "scenario": {"title": run.scenario.title, "method": run.scenario.method, "severity": run.scenario.severity,
                         "analogs": run.scenario.analogs, "epicentre": run.scenario.epicentre}}


def make_gate(universe: Universe, taxonomy: Taxonomy) -> Gate:
    return Gate(EntityLinker(universe), CueMatcher(taxonomy), [e.id for e in universe.entities if e.type == "company"])
