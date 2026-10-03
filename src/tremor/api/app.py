"""FastAPI service: the engine's signals over REST + a live server-sent-event stream, both modules,
replay controls, and the dashboard (static files, no build step).

Interactive API docs are served at /docs.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tremor import __version__
from tremor.config import load_settings, load_taxonomy, load_universe
from tremor.engine.runtime import Runtime, event_summary, load_replay_prices, make_gate, stress_summary
from tremor.ingestion.replay import ReplaySource, list_packs
from tremor.nlp.models import load_text_model
from tremor.paths import DOCS_DIR, STATIC_DIR
from tremor.schemas import SourceKind


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=3, max_length=2000, examples=["Moody's downgrades Boeing to junk as 737 MAX deliveries stall"])
    kind: Literal["news", "social"] = "news"


class StressRequest(BaseModel):
    event_id: str | None = Field(default=None, description="stress the scenario derived from this event")
    shocks: dict[str, float] | None = Field(default=None, description="factor shocks for a custom scenario (overrides)")
    method: Literal["analog", "template"] = "analog"
    title: str = "Custom scenario"


class ReplayControl(BaseModel):
    action: Literal["pause", "resume", "speed", "restart"]
    speed: float | None = None
    pack: str | None = None


def build_sources(mode: str, pack: str | None, speed: float):
    """Replay: one recorded pack plus historical prices. Live: every keyless feed plus live prices."""
    if mode == "replay":
        return [ReplaySource(pack or default_pack(), speed=speed)], load_replay_prices()
    from tremor.ingestion.gdelt import GdeltSource
    from tremor.ingestion.prices import LivePriceSource
    from tremor.ingestion.rss import RssSource, default_feeds
    from tremor.ingestion.social import BlueskySource, StockTwitsSource

    universe = load_universe()
    gate = make_gate(universe, load_taxonomy())
    tickers = list(universe.index.constituents)
    live_prices = LivePriceSource(tickers)
    return [GdeltSource(gate), RssSource(default_feeds(tickers), gate), StockTwitsSource(tickers), BlueskySource(tickers),
            live_prices], live_prices.book


def default_pack() -> str:
    packs = [p["name"] for p in list_packs()]
    if not packs:
        raise RuntimeError("no replay packs in data/replay; run scripts/build_replay.py or start in live mode")
    return "ukraine_2022" if "ukraine_2022" in packs else packs[0]


def create_app(mode: str = "replay", pack: str | None = None, speed: float = 900.0) -> FastAPI:
    settings, universe, taxonomy = load_settings(), load_universe(), load_taxonomy()
    model = load_text_model(settings, taxonomy)
    state: dict = {}

    def make_runtime(pack_name: str | None, replay_speed: float) -> Runtime:
        sources, prices = build_sources(mode, pack_name, replay_speed)
        return Runtime(settings, universe, taxonomy, model, sources, prices=prices, mode=mode)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["runtime"] = make_runtime(pack, speed)
        state["runtime"].start(asyncio.get_running_loop())
        yield
        await state["runtime"].stop()

    app = FastAPI(title="TREMOR - AI/NLP Risk Engine", version=__version__, lifespan=lifespan,
                  description="Real-time risk signals from news and social media, a sentiment-tilted index (Module A) "
                              "and event-triggered stress testing of a wholesale banking book (Module B).")

    def rt() -> Runtime:
        return state["runtime"]

    def replay_source() -> ReplaySource | None:
        return next((s for s in rt().sources if isinstance(s, ReplaySource)), None)

    # ------------------------------------------------------------------ engine
    @app.get("/api/status", tags=["engine"])
    def status():
        r = rt()
        replay = replay_source()
        return {"version": __version__, "mode": r.mode, "model": r.model.name, "clock": r.engine.now.isoformat() if r.engine.now else None,
                "counters": r.store.counters, "engine_stats": r.engine.stats, "sources": [s.name for s in r.sources],
                "replay": replay.status() if replay else None, "busy": r.busy, "events_tracked": len(r.store.events),
                "stress_runs": len(r.store.stress_runs), "trigger_threshold": settings.impact.trigger_threshold}

    @app.get("/api/config", tags=["engine"])
    def config():
        ent = universe.by_id()
        return {"index": [{"ticker": t, "name": ent[t].name, "sector": ent[t].sector} for t in universe.index.constituents],
                "event_types": {k: {"label": v.label, "description": v.description, "base_severity": v.base_severity,
                                    "market_wide": v.market_wide} for k, v in taxonomy.event_types.items()},
                "trigger_threshold": settings.impact.trigger_threshold, "risk_factors": rt().library.factors,
                "templates": rt().library.templates}

    @app.get("/api/events", tags=["signals"])
    def events(min_impact: float = 0.0, event_type: str | None = None, limit: int = Query(50, le=500)):
        items = [e for e in rt().store.events.values() if e.impact_score >= min_impact and (event_type is None or e.event_type == event_type)]
        items.sort(key=lambda e: (-e.impact_score, -e.n_docs))
        return [event_summary(e) for e in items[:limit]]

    @app.get("/api/events/{event_id}", tags=["signals"])
    def event(event_id: str):
        e = rt().store.events.get(event_id)
        if e is None:
            raise HTTPException(404, "unknown event")
        return {"signal": e.model_dump(mode="json"), "history": rt().store.event_history.get(event_id, [])}

    @app.get("/api/entities", tags=["signals"])
    def entities(entity_type: str | None = None):
        r = rt()
        ids = universe.index.constituents if entity_type == "index" else [e.id for e in universe.entities
                                                                          if entity_type in (None, e.type)]
        return [r.engine.entity_signal(i).model_dump(mode="json") for i in ids]

    @app.get("/api/entities/{entity_id}", tags=["signals"])
    def entity(entity_id: str):
        if entity_id not in universe.by_id():
            raise HTTPException(404, "unknown entity")
        r = rt()
        docs = [d.model_dump(mode="json") for d in r.store.docs if entity_id in d.entities][-40:]
        return {"signal": r.engine.entity_signal(entity_id).model_dump(mode="json"),
                "history": r.store.entity_history.get(entity_id, [])[-1000:], "documents": docs}

    @app.get("/api/documents", tags=["signals"])
    def documents(limit: int = Query(100, le=1000), event_id: str | None = None, include_noise: bool = False):
        docs = [d for d in rt().store.docs if (include_noise or d.event_id) and (event_id is None or d.event_id == event_id)]
        return [d.model_dump(mode="json") for d in docs[-limit:]]

    @app.post("/api/analyze", tags=["engine"])
    def analyze(req: AnalyzeRequest):
        """Analyse any text without changing engine state: sentiment, event type, impact (as a single report), entities."""
        return rt().engine.analyse_text(req.text, SourceKind(req.kind))

    @app.get("/api/stream", tags=["engine"])
    async def stream(request: Request):
        queue = rt().bus.open_stream()

        async def events_gen():
            try:
                yield "retry: 3000\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        message = await asyncio.wait_for(queue.get(), timeout=15)
                        yield f"data: {message}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
            finally:
                rt().bus.close_stream(queue)

        return StreamingResponse(events_gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------------ module A
    @app.get("/api/index", tags=["module A - index"])
    def index_snapshot():
        return rt().index.snapshot()

    @app.get("/api/index/history", tags=["module A - index"])
    def index_history(limit: int = Query(2000, le=5000)):
        return rt().index.history[-limit:]

    @app.get("/api/index/rebalances", tags=["module A - index"])
    def index_rebalances(limit: int = Query(100, le=1000)):
        return list(reversed(rt().index.rebalances[-limit:]))

    @app.get("/api/backtest", tags=["module A - index"])
    def backtest():
        path = DOCS_DIR / "results" / "backtest.json"
        if not path.exists():
            return {"available": False, "hint": "run `python main.py backtest`"}
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ module B
    @app.get("/api/portfolio", tags=["module B - stress"])
    def portfolio():
        book = rt().stress.book
        def summary(col):
            g = book.groupby(col)["notional"].agg(["count", "sum"]).sort_values("sum", ascending=False)
            return [{"name": k, "positions": int(v["count"]), "notional": float(v["sum"])} for k, v in g.iterrows()]
        return {"positions": len(book), "notional": float(book["notional"].sum()), "by_asset_class": summary("asset_class"),
                "by_sector": summary("sector"), "by_region": summary("region"), "by_rating": summary("rating"),
                "by_source": summary("source")}

    @app.get("/api/stress/runs", tags=["module B - stress"])
    def stress_runs():
        return [stress_summary(run) for run in reversed(rt().store.stress_runs)]

    @app.get("/api/stress/runs/{run_id}", tags=["module B - stress"])
    def stress_run(run_id: str, positions: bool = False):
        run = next((r for r in rt().store.stress_runs if r.run_id == run_id), None)
        if run is None:
            raise HTTPException(404, "unknown run")
        return run.to_dict(include_positions=positions)

    @app.post("/api/stress/run", tags=["module B - stress"])
    def stress_run_now(req: StressRequest):
        """Run a stress test on demand: for an event, for a template, or with analyst-edited shocks."""
        from tremor.modules.stress.engine import run_stress

        r = rt()
        if req.event_id:
            event = r.store.events.get(req.event_id)
            if event is None:
                raise HTTPException(404, "unknown event")
            scenario = r.library.for_event(event, prefer=req.method)
            if req.shocks:
                scenario.shocks.update(req.shocks)
                scenario.method, scenario.narrative = "custom", scenario.narrative + " Analyst overrides applied."
            trigger = {"event_id": event.event_id, "headline": event.headline, "event_type": event.event_type,
                       "event_type_label": event.event_type_label, "impact_score": event.impact_score, "rule": "manual run"}
        else:
            if not req.shocks:
                raise HTTPException(422, "give an event_id or a set of shocks")
            try:
                scenario = r.library.custom(req.title, req.shocks)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            trigger = None
        result = run_stress(r.stress.book, scenario, r.library, trigger)
        return result.to_dict()

    # ------------------------------------------------------------------ evaluation and replay
    @app.get("/api/evaluation", tags=["evaluation"])
    def evaluation():
        out = {}
        for name in ("evaluation", "finbert_reference", "encoder_selection", "finetune", "backtest_summary"):
            path = DOCS_DIR / "results" / f"{name}.json"
            if path.exists():
                out[name] = json.loads(path.read_text(encoding="utf-8"))
        return out

    @app.get("/api/replay/packs", tags=["replay"])
    def packs():
        return list_packs()

    @app.post("/api/replay/control", tags=["replay"])
    async def replay_control(cmd: ReplayControl):
        if mode != "replay":
            raise HTTPException(409, "the service is running on live feeds")
        source = replay_source()
        if cmd.action == "pause":
            source.pause()
        elif cmd.action == "resume":
            source.resume()
        elif cmd.action == "speed":
            source.set_speed(cmd.speed or source.speed)
        else:  # restart, optionally with another pack
            await rt().stop()
            state["runtime"] = make_runtime(cmd.pack or source.pack, cmd.speed or source.speed)
            state["runtime"].start(asyncio.get_running_loop())
            source = replay_source()
        return source.status()

    # ------------------------------------------------------------------ dashboard
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def home():
            return FileResponse(STATIC_DIR / "index.html")

    return app
