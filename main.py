"""TREMOR - Text-driven Risk Engine for Market Observation and Response.

    python main.py                      dashboard on the Feb-2022 crisis replay (http://127.0.0.1:8000)
    python main.py serve --mode live    dashboard on live feeds (GDELT, RSS, StockTwits, Bluesky)
    python main.py replay               run a replay headless and print what the engine detected
    python main.py analyze "headline"   analyse one piece of text
    python main.py evaluate             accuracy / speed of every model on held-out data
    python main.py validate             point-in-time backtest of the stress scenarios on 21 historical crises
    python main.py impact               event study: does the impact score line up with the market's reaction?
    python main.py backtest             one-year backtest of the sentiment-tilted index
    python main.py train | finetune     rebuild the models (finetune needs requirements-dev.txt)
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from tremor.api.app import create_app

    app = create_app(mode=args.mode, pack=args.pack, speed=args.speed)
    url = f"http://{args.host}:{args.port}"
    print(f"\n  TREMOR dashboard  ->  {url}\n  API docs          ->  {url}/docs\n  mode: {args.mode}"
          + (f" (speed x{args.speed:g})" if args.mode == "replay" else "") + "\n  Ctrl+C to stop\n", flush=True)
    if not args.no_browser:
        threading.Timer(3.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Push a whole replay pack through the full stack (engine + both modules) as fast as possible."""
    from tremor.api.app import default_pack
    from tremor.config import load_settings, load_taxonomy, load_universe
    from tremor.engine.runtime import Runtime, load_replay_prices
    from tremor.ingestion.replay import list_packs, load_pack
    from tremor.nlp.models import load_text_model
    from tremor.paths import DOCS_DIR

    settings, universe, taxonomy = load_settings(), load_universe(), load_taxonomy()
    pack = args.pack or default_pack()
    runtime = Runtime(settings, universe, taxonomy, load_text_model(settings, taxonomy), sources=[], prices=load_replay_prices())
    docs = load_pack(pack)
    print(f"replaying {pack}: {len(docs):,} documents, model = {runtime.model.name}", flush=True)
    t0 = time.perf_counter()
    for start in range(0, len(docs), 400):
        runtime.ingest(docs[start:start + 400])
    elapsed = time.perf_counter() - t0
    store = runtime.store
    top = sorted(store.events.values(), key=lambda e: (-e.impact_score, -e.n_docs))[:8]
    print(f"\n{store.counters['documents']:,} documents in {elapsed:.1f}s ({store.counters['documents'] / elapsed:,.0f}/s): "
          f"{store.counters['duplicates']:,} duplicates folded, {store.counters['noise']:,} judged noise, {len(store.events):,} events")
    print("\nTop events")
    for e in top:
        print(f"  [{e.impact_score:4.1f}] {e.event_type_label:20s} sent {e.sentiment_score:+.2f}  {e.n_stories:4d} stories "
              f"{e.n_publishers:4d} publishers | {e.headline[:80]}")
    print(f"\nModule B - {len(store.stress_runs)} stress test(s) triggered")
    for run in store.stress_runs:
        print(f"  {run.trigger['detected_at'][:16]}  {run.trigger['rule']}\n      {run.trigger['headline'][:90]}\n"
              f"      scenario: {run.scenario.narrative[:150]}\n"
              f"      P&L {run.totals['pnl'] / 1e6:,.0f}m ({run.totals['pnl_pct']:.2f}%), CET1 {run.capital['cet1_ratio_before']}% -> "
              f"{run.capital['cet1_ratio_after']}%")
    watch = runtime.watch.snapshot()
    first = sorted(((min(v.values()), k, v) for k, v in watch["first_flagged"].items()), key=lambda t: t[0])
    names = {e.id: e.name for e in universe.entities}
    print(f"\nCredit watch - {len(first)} companies flagged at some point; first flags:")
    for when, entity_id, statuses in first[:12]:
        print(f"  {when[:16]}  {names[entity_id]:28s} " + ", ".join(f"{s} {t[5:16]}" for s, t in sorted(statuses.items(), key=lambda kv: kv[1])))
    snap = runtime.index.snapshot()
    movers = sorted(snap["constituents"], key=lambda r: -abs(r["active"]))[:6]
    print(f"\nModule A - {snap['rebalances']} rebalances, turnover {snap['total_turnover']:.1%}, level {snap['level']:.1f} vs "
          f"benchmark {snap['benchmark']:.1f}")
    for r in movers:
        print(f"  {r['ticker']:5s} weight {r['weight']:.2%} (parent {r['parent']:.2%})  sentiment {r['sentiment']:+.2f}")
    manifest = next((p for p in list_packs() if p["name"] == pack), {})
    out = DOCS_DIR / "results" / f"replay_{pack}.json"
    out.write_text(json.dumps({"pack": pack, "title": manifest.get("title", pack), "documents": store.counters, "seconds": round(elapsed, 1),
                               "window": {"start": manifest.get("start"), "end": manifest.get("end")},
                               "sources": {"news": manifest.get("news"), "social": manifest.get("social")},
                               "top_events": [e.model_dump(mode="json", exclude={"evidence"}) for e in top],
                               "stress_runs": [r.to_dict() for r in store.stress_runs], "index": snap,
                               "rebalances": runtime.index.rebalances, "watchlist": watch}, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {out}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from tremor.config import load_settings, load_taxonomy, load_universe
    from tremor.engine.pipeline import RiskEngine
    from tremor.nlp.models import load_text_model
    from tremor.schemas import SourceKind

    settings, universe, taxonomy = load_settings(), load_universe(), load_taxonomy()
    engine = RiskEngine(settings, universe, taxonomy, load_text_model(settings, taxonomy))
    print(json.dumps(engine.analyse_text(" ".join(args.text), SourceKind(args.kind)), indent=2))
    return 0


def cmd_evaluate(_: argparse.Namespace) -> int:
    from tremor.training.evaluate import calibrate, evaluate

    print("temperatures:", calibrate())
    evaluate()
    return 0


def cmd_validate(_: argparse.Namespace) -> int:
    """Point-in-time backtest of Module B's scenario generator on the historical crises in the library."""
    from tremor.config import load_settings, load_taxonomy
    from tremor.modules.stress.portfolio import load_portfolio
    from tremor.modules.stress.scenarios import ScenarioLibrary
    from tremor.modules.stress.validation import backtest
    from tremor.nlp.models import load_text_model
    from tremor.paths import DOCS_DIR

    model = load_text_model(load_settings(), load_taxonomy())
    result = backtest(ScenarioLibrary(encode=lambda texts: model.predict(texts).embeddings), load_portfolio())
    ctx = result["summary"]["context"]
    print(f"\n{ctx['episodes']} crises ({ctx['first']} .. {ctx['last']}), each scenario built only from episodes that ended "
          f"before it began.\nUSD 10Y yield fell in {ctx['rates_fell']} of them and rose in {ctx['rates_rose']}.\n")
    print(f"  {'method':42s} {'direction':>9s} {'P&L error mean':>15s} {'median':>7s} {'CET1 err':>9s} {'closer than naive':>18s}")
    for key, s in result["summary"].items():
        if key != "context":
            print(f"  {s['label']:42s} {s['direction_hit_rate']:9.1%} {s['pnl_error_mean_musd']:13,.0f}m {s['pnl_error_median_musd']:6,.0f}m "
                  f"{s['cet1_error_mean_pp']:7.2f}pp {s['closer_than_naive']:>12d}/{ctx['episodes']}")
    out = DOCS_DIR / "results" / "scenario_backtest.json"
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {out}")
    return 0


def cmd_impact(args: argparse.Namespace) -> int:
    """Event study: does the impact score line up with the size of the market's reaction?"""
    from tremor.training.impact_study import run

    result = run(rebuild=args.rebuild)
    print(f"\n{result['with_news']:,} company-sessions with news ({result['company_sessions']:,} in all), "
          f"{result['window'][0]} .. {result['window'][1]}\n")
    print(f"  {'measure':48s} {'Spearman vs |abnormal return|':>30s} {'top-decile |AR|':>16s}")
    for s in result["spearman"].values():
        print(f"  {s['label']:48s} {s['rho']:+21.3f} (p {s['p_value']:.1g}) {s['top_decile_abs_car_pct']:13.2f}%")
    print("\n  impact bucket       sessions   mean |AR|")
    for b in result["buckets"]:
        print(f"  {b['bucket']:18s} {b['n']:9,d} {b['mean_abs_car_pct']:9.2f}%")
    coef = result["regression"]["coefficients"]
    print("\n  |AR| on standardised predictors (robust t): " + ", ".join(
        f"{k} {v['per_sd']:+.2f} (t {v['t_robust']:+.1f})" for k, v in coef.items() if k != "intercept"))
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from tremor.modules.rebalancer.backtest import run_backtest

    run_backtest(rebuild_signals=args.rebuild)
    return 0


def cmd_train(_: argparse.Namespace) -> int:
    from tremor.training.train_heads import train

    train()
    return 0


def cmd_finetune(_: argparse.Namespace) -> int:
    from tremor.training.finetune import finetune

    finetune()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="python main.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="start the API and dashboard (default)")
    serve.add_argument("--mode", choices=["replay", "live"], default="replay")
    serve.add_argument("--pack", default=None, help="replay pack in data/replay (default: the Ukraine 2022 pack)")
    serve.add_argument("--speed", type=float, default=1800.0, help="replay speed: virtual seconds per real second")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--no-browser", action="store_true")
    serve.set_defaults(func=cmd_serve)

    replay = sub.add_parser("replay", help="run a replay pack headless and print the results")
    replay.add_argument("--pack", default=None)
    replay.set_defaults(func=cmd_replay)

    analyze = sub.add_parser("analyze", help="analyse a piece of text")
    analyze.add_argument("text", nargs="+")
    analyze.add_argument("--kind", choices=["news", "social"], default="news")
    analyze.set_defaults(func=cmd_analyze)

    sub.add_parser("evaluate", help="evaluate every model on held-out data").set_defaults(func=cmd_evaluate)
    sub.add_parser("validate", help="point-in-time backtest of the stress-scenario generator on 21 historical crises"
                   ).set_defaults(func=cmd_validate)
    impact = sub.add_parser("impact", help="event study: the impact score against the market's reaction (Oct 2021 - Sep 2022)")
    impact.add_argument("--rebuild", action="store_true", help="replay the corpus through both engines (needs data/raw)")
    impact.set_defaults(func=cmd_impact)
    backtest = sub.add_parser("backtest", help="backtest the sentiment-tilted index (Oct 2021 - Sep 2022)")
    backtest.add_argument("--rebuild", action="store_true", help="recompute the daily sentiment signals from the corpus")
    backtest.set_defaults(func=cmd_backtest)
    sub.add_parser("train", help="re-train the frozen-encoder baseline heads").set_defaults(func=cmd_train)
    sub.add_parser("finetune", help="fine-tune the multi-task encoder (needs requirements-dev.txt)").set_defaults(func=cmd_finetune)

    args = parser.parse_args()
    if args.command is None:
        args = parser.parse_args(["serve"])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
