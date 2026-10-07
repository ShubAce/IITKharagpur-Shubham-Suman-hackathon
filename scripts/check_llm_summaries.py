"""Ask the local language model for the executive summary of every replayed stress test, and keep the record.

The risk memo's opening paragraph can be drafted by a local model (``tremor.modules.stress.narrative``):
the model writes placeholders, the engine inserts every figure, checks reject a draft that writes a number
of its own or misstates the capital position, and a rejected draft goes back with the reasons at most
twice before the memo falls back to a template. This script measures how often that works on real
material - the stress tests of both crisis replays - and saves every accepted summary and every rejected
draft to ``docs/results/llm_summaries.json``, so the accepted ones can be read and audited by hand: the
checks catch numbers and capital claims, not every false sentence.

    python scripts/check_llm_summaries.py        (needs Ollama running with the model in configs/settings.yaml;
                                                  runs the model even though the memo ships with it switched off)
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tremor.config import load_settings, load_taxonomy, load_universe  # noqa: E402
from tremor.engine.runtime import Runtime, load_replay_prices  # noqa: E402
from tremor.ingestion.replay import list_packs, load_pack  # noqa: E402
from tremor.modules.stress.narrative import _ollama, executive_summary  # noqa: E402
from tremor.nlp.models import load_text_model  # noqa: E402

OUT = ROOT / "docs" / "results" / "llm_summaries.json"


def main() -> int:
    settings, universe, taxonomy = load_settings(), load_universe(), load_taxonomy()
    llm = settings.llm.model_copy(update={"enabled": True})  # measured here even though the memo ships with it off
    if _ollama(llm) is None:
        print(f"no language model at {llm.url} serving {llm.model} - start Ollama first")
        return 1
    model = load_text_model(settings, taxonomy)
    rows = []
    for pack in [p["name"] for p in list_packs()]:
        runtime = Runtime(settings, universe, taxonomy, model, sources=[], prices=load_replay_prices())
        docs = load_pack(pack)
        for start in range(0, len(docs), 400):
            runtime.ingest(docs[start:start + 400])
        watch = runtime.watch.snapshot()["entries"]
        print(f"{pack}: {len(runtime.store.stress_runs)} stress tests", flush=True)
        for run in runtime.store.stress_runs:
            t0 = time.perf_counter()
            out = executive_summary(run, watch, llm)
            rows.append({"pack": pack, "detected_at": run.trigger["detected_at"], "headline": run.trigger["headline"],
                         "impact": run.trigger["impact_score"], "capital_breach": run.capital["breach"], "source": out["source"],
                         "attempts": out.get("attempts"), "seconds": round(time.perf_counter() - t0, 1), "note": out["note"],
                         "summary": out["text"] if out["source"] == "llm" else None, "rejected_draft": out.get("rejected_draft")})
            print(f"  impact {run.trigger['impact_score']:.1f}  {out['source']:8s}  {out['note'][:100]}", flush=True)
    accepted = [r for r in rows if r["source"] == "llm"]
    summary = {"model": llm.model, "stress_tests": len(rows), "accepted": len(accepted),
               "accepted_on_attempt": dict(sorted(Counter(r["attempts"] for r in accepted).items())),
               "rejected_last_reason": dict(Counter(r["note"].split("the last because ", 1)[-1].rstrip(")")
                                                    for r in rows if r["source"] != "llm").most_common()),
               "median_seconds": sorted(r["seconds"] for r in rows)[len(rows) // 2] if rows else None}
    OUT.write_text(json.dumps({"summary": summary, "runs": rows}, indent=2, default=str), encoding="utf-8")
    print(f"\naccepted {len(accepted)} of {len(rows)}; saved {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
