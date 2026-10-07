"""Write docs/RESULTS.md from the JSON results in docs/results/ (so every number has a source file).

    python scripts/make_results_md.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"
sys.path.insert(0, str(ROOT / "src"))

from tremor.config import load_universe  # noqa: E402


def load(name: str) -> dict | None:
    path = RES / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _numeric(value) -> bool:
    text = str(value).replace(",", "").replace("%", "").replace("+", "").replace("-", "").replace(" ", "")
    head = text.split("(")[0].split(">")[0]
    return head.replace(".", "", 1).isdigit() or str(value).strip() == "-"


def table(header: list[str], rows: list[list]) -> str:
    """Markdown table; a column is right-aligned when every value in it is a number."""
    align = ["---:" if i and rows and all(_numeric(r[i]) for r in rows) else "---" for i in range(len(header))]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def f3(v) -> str:
    return "-" if v is None else f"{v:.3f}"


REPLAYS = {  # pack -> (title, analog-library episode holding what markets actually did)
    "ukraine_2022": ("Russia invades Ukraine (21-24 Feb 2022)", "ukraine_2022"),
    "svb_2023": ("Silicon Valley Bank and the banking contagion (8-15 Mar 2023)", "svb_2023"),
}
REALISM_FACTORS = ["EQ_US", "EQ_EU", "IR_USD_10Y", "CS_IG", "CS_HY", "FX_EUR", "CMD_OIL", "CMD_GOLD", "VOL_VIX"]


def main_situation(rp: dict) -> list[dict]:
    """Stress runs of the replay's most severe situation (the one whose run had the highest impact), oldest first."""
    runs = [r for r in rp["stress_runs"] if r.get("trigger")]
    if not runs:
        return []
    top = max(runs, key=lambda r: r["trigger"]["impact_score"])
    key = (top["trigger"]["event_type"], tuple(top["trigger"]["situation"]))
    return [r for r in runs if (r["trigger"]["event_type"], tuple(r["trigger"]["situation"])) == key]


def realism(rp: dict, realized: dict) -> tuple[list[list], int, int, dict]:
    """The main situation's last scenario against what markets did over the episode window."""
    runs = main_situation(rp)
    if not runs or not realized:
        return [], 0, 0, {}
    last = runs[-1]["scenario"]["shocks"]
    rows, hits = [], 0
    for k in REALISM_FACTORS:
        if realized.get(k):
            ok = last.get(k, 0) * float(realized[k]) > 0
            hits += ok
            rows.append([k, f"{last.get(k, 0):+.1f}", f"{float(realized[k]):+.1f}", "yes" if ok else "no"])
    return rows, hits, len(rows), runs[-1]


def lead_times(rp: dict, actions: list[dict], universe_country: dict[str, str]) -> list[list]:
    """Each public rating action in the replay window against the watchlist's first flag on the same name
    (for a sovereign: the first flag on a company of that country)."""
    first = rp.get("watchlist", {}).get("first_flagged", {})
    window = rp.get("window") or {}
    lo, hi = window.get("start", "0000")[:10], _plus_days(window.get("end"), 7)
    rows = []
    for a in actions:
        day = a["date"]
        if not lo <= day <= hi:
            continue
        names = [a["entity_id"]] if a["entity_id"] in first else [e for e, c in universe_country.items() if c == a["entity_id"] and e in first]
        flags = [(t, s, n) for n in names for s, t in first[n].items()]
        if not flags:
            rows.append([day, a["agency"], a["entity"], a["action"], "not flagged", "-", "-"])
            continue
        when, _, name = min(flags)
        watch = min((t for t, s, _ in flags if s == "Watch Negative"), default=None)
        # Rating actions carry a date, not a time, so lead time is counted in calendar days.
        days = (_parse(day).date() - _parse(when).date()).days
        lead = f"{days} day{'s' if days != 1 else ''} before" if days > 0 else ("same day" if days == 0 else f"{-days} day{'s' if days != -1 else ''} after")
        rows.append([day, a["agency"], a["entity"], a["action"], f"{name} {when[5:16].replace('T', ' ')}",
                     watch[5:16].replace("T", " ") if watch else "-", lead])
    return rows


def _parse(value: str):
    from datetime import datetime, timezone
    moment = datetime.fromisoformat(value if "T" in value else value + "T00:00:00")
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _plus_days(value: str | None, days: int) -> str:
    from datetime import timedelta
    return (_parse(value) + timedelta(days=days)).date().isoformat() if value else "9999-12-31"


def main() -> None:
    ev, fin, ft, enc, bt, sb, pmv = (load(n) for n in ("evaluation", "finbert_reference", "finetune", "encoder_selection", "backtest",
                                                         "scenario_backtest", "price_moves_validation"))
    md = ["# Results", "", "Generated by `python scripts/make_results_md.py` from the JSON files in `docs/results/`.", ""]

    if ev:
        names = list(ev["models"])
        ours = next(n for n in names if n.startswith("TREMOR"))
        order = [ours] + [n for n in names if n != ours]
        md += ["## 1. NLP accuracy on held-out data", "",
               "Same test splits, same inference path as production (the fine-tuned model runs as int8 ONNX on CPU). "
               "No test text was used for training or model selection. FinBERT was itself trained on PhraseBank, so its "
               "PhraseBank score is partly in-sample.", ""]
        tasks = [("News sentiment - PhraseBank (accuracy)", lambda m: m["sentiment"].get("fpb", {}).get("accuracy"), "fpb", "accuracy"),
                 ("Tweet sentiment - TFNS (accuracy)", lambda m: m["sentiment"].get("tfns", {}).get("accuracy"), "tfns", "accuracy"),
                 ("Indian news - SEntFiN (accuracy)", lambda m: m["sentiment"].get("sentfin", {}).get("accuracy"), "sentfin", "accuracy"),
                 ("Targeted - FiQA (accuracy)", lambda m: m["sentiment"].get("fiqa", {}).get("accuracy"), "fiqa", "accuracy"),
                 ("StockTwits self-labels (direction)", lambda m: m["sentiment"].get("stocktwits", {}).get("directional_accuracy"),
                  "stocktwits", "directional_accuracy"),
                 ("Event type, human-labelled (macro-F1)", lambda m: m["event"]["macro_f1"], None, None),
                 ("Entity sentiment, opposite-sentiment headlines (accuracy)", lambda m: m["entity_sentiment"]["conflicting_accuracy"], None, None)]
        header = ["Task"] + [n for n in order] + (["FinBERT (reference)"] if fin else [])
        rows = []
        for label, get, key, metric in tasks:
            row = [label] + [f3(get(ev["models"][n])) for n in order]
            if fin:
                row.append(f3(fin["sentiment"].get(key, {}).get(metric)) if key else "-")
            rows.append(row)
        n_test = {k: v["n"] for k, v in ev["models"][ours]["sentiment"].items()}
        md += [table(header, rows), "", f"Test-set sizes: {n_test}; event types n = {ev['models'][ours]['event']['n']}; "
               f"entity-level n = {ev['models'][ours]['entity_sentiment']['n']} "
               f"({ev['models'][ours]['entity_sentiment']['conflicting_n']} on headlines whose entities have opposite labels).", ""]
        eff_rows = [[n, f"{ev['models'][n]['efficiency']['docs_per_second_batched']:,.0f}",
                     f"{ev['models'][n]['efficiency']['latency_ms_single_p50']:.1f}", ev["models"][n]["efficiency"].get("model_mb", "-")]
                    for n in order]
        if fin:
            eff_rows.append(["FinBERT (PyTorch, CPU)", f"{fin['docs_per_second_batched_cpu']:,.0f}", "-", "438"])
        md += ["## 2. Efficiency (laptop CPU, 8 threads)", "", table(["Model", "Docs / second (batched)", "Latency ms (1 doc, p50)", "Size MB"], eff_rows), ""]
        per_class = ev["models"][ours]["event"]["per_class"]
        md += ["### Event-type F1 per class (TREMOR, human-labelled test set)", "",
               table(["Class", "Precision", "Recall", "F1", "n"], [[k, v["precision"], v["recall"], v["f1"], v["n"]] for k, v in per_class.items()]), ""]

    if enc:
        md += ["## 3. Why fine-tune: frozen-encoder probes plateau", "",
               "Logistic-regression probe on frozen embeddings of 8 candidate encoders (`scripts/select_encoder.py`).", "",
               table(["Encoder", "PhraseBank acc", "TFNS acc", "Topic acc", "Docs / s"],
                     [[k, f3(v["news_sentiment (PhraseBank)"]["accuracy"]), f3(v["tweet_sentiment (TFNS)"]["accuracy"]),
                       f3(v["topic (20 classes)"]["accuracy"]), v["docs_per_second"]] for k, v in enc.items()]), ""]
    if ft:
        md += ["## 4. Fine-tuning run", "",
               f"- Base: `{ft['config']['base_model']}`; {ft['config']['epochs']} epochs; {ft['minutes']} minutes on an RTX 4050 laptop GPU.",
               f"- Training rows: {ft['train_rows']}.",
               f"- Similarity preservation (Spearman of pairwise cosine vs the original encoder): {ft['similarity_preservation_spearman']:.3f}.",
               f"- Exported files (MB): {ft['files']}; softmax temperatures after calibration: {ft.get('temperature', 'see models/tremor-encoder/model.json')}.", ""]

    if bt:
        md += ["## 5. Module A - one-year backtest (Oct 2021 - Sep 2022)", "", bt["description"], ""]
        rows = []
        for name, m in bt["strategies"].items():
            ic = m.get("ic")
            rows.append([name, f"{m['annual_return']:+.2%}", f"{m['annual_volatility']:.1%}", f"{m['max_drawdown']:+.1%}",
                         f"{m['excess_return_vs_benchmark']:+.2%}", f"{m['information_ratio']:+.2f}", f"{m['avg_daily_turnover']:.2%}",
                         f"{ic['mean_ic']:+.4f} (t = {ic['t_stat']:+.2f})" if ic else "-"])
        md += [table(["Strategy", "Annual return", "Volatility", "Max drawdown", "Excess vs EW", "Info. ratio", "Avg daily turnover",
                      "Mean daily IC"], rows), ""]

    if sb:
        s, ctx = sb["summary"], sb["summary"]["context"]
        md += ["## 6. Module B - are the stress scenarios any good? A point-in-time backtest on real crises", "",
               f"Each of {ctx['episodes']} crises ({ctx['first'][:4]}-{ctx['last'][:4]}) is treated as breaking news: its scenario is rebuilt from "
               "the crisis's day-one headline (the trigger, with no market outcome in it: `headline` in `configs/scenarios.yaml`) using only "
               "the episodes that had ended before it began, then compared with what markets did over the episode. Scenarios are compared "
               "unscaled (impact 8.5) and without epicentre notching. *Directions right* = share of the materially moving headline factors "
               f"({', '.join(sb['headline_factors'])}) whose sign the scenario got right (a factor left at zero is a miss); *P&L error* = "
               "|P&L of the book under the scenario - P&L under the realised moves|. `python main.py validate` reproduces it.", "",
               f"The brief's example shock (equities -10%, rates +200 bp) assumes rates rise in a crisis: the 10-year US yield fell in "
               f"{ctx['rates_fell']} of the {ctx['episodes']} crises and rose in {ctx['rates_rose']}.", ""]
        order = [k for k in ("naive", "template", "all_mean", "same_type_mean", "nearest", "tremor") if k in s]
        md += [table(["Method", "Directions right", "Rates right", "Oil right", "P&L error, mean $m", "median $m", "CET1 error pp",
                      "Closer than naive", "Closer than the average crisis"],
                     [[s[k]["label"], f"{s[k]['direction_hit_rate']:.1%}", f"{s[k]['per_factor_hit_rate'].get('IR_USD_10Y', 0):.0%}",
                       f"{s[k]['per_factor_hit_rate'].get('CMD_OIL', 0):.0%}", f"{s[k]['pnl_error_mean_musd']:,.0f}",
                       f"{s[k]['pnl_error_median_musd']:,.0f}", f"{s[k]['cet1_error_mean_pp']:.2f}",
                       f"{s[k]['closer_than_naive']}/{ctx['episodes']}", f"{s[k]['closer_than_average']}/{ctx['episodes']}"] for k in order]), ""]
        rows = []
        for r in sb["episodes"]:
            m = r["methods"]
            analog_mix = ", ".join(f"{a['title'].split(' (')[0]} {a['weight']:.0%}" for a in m["tremor"].get("analogs", [])[:2])
            rows.append([r["title"], r["start"], f"{r['realised_pnl_musd']:,.0f}", f"{m['tremor']['pnl_musd']:,.0f}", f"{m['naive']['pnl_musd']:,.0f}",
                         f"{m['tremor']['hits']}/{m['tremor']['scored']}", f"{m['naive']['hits']}/{m['naive']['scored']}", analog_mix])
        md += ["### Crisis by crisis", "", table(["Crisis", "Start", "Book P&L, realised $m", "TREMOR $m", "Naive $m", "TREMOR directions",
                                                 "Naive directions", "Closest analogs (TREMOR)"], rows), ""]

    if pmv:
        md += ["## 7. Price direction is not sentiment: the direction extractor, checked against prices", "",
               "\"Oil soars on war fears\" is negative in tone but says oil is going up, so the stress scenario reads the direction of the "
               "prices a report names from verbs of movement (`src/tremor/nlp/price_moves.py`), not from sentiment. The rules are checked "
               f"where the truth is known: {pmv['headlines']} company headlines from the Module A corpus that report a move in the company's "
               f"own share price ({pmv['up']} up, {pmv['down']} down), against the stock's actual close-to-close move.", "",
               table(["Reference move", "Agreement"],
                     [["Two sessions around the headline (D-1 to D+1)", f"{pmv['agreement']:.1%}"],
                      [f"... moves above 3% only (n = {pmv['moves_over_3pct']['headlines']})", f"{pmv['moves_over_3pct']['agreement']:.1%}"],
                      ["Headline day only", f"{pmv['agreement_by_session']['headline_day']:.1%}"],
                      ["Always guessing the more common direction (baseline)", f"{pmv['majority_class_baseline']:.1%}"]]), ""]

    imp = load("impact_study")
    if imp:
        sp, reg = imp["spearman"], imp["regression"]["coefficients"]
        md += ["## 7b. Does the impact score measure market impact? An event study on single stocks", "",
               f"The engine replayed over the Module A corpus ({imp['window'][0]} to {imp['window'][1]}): every company-session gets "
               "the highest impact of an event *about* that company (its epicentre); the market's answer is the company's abnormal "
               "move over the sessions before and after the news (return minus the S&P 500's). "
               f"{imp['with_news']:,} company-sessions with news, of {imp['company_sessions']:,}. `python main.py impact` reproduces it "
               "from `data/backtest/impact_daily.csv`.", "",
               table(["Measure", "Spearman with abnormal move", "p-value", "Mean move, top decile"],
                     [[v["label"], f"{v['rho']:+.3f}", f"{v['p_value']:.1g}", f"{v['top_decile_abs_car_pct']:.2f}%"] for v in sp.values()]), "",
               table(["TREMOR impact", "Company-sessions", "Mean abnormal move", "Median"],
                     [[b["bucket"], f"{b['n']:,}", f"{b['mean_abs_car_pct']:.2f}%", f"{b['median_abs_car_pct']:.2f}%"] for b in imp["buckets"]]), "",
               "Regression of the abnormal move (percent) on the standardised measures, robust t-statistics: " + ", ".join(
                   f"{k} {v['per_sd']:+.2f} (t = {v['t_robust']:+.1f})" for k, v in reg.items() if k != "intercept") + ".", "",
               "Reading: the impact score ranks market-moving company news (sessions scored 6 or more move more than lower-scored "
               "news), but on single stocks it adds nothing beyond attention (headline volume) and tone. The scorecard was built to "
               "triage systemic events for stress tests; its event-type priors (earnings, product launches) carry no single-stock "
               "information, and archive headlines carry dates, not times. Calibrating the scorecard's weights on market reactions "
               "is the next step; Module A, the single-stock application, runs on sentiment, which carries the stronger signal here.", ""]

    llm = load("llm_summaries")
    if llm:
        s = llm["summary"]
        md += ["## 7c. A language model for the memo's summary - measured, and switched off", "",
               f"A local {s['model']} can draft the risk memo's opening paragraph: it writes placeholders and the engine inserts every "
               "figure; checks reject a draft with a number of its own, a wrong capital statement, a flipped CET1 direction or a "
               "claim put in the reports' mouth, and a rejected draft goes back with the reasons at most twice before the memo "
               f"falls back to its template. Asked for all {s['stress_tests']} stress tests of the two replays "
               "(`python scripts/check_llm_summaries.py`):", "",
               table(["Outcome", "Stress tests"],
                     [[f"Accepted at attempt {k}", str(v)] for k, v in s["accepted_on_attempt"].items()]
                     + [[f"Template (last reason: {k})", str(v)] for k, v in s["rejected_last_reason"].items()]), "",
               f"Median time per summary: {s['median_seconds']:.0f} s on a laptop GPU. Read by hand (every summary and rejected draft "
               "is in `docs/results/llm_summaries.json`): no accepted summary carries a wrong figure or capital statement, but about "
               "half tie the largest loss channel to the hardest-hit borrowers more tightly than the numbers do, and earlier prompts "
               "invented facts outright (\"the bank harbors dirty money\" - the story was about another bank). Small prompt changes "
               "moved acceptance between 0 and 31 of 33. A risk memo cannot carry prose that needs a second read, so the memo's "
               "summary is deterministic and the model is off by default (`llm.enabled` in `configs/settings.yaml`).", ""]

    analogs = {row["episode_id"]: row for row in csv.DictReader((ROOT / "data" / "scenarios" / "analog_shocks.csv").open(encoding="utf-8"))}
    actions_path = ROOT / "data" / "reference" / "rating_actions.csv"
    actions = list(csv.DictReader(actions_path.open(encoding="utf-8"))) if actions_path.exists() else []
    universe = load_universe()
    countries = {e.id: e.country for e in universe.entities if e.type == "company"}
    names = {e.id: e.name for e in universe.entities}
    section = 7
    for pack, (title, episode) in REPLAYS.items():
        rp = load(f"replay_{pack}")
        if not rp:
            continue
        section += 1
        c, src = rp["documents"], rp.get("sources") or {}
        news, social = src.get("news"), src.get("social")  # counts from the pack manifest
        skipped = (news or 0) + (social or 0) - c["documents"] if news else 0  # no readable text: a file name or a fragment
        md += [f"## {section}. Crisis replay - {title}", "",
               f"{c['documents']:,} documents (" + (f"{news:,} news headlines from GDELT" if news else "news") + (f", {social:,} social posts" if social else "")
               + (f"; {skipped:,} with no readable text, such as a file name, skipped" if skipped > 0 else "")
               + f") processed in {rp['seconds']} s; {c['duplicates']:,} syndicated copies folded into corroboration; {c['noise']:,} judged "
               "noise. The same code and thresholds as every other replay and the live mode.", "", "### Stress tests triggered (Module B)", ""]
        rows = []
        for r in rp["stress_runs"]:
            t = r["trigger"]
            rows.append([t["detected_at"][:16].replace("T", " "), t["event_type_label"], f"{t['impact_score']:.1f}",
                         " ".join(t["situation"]), t.get("reason", ""), t["headline"][:70],
                         f"{r['totals']['pnl'] / 1e6:,.0f}", f"{r['capital']['cet1_ratio_before']} -> {r['capital']['cet1_ratio_after']}"])
        md += [table(["Detected (UTC)", "Type", "Impact", "Situation", "Why it ran", "Event headline", "P&L $m", "CET1 %"], rows), ""]
        snap = rp["index"]
        md += ["### Module A on the replay", "",
               f"{snap['rebalances']} rebalances, one-way turnover {snap['total_turnover']:.1%}, index {snap['level']:.1f} vs equal-weight "
               f"{snap['benchmark']:.1f} (base 1,000, daily prices).", ""]
        rows, hits, total, last = realism(rp, analogs.get(episode))
        if rows:
            t = last["trigger"]
            md += ["### Scenario realism check", "",
                   f"The last stress scenario of the replay's most severe situation ({' '.join(t['situation'])}, impact {t['impact_score']:.1f}, "
                   f"{t['detected_at'][:16].replace('T', ' ')} UTC), built point-in-time from analogs that ended before the crisis, against what "
                   f"markets did over the episode ({analogs[episode]['start']} to {analogs[episode]['end']}). Same sign = direction right: "
                   f"**{hits} of {total}**. The scenario is scaled x{last['scenario']['severity']:.2f} for its impact; the realised moves are not.", "",
                   table(["Factor", "Scenario", "Realised", "Same direction"], rows), ""]
        watch = rp.get("watchlist") or {}
        first = sorted((min(v.values()), k, v) for k, v in (watch.get("first_flagged") or {}).items())
        if first:
            md += ["### Credit early warning", "",
                   "When each company was first flagged, and first put on Watch Negative (the scorecard is in "
                   "`src/tremor/modules/watchlist.py`):", "",
                   table(["Company", "First flagged, Monitor or higher (UTC)", "First Watch Negative (UTC)"],
                         [[names.get(k, k), when[5:16].replace("T", " "), (v.get("Watch Negative") or "-")[5:16].replace("T", " ")]
                          for when, k, v in first[:12]]), ""]
            leads = lead_times(rp, actions, countries)
            if leads:
                md += ["Against the public rating actions of the same weeks (`data/reference/rating_actions.csv`, each with its source). "
                       "Ratings are through-the-cycle and decided by committee by design; early warning tells surveillance teams which "
                       "names to review first:", "",
                       table(["Rating action", "Agency", "Name", "Action", "TREMOR first flag (UTC)", "TREMOR Watch Negative (UTC)", "Lead of first flag"], leads), ""]

    out = ROOT / "docs" / "RESULTS.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
