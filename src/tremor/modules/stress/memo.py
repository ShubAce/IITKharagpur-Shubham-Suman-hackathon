"""The risk memo: one printable page per stress test, written for a chief risk officer.

What a CRO asks the morning a crisis breaks - what happened, how sure are we, what does it do to our
book and our capital, who do we need to look at, what should we do - answered from the signals alone.
Deterministic templates rather than a language model: every sentence comes from a number the engine
produced, so nothing in the memo can be invented, and the same run always gives the same memo.
"""

from __future__ import annotations

from datetime import datetime
from html import escape

from tremor.modules.stress.engine import CHANNELS, StressResult, event_snapshot
from tremor.modules.stress.scenarios import PRICE_NAMES, UNIT_SUFFIX
from tremor.schemas import EventSignal

KEY_FACTORS = ("EQ_US", "EQ_EU", "EQ_EM", "EQ_IN", "IR_USD_10Y", "CS_IG", "CS_HY", "CS_EM", "FX_EUR", "FX_INR", "CMD_OIL", "CMD_GOLD", "VOL_VIX")

_CSS = """
body { font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b; background: #fff; margin: 0; }
main { max-width: 860px; margin: 0 auto; padding: 28px 24px 40px; }
h1 { font-size: 20px; margin: 0 0 4px; } h2 { font-size: 14px; text-transform: uppercase; letter-spacing: .06em; color: #52514e;
  border-bottom: 1px solid #e1e0d9; padding-bottom: 4px; margin: 22px 0 8px; }
.meta { color: #52514e; font-size: 13px; } .bottom { background: #f3f2ee; border-left: 4px solid #d03b3b; padding: 10px 14px; margin: 14px 0; }
table { border-collapse: collapse; width: 100%; font-size: 13px; } td, th { padding: 3px 8px; border-bottom: 1px solid #eeede8; text-align: left; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; } .neg { color: #b42d2d; } .pos { color: #1d6fd1; }
ul { margin: 4px 0; padding-left: 20px; } li { margin: 3px 0; } .small { font-size: 12px; color: #6b6a65; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; } @media (max-width: 700px) { .grid { grid-template-columns: 1fr; } }
@media print { main { padding: 0; } .noprint { display: none; } }
"""


def _usd(value: float) -> str:
    sign, a = ("-" if value < 0 else ""), abs(value)
    return f"{sign}${a / 1e9:.2f}bn" if a >= 1e9 else f"{sign}${a / 1e6:,.0f}m"


def _when(iso: str | datetime | None) -> str:
    if not iso:
        return "-"
    moment = iso if isinstance(iso, datetime) else datetime.fromisoformat(str(iso))
    return moment.strftime("%d %b %Y %H:%M UTC")


def names_to_review(run: StressResult, watch: list[dict]) -> list[dict]:
    """Early-warning Watch Negative names with exposure in the book, those this scenario hits hardest first.

    The watchlist is bank-wide (a Russia memo would otherwise open with China Evergrande); names the scenario
    leaves untouched keep the watchlist's own order after the ones it hits."""
    pnl = run.positions.groupby("obligor_id")["pnl"].sum()
    flagged = [w for w in watch if w["status"] == "Watch Negative" and w["exposure_musd"] > 0]
    return sorted(flagged, key=lambda w: min(float(pnl.get(w.get("entity_id"), 0.0)), 0.0))


def suggested_actions(run: StressResult, watch: list[dict]) -> list[str]:
    """Rule-based next steps, each tied to a number in the run."""
    cap, credit, out = run.capital, run.credit, []
    headroom_bp = (cap["cet1_ratio_after"] - cap["requirement_pct"]) * 100
    if cap["breach"]:
        out.append(f"Capital: post-stress CET1 of {cap['cet1_ratio_after']:.2f}% breaches the {cap['requirement_pct']:.1f}% requirement - "
                   "escalate to ALCO and prepare capital actions.")
    else:
        out.append(f"Capital: post-stress CET1 of {cap['cet1_ratio_after']:.2f}% leaves {headroom_bp:,.0f} bp of headroom over the "
                   f"{cap['requirement_pct']:.1f}% requirement ({cap['capital_depletion_bp']:.0f} bp of capital consumed).")
    channels = {c["name"]: c["pnl"] for c in run.by_channel}
    worst = min(channels, key=channels.get)
    if channels[worst] < 0:
        out.append(f"Largest loss channel: {worst}, {_usd(channels[worst])} - review the limits and hedges behind it.")
    pos = run.positions
    # Loans, bonds and CDS add up per borrower; derivatives facing the clearing house (CCP) are separate trades.
    key = pos["obligor_id"].where(pos["obligor_id"] != "CCP", pos["obligor_name"])
    by_obligor = pos.groupby(key).agg(pnl=("pnl", "sum"), name=("obligor_name", "first")).sort_values("pnl")
    losers = by_obligor[by_obligor["pnl"] < 0]
    total = float(losers["pnl"].sum()) or -1.0
    if len(losers):
        top3 = losers.head(3)
        out.append(f"Concentration: the three largest losses by obligor ({', '.join(top3['name'])}) are "
                   f"{float(top3['pnl'].sum()) / total:.0%} of all losses.")
    if credit["defaults"] or credit["downgraded_obligors"]:
        out.append(f"Credit: {credit['downgraded_obligors']} borrowers downgraded and {credit['defaults']} in default under the scenario; "
                   f"expected credit loss rises {_usd(credit['ecl_after'] - credit['ecl_before'])} - pre-position IFRS 9 overlays.")
    flagged = names_to_review(run, watch)
    if flagged:
        names = ", ".join(f"{w['name']} ({_usd(w['exposure_musd'] * 1e6)})" for w in flagged[:5])
        out.append(f"Review now (early-warning Watch Negative with exposure in the book, hardest hit by this scenario first): {names}.")
    epicentre = run.scenario.epicentre
    if epicentre:
        out.append(f"Epicentre: {', '.join(sorted(epicentre))} - confirm exposures and sanctions screening; consider buying protection where "
                   "the book is unhedged.")
    out.append("Model governance: the scenario is a point-in-time blend of historical analogs; check any factor the analyst disagrees "
               "with in the what-if editor before this number is used for decisions.")
    return out


def render_memo(run: StressResult, event: EventSignal | None, watch: list[dict], factors: dict[str, dict],
                summary: dict | None = None) -> str:
    """HTML for one stress test. ``factors``: risk-factor metadata (labels and units) from configs/scenarios.yaml;
    ``summary``: the executive summary from ``narrative`` (shown when a language model drafted it)."""
    t, cap, credit, tot, sc = run.trigger or {}, run.capital, run.credit, run.totals, run.scenario
    e = escape
    headline = t.get("headline") or sc.title
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
             f"<title>Risk memo - {e(headline[:60])}</title><style>{_CSS}</style></head><body><main>",
             "<p class='noprint small'>Print or save as PDF from the browser (Ctrl+P).</p>",
             "<h1>Risk memo: stress test on a detected event</h1>",
             f"<div class='meta'>{e(t.get('event_type_label', 'Analyst scenario'))} event · impact {t.get('impact_score', 0):.1f} · "
             f"detected {_when(t.get('detected_at'))} · run {e(run.run_id)}</div>",
             f"<div class='bottom'><b>Bottom line.</b> Under this scenario the book loses {_usd(-tot['pnl']) if tot['pnl'] < 0 else _usd(tot['pnl'])} "
             f"({tot['pnl_pct']:.2f}% of value){' ' if tot['pnl'] < 0 else ' (a gain) '}and CET1 moves from {cap['cet1_ratio_before']:.2f}% to "
             f"{cap['cet1_ratio_after']:.2f}% against a {cap['requirement_pct']:.1f}% requirement"
             f"{' - a breach' if cap['breach'] else ''}.</div>"]
    if summary and summary.get("source") == "llm":
        parts.append(f"<h2>Executive summary</h2><p>{e(summary['text'])}</p><p class='small'>{e(summary['note'])}.</p>")
    elif summary and summary.get("note"):
        parts.append(f"<p class='small noprint'>Executive summary: {e(summary['note'])}.</p>")

    parts.append("<h2>1. What happened</h2>")
    parts.append(f"<p><b>{e(headline)}</b></p>")
    # The evidence as it stood when the test ran (older runs carry no snapshot: fall back to the event now).
    snap = t.get("snapshot") or (event_snapshot(event) if event is not None else None)
    if snap:
        parts.append(f"<p class='meta'>First seen {_when(snap['first_seen'])}; when the test ran: {snap['n_docs']:,} reports, "
                     f"{snap['n_stories']:,} independent stories, {snap['n_publishers']:,} publishers ({snap['n_news']:,} news / "
                     f"{snap['n_social']:,} social), {snap['reports_last_hour']} reports in the last hour.</p>")
        if snap["stories"]:
            parts.append("<ul>" + "".join(f"<li>{e(s['headline'])} <span class='small'>({s['n_publishers']} publishers)</span></li>"
                                          for s in snap["stories"]) + "</ul>")
        parts.append("<table><tr><th>Impact scorecard</th><th>Evidence</th><th class='num'>Points</th></tr>" + "".join(
            f"<tr><td>{e(f['name'])}</td><td>{e(f['detail'])}</td><td class='num'>{f['points']:+.2f}</td></tr>"
            for f in snap["impact_factors"]) +
            f"<tr><td><b>Impact</b></td><td class='small'>sum of the points, clipped to 1-10</td>"
            f"<td class='num'><b>{t.get('impact_score', 0):.1f}</b></td></tr></table>")
        moves = {k: v for k, v in snap["price_moves"].items() if v.get("up", 0) + v.get("down", 0) > 0}
        if moves:
            parts.append("<p class='small'>Prices the reports say are moving: " + "; ".join(
                f"{e(PRICE_NAMES.get(k, k))} {v['up']} up / {v['down']} down" for k, v in moves.items()) + ".</p>")
    parts.append(f"<p class='small'>Why the test ran: {e(t.get('rule', 'manual run'))}.</p>")

    parts.append("<h2>2. The scenario</h2>")
    parts.append(f"<p>{e(sc.narrative)}</p>")
    rows = []
    for f in KEY_FACTORS:
        if f in sc.shocks:
            meta = factors.get(f, {})
            value = sc.shocks[f]
            rows.append(f"<tr><td>{e(meta.get('label', f))}</td><td class='num {'neg' if value < 0 else 'pos'}'>{value:+.1f}"
                        f"{UNIT_SUFFIX.get(meta.get('unit', ''), '')}</td></tr>")
    parts.append("<table><tr><th>Risk factor</th><th class='num'>Shock</th></tr>" + "".join(rows) + "</table>")

    parts.append("<h2>3. Impact on the book</h2><div class='grid'><div>")
    parts.append("<table><tr><th>By risk channel</th><th class='num'>P&amp;L</th></tr>" + "".join(
        f"<tr><td>{e(c['name'])}</td><td class='num {'neg' if c['pnl'] < 0 else 'pos'}'>{_usd(c['pnl'])}</td></tr>"
        for c in run.by_channel if c["name"] in CHANNELS and abs(c["pnl"]) >= 1e5) +
        f"<tr><td><b>Total</b></td><td class='num'><b>{_usd(tot['pnl'])}</b></td></tr></table></div><div>")
    parts.append("<table>"
                 f"<tr><td>Portfolio value</td><td class='num'>{_usd(tot['value_before'])} &rarr; {_usd(tot['value_after'])}</td></tr>"
                 f"<tr><td>CET1 ratio</td><td class='num'>{cap['cet1_ratio_before']:.2f}% &rarr; {cap['cet1_ratio_after']:.2f}%</td></tr>"
                 f"<tr><td>Risk-weighted assets</td><td class='num'>{_usd(cap['rwa_before'])} &rarr; {_usd(cap['rwa_after'])}</td></tr>"
                 f"<tr><td>Expected credit loss</td><td class='num'>{_usd(credit['ecl_before'])} &rarr; {_usd(credit['ecl_after'])}</td></tr>"
                 f"<tr><td>Stage 2 loans</td><td class='num'>{credit['stage2_before']} &rarr; {credit['stage2_after']}</td></tr>"
                 f"<tr><td>Downgrades / defaults</td><td class='num'>{credit['downgraded_obligors']} / {credit['defaults']}</td></tr>"
                 "</table></div></div>")
    parts.append("<table><tr><th>Largest losses</th><th>Type</th><th>Rating</th><th class='num'>P&amp;L</th></tr>" + "".join(
        f"<tr><td>{e(x['obligor'])}</td><td>{e(x['asset_class'])}</td><td>{e(x['rating'])}"
        f"{'' if x['rating'] == x['rating_after'] else ' &rarr; ' + e(x['rating_after'])}</td><td class='num neg'>{_usd(x['pnl'])}</td></tr>"
        for x in run.top_losses[:6]) + "</table>")

    in_book = [w for w in watch if w["exposure_musd"] > 0][:8]
    if in_book:
        parts.append("<h2>4. Names to review (credit early warning, now)</h2>")
        parts.append("<table><tr><th>Status</th><th>Name</th><th>Rating</th><th class='num'>Exposure</th><th>Main reason</th></tr>" + "".join(
            f"<tr><td>{e(w['status'])}</td><td>{e(w['name'])}</td><td>{e(w.get('rating') or '-')}</td>"
            f"<td class='num'>{_usd(w['exposure_musd'] * 1e6)}</td>"
            f"<td class='small'>{e(max(w['factors'], key=lambda f: f['points'])['detail'][:110]) if w.get('factors') else ''}</td></tr>"
            for w in in_book) + "</table>")

    parts.append(f"<h2>{5 if in_book else 4}. Suggested actions</h2><ul>" + "".join(f"<li>{e(a)}</li>" for a in suggested_actions(run, watch)) + "</ul>")
    origin = ("Generated by TREMOR from engine signals; the executive summary is worded by a local language model, every figure in it "
              "inserted by the engine." if summary and summary.get("source") == "llm"
              else "Generated by TREMOR from engine signals only (no language model).")
    parts.append(f"<p class='small'>{origin} The book, its ratings and exposures are synthetic and illustrative; scenario shocks are "
                 "measured from public market data.</p></main></body></html>")
    return "".join(parts)
