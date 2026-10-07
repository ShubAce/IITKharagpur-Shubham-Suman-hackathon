// TREMOR dashboard: live risk radar, index rebalancer (Module A), stress lab (Module B),
// text playground and model results. Data arrives over server-sent events and REST.
import { h, lineChart, hBars, heatmap, waterfall, groupedColumns } from "./charts.js";

const state = {
  status: null, config: null, events: new Map(), selectedEvent: null, eventDetail: null, entities: new Map(),
  docs: [], index: null, indexHistory: [], rebalances: [], runs: [], selectedRun: null, runDetail: null,
  evaluation: null, backtest: null, portfolio: null, tab: "radar", dirty: new Set(["radar"]), knownRuns: new Set(),
};
const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ formatting
const fmtTime = (iso) => (iso ? new Date(iso).toLocaleString("en-GB", { timeZone: "UTC", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) + " UTC" : "-");
const fmtShort = (d) => d.toLocaleString("en-GB", { timeZone: "UTC", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
const fmtDay = (d) => d.toLocaleDateString("en-GB", { timeZone: "UTC", day: "numeric", month: "short" });
const fmtInt = (v) => Math.round(v).toLocaleString("en-US");
const fmtSigned = (v, digits = 2) => `${v >= 0 ? "+" : ""}${v.toFixed(digits)}`;
const fmtUsd = (v) => {
  const a = Math.abs(v), s = v < 0 ? "-" : "";
  if (a >= 1e9) return `${s}$${(a / 1e9).toFixed(2)}bn`;
  if (a >= 1e6) return `${s}$${(a / 1e6).toFixed(0)}m`;
  return `${s}$${(a / 1e3).toFixed(0)}k`;
};
const safeUrl = (u) => (typeof u === "string" && /^https?:\/\//i.test(u) ? u : null);
const shortName = (name) => name.replace(/,? (Inc\.?|Corp\.?|Co\.?|Corporation|Company|Group|plc|PJSC|AG|SE|Holdings|Communications|Wholesale Corp\.?|& Co\.?)$/i, "")
  .replace(/,? (Inc\.?|Corp\.?|Co\.?|& Co\.?)$/i, "");
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const sentColor = (v) => (v >= 0 ? "var(--div-pos)" : "var(--div-neg)");
const threshold = () => state.status?.trigger_threshold ?? 7;
const sevClass = (impact) => (impact >= threshold() ? "sev-high" : impact >= 5 ? "sev-mid" : "sev-low");

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

function sentimentChip(v) {
  return h("span", { class: "sent" }, h("span", { class: "dot", style: `background:${sentColor(v)}` }), fmtSigned(v));
}
function tile(label, value, sub, extra = "") {
  return h("div", { class: `tile ${extra}` }, h("div", { class: "tile-label" }, label), h("div", { class: `tile-value ${extra.includes("hero") ? "hero" : ""}` }, value),
    sub ? (sub instanceof Node ? sub : h("div", { class: "tile-sub" }, sub)) : null);
}

// ------------------------------------------------------------------ header & tabs
function renderHeader() {
  const st = state.status;
  if (!st) return;
  const replay = st.replay;
  $("mode-badge").textContent = replay ? `REPLAY · ${replay.pack.replaceAll("_", " ")} · model: ${st.model}` : `LIVE FEEDS · ${st.sources.join(", ")} · model: ${st.model}`;
  $("clock").textContent = replay ? fmtTime(replay.virtual_now) : fmtTime(st.clock || new Date().toISOString());
  $("progress-wrap").hidden = !replay;
  $("replay-controls").hidden = !replay;
  if (replay) {
    $("progress-bar").style.width = `${(replay.progress * 100).toFixed(1)}%`;
    $("btn-play").textContent = replay.paused ? "Resume" : replay.finished ? "Finished" : "Pause";
    const sel = $("speed");
    if (document.activeElement !== sel && [...sel.options].some((o) => +o.value === replay.speed)) sel.value = String(replay.speed);
    if (document.activeElement !== $("pack")) $("pack").value = replay.pack;
  }
  const c = st.counters || {};
  $("counters").replaceChildren(
    h("div", {}, h("b", {}, fmtInt(c.documents || 0)), "documents"),
    h("div", {}, h("b", {}, fmtInt(c.duplicates || 0)), "duplicates folded"),
    h("div", {}, h("b", {}, fmtInt(c.noise || 0)), "judged noise"),
    h("div", {}, h("b", {}, fmtInt(st.events_tracked || 0)), "events"),
    h("div", {}, h("b", {}, String(st.stress_runs || 0)), "stress tests"),
  );
}

function showTab(name) {
  state.tab = name;
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".view").forEach((v) => (v.hidden = v.id !== `view-${name}`));
  try { history.replaceState(null, "", `#${name}`); } catch { /* sandboxed */ }
  state.dirty.add(name);
  if (name === "results") loadResults();
  if (name === "stress") { loadPortfolio(); loadValidation(); }
  if (name === "watch") loadWatch();
  renderDirty();
}

// ------------------------------------------------------------------ radar
function renderRadar() {
  const events = [...state.events.values()].sort((a, b) => b.impact_score - a.impact_score || b.n_docs - a.n_docs);
  const top = events[0];
  const c = state.status?.counters || {};
  const tiles = [
    tile("Highest impact now", top ? top.impact_score.toFixed(1) : "-", top ? `${top.event_type_label} · ${top.headline.slice(0, 70)}` : "waiting for documents", "hero wide"),
    tile("Events above stress threshold", String(events.filter((e) => e.impact_score >= threshold()).length), `impact ≥ ${threshold().toFixed(1)}`),
    tile("Documents analysed", fmtInt((c.documents || 0) - (c.duplicates || 0) - (c.noise || 0)), `of ${fmtInt(c.documents || 0)} received`),
    tile("Stress tests run", String(state.runs.length), state.runs[0] ? `latest ${fmtTime(state.runs[0].trigger?.detected_at)}` : "none yet"),
  ];
  $("radar-tiles").replaceChildren(...tiles);

  const list = $("event-list");
  list.replaceChildren(...events.slice(0, 30).map((e) => h("div", {
    class: `event-row ${state.selectedEvent === e.event_id ? "selected" : ""}`, onclick: () => selectEvent(e.event_id),
  },
  h("div", {}, h("div", { class: "impact-badge" }, e.impact_score.toFixed(1), h("small", {}, "impact")),
    h("div", { class: `impact-meter ${sevClass(e.impact_score)}` }, h("div", { style: `width:${e.impact_score * 10}%` }))),
  h("div", {},
    h("div", { class: "event-headline" }, e.headline),
    h("div", { class: "event-meta" },
      h("span", { class: `chip ${e.impact_score >= threshold() ? "trigger" : ""}` }, e.event_type_label),
      sentimentChip(e.sentiment_score),
      h("span", {}, `${e.n_stories} stories · ${e.n_publishers} publishers`),
      e.regions?.length ? h("span", {}, e.regions.join(" ")) : null,
      h("span", {}, fmtTime(e.last_updated))))),
  ));
  if (!events.length) list.replaceChildren(h("div", { class: "muted pad" }, "Waiting for the first events..."));
  if (!state.selectedEvent && top) selectEvent(top.event_id);

  const idx = state.config?.index || [];
  const rows = idx.map((c2) => state.entities.get(c2.ticker)).filter(Boolean).sort((a, b) => b.sentiment_score - a.sentiment_score)
    .map((s) => ({ label: `${s.entity_id} · ${shortName(s.name)}`, value: s.sentiment_score, err: s.sentiment_uncertainty,
      sub: `±${s.sentiment_uncertainty.toFixed(2)} uncertainty · evidence weight ${s.evidence_weight.toFixed(1)}` }));
  const maxAbs = Math.max(0.2, ...rows.map((r) => Math.abs(r.value) + r.err));
  hBars($("sentiment-chart"), { rows, diverging: true, format: (v) => fmtSigned(v), maxAbs, labelWidth: 150, rowHeight: 21 });
}

function renderDocFeed() {
  const feed = $("doc-feed");
  const shown = state.shownDocs || new Set();
  feed.replaceChildren(...state.docs.slice(0, 60).map((d) => h("div", { class: shown.has(d.doc_id) ? "doc" : "doc fresh" },
    safeUrl(d.url) ? h("a", { href: safeUrl(d.url), target: "_blank", rel: "noopener noreferrer" }, d.text) : h("span", {}, d.text),
    h("div", { class: "doc-meta" },
      h("span", { class: "chip" }, d.kind === "social" ? `social · ${d.source}` : `news · ${d.publisher || d.source}`),
      sentimentChip(d.sentiment_score), h("span", {}, d.event_type.replaceAll("_", " ").toLowerCase()),
      d.entities?.length ? h("span", {}, d.entities.slice(0, 4).join(" ")) : null,
      h("span", {}, `novelty ${d.novelty.toFixed(2)}`), h("span", {}, fmtTime(d.published_at))))));
  if (!state.docs.length) feed.replaceChildren(h("div", { class: "muted pad" }, "Waiting for documents..."));
  state.shownDocs = new Set(state.docs.slice(0, 60).map((d) => d.doc_id));
}

async function selectEvent(id) {
  state.selectedEvent = id;
  try {
    state.eventDetail = await api(`/api/events/${id}`);
  } catch { return; }
  renderEventDetail();
  state.dirty.add("radar");
}

const PRICE_LABELS = { OIL: "Oil", GAS: "Gas", GOLD: "Gold", SPX: "Stock market" };

// Direction, not tone: how many reports say each named price is rising or falling (feeds the stress scenario).
function priceMoves(moves) {
  const items = Object.entries(moves || {}).filter(([, m]) => m.up + m.down > 0);
  if (!items.length) return null;
  return h("div", {}, h("h4", {}, "Prices the reports say are moving"),
    h("div", { class: "moves" }, ...items.map(([id, m]) => h("span", { class: "chip", title: "reports that state a direction for this price" },
      `${PRICE_LABELS[id] || id} `, h("span", { class: "move-up" }, `▲ ${m.up}`), " ", h("span", { class: "move-down" }, `▼ ${m.down}`)))));
}

function renderEventDetail() {
  const d = state.eventDetail;
  if (!d) return;
  const e = d.signal;
  const total = e.impact_factors.reduce((s, f) => s + f.points, 0);
  const box = h("div", { class: "detail" },
    h("h3", {}, e.headline),
    h("div", { class: "event-meta" },
      h("span", { class: `chip ${e.impact_score >= threshold() ? "trigger" : ""}` }, `${e.event_type_label} · impact ${e.impact_score.toFixed(1)}`),
      sentimentChip(e.sentiment_score), h("span", {}, `${e.n_docs} reports · ${e.n_stories} stories · ${e.n_publishers} publishers · ${e.n_news} news / ${e.n_social} social`),
      h("span", {}, `first seen ${fmtTime(e.first_seen)}`)),
    h("h4", {}, "Impact scorecard"),
    h("table", { class: "data" },
      h("thead", {}, h("tr", {}, h("th", {}, "Factor"), h("th", {}, "Evidence"), h("th", { class: "num" }, "Points"))),
      h("tbody", {},
        ...e.impact_factors.map((f) => h("tr", {}, h("td", {}, f.name), h("td", {}, f.detail),
          h("td", { class: `num ${f.points < 0 ? "points-neg" : "points-pos"}` }, fmtSigned(f.points)))),
        h("tr", { class: "total" }, h("td", {}, "Impact score"), h("td", { class: "muted" }, "clipped to 1-10"),
          h("td", { class: "num" }, `${e.impact_score.toFixed(2)}${Math.abs(total - e.impact_score) > 0.01 ? ` (raw ${total.toFixed(2)})` : ""}`)))),
    h("div", { id: "impact-history", class: "chart" }),
    priceMoves(e.price_moves),
    h("h4", {}, "Most widely reported stories"),
    h("ul", { class: "evidence" }, ...e.stories.map((s) => h("li", {}, s.headline,
      h("div", { class: "src" }, `${s.n_docs} reports · ${s.n_publishers} publishers · sentiment ${fmtSigned(s.sentiment_score)} · ${fmtTime(s.first_seen)}`)))),
    h("h4", {}, "Evidence (first and latest reports)"),
    h("ul", { class: "evidence" }, ...e.evidence.slice(-8).reverse().map((ev) => h("li", {},
      safeUrl(ev.url) ? h("a", { href: safeUrl(ev.url), target: "_blank", rel: "noopener noreferrer" }, ev.text) : ev.text,
      h("div", { class: "src" }, `${ev.kind} · ${ev.publisher || ev.source} · ${fmtTime(ev.published_at)} · sentiment ${fmtSigned(ev.sentiment_score)}`)))),
    e.entities.length ? h("h4", {}, "Entities") : null,
    e.entities.length ? h("div", { class: "event-meta" }, ...e.entities.map((x) => h("span", { class: "chip" }, x))) : null,
  );
  $("event-detail").replaceChildren(box);
  $("event-detail").classList.remove("muted", "pad");
  const hist = d.history || [];
  if (hist.length > 1) {
    lineChart($("impact-history"), { height: 150, series: [{ name: "Impact", color: "var(--series-1)", points: hist.map((p) => ({ x: new Date(p.time), y: p.impact })) }],
      yFormat: (v) => v.toFixed(1), xFormat: fmtShort, yLabel: "impact over time", baselineY: threshold() });
  }
}

// ------------------------------------------------------------------ index (Module A)
function renderIndex() {
  const s = state.index;
  if (!s) return;
  const excess = s.excess_return_pct;
  $("index-tiles").replaceChildren(
    tile("TREMOR-20 level", s.level.toFixed(2), `base 1,000 · ${s.rebalances} rebalances`, "hero"),
    tile("Equal-weight benchmark", s.benchmark.toFixed(2), "same stocks, same costs"),
    tile("Excess return", `${fmtSigned(excess, 2)}%`, h("div", { class: `tile-sub ${excess >= 0 ? "delta-up" : "delta-down"}` }, excess >= 0 ? "▲ ahead of benchmark" : "▼ behind benchmark")),
    tile("Turnover (one-way)", `${(s.total_turnover * 100).toFixed(1)}%`, `cost ${s.total_cost_bps.toFixed(1)} bp`),
    tile("Last rebalance", s.last_rebalance ? fmtTime(s.last_rebalance) : "-", "scheduled hourly + tactical"),
  );
  const hist = state.indexHistory;
  lineChart($("index-line"), {
    height: 230, xFormat: fmtShort, yFormat: (v) => v.toFixed(1), yLabel: "index level",
    series: [
      { name: "TREMOR-20", color: "var(--series-1)", points: hist.map((p) => ({ x: new Date(p.time), y: p.level })) },
      { name: "Equal weight", color: "var(--series-2)", points: hist.map((p) => ({ x: new Date(p.time), y: p.benchmark })) },
    ],
  });
  // Heatmap of active weights, one column per recorded point (thinned to <= 160 columns).
  const tickers = s.constituents.map((c) => c.ticker);
  const step = Math.max(1, Math.ceil(hist.length / 160));
  const cols = hist.filter((_, i) => i % step === 0 || i === hist.length - 1);
  const parent = Object.fromEntries(s.constituents.map((c) => [c.ticker, c.parent]));
  heatmap($("weight-heatmap"), {
    rows: tickers, cols: cols.map((p) => new Date(p.time)), colFormat: fmtShort,
    matrix: tickers.map((t) => cols.map((p) => (p.weights[t] ?? parent[t]) - parent[t])), maxAbs: 0.05,
    format: (v) => `${fmtSigned(v * 100, 2)} pts`, rowLabels: s.constituents.map((c) => `${c.ticker} ${c.name}`),
  });
  const rows = [...s.constituents].sort((a, b) => b.active - a.active);
  const maxW = Math.max(...rows.map((r) => r.weight), 0.1);
  $("weights-table").replaceChildren(h("div", { class: "weights-wrap" }, h("table", { class: "data" },
    h("thead", {}, h("tr", {}, h("th", {}, "Stock"), h("th", {}, "Weight vs parent"), h("th", { class: "num" }, "Weight"), h("th", { class: "num" }, "Active"),
      h("th", { class: "num" }, "Sentiment"), h("th", {}, "Driver"))),
    h("tbody", {}, ...rows.map((r) => h("tr", {},
      h("td", {}, h("b", {}, r.ticker), h("div", { class: "muted" }, r.sector)),
      h("td", {}, h("div", { class: "bar-cell" }, h("div", { class: "w", style: `width:${(r.weight / maxW) * 100}%` }),
        h("div", { class: "p", style: `left:${(r.parent / maxW) * 100}%` }))),
      h("td", { class: "num" }, `${(r.weight * 100).toFixed(2)}%`),
      h("td", { class: `num ${r.active >= 0 ? "delta-up" : "delta-down"}` }, `${fmtSigned(r.active * 100, 2)}`),
      h("td", { class: "num" }, sentimentChip(r.sentiment)),
      h("td", { class: "muted" }, r.driver ? `${r.driver.headline.slice(0, 80)}` : "-")))))));
  $("rebalance-log").replaceChildren(...state.rebalances.slice(0, 40).map((r) => h("div", { class: "log-item" },
    h("div", { class: "when" }, `${fmtTime(r.time)} · turnover ${(r.turnover * 100).toFixed(2)}% · cost ${r.cost_bps.toFixed(2)} bp`),
    h("div", {}, r.reason),
    ...r.changes.slice(0, 4).map((c) => h("div", { class: "change" },
      `${c.ticker}: ${(c.from * 100).toFixed(2)}% → ${(c.to * 100).toFixed(2)}% (sentiment ${fmtSigned(c.sentiment)})`,
      c.driver ? ` · ${c.driver.headline.slice(0, 70)}` : "")))));
  if (!state.rebalances.length) $("rebalance-log").replaceChildren(h("div", { class: "muted pad" }, "No rebalances yet."));
  renderBacktest();
}

function renderBacktest() {
  const b = state.backtest;
  $("backtest-card").hidden = !b;
  if (!b) return;
  const m = b.metrics;
  $("backtest-hint").textContent = b.description || "";
  $("backtest-tiles").replaceChildren(
    ...Object.entries(m).map(([name, v]) => tile(name, typeof v === "number" ? v.toFixed(2) : String(v), b.metric_notes?.[name] || "")));
  lineChart($("backtest-line"), {
    height: 240, xFormat: fmtDay, yFormat: (v) => v.toFixed(1), yLabel: "backtest",
    series: b.series.map((s, i) => ({ name: s.name, color: `var(--series-${i + 1})`, points: s.points.map((p) => ({ x: new Date(p[0]), y: p[1] })) })),
  });
}

// ------------------------------------------------------------------ stress (Module B)
function renderRuns() {
  const list = $("run-list");
  if (!state.runs.length) return;
  list.replaceChildren(...state.runs.map((r) => h("div", {
    class: `run-card ${state.selectedRun === r.run_id ? "selected" : ""}`, onclick: () => selectRun(r.run_id),
  },
  h("div", { class: "when" }, `${fmtTime(r.trigger?.detected_at || r.created_at)} · ${r.trigger?.event_type_label || "custom"} · impact ${r.trigger?.impact_score?.toFixed(1) ?? "-"}`),
  h("div", { class: "what" }, r.trigger?.headline || r.scenario.title),
  h("div", { class: "pnl" }, h("b", {}, fmtUsd(r.totals.pnl)), ` (${r.totals.pnl_pct.toFixed(2)}%) · CET1 ${r.capital.cet1_ratio_before}% → ${r.capital.cet1_ratio_after}%`))));
  if (!state.selectedRun) {
    const severe = state.runs.reduce((best, r) => ((r.trigger?.impact_score ?? 0) > (best.trigger?.impact_score ?? 0) ? r : best), state.runs[0]);
    selectRun(severe.run_id);
  }
}

async function selectRun(id) {
  state.selectedRun = id;
  try { state.runDetail = await api(`/api/stress/runs/${id}`); } catch { return; }
  renderRunDetail();
  renderRuns();
}

const KEY_FACTORS = ["EQ_US", "EQ_EU", "EQ_EM", "EQ_IN", "IR_USD_5Y", "IR_USD_10Y", "CS_IG", "CS_HY", "CS_EM", "FX_EUR", "FX_INR", "CMD_OIL", "CMD_GOLD", "SEC_FINANCIALS", "SEC_ENERGY"];

function renderRunDetail() {
  const r = state.runDetail;
  if (!r) return;
  $("run-detail").hidden = false;
  const stored = state.knownRuns.has(r.run_id);  // what-if re-runs are not stored, so they have no memo
  $("memo-link").hidden = !stored;
  if (stored) $("memo-link").href = `/api/stress/runs/${encodeURIComponent(r.run_id)}/memo`;
  const cap = r.capital, tot = r.totals, cr = r.credit;
  $("run-tiles").replaceChildren(
    tile("Portfolio value before", fmtUsd(tot.value_before), `${tot.positions} positions`, "hero"),
    tile("Portfolio value after", fmtUsd(tot.value_after), h("div", { class: `tile-sub ${tot.pnl >= 0 ? "delta-up" : "delta-down"}` }, `${tot.pnl >= 0 ? "▲" : "▼"} ${fmtUsd(tot.pnl)} (${tot.pnl_pct.toFixed(2)}%)`), "hero"),
    tile("CET1 ratio", `${cap.cet1_ratio_before}% → ${cap.cet1_ratio_after}%`,
      h("div", { class: `tile-sub ${cap.breach ? "delta-down" : ""}` }, `${cap.breach ? "▼ below" : "requirement"} ${cap.requirement_pct}% · depletion ${cap.capital_depletion_bp} bp`)),
    tile("Expected credit loss", `${fmtUsd(cr.ecl_before)} → ${fmtUsd(cr.ecl_after)}`, `stage 2 loans ${cr.stage2_before} → ${cr.stage2_after}`),
    tile("Borrowers downgraded", String(cr.downgraded_obligors), `${cr.defaults} default(s) · avg PD ${cr.avg_pd_before_bp} → ${cr.avg_pd_after_bp} bp`),
  );
  waterfall($("waterfall"), {
    start: { label: "Value before", value: tot.value_before / 1e6 },
    steps: r.by_channel.filter((c) => Math.abs(c.pnl) >= 1e5).map((c) => ({ label: c.name, value: c.pnl / 1e6 })),
    endLabel: "Value after", format: (v) => fmtInt(v),
  });
  const sc = r.scenario;
  $("scenario-hint").textContent = sc.narrative;
  // Composition of the historical blend: the closest analogs (credibility-weighted) and the average earlier crisis.
  const cred = sc.credibility ?? 1;
  const part = (label, period, share, cls = "analog") => h("div", { class: "analogs" }, h("div", { class: cls },
    h("span", {}, `${label} `, period ? h("span", { class: "muted" }, period) : null),
    h("div", { class: "track" }, h("div", { class: "fill", style: `width:${share * 100}%` })), h("span", {}, `${Math.round(share * 100)}%`)));
  $("scenario-analogs").replaceChildren(
    ...(sc.analogs || []).map((a) => part(a.title, `(${a.start} → ${a.end})`, a.weight * cred)),
    sc.analogs?.length && cred < 1 ? part(`Average of ${sc.prior_episodes} earlier crises`, "(credibility weighting)", 1 - cred, "analog prior") : "",
    Object.keys(sc.epicentre || {}).length ? h("div", { class: "analogs muted" }, `Epicentre (notched down): ${Object.entries(sc.epicentre).map(([k, v]) => `${k} −${v}`).join(", ")}`) : "");
  const factors = state.config?.risk_factors || {};
  const shockRows = KEY_FACTORS.filter((f) => sc.shocks[f] !== undefined).map((f) => ({
    label: factors[f]?.label || f, value: sc.shocks[f], sub: factors[f]?.unit === "bp" ? "basis points" : factors[f]?.unit === "pts" ? "points" : "percent" }));
  hBars($("shock-chart"), { rows: shockRows, diverging: true, labelWidth: 170, rowHeight: 20,
    format: (v) => fmtSigned(v, 1), maxAbs: Math.max(...shockRows.map((x) => Math.abs(x.value)), 1) });
  $("top-losses").replaceChildren(h("div", { class: "weights-wrap" }, h("table", { class: "data" },
    h("thead", {}, h("tr", {}, h("th", {}, "Position"), h("th", {}, "Type"), h("th", {}, "Rating"), h("th", { class: "num" }, "P&L"))),
    h("tbody", {}, ...r.top_losses.map((t) => h("tr", {}, h("td", {}, t.obligor), h("td", {}, t.asset_class),
      h("td", {}, t.rating === t.rating_after ? t.rating : `${t.rating} → ${t.rating_after}`), h("td", { class: "num" }, fmtUsd(t.pnl))))))));
  hBars($("sector-chart"), { rows: r.by_sector.filter((s) => Math.abs(s.pnl) > 5e4).map((s) => ({ label: s.name, value: s.pnl / 1e6 })),
    diverging: true, format: (v) => `${fmtSigned(v, 1)}m`, labelWidth: 170 });
  $("shock-editor").replaceChildren(...KEY_FACTORS.map((f) => h("label", {}, `${factors[f]?.label || f} (${factors[f]?.unit || ""})`,
    h("input", { type: "number", step: "0.5", "data-factor": f, value: (sc.shocks[f] ?? 0).toFixed(1) }))));
  $("btn-rerun").dataset.event = r.trigger?.event_id || "";
}

async function rerun() {
  const shocks = {};
  document.querySelectorAll("#shock-editor input").forEach((i) => { shocks[i.dataset.factor] = parseFloat(i.value) || 0; });
  const eventId = $("btn-rerun").dataset.event || null;
  $("rerun-status").textContent = "running...";
  try {
    const t0 = performance.now();
    const result = await post("/api/stress/run", eventId ? { event_id: eventId, shocks } : { shocks, title: "Analyst scenario" });
    state.runDetail = result;
    renderRunDetail();
    $("rerun-status").textContent = `revalued ${result.totals.positions} positions in ${Math.round(performance.now() - t0)} ms (including the network round-trip)`;
  } catch (err) {
    $("rerun-status").textContent = String(err);
  }
}

// Point-in-time backtest of the scenario generator (docs/results/scenario_backtest.json).
async function loadValidation() {
  if (!state.evaluation) {
    try { state.evaluation = await api("/api/evaluation"); } catch { return; }
  }
  const v = state.evaluation?.scenario_backtest;
  $("validation-card").hidden = !v;
  if (!v) return;
  const s = v.summary, ctx = s.context;
  const order = ["naive", "template", "all_mean", "same_type_mean", "nearest", "tremor"].filter((k) => s[k]);
  const color = (k) => (k === "tremor" ? "var(--series-1)" : k === "naive" ? "var(--div-neg)" : "var(--text-muted)");
  $("validation-hint").textContent = `${ctx.episodes} crises, ${ctx.first.slice(0, 4)}-${ctx.last.slice(0, 4)}: each scenario was rebuilt from the crisis's ` +
    `day-one headline using only episodes that had ended before it began, then compared with what markets actually did. ` +
    `The brief's example shock assumes rates rise, but the 10-year Treasury yield fell in ${ctx.rates_fell} of the ${ctx.episodes} crises (rose in ${ctx.rates_rose}).`;
  hBars($("validation-direction"), { rows: order.map((k) => ({ label: s[k].label, value: s[k].direction_hit_rate, color: color(k),
    sub: `rates ${((s[k].per_factor_hit_rate?.IR_USD_10Y ?? 0) * 100).toFixed(0)}% · oil ${((s[k].per_factor_hit_rate?.CMD_OIL ?? 0) * 100).toFixed(0)}% right` })),
    format: (x) => `${(x * 100).toFixed(0)}%`, maxAbs: 1, labelWidth: 260, rowHeight: 22 });
  hBars($("validation-error"), { rows: order.map((k) => ({ label: s[k].label, value: s[k].pnl_error_mean_musd, color: color(k),
    sub: `median ${s[k].pnl_error_median_musd}m · CET1 off by ${s[k].cet1_error_mean_pp} pp` })), format: (x) => `${fmtInt(x)}m`, labelWidth: 260, rowHeight: 22 });
  $("validation-table").replaceChildren(h("div", { class: "weights-wrap" }, h("table", { class: "data" },
    h("thead", {}, h("tr", {}, h("th", {}, "Method"), h("th", { class: "num" }, "Directions right"), h("th", { class: "num" }, "P&L error (mean)"),
      h("th", { class: "num" }, "median"), h("th", { class: "num" }, "CET1 error"), h("th", { class: "num" }, "Closer than naive"))),
    h("tbody", {}, ...order.map((k) => h("tr", { class: k === "tremor" ? "ours" : "" }, h("td", {}, s[k].label),
      h("td", { class: "num" }, `${(s[k].direction_hit_rate * 100).toFixed(1)}%`), h("td", { class: "num" }, `$${fmtInt(s[k].pnl_error_mean_musd)}m`),
      h("td", { class: "num" }, `$${fmtInt(s[k].pnl_error_median_musd)}m`), h("td", { class: "num" }, `${s[k].cet1_error_mean_pp} pp`),
      h("td", { class: "num" }, `${s[k].closer_than_naive}/${ctx.episodes}`)))))));
}

async function loadPortfolio() {
  if (state.portfolio) return;
  try { state.portfolio = await api("/api/portfolio"); } catch { return; }
  const p = state.portfolio;
  $("book-hint").textContent = `${p.positions} positions, ${fmtUsd(p.notional)} notional. Mid-market loans are derived from the Kaggle Financial Transactions dataset (merchant cash-flow profiles).`;
  hBars($("book-chart"), { rows: p.by_asset_class.map((a) => ({ label: `${a.name} (${a.positions})`, value: a.notional / 1e9 })),
    format: (v) => `$${v.toFixed(2)}bn`, labelWidth: 150 });
}

// ------------------------------------------------------------------ credit watch (early warning)
async function loadWatch() {
  try { state.watch = await api("/api/watchlist"); } catch { return; }
  state.dirty.add("watch");
}

const statusChip = (status) => h("span", { class: `chip ${status === "Watch Negative" ? "watch-neg" : "watch-mon"}` }, status);

function renderWatch() {
  const w = state.watch;
  if (!w) return;
  const entries = w.entries || [];
  const neg = entries.filter((e) => e.status === "Watch Negative"), mon = entries.filter((e) => e.status === "Monitor");
  const names = (list) => (list.length ? list.slice(0, 3).map((e) => shortName(e.name)).join(", ") + (list.length > 3 ? "…" : "") : "none");
  $("watch-tiles").replaceChildren(
    tile("Watch Negative", String(neg.length), names(neg), "hero"),
    tile("Monitor", String(mon.length), names(mon)),
    tile("Book exposure under review", fmtUsd(w.exposure_flagged_musd * 1e6), `${entries.filter((e) => e.exposure_musd > 0).length} flagged names are obligors in the book`),
    tile("Names flagged so far", String(Object.keys(w.first_flagged || {}).length), w.as_of ? `as of ${fmtTime(w.as_of)}` : ""),
  );
  if (!entries.length) {
    $("watch-list").replaceChildren(h("div", { class: "muted pad" }, "No name is flagged right now."));
  } else {
    $("watch-list").replaceChildren(h("div", { class: "weights-wrap" }, h("table", { class: "data" },
      h("thead", {}, h("tr", {}, h("th", {}, "Status"), h("th", {}, "Name"), h("th", {}, "Rating"), h("th", { class: "num" }, "Exposure"),
        h("th", { class: "num" }, "Score"), h("th", {}, "Main reason"))),
      h("tbody", {}, ...entries.map((e) => h("tr", { class: `clickable ${state.selectedWatch === e.entity_id ? "selected" : ""}`,
        onclick: () => { state.selectedWatch = e.entity_id; renderWatch(); } },
        h("td", {}, statusChip(e.status)), h("td", {}, h("b", {}, shortName(e.name)), h("div", { class: "muted" }, [e.sector, e.country].filter(Boolean).join(" · "))),
        h("td", {}, e.rating || h("span", { class: "muted" }, "not held")),
        h("td", { class: "num" }, e.exposure_musd ? fmtUsd(e.exposure_musd * 1e6) : "-"), h("td", { class: "num" }, e.score.toFixed(1)),
        h("td", { class: "muted" }, (e.factors.slice().sort((a, b) => b.points - a.points)[0]?.detail || "").slice(0, 90))))))));
  }
  const sel = entries.find((e) => e.entity_id === state.selectedWatch) || entries[0];
  if (sel) {
    state.selectedWatch = sel.entity_id;
    $("watch-detail-hint").textContent = `${sel.name} · ${sel.status} · ${sel.rating ? `rated ${sel.rating} in the book, exposure ${fmtUsd(sel.exposure_musd * 1e6)}` +
      (sel.protection_musd ? `, CDS protection bought ${fmtUsd(sel.protection_musd * 1e6)}` : "") : "no exposure in the book"}`;
    const total = sel.factors.reduce((s, f) => s + f.points, 0);
    $("watch-detail").classList.remove("muted");
    $("watch-detail").replaceChildren(h("table", { class: "data" },
      h("thead", {}, h("tr", {}, h("th", {}, "Signal"), h("th", {}, "Evidence"), h("th", { class: "num" }, "Points"))),
      h("tbody", {}, ...sel.factors.map((f) => h("tr", {}, h("td", {}, f.name), h("td", {}, f.detail), h("td", { class: "num points-pos" }, fmtSigned(f.points)))),
        h("tr", { class: "total" }, h("td", {}, "Score"), h("td", { class: "muted" }, "≥ 3 Watch Negative · ≥ 1.5 Monitor"), h("td", { class: "num" }, total.toFixed(2))))),
      h("div", { class: "muted", style: "margin-top:10px" }, Object.entries(sel.first_flagged || {}).map(([s, t]) => `${s} since ${fmtTime(t)}`).join(" · ")));
  }
  $("watch-timeline").replaceChildren(...(w.timeline || []).slice().reverse().slice(0, 60).map((c) => h("div", { class: "log-item" },
    h("div", { class: "when" }, fmtTime(c.at)),
    h("div", {}, h("b", {}, shortName(c.name)), ` ${c.from || "not listed"} → `, c.to ? statusChip(c.to) : "no longer listed", ` (score ${c.score.toFixed(1)})`),
    h("div", { class: "change" }, c.reason.slice(0, 140)))));
  if (!(w.timeline || []).length) $("watch-timeline").replaceChildren(h("div", { class: "muted pad" }, "No flags yet."));
}

// ------------------------------------------------------------------ analyze
const EXAMPLES = [
  "Moody's downgrades Boeing to junk as 737 MAX deliveries stall, while Airbus wins record orders",
  "Apple shares jump as iPhone sales beat estimates; Intel slides on weak guidance",
  "Russia launches full-scale invasion of Ukraine; oil surges past $100 as West readies sanctions",
  "Fed raises rates by 75 basis points, signals more hikes as inflation hits 40-year high",
  "$TSLA to the moon 🚀 deliveries crushed it, shorts getting squeezed",
  "Silicon Valley Bank collapses after a run on deposits; regulators seize the lender",
];

async function analyze() {
  const text = $("analyze-text").value.trim();
  if (!text) return;
  const kind = document.querySelector('input[name="kind"]:checked').value;
  $("analyze-result").replaceChildren(h("div", { class: "muted pad" }, "analysing..."));
  let r;
  try { r = await post("/api/analyze", { text, kind }); } catch (err) { $("analyze-result").replaceChildren(h("div", { class: "pad" }, String(err))); return; }
  const probs = Object.entries(r.event_type_probs).map(([k, v]) => ({ label: state.config?.event_types?.[k]?.label || k, value: v }));
  const sentBox = h("div", { class: "result-box" }, h("h4", {}, "Sentiment"), h("div", { class: "big" }, fmtSigned(r.sentiment_score)),
    h("div", { class: "muted" }, `negative ${(r.sentiment_probs.negative * 100).toFixed(0)}% · neutral ${(r.sentiment_probs.neutral * 100).toFixed(0)}% · positive ${(r.sentiment_probs.positive * 100).toFixed(0)}%`),
    r.entities.length ? h("h4", { style: "margin-top:12px" }, r.entity_level_sentiment ? "Sentiment per entity (entity-conditioned model)" : "Entities") : null,
    ...r.entities.map((e) => h("div", { class: "entity-row" }, h("span", { class: "chip" }, e.id), h("span", {}, e.name), sentimentChip(e.sentiment))),
    Object.keys(r.price_moves || {}).length ? h("h4", { style: "margin-top:12px" }, "Price direction (not tone)") : null,
    ...Object.entries(r.price_moves || {}).map(([id, dir]) => h("div", { class: "entity-row" }, h("span", { class: "chip" }, id),
      h("span", { class: dir === "up" ? "move-up" : "move-down" }, dir === "up" ? "▲ rising" : "▼ falling"))));
  const typeBox = h("div", { class: "result-box" }, h("h4", {}, "Event classification"), h("div", { class: "big" }, r.event_type_label), h("div", { id: "probs-chart" }));
  const impactBox = h("div", { class: "result-box" }, h("h4", {}, "Impact (single report)"), h("div", { class: "big" }, r.impact_score.toFixed(1)),
    h("table", { class: "data" }, h("tbody", {}, ...r.impact_factors.map((f) => h("tr", {}, h("td", {}, f.name), h("td", { class: "muted" }, f.detail),
      h("td", { class: `num ${f.points < 0 ? "points-neg" : ""}` }, fmtSigned(f.points)))))),
    h("div", { class: "muted", style: "margin-top:8px" }, r.note));
  $("analyze-result").replaceChildren(h("div", { class: "result-grid" }, sentBox, typeBox, impactBox));
  hBars($("probs-chart"), { rows: probs, format: (v) => `${(v * 100).toFixed(0)}%`, maxAbs: 1, labelWidth: 150, rowHeight: 20 });
}

// ------------------------------------------------------------------ results
async function loadResults() {
  if (!state.evaluation) {
    try { state.evaluation = await api("/api/evaluation"); } catch { return; }
  }
  renderResults();
}

function renderResults() {
  const ev = state.evaluation?.evaluation;
  if (!ev) { $("eval-chart").replaceChildren(h("div", { class: "muted pad" }, "Run `python main.py evaluate` to produce results.")); return; }
  const fin = state.evaluation.finbert_reference;
  const names = Object.keys(ev.models);
  const ours = names.find((n) => n.startsWith("TREMOR")), probe = names.find((n) => n.startsWith("frozen")), naive = names.find((n) => n.startsWith("keyword"));
  const tasks = [
    ["News sentiment (PhraseBank)", (m) => m.sentiment.fpb?.accuracy, fin?.sentiment.fpb?.accuracy],
    ["Tweet sentiment (TFNS)", (m) => m.sentiment.tfns?.accuracy, fin?.sentiment.tfns?.accuracy],
    ["Indian news (SEntFiN)", (m) => m.sentiment.sentfin?.accuracy, fin?.sentiment.sentfin?.accuracy],
    ["Targeted (FiQA)", (m) => m.sentiment.fiqa?.accuracy, fin?.sentiment.fiqa?.accuracy],
    ["StockTwits direction", (m) => m.sentiment.stocktwits?.directional_accuracy, fin?.sentiment.stocktwits?.directional_accuracy],
    ["Event type (macro-F1)", (m) => m.event.macro_f1, null],
    ["Opposite-sentiment entities", (m) => m.entity_sentiment.conflicting_accuracy, null],
  ];
  const series = [
    { name: "TREMOR fine-tuned", color: "var(--series-1)", values: tasks.map(([, f]) => f(ev.models[ours])) },
    fin ? { name: "FinBERT (reference)", color: "var(--series-2)", values: tasks.map(([, , v]) => v ?? null) } : null,
    probe ? { name: "Frozen encoder + head", color: "var(--series-3)", values: tasks.map(([, f]) => f(ev.models[probe])) } : null,
    { name: "Keyword baseline (naive)", color: "var(--text-muted)", values: tasks.map(([, f]) => f(ev.models[naive])) },
  ].filter(Boolean);
  groupedColumns($("eval-chart"), { categories: tasks.map(([n]) => n), series, format: (v) => v.toFixed(2), yMin: 0, yMax: 1, labelSeries: 0, height: 280 });
  $("eval-table").replaceChildren(h("div", { class: "weights-wrap" }, h("table", { class: "data" },
    h("thead", {}, h("tr", {}, h("th", {}, "Task (held-out)"), ...series.map((s) => h("th", { class: "num" }, s.name)))),
    h("tbody", {}, ...tasks.map(([n], i) => h("tr", {}, h("td", {}, n), ...series.map((s) => h("td", { class: "num" }, s.values[i] == null ? "-" : s.values[i].toFixed(3)))))))));
  const speed = names.map((n) => ({ label: n.replace(" (int8 ONNX)", ""), value: ev.models[n].efficiency.docs_per_second_batched }));
  if (fin) speed.push({ label: "FinBERT (PyTorch)", value: fin.docs_per_second_batched_cpu });
  hBars($("speed-chart"), { rows: speed.sort((a, b) => b.value - a.value), format: (v) => `${fmtInt(v)}/s`, labelWidth: 190 });
  const ft = state.evaluation.finetune;
  $("model-notes").replaceChildren(h("ul", {},
    h("li", {}, "One compact encoder (BAAI bge-small, 33M parameters) fine-tuned on sentiment, event type and entity-level sentiment at once."),
    ft ? h("li", {}, `Training rows: ${Object.entries(ft.train_rows).map(([k, v]) => `${k} ${fmtInt(v)}`).join(", ")}; ${ft.minutes} minutes on a laptop GPU.`) : null,
    ft ? h("li", {}, `Clustering geometry preserved by distillation: rank correlation ${ft.similarity_preservation_spearman.toFixed(3)} with the original encoder's similarities.`) : null,
    h("li", {}, `Runs on CPU through ONNX Runtime, int8-quantised: ${ev.models[ours].efficiency.model_mb} MB, ${ev.models[ours].efficiency.latency_ms_single_p50} ms per headline.`),
    h("li", {}, "The keyword baseline is the naive approach; FinBERT is the most-used open financial model (it was trained on PhraseBank, so its PhraseBank score is partly in-sample)."),
  ));
}

// ------------------------------------------------------------------ live updates
function handle(topic, data) {
  if (topic === "batch") {
    if (state.status) {
      state.status.counters = data.counters;
      if (state.status.replay && data.clock) state.status.clock = data.clock;
    }
    for (const d of data.docs) state.docs.unshift(d);
    state.docs.length = Math.min(state.docs.length, 200);
    for (const e of data.events) state.events.set(e.event_id, e);
    for (const s of data.entities) state.entities.set(s.entity_id, s);
    state.dirty.add("radar");
    if (state.selectedEvent && data.events.some((e) => e.event_id === state.selectedEvent)) state.refreshDetail = true;
  } else if (topic === "index") {
    state.rebalances.unshift(data);
    state.dirty.add("index");
  } else if (topic === "watch") {
    state.dirty.add("watch");
  } else if (topic === "stress") {
    if (!state.knownRuns.has(data.run_id)) {
      state.knownRuns.add(data.run_id);
      state.runs.unshift(data);
      toast(data);
    }
    state.dirty.add("stress");
    state.dirty.add("radar");
  }
}

function toast(run) {
  const t = $("toast");
  t.replaceChildren(h("div", { class: "t-title" }, `Stress test triggered · ${run.trigger?.event_type_label} impact ${run.trigger?.impact_score?.toFixed(1)}`),
    h("div", { class: "t-body" }, run.trigger?.headline || ""),
    h("div", { class: "t-body" }, `P&L ${fmtUsd(run.totals.pnl)} (${run.totals.pnl_pct.toFixed(2)}%) · CET1 ${run.capital.cet1_ratio_before}% → ${run.capital.cet1_ratio_after}%`));
  t.hidden = false;
  t.onclick = () => { t.hidden = true; state.selectedRun = run.run_id; showTab("stress"); selectRun(run.run_id); };
  clearTimeout(t._timer);
  t._timer = setTimeout(() => { t.hidden = true; }, 9000);
}

function renderDirty() {
  renderHeader();
  if (state.dirty.has("radar") && state.tab === "radar") { renderRadar(); renderDocFeed(); state.dirty.delete("radar"); }
  if (state.dirty.has("index") && state.tab === "index") { renderIndex(); state.dirty.delete("index"); }
  if (state.dirty.has("stress") && state.tab === "stress") { renderRuns(); state.dirty.delete("stress"); }
  if (state.dirty.has("results") && state.tab === "results") { renderResults(); state.dirty.delete("results"); }
  if (state.dirty.has("watch") && state.tab === "watch") { renderWatch(); state.dirty.delete("watch"); }
}

async function poll() {
  try {
    state.status = await api("/api/status");
    const events = await api("/api/events?limit=60");
    for (const e of events) state.events.set(e.event_id, e);
    if (state.events.size > 400) {
      const keep = [...state.events.values()].sort((a, b) => b.impact_score - a.impact_score).slice(0, 300);
      state.events = new Map(keep.map((e) => [e.event_id, e]));
    }
    if (state.tab === "index" || !state.index) {
      [state.index, state.indexHistory] = await Promise.all([api("/api/index"), api("/api/index/history?limit=3000")]);
      if (!state.rebalances.length) state.rebalances = await api("/api/index/rebalances?limit=60");
      state.dirty.add("index");
    }
    if (state.refreshDetail && state.selectedEvent) { state.refreshDetail = false; selectEvent(state.selectedEvent); }
    if (state.tab === "watch") await loadWatch();
    state.dirty.add("radar");
  } catch { /* server restarting: keep the last frame */ }
}

function connectStream() {
  state.stream?.close();
  state.stream = new EventSource("/api/stream");
  state.stream.onmessage = (m) => { try { const msg = JSON.parse(m.data); handle(msg.topic, msg.data); } catch { /* ignore malformed */ } };
}

async function init() {
  try {
    if (localStorage.getItem("tremor-theme")) document.documentElement.dataset.theme = localStorage.getItem("tremor-theme");
  } catch { /* storage blocked */ }
  $("theme-toggle").onclick = () => {
    const cur = document.documentElement.dataset.theme;
    const dark = cur ? cur === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("tremor-theme", document.documentElement.dataset.theme); } catch { /* ignore */ }
    for (const v of ["radar", "index", "stress", "results", "watch"]) state.dirty.add(v);
    renderDirty();
  };
  document.querySelectorAll(".tab").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
  $("btn-play").onclick = async () => {
    const paused = state.status?.replay?.paused;
    state.status.replay = await post("/api/replay/control", { action: paused ? "resume" : "pause" });
    renderHeader();
  };
  $("speed").onchange = async (e) => { state.status.replay = await post("/api/replay/control", { action: "speed", speed: +e.target.value }); };
  const restart = async (pack) => {
    await post("/api/replay/control", { action: "restart", speed: +$("speed").value, ...(pack ? { pack } : {}) });
    connectStream();  // a restart builds a new runtime with its own signal bus: re-attach the live stream
    Object.assign(state, { events: new Map(), docs: [], entities: new Map(), runs: [], rebalances: [], selectedEvent: null, eventDetail: null,
      selectedRun: null, runDetail: null, index: null, indexHistory: [], knownRuns: new Set(), watch: null, selectedWatch: null });
    $("event-detail").replaceChildren(h("div", { class: "muted pad" }, "No event selected yet."));
    $("run-detail").hidden = true;
    $("run-list").replaceChildren(h("div", { class: "muted pad" }, "No stress test yet: waiting for an event to cross the threshold."));
  };
  $("btn-restart").onclick = () => restart();
  $("pack").onchange = (e) => restart(e.target.value);
  $("btn-rerun").onclick = rerun;
  $("btn-analyze").onclick = analyze;
  $("examples").replaceChildren(...EXAMPLES.map((x) => h("button", { onclick: () => { $("analyze-text").value = x; analyze(); } }, x.length > 60 ? `${x.slice(0, 58)}...` : x)));

  [state.status, state.config] = await Promise.all([api("/api/status"), api("/api/config")]);
  if (state.status.replay) {
    const packs = await api("/api/replay/packs");
    $("pack").replaceChildren(...packs.map((p) => h("option", { value: p.name }, p.title)));
  }
  state.runs = await api("/api/stress/runs");
  state.runs.forEach((r) => state.knownRuns.add(r.run_id));
  const backtest = await api("/api/backtest");
  state.backtest = backtest.available === false ? null : backtest;
  state.docs = (await api("/api/documents?limit=80")).reverse();
  for (const e of await api("/api/entities?entity_type=index")) state.entities.set(e.entity_id, e);
  await poll();
  connectStream();
  showTab(location.hash.replace("#", "") || "radar");
  setInterval(renderDirty, 1000);
  setInterval(poll, 3000);
}

init();
