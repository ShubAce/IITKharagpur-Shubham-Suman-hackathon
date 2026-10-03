// Minimal SVG charts for the dashboard: no dependencies, theme-aware through CSS custom
// properties, thin marks, hairline grids, a hover layer on every chart. All text from the
// data (headlines, names) is inserted with textContent - it is untrusted.

const SVG = "http://www.w3.org/2000/svg";
const tip = () => document.getElementById("tooltip");

export function svgEl(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

export function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "style") node.setAttribute("style", v);
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

// ------------------------------------------------------------------ tooltip
export function showTip(evt, title, rows) {
  const t = tip();
  t.replaceChildren();
  if (title) t.appendChild(h("div", { class: "tt-title" }, title));
  for (const r of rows) {
    t.appendChild(h("div", { class: "tt-row" },
      r.color ? h("span", { class: "key", style: `background:${r.color}` }) : null,
      h("b", {}, r.value), h("span", {}, r.label || "")));
  }
  t.hidden = false;
  const pad = 14;
  const { innerWidth: W, innerHeight: H } = window;
  const box = t.getBoundingClientRect();
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + box.width > W - 8) x = evt.clientX - box.width - pad;
  if (y + box.height > H - 8) y = evt.clientY - box.height - pad;
  t.style.left = `${x}px`;
  t.style.top = `${y}px`;
}
export function hideTip() { tip().hidden = true; }

// ------------------------------------------------------------------ helpers
const scale = (d0, d1, r0, r1) => (v) => (d1 === d0 ? (r0 + r1) / 2 : r0 + ((v - d0) / (d1 - d0)) * (r1 - r0));

// Clean round ticks whose first and last values *enclose* [min, max], so no mark can fall outside the axis.
export function niceTicks(min, max, count = 5) {
  if (min === max) { min -= 1; max += 1; }
  const span = max - min;
  const mag = 10 ** Math.floor(Math.log10(span / count));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) || 10 * mag;
  const ticks = [];
  const first = Math.floor(min / step) * step, last = Math.ceil(max / step) * step;
  for (let v = first; v <= last + step * 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

function wrapWords(text, maxChars = 11) {
  const lines = [];
  for (const word of text.split(" ")) {
    if (lines.length && (lines.at(-1) + " " + word).length <= maxChars) lines[lines.length - 1] += " " + word;
    else lines.push(word);
  }
  return lines.slice(0, 2);
}

export function divColor(value, maxAbs) {
  const share = Math.min(1, Math.abs(value) / (maxAbs || 1));
  const pole = value >= 0 ? "var(--div-pos)" : "var(--div-neg)";
  return `color-mix(in oklab, ${pole} ${Math.round(share * 100)}%, var(--div-mid))`;
}

// Column/bar with a 4px rounded data end and a square baseline end.
function barPath(x, y, w, hgt, r, end) {
  r = Math.max(0, Math.min(r, w / 2, hgt / 2));
  if (end === "top") return `M${x},${y + hgt}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + hgt}Z`;
  if (end === "bottom") return `M${x},${y}V${y + hgt - r}Q${x},${y + hgt} ${x + r},${y + hgt}H${x + w - r}Q${x + w},${y + hgt} ${x + w},${y + hgt - r}V${y}Z`;
  if (end === "right") return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + hgt - r}Q${x + w},${y + hgt} ${x + w - r},${y + hgt}H${x}Z`;
  return `M${x + w},${y}H${x + r}Q${x},${y} ${x},${y + r}V${y + hgt - r}Q${x},${y + hgt} ${x + r},${y + hgt}H${x + w}Z`; // left
}

function prepare(container, height) {
  container.replaceChildren();
  const width = Math.max(280, container.clientWidth - 24);
  return { width, height };
}

function legend(container, series, shape = "line") {
  if (series.length < 2) return;
  container.appendChild(h("div", { class: "legend" },
    series.map((s) => h("span", {}, h("span", { class: `key ${shape === "rect" ? "rect" : ""}`, style: `background:${s.color}` }), s.name))));
}

function remember(container, fn, args) { container._chart = { fn, args }; }
const observer = new ResizeObserver((entries) => {
  for (const entry of entries) {
    const c = entry.target;
    if (c._chart && Math.abs((c._lastWidth || 0) - c.clientWidth) > 8) {
      c._lastWidth = c.clientWidth;
      c._chart.fn(c, c._chart.args);
    }
  }
});
function observe(container) { if (!container._observed) { observer.observe(container); container._observed = true; container._lastWidth = container.clientWidth; } }

// ------------------------------------------------------------------ line chart
export function lineChart(container, args) {
  remember(container, lineChart, args); observe(container);
  const { series, height = 240, yFormat = (v) => v.toFixed(2), xFormat = (d) => d.toLocaleString(), yLabel = "", baselineY = null } = args;
  const { width } = prepare(container, height);
  legend(container, series);
  const all = series.flatMap((s) => s.points);
  if (!all.length) { container.appendChild(h("div", { class: "muted pad" }, "Waiting for data...")); return; }
  const m = { l: 52, r: 96, t: 10, b: 26 };
  const xs = all.map((p) => +p.x), ys = all.map((p) => p.y);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  if (baselineY !== null) { yMin = Math.min(yMin, baselineY); yMax = Math.max(yMax, baselineY); }
  const padY = (yMax - yMin) * 0.08 || Math.abs(yMax) * 0.01 || 1;
  const ticks = niceTicks(yMin - padY, yMax + padY, 5);
  const x = scale(Math.min(...xs), Math.max(...xs), m.l, width - m.r);
  const y = scale(ticks[0], ticks[ticks.length - 1], height - m.b, m.t);
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": yLabel || "line chart" }, container);
  for (const t of ticks) {
    svgEl("line", { x1: m.l, x2: width - m.r, y1: y(t), y2: y(t), class: "gridline" }, svg);
    svgEl("text", { x: m.l - 6, y: y(t) + 4, "text-anchor": "end", class: "axis-text" }, svg).textContent = yFormat(t);
  }
  if (baselineY !== null) svgEl("line", { x1: m.l, x2: width - m.r, y1: y(baselineY), y2: y(baselineY), class: "baseline" }, svg);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  for (let i = 0; i <= 4; i++) {
    const v = xmin + ((xmax - xmin) * i) / 4;
    svgEl("text", { x: x(v), y: height - 6, "text-anchor": i === 0 ? "start" : i === 4 ? "end" : "middle", class: "axis-text" }, svg).textContent = xFormat(new Date(v));
  }
  const ends = [];
  for (const s of series) {
    if (!s.points.length) continue;
    const d = s.points.map((p, i) => `${i ? "L" : "M"}${x(+p.x).toFixed(1)},${y(p.y).toFixed(1)}`).join("");
    if (s.area) svgEl("path", { d: `${d}L${x(+s.points.at(-1).x)},${y(ticks[0])}L${x(+s.points[0].x)},${y(ticks[0])}Z`, fill: s.color, "fill-opacity": 0.1, stroke: "none" }, svg);
    svgEl("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    const last = s.points.at(-1);
    svgEl("circle", { cx: x(+last.x), cy: y(last.y), r: 4, fill: s.color, stroke: "var(--surface-1)", "stroke-width": 2 }, svg);
    ends.push({ y: y(last.y), text: `${s.name} ${yFormat(last.y)}` });
  }
  // End labels only when they do not collide; otherwise the legend and tooltip carry identity.
  ends.sort((a, b) => a.y - b.y);
  if (series.length <= 4 && ends.every((e, i) => i === 0 || e.y - ends[i - 1].y > 14)) {
    for (const e of ends) svgEl("text", { x: width - m.r + 8, y: e.y + 4, class: "label-text" }, svg).textContent = e.text;
  }
  // Crosshair + tooltip listing every series at the nearest x.
  const cross = svgEl("line", { y1: m.t, y2: height - m.b, class: "crosshair", visibility: "hidden" }, svg);
  const hit = svgEl("rect", { x: m.l, y: m.t, width: width - m.l - m.r, height: height - m.t - m.b, fill: "transparent" }, svg);
  const toData = (evt) => {
    const box = svg.getBoundingClientRect();
    const px = ((evt.clientX - box.left) / box.width) * width;
    return xmin + ((px - m.l) / (width - m.l - m.r)) * (xmax - xmin);
  };
  hit.addEventListener("pointermove", (evt) => {
    const xv = toData(evt);
    const rows = [];
    let snapX = null;
    for (const s of series) {
      if (!s.points.length) continue;
      let best = s.points[0];
      for (const p of s.points) if (Math.abs(+p.x - xv) < Math.abs(+best.x - xv)) best = p;
      snapX = snapX ?? +best.x;
      rows.push({ color: s.color, value: yFormat(best.y), label: s.name });
    }
    cross.setAttribute("x1", x(snapX)); cross.setAttribute("x2", x(snapX)); cross.setAttribute("visibility", "visible");
    showTip(evt, xFormat(new Date(snapX)), rows);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
}

// ------------------------------------------------------------------ horizontal bars
export function hBars(container, args) {
  remember(container, hBars, args); observe(container);
  const { rows, format = (v) => v.toFixed(2), diverging = false, labelWidth = 150, rowHeight = 24, color = "var(--series-1)", maxAbs: forcedMax } = args;
  const { width } = prepare(container, 0);
  if (!rows.length) { container.appendChild(h("div", { class: "muted pad" }, "No data yet.")); return; }
  const m = { l: labelWidth, r: 64, t: 6, b: 6 };
  const height = m.t + m.b + rows.length * rowHeight;
  const maxAbs = forcedMax || Math.max(...rows.map((r) => Math.abs(r.value) + (r.err || 0)), 1e-9);
  const x = diverging ? scale(-maxAbs, maxAbs, m.l, width - m.r) : scale(0, maxAbs, m.l, width - m.r);
  const zero = x(0);
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}` }, container);
  svgEl("line", { x1: zero, x2: zero, y1: m.t, y2: height - m.b, class: "baseline" }, svg);
  const thick = Math.min(14, rowHeight - 8);
  rows.forEach((r, i) => {
    const cy = m.t + i * rowHeight + rowHeight / 2;
    const label = svgEl("text", { x: m.l - 8, y: cy + 4, "text-anchor": "end", class: "label-text" }, svg);
    label.textContent = r.label;
    const x1 = x(r.value);
    const left = Math.min(zero, x1), w = Math.max(1.5, Math.abs(x1 - zero));
    const fill = r.color || (diverging ? (r.value >= 0 ? "var(--div-pos)" : "var(--div-neg)") : color);
    svgEl("path", { d: barPath(left, cy - thick / 2, w, thick, 4, r.value >= 0 ? "right" : "left"), fill }, svg);
    if (r.err) {
      svgEl("line", { x1: x(r.value - r.err), x2: x(r.value + r.err), y1: cy, y2: cy, stroke: "var(--text-secondary)", "stroke-width": 1 }, svg);
    }
    const tipX = r.value >= 0 ? Math.max(x1, r.err ? x(r.value + r.err) : x1) + 6 : Math.min(x1, r.err ? x(r.value - r.err) : x1) - 6;
    const text = format(r.value);
    const room = r.value >= 0 ? width - tipX : tipX - m.l;
    if (room >= text.length * 6.5) {
      svgEl("text", { x: tipX, y: cy + 4, "text-anchor": r.value >= 0 ? "start" : "end", class: "label-text" }, svg).textContent = text;
    } else {
      // No room outside the bar end: write the value inside the bar, near the baseline, in white.
      const inside = svgEl("text", { x: r.value >= 0 ? zero + 6 : zero - 6, y: cy + 4, "text-anchor": r.value >= 0 ? "start" : "end",
        class: "label-text", style: "fill:#ffffff" }, svg);
      inside.textContent = text;
    }
    const hit = svgEl("rect", { x: 0, y: cy - rowHeight / 2, width, height: rowHeight, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", (evt) => showTip(evt, r.label, [{ color: fill, value: format(r.value), label: r.sub || "" }]));
    hit.addEventListener("pointerleave", hideTip);
    if (r.onclick) { hit.style.cursor = "pointer"; hit.addEventListener("click", r.onclick); }
  });
}

// ------------------------------------------------------------------ heatmap (diverging)
export function heatmap(container, args) {
  remember(container, heatmap, args); observe(container);
  const { rows, cols, matrix, format = (v) => v.toFixed(3), colFormat = (d) => d.toLocaleString(), maxAbs: forcedMax, rowLabels } = args;
  const { width } = prepare(container, 0);
  if (!cols.length) { container.appendChild(h("div", { class: "muted pad" }, "Waiting for the first rebalance...")); return; }
  const maxAbs = forcedMax || Math.max(...matrix.flat().map(Math.abs), 1e-9);
  container.appendChild(h("div", { class: "scale-legend" }, h("span", {}, `underweight ${format(-maxAbs)}`), h("span", { class: "ramp" }),
    h("span", {}, `overweight ${format(maxAbs)}`)));
  const m = { l: 64, r: 8, t: 4, b: 22 };
  const cellH = 15;
  const height = m.t + m.b + rows.length * cellH;
  const cw = (width - m.l - m.r) / cols.length;
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}` }, container);
  rows.forEach((name, i) => {
    svgEl("text", { x: m.l - 6, y: m.t + i * cellH + cellH / 2 + 4, "text-anchor": "end", class: "label-text" }, svg).textContent = name;
    cols.forEach((c, j) => {
      const v = matrix[i][j];
      const cell = svgEl("rect", { x: m.l + j * cw, y: m.t + i * cellH, width: Math.max(1, cw - (cw > 4 ? 1 : 0)), height: cellH - 1,
        style: `fill:${divColor(v, maxAbs)}` }, svg);
      cell.addEventListener("pointermove", (evt) => showTip(evt, `${rowLabels ? rowLabels[i] : name} · ${colFormat(c)}`,
        [{ color: divColor(v, maxAbs), value: format(v), label: "active weight" }]));
      cell.addEventListener("pointerleave", hideTip);
    });
  });
  const n = Math.min(5, cols.length);
  for (let k = 0; k < n; k++) {
    const j = n === 1 ? 0 : Math.round((k * (cols.length - 1)) / (n - 1));
    svgEl("text", { x: m.l + j * cw + cw / 2, y: height - 6, "text-anchor": k === 0 ? "start" : k === n - 1 ? "end" : "middle", class: "axis-text" }, svg)
      .textContent = colFormat(cols[j]);
  }
}

// ------------------------------------------------------------------ waterfall
export function waterfall(container, args) {
  remember(container, waterfall, args); observe(container);
  const { start, steps, endLabel, format = (v) => v.toFixed(0), height = 280 } = args;
  const { width } = prepare(container, height);
  const items = [{ label: start.label, base: 0, value: start.value, kind: "total" }];
  let running = start.value;
  for (const s of steps) { items.push({ label: s.label, base: running, value: s.value, kind: "step" }); running += s.value; }
  items.push({ label: endLabel, base: 0, value: running, kind: "total" });
  const lows = items.map((it) => Math.min(it.base, it.base + it.value)), highs = items.map((it) => Math.max(it.base, it.base + it.value));
  // Zoom the y-axis onto the band where the changes happen (a truncated axis, so the two total
  // columns are labelled with their full values); keep a margin so labels never touch the frame.
  const changeLo = Math.min(...lows.slice(1, -1), start.value, running), changeHi = Math.max(...highs.slice(1, -1), start.value, running);
  const margin = Math.max((changeHi - changeLo) * 0.25, Math.abs(start.value) * 0.001);
  const ticks = niceTicks(changeLo - margin, changeHi + margin, 4);
  const m = { l: 56, r: 8, t: 18, b: 44 };
  const y = scale(ticks[0], ticks.at(-1), height - m.b, m.t);
  const band = (width - m.l - m.r) / items.length;
  const bw = Math.min(40, band * 0.6);
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}` }, container);
  for (const t of ticks) {
    svgEl("line", { x1: m.l, x2: width - m.r, y1: y(t), y2: y(t), class: "gridline" }, svg);
    svgEl("text", { x: m.l - 6, y: y(t) + 4, "text-anchor": "end", class: "axis-text" }, svg).textContent = format(t);
  }
  items.forEach((it, i) => {
    const cx = m.l + band * i + band / 2;
    const top = y(it.kind === "total" ? it.value : Math.max(it.base, it.base + it.value));
    const bottom = it.kind === "total" ? y(ticks[0]) : y(Math.min(it.base, it.base + it.value));
    const fill = it.kind === "total" ? "var(--series-1)" : it.value >= 0 ? "var(--div-pos)" : "var(--div-neg)";
    svgEl("path", { d: barPath(cx - bw / 2, top, bw, Math.max(1.5, bottom - top), 4, it.kind === "step" && it.value < 0 ? "bottom" : "top"), fill }, svg);
    const valueText = it.kind === "total" ? format(it.value) : `${it.value >= 0 ? "+" : ""}${format(it.value)}`;
    const ly = it.kind === "step" && it.value < 0 ? bottom + 13 : top - 5;
    svgEl("text", { x: cx, y: ly, "text-anchor": "middle", class: it.kind === "total" ? "label-strong" : "label-text" }, svg).textContent = valueText;
    wrapWords(it.label, Math.max(8, Math.floor(band / 6.5))).forEach((line, k) => {
      svgEl("text", { x: cx, y: height - m.b + 14 + k * 13, "text-anchor": "middle", class: "axis-text" }, svg).textContent = line;
    });
    if (i < items.length - 1) {
      const endY = y(it.kind === "total" ? it.value : it.base + it.value);
      svgEl("line", { x1: cx + bw / 2, x2: cx + band - bw / 2, y1: endY, y2: endY, class: "baseline" }, svg);
    }
    const hit = svgEl("rect", { x: cx - band / 2, y: m.t, width: band, height: height - m.t - m.b, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", (evt) => showTip(evt, it.label, [{ color: fill, value: valueText, label: "USD m" }]));
    hit.addEventListener("pointerleave", hideTip);
  });
}

// ------------------------------------------------------------------ grouped columns
export function groupedColumns(container, args) {
  remember(container, groupedColumns, args); observe(container);
  const { categories, series, format = (v) => v.toFixed(2), height = 260, yMin = 0, yMax = 1, labelSeries = null } = args;
  const { width } = prepare(container, height);
  legend(container, series, "rect");
  const m = { l: 40, r: 8, t: 16, b: 40 };
  const ticks = niceTicks(yMin, yMax, 5);
  const y = scale(ticks[0], ticks.at(-1), height - m.b, m.t);
  const band = (width - m.l - m.r) / categories.length;
  const bw = Math.min(24, (band * 0.75) / series.length);
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}` }, container);
  for (const t of ticks) {
    svgEl("line", { x1: m.l, x2: width - m.r, y1: y(t), y2: y(t), class: t === ticks[0] ? "baseline" : "gridline" }, svg);
    svgEl("text", { x: m.l - 6, y: y(t) + 4, "text-anchor": "end", class: "axis-text" }, svg).textContent = format(t);
  }
  categories.forEach((cat, i) => {
    const gx = m.l + band * i + band / 2 - (bw * series.length + 2 * (series.length - 1)) / 2;
    series.forEach((s, k) => {
      const v = s.values[i];
      if (v === null || v === undefined) return;
      const bx = gx + k * (bw + 2), top = y(v), bh = Math.max(1.5, y(ticks[0]) - top);
      svgEl("path", { d: barPath(bx, top, bw, bh, 4, "top"), fill: s.color }, svg);
      if (labelSeries === null || labelSeries === k) {
        svgEl("text", { x: bx + bw / 2, y: top - 4, "text-anchor": "middle", class: "label-text" }, svg).textContent = format(v);
      }
      const hit = svgEl("rect", { x: bx - 1, y: m.t, width: bw + 2, height: height - m.t - m.b, fill: "transparent" }, svg);
      hit.addEventListener("pointermove", (evt) => showTip(evt, cat, series.map((ss) => ({ color: ss.color, value: ss.values[i] == null ? "-" : format(ss.values[i]), label: ss.name }))));
      hit.addEventListener("pointerleave", hideTip);
    });
    wrapWords(cat, Math.max(10, Math.floor(band / 6.5))).forEach((line, k) => {
      svgEl("text", { x: m.l + band * i + band / 2, y: height - m.b + 15 + k * 13, "text-anchor": "middle", class: "axis-text" }, svg).textContent = line;
    });
  });
}
