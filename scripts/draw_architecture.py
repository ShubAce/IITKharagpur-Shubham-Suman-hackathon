"""Render docs/architecture.png (high resolution) from code, so the diagram stays in sync with the system.

    python scripts/draw_architecture.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "architecture.png"

INK, MUTED, LINE = "#0b0b0b", "#52514e", "#c3c2b7"
BLUE, BLUE_WASH = "#2a78d6", "#e8f1fc"
ORANGE, ORANGE_WASH = "#eb6834", "#fdeee7"
AQUA, AQUA_WASH = "#1baf7a", "#e5f6f0"
GRAY_WASH = "#f3f2ee"


def box(ax, x, y, w, h, title, lines=(), face=GRAY_WASH, edge=LINE, title_color=INK, size=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012", linewidth=1.2,
                                facecolor=face, edgecolor=edge))
    ax.text(x + 0.012, y + h - 0.022, title, fontsize=size, fontweight="bold", color=title_color, va="top", ha="left")
    for i, line in enumerate(lines):
        ax.text(x + 0.012, y + h - 0.055 - i * 0.026, line, fontsize=8.6, color=MUTED, va="top", ha="left")


def arrow(ax, x1, y1, x2, y2, label=None, color=MUTED, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.012, label, fontsize=8, color=color, ha="center", va="bottom")


def main() -> None:
    fig = plt.figure(figsize=(18, 10), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#fcfcfb")

    ax.text(0.02, 0.965, "TREMOR - Text-driven Risk Engine for Market Observation & Response", fontsize=17, fontweight="bold", color=INK)
    ax.text(0.02, 0.935, "Unstructured news and social media in, structured risk signals out, consumed by a tactical index (Module A), "
            "a strategic stress test (Module B) and a credit early-warning watchlist.", fontsize=10.5, color=MUTED)

    # ------------------------------------------------------------------ sources
    ax.text(0.02, 0.885, "SOURCES", fontsize=9, color=MUTED, fontweight="bold")
    sources = [("GDELT 2.0 (news)", ["global news, every 15 minutes", "raw GKG files, no API key"]),
               ("RSS (news)", ["Yahoo Finance per ticker,", "Google News topics, Federal Reserve"]),
               ("StockTwits + Bluesky (social)", ["live cashtag streams, no API key;", "replays: tweet archive, Hacker News"]),
               ("Replay packs (history)", ["Russia-Ukraine, 21-24 Feb 2022;", "SVB + bank contagion, 8-15 Mar 2023"]),
               ("Prices (yfinance)", ["constituents, risk factors,", "26 historical episodes"])]
    for i, (title, lines) in enumerate(sources):
        box(ax, 0.02, 0.765 - i * 0.135, 0.17, 0.11, title, lines, face="#ffffff", size=10)

    # ------------------------------------------------------------------ engine
    ax.add_patch(FancyBboxPatch((0.225, 0.20), 0.43, 0.69, boxstyle="round,pad=0.01,rounding_size=0.02", linewidth=1.6,
                                facecolor=BLUE_WASH, edgecolor=BLUE))
    ax.text(0.24, 0.872, "AI / NLP RISK ENGINE", fontsize=12, fontweight="bold", color=BLUE)
    ax.text(0.24, 0.848, "synchronous, deterministic, event-time clock - the same code runs live and on replays", fontsize=8.5, color=MUTED)

    steps = [
        ("1  Gate", ["drop text naming no tracked entity / risk theme", "(~10x volume cut, zero model cost)"]),
        ("2  De-duplicate", ["fingerprint: syndicated copies = corroboration,", "not new information"]),
        ("3  Entity linking", ["88 entities: companies, banks, countries, central", "banks, commodities; ambiguity + context guards"]),
        ("4  One encoder, many heads (ONNX int8, CPU)", ["fine-tuned bge-small, 34 MB, ~4 ms/headline:", "embedding | sentiment | event type | entity sentiment"]),
    ]
    top, gap, h, w = 0.715, 0.118, 0.096, 0.188
    for i, (title, lines) in enumerate(steps):
        box(ax, 0.238, top - i * gap, w, h, title, lines, face="#ffffff", edge=BLUE, size=9.5)
    steps2 = [
        ("5  Story & event clustering", ["single-pass, online: stories (same report),", "events (related stories) -> novelty"]),
        ("6  Impact scorecard (1-10)", ["base severity + intensity + corroboration", "+ velocity + breadth + market linkage"]),
        ("7  Entity sentiment state", ["Bayesian filter with decay + uncertainty;", "news vs social divergence"]),
        ("8  Signals", ["Doc / Event / Entity signals: sentiment -1..1,", "event type, impact 1-10, price direction"]),
    ]
    for i, (title, lines) in enumerate(steps2):
        box(ax, 0.452, top - i * gap, w, h, title, lines, face="#ffffff", edge=BLUE, size=9.5)
    for i in range(3):
        arrow(ax, 0.332, top - i * gap - 0.002, 0.332, top - (i + 1) * gap + h + 0.012, color=BLUE)
        arrow(ax, 0.546, top - i * gap - 0.002, 0.546, top - (i + 1) * gap + h + 0.012, color=BLUE)
    arrow(ax, 0.441, top - 3 * gap + h / 2, 0.441, top + h / 2, color=BLUE)  # step 4 feeds step 5, up the gap
    box(ax, 0.24, 0.215, 0.4, 0.07, "Pub/sub bus  ->  REST API (/docs)  +  live event stream (SSE)  +  JSONL signal file",
        ["in-process today; the integration point where Kafka / Redis Streams would sit in production"], face="#ffffff", edge=BLUE, size=9.5)
    arrow(ax, 0.546, top - 3 * gap - 0.004, 0.546, 0.29, color=BLUE)

    for i in range(5):
        arrow(ax, 0.19, 0.82 - i * 0.135, 0.236, top + h / 2, color=MUTED, rad=0.0)

    # ------------------------------------------------------------------ modules
    ax.text(0.69, 0.885, "DOWNSTREAM APPLICATIONS", fontsize=9, color=MUTED, fontweight="bold")
    box(ax, 0.69, 0.64, 0.29, 0.23, "Module A - Tactical index rebalancer",
        ["subscribes to: entity sentiment (with uncertainty)",
         "TREMOR-20: 20 S&P 100 stocks; w ~ parent x exp(2s)",
         "caps 0.4x-2x parent, max 12%, sector bands +/-8 pts",
         "no-trade band 25 bp, turnover cap 10%, cost 5 bp",
         "hourly + event-driven rebalances vs equal weight",
         "every weight change attributed to a headline"],
        face=AQUA_WASH, edge=AQUA, title_color=INK, size=10.5)
    box(ax, 0.69, 0.325, 0.29, 0.295, "Module B - Strategic stress test",
        ["subscribes to: event type + impact score",
         "trigger: Geopolitical / Credit > 7, Macro > 8, per situation",
         "scenario: closest analogs (point-in-time), credibility-",
         "  weighted with the average crisis, scaled by impact;",
         "  prices the reports say are moving override history",
         "book: 388 positions, $22bn (loans from 13M card txns)",
         "Vasicek PD, IFRS 9 ECL, Basel IRB RWA, CET1 -> risk memo",
         "  (optional LLM summary: guarded, off by default)",
         "backtest, 21 crises: 88% of directions right (naive 35%)"],
        face=ORANGE_WASH, edge=ORANGE, title_color=INK, size=10.5)
    box(ax, 0.69, 0.165, 0.29, 0.14, "Credit early-warning watchlist",
        ["subscribes to: events + the entity sentiment state",
         "points scorecard -> Watch Negative / Monitor, exposure",
         "lead time measured against public rating actions"],
        face="#f6eefb", edge="#8a5cc4", title_color=INK, size=10.5)
    arrow(ax, 0.64, 0.25, 0.69, 0.72, label="entity", color=AQUA, rad=0.15)
    arrow(ax, 0.64, 0.245, 0.69, 0.45, label="event", color=ORANGE, rad=-0.05)
    arrow(ax, 0.64, 0.235, 0.69, 0.235, color="#8a5cc4")

    box(ax, 0.69, 0.03, 0.29, 0.115, "Dashboard (no build step)",
        ["radar - index - stress lab - credit watch - analyze -", "results; live via server-sent events; light / dark"],
        face="#ffffff", size=10.5)
    arrow(ax, 0.835, 0.165, 0.835, 0.147, color=MUTED)

    # ------------------------------------------------------------------ offline
    box(ax, 0.02, 0.06, 0.635, 0.12, "Offline: training, evaluation, validation  (python main.py finetune | evaluate | backtest | validate | impact)",
        ["public labelled data (PhraseBank, TFNS, FiQA, SEntFiN, StockTwits self-labels, topic tweets) + weak labels (news searches, GDELT)",
         "-> multi-task fine-tune (8 min, laptop GPU) with embedding distillation -> ONNX int8 -> held-out evaluation vs keyword baseline, FinBERT;",
         "Module A backtest; point-in-time scenario backtest (21 crises); impact event study vs market moves; price-direction check"],
        face=GRAY_WASH, size=10)
    arrow(ax, 0.215, 0.185, 0.236, top - 3 * gap + h / 2, label="", color=MUTED, rad=-0.3)
    ax.text(0.205, 0.255, "fine-tuned model", fontsize=8, color=MUTED, ha="center", rotation=90)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor=fig.get_facecolor())
    print("saved", OUT)


if __name__ == "__main__":
    main()
