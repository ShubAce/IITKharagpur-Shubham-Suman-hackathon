"""Render docs/results_at_a_glance.png: TREMOR against the naive approach, one metric per panel.

Every number is read from docs/results/*.json, so the figure cannot drift from the results. Small multiples
(no shared axis between different units), TREMOR the only coloured bar, baselines in grey, every bar labelled.

    python scripts/draw_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"
OUT = ROOT / "docs" / "results_at_a_glance.png"

SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
TREMOR, BASELINE = "#2a78d6", "#c3c2b7"


def load(name: str) -> dict:
    return json.loads((RES / f"{name}.json").read_text(encoding="utf-8"))


def panel(ax, title: str, note: str, rows: list[tuple[str, float, bool]], fmt, lower_is_better: bool = False) -> None:
    """Horizontal bars; ``rows`` = (label, value, is_tremor), drawn top to bottom."""
    labels, values, ours = zip(*rows)
    y = range(len(rows))[::-1]
    ax.barh(list(y), values, height=0.55, color=[TREMOR if o else BASELINE for o in ours], zorder=2)
    top = max(values)
    for yy, v, o in zip(y, values, ours):
        ax.text(v + top * 0.02, yy, fmt(v), va="center", ha="left", fontsize=10.5, color=INK,
                fontweight="bold" if o else "normal")
    ax.set_yticks(list(y), labels, fontsize=9.5, color=INK_2)
    ax.set_xlim(0, top * 1.32)
    ax.xaxis.set_visible(False)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, fontweight="bold", pad=18)
    ax.text(0, 1.02, note + ("  (lower is better)" if lower_is_better else ""), transform=ax.transAxes, fontsize=8.8,
            color=INK_3, va="bottom")


def main() -> None:
    ev, fin, sb, pm, rp = (load(n) for n in ("evaluation", "finbert_reference", "scenario_backtest", "price_moves_validation",
                                             "replay_ukraine_2022"))
    models = ev["models"]
    ours = next(k for k in models if k.startswith("TREMOR"))
    naive = next(k for k in models if k.startswith("keyword"))
    s = sb["summary"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
    fig.text(0.02, 0.965, "TREMOR against the naive approach", fontsize=17, fontweight="bold", color=INK)
    fig.text(0.02, 0.928, "Held-out data and point-in-time history only. Every number comes from docs/results/*.json; "
             "stronger baselines and all metrics are in docs/RESULTS.md.", fontsize=10, color=INK_2)

    panel(axes[0, 0], "Event type", "macro-F1, human-labelled test set",
          [("Keyword baseline", models[naive]["event"]["macro_f1"], False), ("TREMOR", models[ours]["event"]["macro_f1"], True)],
          lambda v: f"{v:.2f}")
    panel(axes[0, 1], "Tweet sentiment", "accuracy, held-out financial tweets (TFNS)",
          [("Keyword baseline", models[naive]["sentiment"]["tfns"]["accuracy"], False),
           ("FinBERT", fin["sentiment"]["tfns"]["accuracy"], False),
           ("TREMOR", models[ours]["sentiment"]["tfns"]["accuracy"], True)], lambda v: f"{v:.2f}")
    panel(axes[0, 2], "Price direction from headlines", f"agreement with the real move, {pm['headlines']} headlines",
          [("Always the commoner direction", pm["majority_class_baseline"], False), ("TREMOR", pm["agreement"], True)],
          lambda v: f"{v:.0%}")
    n = s["context"]["episodes"]
    panel(axes[1, 0], "Stress scenarios: directions", f"risk-factor moves called right, {n} past crises",
          [("The brief's example shock", s["naive"]["direction_hit_rate"], False), ("TREMOR", s["tremor"]["direction_hit_rate"], True)],
          lambda v: f"{v:.0%}")
    panel(axes[1, 1], "Stress scenarios: P&L error", f"mean error in the bank's P&L, {n} past crises",
          [("The brief's example shock", s["naive"]["pnl_error_mean_musd"], False), ("TREMOR", s["tremor"]["pnl_error_mean_musd"], True)],
          lambda v: f"${v:,.0f}m", lower_is_better=True)

    tile = axes[1, 2]
    tile.axis("off")
    c = rp["documents"]
    tile.set_title("Signal, not noise", loc="left", fontsize=11.5, color=INK, fontweight="bold", pad=18)
    tile.text(0, 1.02, "the four-day Ukraine replay", transform=tile.transAxes, fontsize=8.8, color=INK_3, va="bottom")
    tile.text(0.0, 0.62, f"{c['documents']:,}", fontsize=34, fontweight="bold", color=INK, transform=tile.transAxes)
    tile.text(0.0, 0.50, "documents in", fontsize=11, color=INK_2, transform=tile.transAxes)
    tile.text(0.0, 0.18, f"{len(rp['stress_runs'])}", fontsize=34, fontweight="bold", color=TREMOR, transform=tile.transAxes)
    tile.text(0.0, 0.06, "stress tests out,", fontsize=11, color=INK_2, transform=tile.transAxes)
    tile.text(0.0, -0.06, "each tied to a real development", fontsize=11, color=INK_2, transform=tile.transAxes)

    fig.subplots_adjust(left=0.13, right=0.98, top=0.82, bottom=0.04, wspace=0.75, hspace=0.62)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print("saved", OUT)


if __name__ == "__main__":
    main()
