"""Turn a detected event into a stress scenario: a vector of risk-factor shocks.

1. **Retrieve analogs.** Rank past episodes by (a) event-type match and (b) semantic similarity
   between the event's headlines and each episode's description, using the engine's own encoder.
   Point-in-time: only episodes that *ended before* the event are eligible.
2. **Blend.** Similarity-weighted average of the top analogs' measured factor moves.
3. **Scale by impact.** Impact 8.5 reproduces the blended history (x1.0); impact 7 gives x0.67,
   impact 10 gives x1.33 - beyond the historical analogs, as a stress test should allow.
4. **Overlay the epicentre.** Obligors the event names (or the countries it names) get an
   idiosyncratic downgrade on top of the systematic shock.

If no analog is eligible the expert template for the event type is used instead, and either way
the analyst can override any factor from the dashboard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from tremor.config import load_universe
from tremor.paths import CONFIG_DIR, DATA_DIR
from tremor.schemas import EventSignal

ANALOGS_PATH = DATA_DIR / "scenarios" / "analog_shocks.csv"
REFERENCE_IMPACT = 8.5  # the impact level at which a scenario reproduces history one-for-one
EXTREME_ANALOG_IMPACT = 9.5  # 2008- and 2020-style episodes are only analogs for events at least this severe
# entity named in the news -> (risk factor it moves, shock in factor units at tone +1.0 and severity 1.0)
TEXT_OVERLAYS = {"OIL": ("CMD_OIL", 30.0), "GAS": ("CMD_OIL", 15.0), "GOLD": ("CMD_GOLD", 10.0), "SPX": ("EQ_US", 8.0)}


@dataclass
class Scenario:
    scenario_id: str
    title: str
    method: str  # "analog" | "template" | "custom"
    severity: float  # multiplier applied to the base shocks
    shocks: dict[str, float]  # factor id -> shock (pct, bp or pts, see configs/scenarios.yaml)
    analogs: list[dict] = field(default_factory=list)  # [{episode_id, title, weight, similarity}]
    epicentre: dict[str, int] = field(default_factory=dict)  # obligor or country id -> rating notches down
    event_id: str | None = None
    event_type: str | None = None
    impact_score: float | None = None
    narrative: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# The bank's home market and the broad blocs that *respond* to most crises are covered by the
# systemic factor shocks; they are never treated as the epicentre of an idiosyncratic hit.
NOT_EPICENTRE = frozenset({"US", "EU"})


def epicentre_of(event: EventSignal) -> list[str]:
    """Who is *directly* hit: the two most-mentioned countries of a systemic event (e.g. RU, UA),
    or the companies a firm-specific event is about. Obligors there get notched down."""
    if event.event_type == "MACROECONOMIC":
        return []  # macro shocks are global: systemic factors only, no single country singled out
    if event.market_wide:
        return [r for r in event.regions if r not in NOT_EPICENTRE][:2]
    companies = {e.id for e in load_universe().entities if e.type == "company"}
    return [e for e in event.entities if e in companies][:3]


def severity_multiplier(impact: float) -> float:
    return float(np.clip((impact - 4.0) / (REFERENCE_IMPACT - 4.0), 0.25, 1.35))


class ScenarioLibrary:
    def __init__(self, encode=None):
        """``encode``: callable(list[str]) -> unit-norm embeddings (the engine's text model)."""
        cfg = yaml.safe_load((CONFIG_DIR / "scenarios.yaml").read_text(encoding="utf-8"))
        self.factors: dict[str, dict] = cfg["risk_factors"]
        self.sector_factor: dict[str, str] = cfg["sector_factor"]
        self.related: dict[str, list[str]] = cfg.get("related_types", {})
        self.templates: dict[str, dict] = cfg["templates"]
        self.episodes = pd.DataFrame(cfg["episodes"])
        self.episodes["end"] = pd.to_datetime(self.episodes["end"])
        self.analogs = pd.read_csv(ANALOGS_PATH).set_index("episode_id") if ANALOGS_PATH.exists() else pd.DataFrame()
        self._encode = encode
        self._episode_vecs: np.ndarray | None = None

    # ------------------------------------------------------------------ retrieval
    def _episode_embeddings(self) -> np.ndarray | None:
        if self._encode is None:
            return None
        if self._episode_vecs is None:
            texts = (self.episodes["title"] + ". " + self.episodes["description"]).tolist()
            self._episode_vecs = np.asarray(self._encode(texts))
        return self._episode_vecs

    def rank_analogs(self, event_type: str, text: str, as_of: datetime, top_k: int = 3, impact: float = 10.0) -> list[dict]:
        if self.analogs.empty:
            return []
        eligible = self.episodes["end"] < pd.Timestamp(as_of).tz_localize(None)
        if "extreme" in self.episodes:
            eligible &= ~(self.episodes["extreme"].fillna(False).astype(bool) & (impact < EXTREME_ANALOG_IMPACT))
        type_weight = self.episodes["event_type"].map(
            lambda t: 1.0 if t == event_type else (0.4 if t in self.related.get(event_type, []) else 0.0))
        vecs = self._episode_embeddings()
        if vecs is not None and text:
            sim = vecs @ np.asarray(self._encode([text]))[0]
        else:
            sim = np.full(len(self.episodes), 0.5)
        rows = []
        for i, ep in self.episodes.iterrows():
            if not eligible.iloc[i] or type_weight.iloc[i] == 0 or ep["id"] not in self.analogs.index:
                continue
            # Sharpen similarities so a clearly closer episode dominates the blend.
            score = type_weight.iloc[i] * math.exp(12.0 * float(sim[i]))
            rows.append({"episode_id": ep["id"], "title": ep["title"], "event_type": ep["event_type"],
                         "start": str(ep["start"]), "end": str(ep["end"].date()), "similarity": round(float(sim[i]), 3), "score": score})
        rows.sort(key=lambda r: -r["score"])
        rows = rows[:top_k]
        total = sum(r["score"] for r in rows) or 1.0
        for r in rows:
            r["weight"] = round(r.pop("score") / total, 3)
        return rows

    # ------------------------------------------------------------------ construction
    def blend(self, analogs: list[dict]) -> dict[str, float]:
        """Weighted average of analog shocks, factor by factor (an episode missing a factor is skipped)."""
        shocks = {}
        for fid in self.factors:
            num = den = 0.0
            for a in analogs:
                value = self.analogs.loc[a["episode_id"], fid]
                if pd.notna(value):
                    num += a["weight"] * float(value)
                    den += a["weight"]
            if den > 0:
                shocks[fid] = num / den
        return shocks

    def for_event(self, event: EventSignal, prefer: str = "analog") -> Scenario:
        severity = severity_multiplier(event.impact_score)
        # Headlines carry a display-only "…" on titles truncated upstream; it is noise for the encoder.
        text = " ".join(h.removesuffix("…") for h in [event.headline] + [s.headline for s in event.stories[:5]])
        analogs = self.rank_analogs(event.event_type, text, event.first_seen, impact=event.impact_score) if prefer == "analog" else []
        if analogs:
            base, method = self.blend(analogs), "analog"
            mix = ", ".join(f"{a['title']} ({a['weight']:.0%})" for a in analogs)
            narrative = f"Blend of historical analogs: {mix}; scaled x{severity:.2f} for impact {event.impact_score:.1f}."
        else:
            template = self.templates.get(event.event_type) or self.templates["GEOPOLITICAL"]
            base, method = dict(template["shocks"]), "template"
            narrative = f"{template['title']}; scaled x{severity:.2f} for impact {event.impact_score:.1f}."
        shocks = {fid: round(v * severity, 2) for fid, v in base.items()}

        # Text-implied overlays: when a meaningful share of the reports talks about a price (oil, gold,
        # the stock market) the direction in which they talk about it adjusts that factor. History supplies
        # the shape of the shock; today's news says which way the prices it names are actually moving.
        overlays = []
        for entity, (factor, scale) in TEXT_OVERLAYS.items():
            mentions = event.entity_mentions.get(entity, 0)
            tone = event.entity_sentiment.get(entity, 0.0)
            if (mentions >= 25 or (mentions >= 5 and mentions / max(1, event.n_docs) >= 0.02)) and abs(tone) >= 0.1:
                delta = round(scale * tone * severity, 2)
                shocks[factor] = round(shocks.get(factor, 0.0) + delta, 2)
                overlays.append(f"{factor} {delta:+.1f} ({mentions} reports on {entity.lower()}, tone {tone:+.2f})")
        if overlays:
            narrative += " Text-implied adjustments: " + "; ".join(overlays) + "."

        epicentre = {e: 1 + int(event.impact_score >= 8) + int(event.impact_score >= 9) for e in epicentre_of(event)}
        return Scenario(scenario_id=f"scn_{event.event_id[4:]}", title=f"{event.event_type_label}: {event.headline[:90]}",
                        method=method, severity=round(severity, 3), shocks=shocks, analogs=analogs, epicentre=epicentre,
                        event_id=event.event_id, event_type=event.event_type, impact_score=event.impact_score,
                        narrative=narrative)

    def custom(self, title: str, shocks: dict[str, float], epicentre: dict[str, int] | None = None) -> Scenario:
        unknown = set(shocks) - set(self.factors)
        if unknown:
            raise ValueError(f"unknown risk factors: {sorted(unknown)}")
        return Scenario(scenario_id="scn_custom", title=title, method="custom", severity=1.0, shocks=dict(shocks),
                        epicentre=epicentre or {}, narrative="Analyst-defined scenario.")
