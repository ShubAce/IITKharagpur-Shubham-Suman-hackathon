"""Turn a detected event into a stress scenario: a vector of risk-factor shocks.

1. **Retrieve analogs.** Rank past episodes by (a) event-type match and (b) semantic similarity
   between the event's headlines and each episode's description, using the engine's own encoder.
   Point-in-time: only episodes that *ended before* the event are eligible.
2. **Blend, with credibility weighting.** Similarity-weighted average of the top analogs' measured
   factor moves, blended 50/50 with the average of *all* earlier crises. A handful of analogs is a
   small sample: the closest ones say which markets this kind of event moves (oil in a supply shock,
   yields in an inflation scare), the average crisis keeps the size of the moves from riding on one
   episode - the credibility weighting actuaries use for thin experience. The point-in-time backtest
   on 21 crises (``validation.py``, ``python main.py validate``) is the evidence: the blend gets the
   most factor directions right of every method tested, with less P&L error than analogs alone.
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
from functools import lru_cache

import numpy as np
import pandas as pd
import yaml

from tremor.config import load_universe
from tremor.paths import CONFIG_DIR, DATA_DIR
from tremor.schemas import EventSignal

ANALOGS_PATH = DATA_DIR / "scenarios" / "analog_shocks.csv"
REFERENCE_IMPACT = 8.5  # the impact level at which a scenario reproduces history one-for-one
EXTREME_ANALOG_IMPACT = 9.5  # 2008- and 2020-style episodes are only analogs for events at least this severe
ANALOG_CREDIBILITY = 0.5  # weight of the closest analogs; the rest goes to the average of all earlier crises
# price named in the news -> (risk factor it moves, move in factor units when every report agrees, at severity 1.0);
# oil comes before gas, so gas only speaks for the oil factor when the reports say nothing about oil itself
TEXT_OVERLAYS = {"OIL": ("CMD_OIL", 30.0), "GAS": ("CMD_OIL", 15.0), "GOLD": ("CMD_GOLD", 10.0), "SPX": ("EQ_US", 8.0)}
PRICE_NAMES = {"OIL": "oil", "GAS": "gas", "GOLD": "gold", "SPX": "the stock market"}
UNIT_SUFFIX = {"pct": "%", "bp": " bp", "pts": " pts"}  # how a shock in each factor unit is written
MIN_PRICE_REPORTS = 5  # reports stating a direction for a price before they overrule history on it
MIN_PRICE_AGREEMENT = 0.2  # |up - down| / (up + down): below this the reports disagree and history stands


@dataclass
class Scenario:
    scenario_id: str
    title: str
    method: str  # "analog" | "template" | "custom"
    severity: float  # multiplier applied to the base shocks
    shocks: dict[str, float]  # factor id -> shock (pct, bp or pts, see configs/scenarios.yaml)
    analogs: list[dict] = field(default_factory=list)  # [{episode_id, title, weight, similarity}]
    credibility: float | None = None  # share of the closest analogs in the blend (analog method only)
    prior_episodes: int = 0  # earlier crises averaged into the rest of the blend
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


@lru_cache(maxsize=1)
def _company_ids() -> frozenset[str]:
    return frozenset(e.id for e in load_universe().entities if e.type == "company")


def epicentre_of(event: EventSignal) -> list[str]:
    """Who is *directly* hit: the two most-mentioned countries of a systemic event (e.g. RU, UA),
    or the companies a firm-specific event is about. Obligors there get notched down.

    A firm-specific event is about its most-named company, plus any company named at least half as
    often ("UBS agrees to buy Credit Suisse"). Banks merely mentioned in coverage of another bank's run
    ("Banks tumble as SVB ignites broader fears") are hit through the sector shock, not notched."""
    if event.event_type == "MACROECONOMIC":
        return []  # macro shocks are global: systemic factors only, no single country singled out
    if event.market_wide:
        return [r for r in event.regions if r not in NOT_EPICENTRE][:2]
    companies = _company_ids()
    named = [e for e in event.entities if e in companies]  # most mentioned first
    if not named or not event.entity_mentions:
        return named[:3]
    top = event.entity_mentions.get(named[0], 0)
    return [e for e in named if event.entity_mentions.get(e, 0) >= 0.5 * top][:3]


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

    def eligible(self, as_of: datetime, impact: float) -> pd.Series:
        """Which episodes a scenario dated ``as_of`` may learn from: measured, ended before it (point-in-time),
        and - unless the event is itself extreme - not a once-in-a-generation crisis."""
        ok = (self.episodes["end"] < pd.Timestamp(as_of).tz_localize(None)) & self.episodes["id"].isin(self.analogs.index)
        if "extreme" in self.episodes and impact < EXTREME_ANALOG_IMPACT:
            ok &= ~self.episodes["extreme"].astype("boolean").fillna(False).astype(bool)
        return ok

    def rank_analogs(self, event_type: str, text: str, as_of: datetime, top_k: int = 3, impact: float = 10.0) -> list[dict]:
        if self.analogs.empty:
            return []
        eligible = self.eligible(as_of, impact)
        type_weight = self.episodes["event_type"].map(
            lambda t: 1.0 if t == event_type else (0.4 if t in self.related.get(event_type, []) else 0.0))
        vecs = self._episode_embeddings()
        if vecs is not None and text:
            sim = vecs @ np.asarray(self._encode([text]))[0]
        else:
            sim = np.full(len(self.episodes), 0.5)
        rows = []
        for i, ep in self.episodes.iterrows():
            if not eligible.iloc[i] or type_weight.iloc[i] == 0:
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

    def average_crisis(self, as_of: datetime, impact: float) -> tuple[dict[str, float], int]:
        """Equal-weighted average of every eligible earlier episode, and how many there were."""
        ids = self.episodes.loc[self.eligible(as_of, impact), "id"].tolist()
        return (self.blend([{"episode_id": i, "weight": 1.0 / len(ids)} for i in ids]) if ids else {}), len(ids)

    def historical_shocks(self, event_type: str, text: str, as_of: datetime, impact: float,
                          credibility: float = ANALOG_CREDIBILITY) -> tuple[dict[str, float], list[dict], int]:
        """Unscaled scenario from history: the closest analogs, credibility-weighted against the average crisis.

        Returns ``(shocks, analogs, n_prior)``; empty shocks when no episode is eligible."""
        analogs = self.rank_analogs(event_type, text, as_of, impact=impact)
        if not analogs:
            return {}, [], 0
        nearest = self.blend(analogs)
        prior, n_prior = self.average_crisis(as_of, impact)
        shocks = {f: credibility * nearest.get(f, prior.get(f, 0.0)) + (1.0 - credibility) * prior.get(f, nearest.get(f, 0.0))
                  for f in self.factors if f in nearest or f in prior}
        return shocks, analogs, n_prior

    def for_event(self, event: EventSignal, prefer: str = "analog") -> Scenario:
        severity = severity_multiplier(event.impact_score)
        # Headlines carry a display-only "…" on titles truncated upstream; it is noise for the encoder.
        text = " ".join(h.removesuffix("…") for h in [event.headline] + [s.headline for s in event.stories[:5]])
        base, analogs, n_prior = (self.historical_shocks(event.event_type, text, event.first_seen, event.impact_score)
                                  if prefer == "analog" else ({}, [], 0))
        credibility = ANALOG_CREDIBILITY if analogs else None
        if analogs:
            method = "analog"
            mix = ", ".join(f"{a['title']} ({a['weight']:.0%})" for a in analogs)
            narrative = (f"Closest historical analogs: {mix} - weighted {ANALOG_CREDIBILITY:.0%}, against the average of "
                         f"{n_prior} earlier crises ({1 - ANALOG_CREDIBILITY:.0%}); scaled x{severity:.2f} for impact "
                         f"{event.impact_score:.1f}.")
        else:
            template = self.templates.get(event.event_type) or self.templates["GEOPOLITICAL"]
            base, method = dict(template["shocks"]), "template"
            narrative = f"{template['title']}; scaled x{severity:.2f} for impact {event.impact_score:.1f}."
        shocks = {fid: round(v * severity, 2) for fid, v in base.items()}

        # Text-implied moves: when enough of the reports say which way a price they name is moving ("oil
        # surges", "gold slips"), their direction replaces history's for that factor. History supplies the
        # shape of the shock; today's reports say where the prices they talk about are actually going. The
        # direction comes from verbs of movement (nlp/price_moves.py), never from tone: "oil soars on war
        # fears" is a negative sentence about a rising price.
        overlays, decided = [], set()
        for entity, (factor, scale) in TEXT_OVERLAYS.items():
            moves = event.price_moves.get(entity) or {}
            up, down = moves.get("up", 0), moves.get("down", 0)
            if factor in decided or up + down < MIN_PRICE_REPORTS or abs(up - down) / (up + down) < MIN_PRICE_AGREEMENT:
                continue
            implied = round(scale * (up - down) / (up + down) * severity, 2)
            meta = self.factors.get(factor, {})
            unit = UNIT_SUFFIX.get(meta.get("unit", ""), "")
            overlays.append(f"{meta.get('label', factor)} {shocks.get(factor, 0.0):+.1f}{unit} -> {implied:+.1f}{unit} ({up} reports say "
                            f"{PRICE_NAMES[entity]} is rising, {down} falling)")
            shocks[factor] = implied
            decided.add(factor)
        if overlays:
            narrative += " Prices the reports say are moving: " + "; ".join(overlays) + "."

        epicentre = {e: 1 + int(event.impact_score >= 8) + int(event.impact_score >= 9) for e in epicentre_of(event)}
        return Scenario(scenario_id=f"scn_{event.event_id[4:]}", title=f"{event.event_type_label}: {event.headline[:90]}",
                        method=method, severity=round(severity, 3), shocks=shocks, analogs=analogs, credibility=credibility,
                        prior_episodes=n_prior, epicentre=epicentre, event_id=event.event_id, event_type=event.event_type,
                        impact_score=event.impact_score, narrative=narrative)

    def custom(self, title: str, shocks: dict[str, float], epicentre: dict[str, int] | None = None) -> Scenario:
        unknown = set(shocks) - set(self.factors)
        if unknown:
            raise ValueError(f"unknown risk factors: {sorted(unknown)}")
        return Scenario(scenario_id="scn_custom", title=title, method="custom", severity=1.0, shocks=dict(shocks),
                        epicentre=epicentre or {}, narrative="Analyst-defined scenario.")
