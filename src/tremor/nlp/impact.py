"""Impact score (1-10): a transparent, points-based scorecard.

Credit analysts will recognise the shape: a base score for the kind of event, then points
added or removed for observable evidence. Every point is reported back as an ``ImpactFactor``,
so a score of 8.5 can always be explained ("+2.0 because 41 independent publishers...").

The score is deliberately *dynamic*. A single early report of a geopolitical event starts
around 5; it only crosses the stress-test threshold of 7 once other outlets corroborate it,
reports accelerate and the language turns severe. One alarming tweet cannot trigger a stress test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tremor.config import ScorecardSettings, Taxonomy
from tremor.schemas import ImpactFactor

MIN_IMPACT, MAX_IMPACT = 1.0, 10.0


@dataclass(frozen=True)
class ImpactInputs:
    event_type: str
    type_confidence: float
    sentiment: float  # -1 .. 1
    n_scored: int  # documents analysed (exact duplicates excluded)
    n_stories: int  # independently worded reports (syndicated copies of one wire story count once)
    n_publishers: int
    n_news: int
    n_social: int
    stories_last_hour: int
    extreme_docs: int  # documents containing an extreme-severity cue
    high_docs: int
    scale_docs: int
    n_sectors: int
    n_regions: int
    market_docs: int = 0  # documents discussing the event in market terms


def score_impact(x: ImpactInputs, taxonomy: Taxonomy, cfg: ScorecardSettings) -> tuple[float, list[ImpactFactor]]:
    spec = taxonomy.event_types[x.event_type]
    factors = [ImpactFactor(name="Event type", points=spec.base_severity, detail=f"{spec.label}: base severity")]
    n = max(1, x.n_scored)

    def share_weight(docs: int) -> float:
        # A cue present in a fifth of the reports counts in full; a lone mention counts half.
        return 0.0 if docs == 0 else (1.0 if docs / n >= 0.2 or n <= 3 else 0.5)

    intensity, notes = 0.0, []
    if x.extreme_docs:
        intensity += cfg.extreme_cue * share_weight(x.extreme_docs)
        notes.append(f"extreme-severity language in {x.extreme_docs}/{n} reports")
    elif x.high_docs:
        intensity += cfg.high_cue * share_weight(x.high_docs)
        notes.append(f"high-severity language in {x.high_docs}/{n} reports")
    if x.scale_docs:
        intensity += cfg.scale_cue * share_weight(x.scale_docs)
        notes.append("amounts in the billions")
    if abs(x.sentiment) >= cfg.strong_sentiment_level:
        intensity += cfg.strong_sentiment
        notes.append(f"strong sentiment ({x.sentiment:+.2f})")
    if intensity:
        factors.append(ImpactFactor(name="Intensity", points=min(cfg.intensity_cap, intensity), detail="; ".join(notes)))

    if x.n_stories > 2:
        # Independent reporting, not syndication: one wire story reprinted by forty outlets is one report.
        points = min(cfg.corroboration_cap, cfg.corroboration_per_doubling * math.log2(x.n_stories / 2))
        factors.append(ImpactFactor(name="Corroboration", points=points,
                                    detail=f"{x.n_stories} independent reports across {x.n_publishers} publishers"))

    velocity = max((pts for threshold, pts in cfg.velocity_steps if x.stories_last_hour >= threshold), default=0.0)
    if velocity:
        factors.append(ImpactFactor(name="Velocity", points=velocity, detail=f"{x.stories_last_hour} new reports in the last hour"))

    breadth, notes = 0.0, []
    if x.n_sectors >= 3:
        breadth += cfg.multi_sector
        notes.append(f"{x.n_sectors} sectors named")
    if x.n_regions >= 2:
        breadth += cfg.multi_region
        notes.append(f"{x.n_regions} countries/regions named")
    if breadth:
        factors.append(ImpactFactor(name="Breadth", points=min(cfg.breadth_cap, breadth), detail="; ".join(notes)))

    market_share = x.market_docs / n
    if market_share >= cfg.market_linked_share or x.market_docs >= cfg.market_linked_docs:
        factors.append(ImpactFactor(name="Market linkage", points=cfg.market_linked,
                                    detail=f"{x.market_docs} report{'s' if x.market_docs != 1 else ''} ({market_share:.0%}) discuss markets, prices or listed companies"))
    elif market_share < cfg.market_unlinked_share and x.market_docs < 5:
        factors.append(ImpactFactor(name="Market linkage", points=cfg.market_unlinked,
                                    detail=f"only {x.market_docs} of {n} reports mention markets or listed companies"))

    if x.n_news == 0:
        factors.append(ImpactFactor(name="Credibility", points=cfg.social_only, detail="social media only - no news outlet has reported it"))
    elif x.n_stories == 1:
        factors.append(ImpactFactor(name="Credibility", points=cfg.single_source,
                                    detail=f"a single report{' (syndicated by ' + str(x.n_publishers) + ' outlets)' if x.n_publishers > 1 else ''}, not independently corroborated"))
    elif x.n_stories == 2:
        factors.append(ImpactFactor(name="Credibility", points=cfg.single_source / 2, detail="only two independent reports so far"))
    if x.type_confidence < cfg.uncertain_type_level:
        factors.append(ImpactFactor(name="Classification", points=cfg.uncertain_type, detail=f"event type uncertain ({x.type_confidence:.0%})"))

    total = sum(f.points for f in factors)
    return max(MIN_IMPACT, min(MAX_IMPACT, total)), [f.model_copy(update={"points": round(f.points, 2)}) for f in factors]
