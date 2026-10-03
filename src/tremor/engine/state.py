"""Per-entity sentiment state: an exponentially-forgetting Bayesian filter.

Each company's sentiment is treated as a hidden quantity observed through noisy documents.
The filter answers two questions the index rebalancer needs: *what is the sentiment now*, and
*how sure are we*. It has three properties a plain average lacks:

* evidence decays (news slower than social chatter), so the state drifts back to neutral;
* a neutral prior means one stray tweet cannot swing a company to +1;
* it reports an uncertainty, which shrinks only as credible, fresh evidence accumulates.

The update is O(1) per document: two running sums per (entity, source kind).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from tremor.config import StateSettings
from tremor.schemas import SourceKind

OBSERVATION_STD = 0.5  # typical disagreement between one document's score and the true sentiment


@dataclass
class _Accumulator:
    weight: float = 0.0  # decayed sum of weights (effective number of documents)
    weighted_score: float = 0.0  # decayed sum of weight * score
    updated_at: datetime | None = None

    def decayed(self, now: datetime, half_life_hours: float) -> tuple[float, float]:
        if self.updated_at is None:
            return 0.0, 0.0
        hours = max(0.0, (now - self.updated_at).total_seconds() / 3600.0)
        factor = 0.5 ** (hours / half_life_hours)
        return self.weight * factor, self.weighted_score * factor

    def add(self, score: float, weight: float, at: datetime, half_life_hours: float) -> None:
        self.weight, self.weighted_score = self.decayed(at, half_life_hours)
        self.weight += weight
        self.weighted_score += weight * score
        if self.updated_at is None or at > self.updated_at:
            self.updated_at = at


@dataclass(frozen=True)
class StateSnapshot:
    sentiment: float  # posterior mean, -1 .. 1
    uncertainty: float  # posterior standard deviation
    evidence_weight: float  # credibility-weighted effective number of documents
    news_sentiment: float | None
    social_sentiment: float | None

    @property
    def divergence(self) -> float | None:
        """Social minus news sentiment: positive = the crowd is more bullish than the press."""
        if self.news_sentiment is None or self.social_sentiment is None:
            return None
        return self.social_sentiment - self.news_sentiment


@dataclass
class _EntityState:
    by_kind: dict[SourceKind, _Accumulator] = field(default_factory=lambda: {k: _Accumulator() for k in SourceKind})


class SentimentState:
    def __init__(self, settings: StateSettings):
        self._cfg = settings
        self._half_life = {SourceKind.NEWS: settings.news_half_life_hours, SourceKind.SOCIAL: settings.social_half_life_hours}
        self._credibility = {SourceKind.NEWS: settings.news_credibility, SourceKind.SOCIAL: settings.social_credibility}
        self._entities: dict[str, _EntityState] = {}

    def update(self, entity_id: str, kind: SourceKind, score: float, weight: float, at: datetime) -> None:
        """Fold one document's sentiment about ``entity_id`` into its state."""
        if weight <= 0:
            return
        state = self._entities.setdefault(entity_id, _EntityState())
        state.by_kind[kind].add(max(-1.0, min(1.0, score)), weight, at, self._half_life[kind])

    def snapshot(self, entity_id: str, now: datetime) -> StateSnapshot:
        state = self._entities.get(entity_id)
        prior = self._cfg.prior_strength
        if state is None:
            return StateSnapshot(0.0, OBSERVATION_STD / math.sqrt(prior), 0.0, None, None)
        total_w = total_ws = 0.0
        per_kind: dict[SourceKind, float | None] = {}
        for kind, acc in state.by_kind.items():
            w, ws = acc.decayed(now, self._half_life[kind])
            per_kind[kind] = ws / (w + 1.0) if w > 0.05 else None  # light shrinkage for display
            total_w += self._credibility[kind] * w
            total_ws += self._credibility[kind] * ws
        mean = total_ws / (total_w + prior)
        std = OBSERVATION_STD / math.sqrt(total_w + prior)
        return StateSnapshot(mean, std, total_w, per_kind[SourceKind.NEWS], per_kind[SourceKind.SOCIAL])

    def known_entities(self) -> list[str]:
        return list(self._entities)
