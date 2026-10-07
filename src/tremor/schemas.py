"""Data contracts.

``RawDocument`` is what every source produces; ``DocSignal``, ``EventSignal`` and
``EntitySignal`` are what the engine publishes. The last two are the machine-readable
"risk signals" downstream applications consume, each carrying the three fields the brief
asks for: a sentiment score, an event classification and an impact score.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "1.0"


def utc(dt: datetime) -> datetime:
    """Normalise to timezone-aware UTC (naive datetimes are assumed to already be UTC)."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


class SourceKind(str, Enum):
    NEWS = "news"
    SOCIAL = "social"


class RawDocument(BaseModel):
    """One headline or post, exactly as a source delivered it."""

    doc_id: str
    source: str  # connector name, e.g. "gdelt", "rss", "stocktwits"
    kind: SourceKind
    published_at: datetime
    text: str
    url: str | None = None
    publisher: str | None = None  # news domain or social handle
    finance_feed: bool = False  # the source itself guarantees a market context
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("published_at")
    @classmethod
    def _to_utc(cls, value: datetime) -> datetime:
        return utc(value)

    @staticmethod
    def make_id(source: str, key: str) -> str:
        return hashlib.sha1(f"{source}|{key}".encode()).hexdigest()[:16]


class DocSignal(BaseModel):
    """Per-document analysis: the audit trail behind every aggregate number."""

    doc_id: str
    published_at: datetime
    source: str
    kind: SourceKind
    publisher: str | None = None
    text: str
    url: str | None = None
    entities: list[str] = Field(default_factory=list)
    sentiment_score: float  # -1 .. 1, document level
    entity_sentiment: dict[str, float] = Field(default_factory=dict)  # -1 .. 1 per mentioned entity
    price_moves: dict[str, int] = Field(default_factory=dict)  # price entity -> +1 / -1: which way the text says it moves
    sentiment_confidence: float
    event_type: str
    event_confidence: float
    relevance: float  # probability the text is financially material
    novelty: float  # 1 = new story, 0 = repeat of a known one
    impact_score: float  # 1 .. 10, of the event this document belongs to
    story_id: str | None = None  # tight cluster of documents saying the same thing
    event_id: str | None = None  # looser cluster of related stories


class ImpactFactor(BaseModel):
    """One line of the impact scorecard: what was observed and how many points it added."""

    name: str
    points: float
    detail: str


class Evidence(BaseModel):
    doc_id: str
    published_at: datetime
    source: str
    kind: SourceKind
    publisher: str | None = None
    text: str
    url: str | None = None
    sentiment_score: float


class StoryBrief(BaseModel):
    """One sub-story of an event: a headline and how widely it was reported."""

    story_id: str
    headline: str
    first_seen: datetime
    n_docs: int
    n_publishers: int
    sentiment_score: float


class EventSignal(BaseModel):
    """Risk signal for an *event*: related stories of one type, rolled up and scored together."""

    schema_version: str = SCHEMA_VERSION
    scope: str = "event"
    event_id: str
    first_seen: datetime
    last_updated: datetime
    headline: str
    sentiment_score: float  # -1 .. 1
    event_type: str
    event_type_label: str
    event_type_probs: dict[str, float] = Field(default_factory=dict)
    impact_score: float  # 1 .. 10
    impact_factors: list[ImpactFactor] = Field(default_factory=list)
    market_wide: bool = False
    entities: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    n_docs: int = 1
    n_stories: int = 1
    n_publishers: int = 1
    n_news: int = 0
    n_social: int = 0
    reports_last_hour: int = 0
    entity_sentiment: dict[str, float] = Field(default_factory=dict)  # how the reports talk about each named entity
    entity_mentions: dict[str, int] = Field(default_factory=dict)
    # Direction, not tone: how many reports say each named price (oil, gas, gold, the stock market) is rising
    # or falling - "oil soars on war fears" is negative in tone but says oil is going up.
    price_moves: dict[str, dict[str, int]] = Field(default_factory=dict)  # e.g. {"OIL": {"up": 31, "down": 4}}
    stories: list[StoryBrief] = Field(default_factory=list)  # most widely reported sub-stories
    evidence: list[Evidence] = Field(default_factory=list)  # individual documents (first and latest)


class EntitySignal(BaseModel):
    """Risk signal for a *company or macro actor*: its filtered sentiment state right now."""

    schema_version: str = SCHEMA_VERSION
    scope: str = "entity"
    entity_id: str
    name: str
    entity_type: str
    sector: str | None = None
    as_of: datetime
    sentiment_score: float  # -1 .. 1, decayed and shrunk towards neutral when evidence is thin
    sentiment_uncertainty: float  # posterior standard deviation; wide = little or stale evidence
    news_sentiment: float | None = None
    social_sentiment: float | None = None
    divergence: float | None = None  # social minus news: a hype / early-warning flag
    evidence_weight: float = 0.0  # effective number of recent, credible documents
    event_type: str | None = None  # dominant active event touching this entity
    impact_score: float = 1.0  # highest impact among those active events
    top_event_id: str | None = None
