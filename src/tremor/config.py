"""Typed configuration, loaded from the YAML files in ``configs/``.

Everything tunable lives in YAML so behaviour can be changed (or audited) without touching code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from tremor.paths import CONFIG_DIR


# --------------------------------------------------------------------------- universe
class EntityDef(BaseModel):
    id: str
    name: str
    type: Literal["company", "institution", "country", "commodity", "index_benchmark"]
    sector: str | None = None
    country: str | None = None
    aliases: list[str] = Field(default_factory=list)
    exact: list[str] = Field(default_factory=list)
    ambiguous: list[str] = Field(default_factory=list)
    cashtags: list[str] = Field(default_factory=list)


class IndexDef(BaseModel):
    name: str
    description: str = ""
    constituents: list[str]


class Universe(BaseModel):
    index: IndexDef
    entities: list[EntityDef]

    @model_validator(mode="after")
    def _check(self) -> Universe:
        ids = [e.id for e in self.entities]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate entity ids in universe.yaml")
        missing = set(self.index.constituents) - set(ids)
        if missing:
            raise ValueError(f"index constituents missing from entity master: {sorted(missing)}")
        return self

    def by_id(self) -> dict[str, EntityDef]:
        return {e.id: e for e in self.entities}


# --------------------------------------------------------------------------- taxonomy
class EventTypeDef(BaseModel):
    label: str
    description: str = ""
    base_severity: float
    half_life_hours: float = 48.0
    market_wide: bool = False
    cues: list[str] = Field(default_factory=list)


class Taxonomy(BaseModel):
    event_types: dict[str, EventTypeDef]
    severity_cues: dict[str, str] = Field(default_factory=dict)

    @property
    def ids(self) -> list[str]:
        return list(self.event_types)


# --------------------------------------------------------------------------- settings
class EncoderSettings(BaseModel):
    repo: str = "sentence-transformers/all-MiniLM-L12-v2"
    onnx_file: str = "onnx/model.onnx"
    pooling: Literal["mean", "cls"] = "mean"
    max_length: int = 96
    batch_size: int = 64
    threads: int | None = None


class ClusteringSettings(BaseModel):
    similarity_threshold: float = 0.62  # documents this similar are the same story
    disjoint_entity_penalty: float = 0.15  # two texts about different companies are rarely one story
    event_threshold: float = 0.42  # a story this similar to an event (of the same type) joins it
    shared_entity_bonus: float = 0.10  # ...and the bar is lower when they name the same entity
    window_hours: float = 48.0
    max_evidence: int = 12
    max_stories: int = 8  # sub-stories listed on an event signal


class StateSettings(BaseModel):
    news_half_life_hours: float = 36.0
    social_half_life_hours: float = 12.0
    prior_strength: float = 2.0  # pseudo-observations of "neutral" that evidence must outweigh
    news_credibility: float = 1.0
    social_credibility: float = 0.35


class ScorecardSettings(BaseModel):
    """Points the impact scorecard adds to (or takes off) an event type's base severity."""

    # intensity: how severe the language is
    extreme_cue: float = 1.0  # "war", "default", "bankruptcy", "collapse" ...
    high_cue: float = 0.5  # "plunge", "record low", "emergency" ...
    scale_cue: float = 0.3  # an amount in billions or trillions
    strong_sentiment: float = 0.3
    strong_sentiment_level: float = 0.6
    intensity_cap: float = 1.5
    # corroboration: independent publishers telling the same story
    corroboration_per_doubling: float = 0.5
    corroboration_cap: float = 2.0
    # velocity: new independent reports in the trailing 60 minutes -> points
    velocity_steps: list[tuple[int, float]] = Field(default_factory=lambda: [(5, 0.25), (15, 0.5), (40, 1.0)])
    # market linkage: is the event being discussed in market terms at all?
    market_linked: float = 0.5
    market_linked_share: float = 0.25  # at least this share of reports mention markets / listed names ...
    market_linked_docs: int = 25  # ... or at least this many reports do (big political stories)
    market_unlinked: float = -1.0
    market_unlinked_share: float = 0.10
    # breadth: how much of the economy the reports actually name
    multi_sector: float = 0.5  # three or more sectors named
    multi_region: float = 0.25  # two or more countries / regions named
    breadth_cap: float = 0.75
    # credibility deductions
    social_only: float = -1.0  # no news outlet has reported it
    single_source: float = -1.0  # one report (however widely syndicated); half of this for two
    uncertain_type: float = -0.5
    uncertain_type_level: float = 0.5


class ImpactSettings(BaseModel):
    trigger_threshold: float = 7.0  # Module B runs a stress test at or above this
    retrigger_delta: float = 1.0  # ...and again if the same event is revised up by this much
    scorecard: ScorecardSettings = Field(default_factory=ScorecardSettings)


class EngineSettings(BaseModel):
    relevance_threshold: float = 0.5
    min_event_confidence: float = 0.35
    dedupe_cache_size: int = 50_000


class ApiSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class Settings(BaseModel):
    encoder: EncoderSettings = Field(default_factory=EncoderSettings)
    clustering: ClusteringSettings = Field(default_factory=ClusteringSettings)
    state: StateSettings = Field(default_factory=StateSettings)
    impact: ImpactSettings = Field(default_factory=ImpactSettings)
    engine: EngineSettings = Field(default_factory=EngineSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def load_universe() -> Universe:
    return Universe.model_validate(_read_yaml(CONFIG_DIR / "universe.yaml"))


@lru_cache(maxsize=1)
def load_taxonomy() -> Taxonomy:
    return Taxonomy.model_validate(_read_yaml(CONFIG_DIR / "taxonomy.yaml"))


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    path = CONFIG_DIR / "settings.yaml"
    return Settings.model_validate(_read_yaml(path) if path.exists() else {})
