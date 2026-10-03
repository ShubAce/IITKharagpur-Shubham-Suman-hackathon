"""Online two-level clustering: documents -> stories -> events.

* A **story** is a tight group of documents saying the same thing (one wire report picked up
  by fifty outlets). ``1 - similarity`` to the closest existing story is a document's *novelty*.
* An **event** is a looser group of related stories of the same event type ("Russia invades
  Ukraine" spans stories about the declaration, the sanctions, the border crossings ...).

The event is the unit downstream applications act on: it is what gets an impact score and what
can trigger a stress test. Rolling stories up fixes two failure modes of scoring articles one by
one - double counting, and a single big event firing dozens of separate alerts - and turns
repetition into evidence: the number of independent publishers becomes corroboration.

Both levels are single-pass and incremental, as streaming requires.
"""

from __future__ import annotations

import hashlib
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from tremor.config import ClusteringSettings
from tremor.schemas import Evidence, RawDocument, SourceKind


@dataclass
class Cluster:
    """Running statistics of a story or an event (both levels track the same things)."""

    cluster_id: str
    first_seen: datetime
    last_seen: datetime
    vec_sum: np.ndarray
    n_types: int
    parent_id: str | None = None  # a story's event
    n_docs: int = 0  # every report, exact duplicates included
    n_scored: int = 0  # reports that were analysed by the model
    n_news: int = 0
    n_social: int = 0
    publishers: set[str] = field(default_factory=set)
    entity_counts: Counter = field(default_factory=Counter)
    type_weight: np.ndarray | None = None  # credibility-weighted sum of event-type probabilities
    sentiment_weight: float = 0.0
    sentiment_sum: float = 0.0
    extreme_docs: int = 0
    high_docs: int = 0
    scale_docs: int = 0
    market_docs: int = 0  # documents that discuss the story in market terms
    report_times: deque = field(default_factory=deque)  # trailing hour, for velocity
    headline: str = ""
    _headline_rank: tuple = (-1, -1.0)
    evidence: list[Evidence] = field(default_factory=list)
    children: set[str] = field(default_factory=set)  # an event's stories
    entity_sent: dict = field(default_factory=dict)  # entity -> [weighted sentiment sum, weight]
    story_times: deque = field(default_factory=deque)  # first-seen time of each story, trailing hour
    impact: float = 1.0
    peak_impact: float = 1.0

    def __post_init__(self) -> None:
        if self.type_weight is None:
            self.type_weight = np.zeros(self.n_types, dtype=np.float64)

    # ------------------------------------------------------------------ updates
    def _touch(self, doc: RawDocument) -> None:
        self.n_docs += 1
        self.last_seen = max(self.last_seen, doc.published_at)
        self.publishers.add(doc.publisher or doc.source)
        if doc.kind is SourceKind.NEWS:
            self.n_news += 1
        else:
            self.n_social += 1
        self.report_times.append(doc.published_at)
        horizon = self.last_seen - timedelta(hours=1)
        while self.report_times and self.report_times[0] < horizon:
            self.report_times.popleft()

    def add_duplicate(self, doc: RawDocument) -> None:
        """An exact copy of a headline already counted: corroboration only."""
        self._touch(doc)

    def add(self, doc: RawDocument, text: str, embedding: np.ndarray, type_probs: np.ndarray, sentiment: float,
            weight: float, severity: dict[str, int], entity_ids: list[str], similarity: float, max_evidence: int,
            market_linked: bool = False) -> None:
        self._touch(doc)
        self.n_scored += 1
        self.market_docs += market_linked
        self.vec_sum += embedding
        self.type_weight += weight * type_probs
        self.sentiment_weight += weight
        self.sentiment_sum += weight * sentiment
        self.entity_counts.update(entity_ids)
        self.extreme_docs += bool(severity.get("extreme"))
        self.high_docs += bool(severity.get("high"))
        self.scale_docs += bool(severity.get("scale"))

        # Representative headline: a news headline beats a social post; then the one closest to the centroid.
        rank = (1 if doc.kind is SourceKind.NEWS else 0, similarity)
        if rank > self._headline_rank or not self.headline:
            self.headline, self._headline_rank = text, rank

        item = Evidence(doc_id=doc.doc_id, published_at=doc.published_at, source=doc.source, kind=doc.kind,
                        publisher=doc.publisher, text=text, url=doc.url, sentiment_score=round(float(sentiment), 3))
        if len(self.evidence) < max_evidence:
            self.evidence.append(item)
        else:  # keep how it began (first third) and the latest reports (the rest)
            keep_head = max_evidence // 3
            self.evidence = self.evidence[:keep_head] + self.evidence[keep_head + 1:] + [item]

    # ------------------------------------------------------------------ views
    @property
    def centroid(self) -> np.ndarray:
        return self.vec_sum / max(np.linalg.norm(self.vec_sum), 1e-12)

    @property
    def type_probs(self) -> np.ndarray:
        total = self.type_weight.sum()
        return self.type_weight / total if total > 0 else np.full(self.n_types, 1.0 / self.n_types)

    @property
    def sentiment(self) -> float:
        return self.sentiment_sum / self.sentiment_weight if self.sentiment_weight > 0 else 0.0

    @property
    def reports_last_hour(self) -> int:
        return len(self.report_times)

    def add_entity_sentiment(self, entity_id: str, score: float, weight: float) -> None:
        acc = self.entity_sent.setdefault(entity_id, [0.0, 0.0])
        acc[0] += weight * score
        acc[1] += weight

    def entity_sentiment(self, entity_id: str) -> float:
        acc = self.entity_sent.get(entity_id)
        return acc[0] / acc[1] if acc and acc[1] > 0 else 0.0

    def add_story(self, story_id: str, at: datetime) -> None:
        """Register a new independent story under this event (velocity counts these, not copies)."""
        self.children.add(story_id)
        self.story_times.append(at)
        horizon = max(self.last_seen, at) - timedelta(hours=1)
        while self.story_times and self.story_times[0] < horizon:
            self.story_times.popleft()

    @property
    def stories_last_hour(self) -> int:
        horizon = self.last_seen - timedelta(hours=1)
        return sum(1 for t in self.story_times if t >= horizon)


class _CentroidIndex:
    """Active clusters plus a matrix of their unit centroids for fast nearest-neighbour lookup."""

    def __init__(self, dim: int, window_hours: float):
        self.clusters: dict[str, Cluster] = {}
        self.rows: list[str] = []
        self.row_of: dict[str, int] = {}
        self.matrix = np.zeros((256, dim), dtype=np.float32)
        self._window = timedelta(hours=window_hours)
        self._since_expiry = 0

    def similarities(self, embedding: np.ndarray) -> np.ndarray:
        return self.matrix[: len(self.rows)] @ embedding

    def insert(self, cluster: Cluster, embedding: np.ndarray) -> None:
        if len(self.rows) == len(self.matrix):
            self.matrix = np.vstack([self.matrix, np.zeros_like(self.matrix)])
        self.matrix[len(self.rows)] = embedding
        self.row_of[cluster.cluster_id] = len(self.rows)
        self.rows.append(cluster.cluster_id)
        self.clusters[cluster.cluster_id] = cluster

    def refresh(self, cluster: Cluster) -> None:
        row = self.row_of.get(cluster.cluster_id)
        if row is not None:
            self.matrix[row] = cluster.centroid

    def maybe_expire(self, now: datetime, every: int = 200) -> list[int] | None:
        """Every ``every`` calls, drop clusters with no report inside the window.

        Returns the surviving row numbers (old numbering) when anything was dropped, else ``None``.
        """
        self._since_expiry += 1
        if self._since_expiry < every:
            return None
        self._since_expiry = 0
        horizon = now - self._window
        keep = [i for i, cid in enumerate(self.rows) if self.clusters[cid].last_seen >= horizon]
        if len(keep) == len(self.rows):
            return None
        kept_ids = {self.rows[i] for i in keep}
        self.clusters = {cid: c for cid, c in self.clusters.items() if cid in kept_ids}
        self.matrix[: len(keep)] = self.matrix[keep]
        self.rows = [self.rows[i] for i in keep]
        self.row_of = {cid: row for row, cid in enumerate(self.rows)}
        return keep


def _new_id(prefix: str, seed: str) -> str:
    return prefix + hashlib.sha1(seed.encode()).hexdigest()[:10]


class StoryClusterer:
    """Level 1: tight clusters of documents that report the same thing."""

    def __init__(self, settings: ClusteringSettings, dim: int, n_types: int, company_ids: frozenset[str]):
        self._cfg = settings
        self._n_types = n_types
        self._company_ids = company_ids
        self._index = _CentroidIndex(dim, settings.window_hours)
        self._companies: list[frozenset[str]] = []

    def get(self, story_id: str | None) -> Cluster | None:
        return self._index.clusters.get(story_id) if story_id else None

    def assign(self, doc: RawDocument, embedding: np.ndarray, entity_ids: list[str]) -> tuple[Cluster, float, bool]:
        """Return ``(story, similarity_to_it_before_joining, is_new)``. Does not add the document."""
        keep = self._index.maybe_expire(doc.published_at)
        if keep is not None:
            self._companies = [self._companies[i] for i in keep]
        companies = self._company_ids.intersection(entity_ids)
        best_row, best_sim = -1, -1.0
        n = len(self._index.rows)
        if n:
            sims = self._index.similarities(embedding)
            # Only the few closest candidates can win, so the entity check stays cheap.
            for row in np.argpartition(-sims, min(5, n - 1))[: min(5, n)]:
                sim = float(sims[row])
                other = self._companies[row]
                if companies and other and companies.isdisjoint(other):
                    sim -= self._cfg.disjoint_entity_penalty  # e.g. two different firms' earnings beats
                if sim > best_sim:
                    best_row, best_sim = int(row), sim
        if best_row >= 0 and best_sim >= self._cfg.similarity_threshold:
            return self._index.clusters[self._index.rows[best_row]], best_sim, False
        story = Cluster(cluster_id=_new_id("sty_", doc.doc_id), first_seen=doc.published_at, last_seen=doc.published_at,
                        vec_sum=np.zeros(embedding.shape, dtype=np.float64), n_types=self._n_types)
        self._index.insert(story, embedding)
        self._companies.append(frozenset())
        return story, max(best_sim, 0.0), True

    def refresh(self, story: Cluster) -> None:
        self._index.refresh(story)
        row = self._index.row_of.get(story.cluster_id)
        if row is not None:
            self._companies[row] = self._company_ids.intersection(story.entity_counts)


class EventClusterer:
    """Level 2: related stories of one event type, rolled up into an event.

    A new story joins the most similar active event of the same type when the similarity clears
    ``event_threshold``; sharing a named entity (a country, a company) lowers that bar, because
    two geopolitical stories that both name Russia are far more likely to be one situation.
    Company-specific event types additionally require a shared company, so Apple's earnings and
    Microsoft's earnings never merge.
    """

    def __init__(self, settings: ClusteringSettings, dim: int, n_types: int, company_ids: frozenset[str],
                 market_wide_types: frozenset[int]):
        self._cfg = settings
        self._n_types = n_types
        self._company_ids = company_ids
        self._market_wide = market_wide_types
        self._index = _CentroidIndex(dim, settings.window_hours)
        self._types: list[int] = []

    def get(self, event_id: str | None) -> Cluster | None:
        return self._index.clusters.get(event_id) if event_id else None

    def active(self) -> list[Cluster]:
        return list(self._index.clusters.values())

    def assign(self, doc: RawDocument, embedding: np.ndarray, type_idx: int, entity_ids: list[str]) -> Cluster:
        keep = self._index.maybe_expire(doc.published_at, every=50)
        if keep is not None:
            self._types = [self._types[i] for i in keep]
        anchors = set(entity_ids)
        companies = self._company_ids.intersection(anchors)
        best: Cluster | None = None
        best_margin = 0.0
        if self._index.rows:
            sims = self._index.similarities(embedding)
            # Vectorised pre-filter: same event type and similar enough to clear the lowest possible bar.
            floor = self._cfg.event_threshold - self._cfg.shared_entity_bonus
            candidates = np.nonzero((np.asarray(self._types) == type_idx) & (sims >= floor))[0]
            for row in candidates:
                event = self._index.clusters[self._index.rows[row]]
                shared = anchors.intersection(event.entity_counts)
                if type_idx not in self._market_wide and not companies.intersection(shared):
                    continue  # firm-specific event types never merge across companies
                threshold = self._cfg.event_threshold - (self._cfg.shared_entity_bonus if shared else 0.0)
                margin = float(sims[row]) - threshold
                if margin >= 0 and (best is None or margin > best_margin):
                    best, best_margin = event, margin
        if best is not None:
            return best
        event = Cluster(cluster_id=_new_id("evt_", doc.doc_id), first_seen=doc.published_at, last_seen=doc.published_at,
                        vec_sum=np.zeros(embedding.shape, dtype=np.float64), n_types=self._n_types)
        self._index.insert(event, embedding)
        self._types.append(type_idx)
        return event

    def refresh(self, event: Cluster) -> None:
        self._index.refresh(event)
