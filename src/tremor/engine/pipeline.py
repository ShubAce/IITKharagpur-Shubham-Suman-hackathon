"""The AI/NLP Risk Engine: raw documents in, structured risk signals out.

    RawDocument --> de-duplicate --> link entities --> encode once --> sentiment + event heads
                --> cluster into stories, roll stories up into events (novelty, corroboration)
                --> score impact --> update entity sentiment state
                --> DocSignal / EventSignal / EntitySignal

The engine is synchronous and deterministic: the same documents in the same order always give
the same signals, and "now" is the timestamp of the latest document rather than the wall clock.
That is what lets a historical crisis be replayed through exactly the code that runs live.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from tremor.config import Settings, Taxonomy, Universe
from tremor.engine.clustering import Cluster, EventClusterer, StoryClusterer
from tremor.engine.state import SentimentState
from tremor.nlp.cues import CueHits, CueMatcher
from tremor.nlp.entities import EntityLinker, has_finance_cue
from tremor.nlp.impact import ImpactInputs, score_impact
from tremor.nlp.models import TextModel
from tremor.nlp.preprocess import clean_text, fingerprint, tidy_headline
from tremor.schemas import DocSignal, EntitySignal, EventSignal, RawDocument, SourceKind, StoryBrief

CUE_PRIOR_BOOST = 0.5  # each rule cue multiplies the model's probability for that type by (1 + boost)
MIN_NOVELTY_WEIGHT = 0.3  # a repeat of a known story still counts a little towards sentiment
MAX_TARGETED_COMPANIES = 5  # beyond this a text is a ticker list, not a statement about each company
MARKET_ENTITY_TYPES = frozenset({"company", "commodity", "index_benchmark"})


@dataclass
class BatchResult:
    """Everything one ``process`` call produced or changed."""

    docs: list[DocSignal] = field(default_factory=list)
    events: list[EventSignal] = field(default_factory=list)
    entities: list[EntitySignal] = field(default_factory=list)
    duplicates: int = 0
    noise: int = 0


class RiskEngine:
    def __init__(self, settings: Settings, universe: Universe, taxonomy: Taxonomy, model: TextModel):
        self.settings, self.universe, self.taxonomy, self.model = settings, universe, taxonomy, model
        self._entities = universe.by_id()
        self._type_ids = taxonomy.ids
        self._other = self._type_ids.index("OTHER")
        self._geo, self._commentary = self._type_ids.index("GEOPOLITICAL"), self._type_ids.index("MARKET_COMMENTARY")
        # Naming a country or the military alliance keeps "war" literal ("a silent war" between retailers is not).
        self._geo_anchors = frozenset(e.id for e in universe.entities if e.type == "country") | {"NATO"}
        self._linker = EntityLinker(universe)
        self._cues = CueMatcher(taxonomy)
        self._company_ids = frozenset(e.id for e in universe.entities if e.type == "company")
        self._price_ids = frozenset(e.id for e in universe.entities if e.type in ("commodity", "index_benchmark"))
        self._stories: StoryClusterer | None = None  # both created on the first batch (need the embedding size)
        self._events: EventClusterer | None = None
        self._state = SentimentState(settings.state)
        self._credibility = {SourceKind.NEWS: settings.state.news_credibility, SourceKind.SOCIAL: settings.state.social_credibility}
        self._seen: OrderedDict[str, str | None] = OrderedDict()  # text fingerprint -> story id (None = judged noise)
        self._entity_events: dict[str, set[str]] = {}
        self.now: datetime | None = None
        self.stats = {"documents": 0, "duplicates": 0, "noise": 0, "scored": 0}

    # ------------------------------------------------------------------ public API
    def process(self, docs: list[RawDocument]) -> BatchResult:
        result = BatchResult()
        fresh: list[tuple[RawDocument, str, str]] = []
        touched: dict[str, Cluster] = {}  # events changed by this batch
        touched_entities: set[str] = set()

        for doc in sorted(docs, key=lambda d: d.published_at):
            self.stats["documents"] += 1
            self.now = doc.published_at if self.now is None else max(self.now, doc.published_at)
            text = clean_text(doc.text)
            if len(text) < 8:
                continue
            fp = fingerprint(text)
            if fp in self._seen:  # exact copy of something already analysed: corroboration, not news
                result.duplicates += 1
                story = self._stories.get(self._seen[fp]) if self._stories else None
                event = self._events.get(story.parent_id) if story is not None else None
                if event is not None:
                    story.add_duplicate(doc)
                    event.add_duplicate(doc)
                    touched[event.cluster_id] = event
                continue
            self._remember(fp, None)
            fresh.append((doc, text, fp))

        if fresh:
            out = self.model.predict([text for _, text, _ in fresh])
            if self._stories is None:
                dim, n_types = out.embeddings.shape[1], len(self._type_ids)
                market_wide = frozenset(i for i, t in enumerate(self._type_ids) if self.taxonomy.event_types[t].market_wide)
                self._stories = StoryClusterer(self.settings.clustering, dim, n_types, self._company_ids)
                self._events = EventClusterer(self.settings.clustering, dim, n_types, self._company_ids, market_wide)
            scores = out.sentiment_score
            linked = [self._link(doc, text) for doc, text, _ in fresh]
            targeted = self._target_scores([text for _, text, _ in fresh], linked)
            for i, (doc, text, fp) in enumerate(fresh):
                signal, story, event = self._analyse(doc, text, out.embeddings[i], out.sentiment[i], float(scores[i]),
                                                     out.event[i], linked[i][0], targeted.get(i, {}))
                result.docs.append(signal)
                if event is None:
                    result.noise += 1
                    continue
                self._remember(fp, story.cluster_id)
                touched[event.cluster_id] = event
                touched_entities.update(signal.entities)

        self.stats["duplicates"] += result.duplicates
        self.stats["noise"] += result.noise
        self.stats["scored"] += len(result.docs) - result.noise

        for event in touched.values():
            signal = self._event_signal(event)
            result.events.append(signal)
            touched_entities.update(signal.entities)
        by_event = {e.event_id: e for e in result.events}
        for sig in result.docs:  # report each document with its event's impact as of this batch
            if sig.event_id in by_event:
                sig.impact_score = by_event[sig.event_id].impact_score
        result.entities = [self.entity_signal(e) for e in sorted(touched_entities)]
        return result

    def entity_signal(self, entity_id: str, now: datetime | None = None) -> EntitySignal:
        """Current risk signal for one entity (also valid for entities with no evidence yet)."""
        now = now or self.now or datetime.now(timezone.utc)
        ent = self._entities[entity_id]
        snap = self._state.snapshot(entity_id, now)
        top: Cluster | None = None
        if self._events is not None:
            horizon = now - timedelta(hours=self.settings.clustering.window_hours)
            live = set()
            for event_id in sorted(self._entity_events.get(entity_id, ())):  # sorted: ties resolve the same every run
                event = self._events.get(event_id)
                if event is not None and event.last_seen >= horizon:
                    live.add(event_id)
                    if top is None or event.impact > top.impact:
                        top = event
            self._entity_events[entity_id] = live
        return EntitySignal(
            entity_id=entity_id, name=ent.name, entity_type=ent.type, sector=ent.sector, as_of=now,
            sentiment_score=round(snap.sentiment, 4), sentiment_uncertainty=round(snap.uncertainty, 4),
            news_sentiment=None if snap.news_sentiment is None else round(snap.news_sentiment, 4),
            social_sentiment=None if snap.social_sentiment is None else round(snap.social_sentiment, 4),
            divergence=None if snap.divergence is None else round(snap.divergence, 4),
            evidence_weight=round(snap.evidence_weight, 3),
            event_type=self._type_ids[int(top.type_probs.argmax())] if top else None,
            impact_score=round(top.impact, 2) if top else 1.0,
            top_event_id=top.cluster_id if top else None,
        )

    def analyse_text(self, text: str, kind: SourceKind = SourceKind.NEWS) -> dict:
        """Stateless one-off analysis of arbitrary text (the dashboard playground). Changes nothing."""
        text = clean_text(text)
        out = self.model.predict([text])
        doc = RawDocument(doc_id="adhoc", source="playground", kind=kind, published_at=datetime.now(timezone.utc), text=text,
                          finance_feed=kind is SourceKind.SOCIAL)
        entity_ids, surfaces = self._link(doc, text)
        hits = self._match_cues(text, entity_ids)
        probs = self._fuse_cues(out.event[0], hits)
        top = int(probs.argmax())
        entity_scores = self._target_scores([text], [(entity_ids, surfaces)]).get(0, {})
        sectors = {self._entities[e].sector for e in entity_ids if self._entities[e].sector}
        regions = {e for e in entity_ids if self._entities[e].type == "country"}
        impact, factors = score_impact(
            ImpactInputs(event_type=self._type_ids[top], type_confidence=float(probs[top]), sentiment=float(out.sentiment_score[0]),
                         n_scored=1, n_stories=1, n_publishers=1, n_news=int(kind is SourceKind.NEWS),
                         n_social=int(kind is SourceKind.SOCIAL), stories_last_hour=1, extreme_docs=int(bool(hits.severity.get("extreme"))),
                         high_docs=int(bool(hits.severity.get("high"))), scale_docs=int(bool(hits.severity.get("scale"))),
                         n_sectors=len(sectors), n_regions=len(regions),
                         market_docs=int(has_finance_cue(text) or any(self._entities[e].type in MARKET_ENTITY_TYPES for e in entity_ids))),
            self.taxonomy, self.settings.impact.scorecard)
        return {
            "text": text,
            "model": self.model.name,
            "sentiment_score": round(float(out.sentiment_score[0]), 4),
            "sentiment_probs": {k: round(float(v), 4) for k, v in zip(("negative", "neutral", "positive"), out.sentiment[0])},
            "event_type": self._type_ids[top],
            "event_type_label": self.taxonomy.event_types[self._type_ids[top]].label,
            "event_type_probs": {t: round(float(p), 4) for t, p in sorted(zip(self._type_ids, probs), key=lambda kv: -kv[1])[:5]},
            "impact_score": round(impact, 2),
            "impact_factors": [f.model_dump() for f in factors],
            "entities": [{"id": e, "name": self._entities[e].name, "type": self._entities[e].type,
                          "sentiment": round(entity_scores.get(e, float(out.sentiment_score[0])), 4)} for e in entity_ids],
            "entity_level_sentiment": bool(entity_scores),
            "rule_cues": hits.by_type,
            "note": "Scored as a single uncorroborated report: impact rises as independent sources report the same event.",
        }

    # ------------------------------------------------------------------ internals
    def _remember(self, fp: str, story_id: str | None) -> None:
        self._seen[fp] = story_id
        self._seen.move_to_end(fp)
        while len(self._seen) > self.settings.engine.dedupe_cache_size:
            self._seen.popitem(last=False)

    def _fuse_cues(self, probs: np.ndarray, hits: CueHits) -> np.ndarray:
        """Rules as a prior: nudge the model towards event types whose cue patterns fire.

        The encoder learned "war" as a strong geopolitical word, so it also reads "a price war" or "God of
        War" that way. When "war" appears only as an idiom and nothing else in the text is geopolitical,
        the probability it gave to Geopolitical moves to Market commentary (competition, not conflict).
        """
        if not hits.by_type and not hits.figurative_war:
            return probs
        fused = probs.astype(np.float64).copy()
        if hits.figurative_war:
            fused[self._commentary] += fused[self._geo]
            fused[self._geo] = 0.0
        for type_id, n in hits.by_type.items():
            fused[self._type_ids.index(type_id)] *= 1.0 + CUE_PRIOR_BOOST * n
        return fused / fused.sum()

    def _match_cues(self, text: str, entity_ids: list[str]) -> CueHits:
        return self._cues.match(text, geo_anchor=not self._geo_anchors.isdisjoint(entity_ids))

    def _link(self, doc: RawDocument, text: str) -> tuple[list[str], dict[str, str]]:
        """Entity ids in order of appearance, and the surface form each was first mentioned by."""
        surfaces: dict[str, str] = {}
        for mention in self._linker.link(text, finance_context=doc.finance_feed):
            surfaces.setdefault(mention.entity_id, mention.surface.lstrip("$"))
        symbol = doc.meta.get("symbol")
        if symbol in self._entities and symbol not in surfaces:
            # Ticker-specific feeds tell us the subject even when the text never names it.
            surfaces[symbol] = self._entities[symbol].name
        return list(surfaces), surfaces

    def _target_scores(self, texts: list[str], linked: list[tuple[list[str], dict[str, str]]]) -> dict[int, dict[str, float]]:
        """Entity-level sentiment for texts naming several companies ("Apple gains as Intel stumbles").

        One extra model pass per (company, text) pair, and only where it can matter: with a single
        company the document score already is the company's score.
        """
        if not self.model.supports_targets:
            return {}
        pairs, owners = [], []
        for i, (entity_ids, surfaces) in enumerate(linked):
            companies = [e for e in entity_ids if e in self._company_ids]
            targets = companies if 2 <= len(companies) <= MAX_TARGETED_COMPANIES else []
            # Prices (oil, gold, the stock market) always get their own score: "stocks dive, oil surges"
            # is negative as a sentence but says oil is going *up*.
            targets += [e for e in entity_ids if e in self._price_ids]
            for entity_id in targets:
                pairs.append((surfaces[entity_id], texts[i]))
                owners.append((i, entity_id))
        scores = self.model.target_sentiment(pairs)
        out: dict[int, dict[str, float]] = {}
        for (i, entity_id), score in zip(owners, scores):
            out.setdefault(i, {})[entity_id] = float(score)
        return out

    def _analyse(self, doc: RawDocument, text: str, embedding: np.ndarray, sent_probs: np.ndarray, sentiment: float,
                 event_probs: np.ndarray, entity_ids: list[str], targeted: dict[str, float],
                 ) -> tuple[DocSignal, Cluster | None, Cluster | None]:
        cues = self._match_cues(text, entity_ids)
        probs = self._fuse_cues(event_probs, cues)
        top = int(probs.argmax())
        relevance = float(1.0 - probs[self._other])

        signal = DocSignal(
            doc_id=doc.doc_id, published_at=doc.published_at, source=doc.source, kind=doc.kind, publisher=doc.publisher,
            text=text, url=doc.url, entities=entity_ids, sentiment_score=round(sentiment, 4),
            sentiment_confidence=round(float(sent_probs.max()), 4), event_type=self._type_ids[top],
            event_confidence=round(float(probs[top]), 4), relevance=round(relevance, 4), novelty=1.0, impact_score=1.0,
        )
        if relevance < self.settings.engine.relevance_threshold:
            return signal, None, None  # judged noise: kept for the audit trail, but moves nothing

        cfg = self.settings.clustering
        story, similarity, is_new = self._stories.assign(doc, embedding, entity_ids)
        if is_new:
            event = self._events.assign(doc, embedding, top, entity_ids)
            story.parent_id = event.cluster_id
            event.add_story(story.cluster_id, doc.published_at)
        else:
            event = self._events.get(story.parent_id)
            if event is None:  # the parent expired while the story lived on: start a fresh event
                event = self._events.assign(doc, embedding, top, entity_ids)
                story.parent_id = event.cluster_id
                event.add_story(story.cluster_id, doc.published_at)
        novelty = 1.0 if is_new else float(np.clip(1.0 - similarity, 0.0, 1.0))
        # Aggregation weight: credible, relevant and *opinionated* reports count most. A wall of neutral
        # wire copy ("Biden announces sanctions") should not wash out the market-moving reports.
        conviction = 1.0 - float(sent_probs[1])
        weight = self._credibility[doc.kind] * relevance * (0.1 + 0.9 * conviction)
        severity = cues.severity
        event_similarity = float(event.centroid @ embedding) if event.n_scored else 1.0
        market_linked = doc.finance_feed or has_finance_cue(text) or any(
            self._entities[e].type in MARKET_ENTITY_TYPES for e in entity_ids)
        story.add(doc, text, embedding, probs, sentiment, weight, severity, entity_ids, 1.0 if is_new else similarity,
                  cfg.max_evidence, market_linked)
        event.add(doc, text, embedding, probs, sentiment, weight, severity, entity_ids, event_similarity, cfg.max_evidence,
                  market_linked)
        for entity_id in entity_ids:
            event.add_entity_sentiment(entity_id, targeted.get(entity_id, sentiment), weight)
        self._stories.refresh(story)
        self._events.refresh(event)

        # Entity sentiment: new information moves the state more than a rehash of a known story,
        # and a post that tags a long list of tickers says little about any one of them.
        state_weight = relevance * (MIN_NOVELTY_WEIGHT + (1.0 - MIN_NOVELTY_WEIGHT) * novelty)
        n_companies = sum(1 for e in entity_ids if e in self._company_ids)
        if n_companies > MAX_TARGETED_COMPANIES:
            state_weight *= MAX_TARGETED_COMPANIES / n_companies
        for entity_id in entity_ids:
            entity_score = targeted.get(entity_id, sentiment)
            self._state.update(entity_id, doc.kind, entity_score, state_weight, doc.published_at)
            self._entity_events.setdefault(entity_id, set()).add(event.cluster_id)
            signal.entity_sentiment[entity_id] = round(entity_score, 4)
        signal.novelty = round(novelty, 4)
        signal.story_id, signal.event_id = story.cluster_id, event.cluster_id
        return signal, story, event

    def _event_signal(self, event: Cluster) -> EventSignal:
        probs = event.type_probs
        top = int(probs.argmax())
        type_id = self._type_ids[top]
        entity_ids = [e for e, _ in event.entity_counts.most_common(12)]  # most mentioned first
        priced = [e for e in self._price_ids if e in event.entity_counts and e not in entity_ids]
        sectors = list(dict.fromkeys(self._entities[e].sector for e in entity_ids if self._entities[e].sector))
        regions = [e for e in entity_ids if self._entities[e].type == "country"]
        impact, factors = score_impact(
            ImpactInputs(event_type=type_id, type_confidence=float(probs[top]), sentiment=event.sentiment,
                         n_scored=event.n_scored, n_stories=len(event.children), n_publishers=len(event.publishers),
                         n_news=event.n_news, n_social=event.n_social, stories_last_hour=event.stories_last_hour,
                         extreme_docs=event.extreme_docs, high_docs=event.high_docs, scale_docs=event.scale_docs,
                         n_sectors=len(sectors), n_regions=len(regions), market_docs=event.market_docs),
            self.taxonomy, self.settings.impact.scorecard)
        event.impact = impact
        event.peak_impact = max(event.peak_impact, impact)
        spec = self.taxonomy.event_types[type_id]
        stories = [s for s in (self._stories.get(sid) for sid in event.children) if s is not None]
        # Headline: the most widely reported story, favouring ones told in market terms and severe language
        # (the reports a risk manager would read first).
        def rank(s):
            return -len(s.publishers) * (1 + (s.market_docs > 0) + (s.extreme_docs + s.high_docs > 0))
        stories.sort(key=lambda s: (rank(s), -s.n_docs, s.first_seen, s.cluster_id))  # id: reproducible ties
        # A long-running event is described by its latest development: prefer widely reported stories
        # that first appeared in the last twelve hours over the day before's most-shared headline.
        recent = [s for s in stories if s.first_seen >= event.last_seen - timedelta(hours=12) and len(s.publishers) >= 3]
        headline = (recent or stories)[0].headline if stories else event.headline
        return EventSignal(
            event_id=event.cluster_id, first_seen=event.first_seen, last_updated=event.last_seen, headline=tidy_headline(headline),
            sentiment_score=round(event.sentiment, 4), event_type=type_id, event_type_label=spec.label,
            event_type_probs={t: round(float(p), 4) for t, p in sorted(zip(self._type_ids, probs), key=lambda kv: -kv[1])[:4]},
            impact_score=round(impact, 2), impact_factors=factors, market_wide=spec.market_wide, entities=entity_ids,
            sectors=sectors, regions=regions, n_docs=event.n_docs, n_stories=len(event.children),
            entity_sentiment={e: round(event.entity_sentiment(e), 3) for e in entity_ids + priced},
            entity_mentions={e: int(event.entity_counts[e]) for e in entity_ids + priced},
            n_publishers=len(event.publishers), n_news=event.n_news, n_social=event.n_social,
            reports_last_hour=event.reports_last_hour,
            stories=[StoryBrief(story_id=s.cluster_id, headline=tidy_headline(s.headline), first_seen=s.first_seen, n_docs=s.n_docs,
                                n_publishers=len(s.publishers), sentiment_score=round(s.sentiment, 3))
                     for s in stories[: self.settings.clustering.max_stories]],
            evidence=[ev.model_copy(update={"text": tidy_headline(ev.text)}) for ev in event.evidence],
        )
