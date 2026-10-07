"""Entity linking: map company names, cashtags and macro actors in text to canonical ids.

A dictionary matcher is deliberate here. The tracked universe is small and fixed, so a compiled
alias automaton is exact, auditable and runs in microseconds - and its one weakness, ambiguous
names, is handled explicitly: "Ford", "Amazon" or "Apple" only count when the text also carries a
finance cue. That is what keeps tweets about Christine Blasey Ford out of Ford Motor's signal.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from tremor.config import Universe

_FINANCE_CUE = re.compile(
    r"\$[A-Za-z]{1,6}\b|\b(stocks?|shares?|shareholders?|earnings|revenue|profits?|investors?|analysts?|"
    r"quarter(ly)?|guidance|dividends?|buyback|market cap|valuation|ipo|ceo|cfo|price target|upgrades?|"
    r"downgrades?|sales|deliveries|nasdaq|nyse|wall street|trading|bullish|bearish|merger|acquisition|"
    r"stake|bonds?|debt|credit rating|automaker|carmaker|retailer|tech giant|inc\.?|corp\.?|plc)\b",
    re.IGNORECASE,
)
_CASHTAG = re.compile(r"\$([A-Za-z]{1,6})(?![A-Za-z])")
_BOUNDARY_L, _BOUNDARY_R = r"(?<![A-Za-z0-9])", r"(?![A-Za-z0-9])"


def has_finance_cue(text: str) -> bool:
    """True when the text talks in market terms (prices, shares, earnings, cashtags ...)."""
    return bool(_FINANCE_CUE.search(text))


@dataclass(frozen=True)
class Mention:
    entity_id: str
    surface: str
    start: int
    end: int
    via: str  # "alias" | "exact" | "ambiguous" | "cashtag"


def _compile(surfaces: Iterable[str], flags: int = 0) -> re.Pattern | None:
    # Longest first, so "JPMorgan Chase" wins over "JPMorgan".
    ordered = sorted(set(surfaces), key=len, reverse=True)
    if not ordered:
        return None
    return re.compile(_BOUNDARY_L + "(" + "|".join(re.escape(s) for s in ordered) + ")" + _BOUNDARY_R, flags)


class EntityLinker:
    def __init__(self, universe: Universe):
        self._alias: dict[str, str] = {}  # lower-cased surface -> entity id
        self._exact: dict[str, str] = {}
        self._ambiguous: dict[str, str] = {}
        self._cashtag: dict[str, str] = {}
        for ent in universe.entities:
            for surface in ent.aliases:
                self._alias[surface.lower()] = ent.id
            for surface in ent.exact:
                self._exact[surface] = ent.id
            for surface in ent.ambiguous:
                self._ambiguous[surface] = ent.id
            for tag in ent.cashtags:
                self._cashtag[tag.upper()] = ent.id
        self._alias_re = _compile(self._alias, re.IGNORECASE)
        self._exact_re = _compile(self._exact)
        self._ambiguous_re = _compile(self._ambiguous)
        # Contexts in which a name means something else ("U.S. Intel shows ..." is intelligence, not Intel Corp).
        self._not_in = {ent.id: [re.compile(p) for p in ent.not_in] for ent in universe.entities if ent.not_in}

    def link(self, text: str, finance_context: bool = False) -> list[Mention]:
        """All non-overlapping mentions, left to right.

        ``finance_context`` should be True when the source itself guarantees a market context
        (a ticker-specific feed, StockTwits); it unlocks the ambiguous aliases.
        """
        found: list[Mention] = []
        for match in _CASHTAG.finditer(text):
            entity_id = self._cashtag.get(match.group(1).upper())
            if entity_id:
                found.append(Mention(entity_id, match.group(0), match.start(), match.end(), "cashtag"))
        if self._alias_re:
            for match in self._alias_re.finditer(text):
                found.append(Mention(self._alias[match.group(1).lower()], match.group(1), match.start(1), match.end(1), "alias"))
        if self._exact_re:
            for match in self._exact_re.finditer(text):
                found.append(Mention(self._exact[match.group(1)], match.group(1), match.start(1), match.end(1), "exact"))
        if self._ambiguous_re and (finance_context or _FINANCE_CUE.search(text)):
            for match in self._ambiguous_re.finditer(text):
                found.append(Mention(self._ambiguous[match.group(1)], match.group(1), match.start(1), match.end(1), "ambiguous"))
        if self._not_in:
            found = [m for m in found if not self._excluded(text, m)]

        # Resolve overlaps: earliest start first, longest span wins.
        found.sort(key=lambda m: (m.start, -(m.end - m.start)))
        kept: list[Mention] = []
        last_end = -1
        for mention in found:
            if mention.start >= last_end:
                kept.append(mention)
                last_end = mention.end
        return kept

    def _excluded(self, text: str, mention: Mention) -> bool:
        return any(m.start() <= mention.start and mention.end <= m.end()
                   for p in self._not_in.get(mention.entity_id, ()) for m in p.finditer(text))

    def entity_ids(self, text: str, finance_context: bool = False) -> list[str]:
        """Distinct entity ids in order of first appearance."""
        return list(dict.fromkeys(m.entity_id for m in self.link(text, finance_context)))
