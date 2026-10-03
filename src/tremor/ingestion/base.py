"""The connector contract and the cheap pre-filter every connector shares."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from tremor.nlp.cues import CueMatcher
from tremor.nlp.entities import EntityLinker
from tremor.schemas import RawDocument

USER_AGENT = {"User-Agent": "Mozilla/5.0 (TREMOR research prototype)"}

# GDELT themes that mark an article as risk-relevant even when it names no tracked company.
RISK_THEMES = frozenset({
    "SANCTIONS", "ARMEDCONFLICT", "BLOCKADE", "CEASEFIRE", "MILITARY", "ECON_STOCKMARKET", "ECON_OILPRICE",
    "ECON_INFLATION", "ECON_INTEREST_RATES", "ECON_BANKRUPTCY", "ECON_DEBT", "ECON_CENTRALBANK",
    "ECON_CURRENCY_EXCHANGE_RATE", "ECON_TRADE_DISPUTE", "ECON_EARNINGSREPORT", "ECON_IPO", "ECON_MONOPOLY",
})


@runtime_checkable
class Source(Protocol):
    """A pollable feed. ``poll`` blocks, returns only documents not returned before, and never raises
    on a network failure - a dead feed must not take the engine down."""

    name: str
    interval_seconds: float

    def poll(self) -> list[RawDocument]: ...


class Gate:
    """Tier-0 filter: decides, without running any model, whether a document is worth analysing.

    A global news firehose is overwhelmingly irrelevant (sport, weather, lifestyle). Keeping only
    text that names a tracked entity cuts volume by roughly an order of magnitude at zero cost.
    Documents that name only a country or commodity must also show risk content (a taxonomy cue
    in the text or a risk theme supplied by the source), so "India wins the cricket" is dropped.
    """

    def __init__(self, linker: EntityLinker, cues: CueMatcher, company_ids: Iterable[str]):
        self._linker = linker
        self._cues = cues
        self._company_ids = frozenset(company_ids)

    def accept(self, text: str, themes: Iterable[str] = (), finance_feed: bool = False) -> bool:
        if finance_feed:
            return True
        entity_ids = self._linker.entity_ids(text)
        if not entity_ids:
            return False
        if self._company_ids.intersection(entity_ids):
            return True
        return bool(RISK_THEMES.intersection(themes)) or bool(self._cues.match(text).by_type)
