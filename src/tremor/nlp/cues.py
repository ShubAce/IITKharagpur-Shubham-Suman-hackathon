"""Rule cues from ``configs/taxonomy.yaml``.

Three jobs: weak labels for training the event head, a transparent fallback classifier when
no model is available, and the "intensity" evidence the impact scorecard reports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tremor.config import Taxonomy

# An equity analyst changing a stock rating is market commentary, not a credit event.
# Without this guard "downgrade" headlines would flood the credit class.
_BROKER_RATING = re.compile(
    r"\b(upgrad\w+|downgrad\w+|initiat\w+|reiterat\w+)\b.{0,60}\b(buy|sell|hold|neutral|overweight|underweight|"
    r"outperform|underperform|price target|equal[- ]weight|market perform)\b"
    r"|\bprice target\b",
    re.IGNORECASE,
)

# The vocabulary of war used for competition: a price war is not an armed conflict. Always figurative:
_WAR_IDIOM = re.compile(
    r"\b(price|pricing|bidding|talent|streaming|console|culture|fare|format|flame|turf|patent|browser|cola|"
    r"subscription|search|cloud|smartphone|delivery|grocery|discount|deposit)\s+wars?\b"
    r"|\bgod of war\b|\bstar wars\b|\bwar (chest|room)s?\b",
    re.IGNORECASE,
)
# Figurative unless the text names a country or an alliance ("a silent war" between two retailers).
_WAR_FIGURE = re.compile(
    r"\b(silent|secret|quiet|stealth)\s+wars?\b"
    r"|\bwar (on|against) (inflation|cash|talent|counterfeits?|fraud|waste|costs?|prices|bots|spam)\b",
    re.IGNORECASE,
)
_WAR_WORD = re.compile(r"\bwars?\b", re.IGNORECASE)


def _drop_war(match: re.Match) -> str:
    """Blank out only the war word, so "oil price war" still reads as oil prices."""
    return _WAR_WORD.sub(" ", match.group(0))


@dataclass(frozen=True)
class CueHits:
    by_type: dict[str, int]  # event type -> number of distinct cue patterns that fired
    severity: dict[str, int]  # "extreme" | "high" | "scale" -> hit count
    figurative_war: bool = False  # "war" only as an idiom, with nothing else geopolitical in the text

    @property
    def best_type(self) -> str | None:
        if not self.by_type:
            return None
        return max(self.by_type, key=lambda k: self.by_type[k])


class CueMatcher:
    def __init__(self, taxonomy: Taxonomy):
        self._type_patterns = {
            type_id: [re.compile(p, re.IGNORECASE) for p in spec.cues] for type_id, spec in taxonomy.event_types.items()
        }
        self._severity_patterns = {name: re.compile(p, re.IGNORECASE) for name, p in taxonomy.severity_cues.items()}

    def match(self, text: str, geo_anchor: bool = True) -> CueHits:
        """``geo_anchor``: the text names a country or an alliance, which keeps "a silent war" literal."""
        literal = text
        if "war" in text.lower():  # cheap pre-check: every idiom contains it
            literal = _WAR_IDIOM.sub(_drop_war, text)
            if not geo_anchor:
                literal = _WAR_FIGURE.sub(_drop_war, literal)
        by_type = {}
        for type_id, patterns in self._type_patterns.items():
            hits = sum(1 for p in patterns if p.search(literal))
            if hits:
                by_type[type_id] = hits
        if "CREDIT_EVENT" in by_type and _BROKER_RATING.search(text):
            # Keep CREDIT only if something other than the word "downgrade" supports it.
            by_type["CREDIT_EVENT"] -= 1
            if by_type["CREDIT_EVENT"] <= 0:
                del by_type["CREDIT_EVENT"]
            by_type["MARKET_COMMENTARY"] = by_type.get("MARKET_COMMENTARY", 0) + 1
        severity = {name: len(p.findall(literal)) for name, p in self._severity_patterns.items()}
        return CueHits(by_type=by_type, severity={k: v for k, v in severity.items() if v},
                       figurative_war=literal != text and "GEOPOLITICAL" not in by_type)
