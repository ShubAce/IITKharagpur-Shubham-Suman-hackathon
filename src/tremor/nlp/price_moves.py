"""Which way are the prices a text talks about moving? Direction, not sentiment.

"Oil soars as Russia invades Ukraine" is a negative sentence about a *rising* price. A stress
scenario needs the direction, so this module reads the verb of movement attached to each price a
text names - "oil prices surge", "gold slips", "Oil, gold cede gains", "stocks tumble", "a jump in
crude", "rising gas prices" - and ignores the quantities that share the vocabulary ("oil output
falls" says nothing about the price, or rather the opposite).

Rules rather than a model, on purpose: the pattern set is small, every decision can be shown, and it
is checked against realised prices (``scripts/validate_price_moves.py``: headline direction vs the
stock's actual move that day).
"""

from __future__ import annotations

import re

# What a price is called in a headline, per tracked price entity. Company share prices are handled
# generically by ``share_direction`` (subject = the name the linker found, plus "shares" / "stock").
PRICE_TERMS = {
    "OIL": r"crude(?: oil)?|oil|brent(?: crude)?|wti",
    "GAS": r"natural gas|gas|lng",
    "GOLD": r"gold|bullion",
    "SPX": r"stocks|equities|wall street|s&p(?: 500)?|dow(?: jones)?|nasdaq|stock markets?|stock futures|equity markets?|world markets",
}
_FILLER = r"(?:prices?|futures|benchmarks?|markets?|contracts?|and|&|,|\$?[\d.,]+%?|a|the|us|u\.s\.|global|world|gold|oil|crude|brent)"
_UP = (r"surg\w*|soar\w*|jump\w*|spik\w*|climb\w*|rall(?:y|ies|ied|ying)|rises?|rose|rising|risen|gain(?:s|ed|ing)?|advanc\w*|"
       r"rebound\w*|leap\w*|rocket\w*|skyrocket\w*|tops?|topped|breach\w*|hits? (?:a |an )?(?:record|new|fresh|multi|\d)|"
       r"extends? gains|firm(?:s|ed)?|edges? (?:up|higher)|ticks? up|heads? higher|spiral\w* (?:up|high)|breaks? (?:above|through|past|\$)|"
       r"broke (?:above|through|past|\$)|near(?:s|ing)? \$|higher")
_DOWN = (r"fall(?:s|ing|en)?|fell|drop\w*|slump\w*|tumbl\w*|plung\w*|slid(?:e|es|ing)?|sinks?|sank|sinking|declin\w*|"
         r"retreat\w*|eas(?:e|es|ed|ing)|dip(?:s|ped|ping)?|crash\w*|loses?|lost|losing|shed\w*|ced(?:e|es|ed|ing)|"
         r"slip(?:s|ped|ping)?|pares? gains|tank(?:s|ed|ing)?|div(?:e|es|ed|ing)|dove|edges? (?:down|lower)|ticks? down|"
         r"heads? lower|turns? (?:lower|tail)|gives? up gains|cool(?:s|ed|ing)?|lower|sell[- ]?off|sells? off|sold off")
# Nouns that put a quantity, not a price, between the subject and the verb.
_QUANTITY = r"output|production|supply|supplies|exports?|imports?|demand|inventor\w*|stockpiles?|reserves?|release|consumption|use|" \
            r"dependence|sales|revenue|profits?|earnings|jobs|costs?|spending|orders|deliveries|shipments|flows?|stocks? build"

_VERB_UP, _VERB_DOWN = re.compile(rf"^(?:{_UP})\b", re.I), re.compile(rf"^(?:{_DOWN})\b", re.I)
_NOUN_BEFORE = re.compile(r"\b(surge|spike|jump|rise|rally|climb|soaring|rising|surging|higher|record|fall|drop|slump|plunge|"
                          r"decline|slide|falling|sliding|lower|cheaper|crash)\s+(?:in\s+)?(?:the\s+)?$", re.I)
_UP_NOUNS = {"surge", "spike", "jump", "rise", "rally", "climb", "soaring", "rising", "surging", "higher", "record"}


def _after(text: str, end: int) -> int:
    """+1 / -1 / 0 from the words following a price subject (up to three filler words in between)."""
    rest = text[end:]
    for _ in range(4):
        rest = rest.lstrip(" ,;:'\"-")
        if re.match(rf"(?:{_QUANTITY})\b", rest, re.I):
            return 0
        up, down = _VERB_UP.match(rest), _VERB_DOWN.match(rest)
        if up or down:
            return 1 if up else -1
        filler = re.match(rf"(?:{_FILLER})(?=\W|$)", rest, re.I)
        if not filler:
            return 0
        rest = rest[filler.end():]
    return 0


def _before(text: str, start: int) -> int:
    """+1 / -1 / 0 from a movement noun or adjective right before the subject ("a jump in crude", "rising gas prices")."""
    match = _NOUN_BEFORE.search(text[:start])
    if not match:
        return 0
    return 1 if match.group(1).lower() in _UP_NOUNS else -1


def _direction_of(text: str, subject: re.Pattern) -> int:
    votes = 0
    for match in subject.finditer(text):
        votes += _after(text, match.end()) or _before(text, match.start())
    return (votes > 0) - (votes < 0)


_SUBJECTS = {eid: re.compile(rf"(?<![\w$])(?:{terms})(?!\w)", re.I) for eid, terms in PRICE_TERMS.items()}


def price_directions(text: str, entity_ids: list[str] | set[str] | None = None) -> dict[str, int]:
    """Direction of each price the text reports moving: entity id -> +1 (up) or -1 (down).

    Only entities in ``entity_ids`` are considered when given (the ones the linker found)."""
    out = {}
    for eid, subject in _SUBJECTS.items():
        if entity_ids is not None and eid not in entity_ids:
            continue
        direction = _direction_of(text, subject)
        if direction:
            out[eid] = direction
    return out


_QUANTIFIED = re.compile(rf"(?:{_UP}|{_DOWN})\s+(?:by\s+)?(?:about\s+|nearly\s+|almost\s+|more than\s+|over\s+)?\d[\d.]*\s?%", re.I)


def share_direction(text: str, surface: str) -> int:
    """Direction a headline reports for one company's share price.

    A company name alone is not a price ("Boeing loses out to Airbus", "Exxon plunges into plastics"), so
    the move must be attached to its shares or stock ("Boeing shares slump", "Netflix share price crash")
    or be quantified ("Tesla jumps 9%")."""
    name = re.escape(surface)
    shares = re.compile(rf"(?<![\w$]){name}(?:'s|s')?\s+(?:shares|stock|stocks|share price|stock price)(?!\w)", re.I)
    direction = _direction_of(text, shares)
    if direction:
        return direction
    for match in re.finditer(rf"(?<![\w$]){name}(?:'s)?(?!\w)", text, re.I):
        rest = text[match.end():].lstrip()
        if _QUANTIFIED.match(rest):
            return 1 if _VERB_UP.match(rest) else -1
    return 0
