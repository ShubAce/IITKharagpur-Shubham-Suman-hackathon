"""GDELT connector, built on the raw 15-minute Global Knowledge Graph (GKG) files.

GDELT publishes one GKG file every 15 minutes covering every article it saw worldwide:
headline, source domain, extracted organisations, themes and tone. We use the raw files
rather than the DOC query API because the raw files are not rate limited and the full
archive back to 2015 is addressable by timestamp - which is what makes crisis replays possible.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests

from tremor.ingestion.base import USER_AGENT, Gate
from tremor.nlp.preprocess import clean_text
from tremor.schemas import RawDocument, SourceKind

log = logging.getLogger(__name__)

BASE = "http://data.gdeltproject.org/gdeltv2"
_TITLE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.DOTALL)
_GKG_COLUMNS = 27
SLOT = timedelta(minutes=15)


def slot_floor(moment: datetime) -> datetime:
    """Start of the 15-minute GDELT slot containing ``moment`` (UTC)."""
    moment = moment.astimezone(timezone.utc)
    return moment.replace(minute=moment.minute - moment.minute % 15, second=0, microsecond=0)


def slot_url(slot: datetime) -> str:
    return f"{BASE}/{slot:%Y%m%d%H%M%S}.gkg.csv.zip"


def fetch_slot(slot: datetime, timeout: float = 120) -> bytes | None:
    """Download one GKG slot; ``None`` if it is not published (yet) or the request fails."""
    try:
        resp = requests.get(slot_url(slot), headers=USER_AGENT, timeout=timeout)
    except requests.RequestException as exc:
        log.warning("GDELT slot %s failed: %s", slot, exc)
        return None
    if resp.status_code != 200 or resp.content[:2] != b"PK":
        return None
    return resp.content


@dataclass(frozen=True)
class GkgArticle:
    published_at: datetime
    domain: str
    url: str
    title: str
    themes: frozenset[str]
    tone: float | None


def iter_gkg(payload: bytes) -> Iterator[GkgArticle]:
    """Every article with a usable headline in one zipped GKG file (no filtering)."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        text = archive.read(archive.namelist()[0]).decode("utf-8", errors="replace")
    for line in text.split("\n"):
        row = line.split("\t")
        if len(row) < _GKG_COLUMNS:
            continue
        match = _TITLE.search(row[26])
        if not match:
            continue
        title = clean_text(match.group(1), strip_publisher_suffix=True)
        if len(title) < 15:
            continue
        try:
            published = datetime.strptime(row[1], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            tone = float(row[15].split(",")[0]) if row[15] else None
        except ValueError:
            continue
        yield GkgArticle(published, row[3], row[4], title, frozenset(t for t in row[7].split(";") if t), tone)


def parse_gkg(payload: bytes, gate: Gate) -> Iterator[RawDocument]:
    """Yield the gated articles of one zipped GKG file as documents (headline only)."""
    for art in iter_gkg(payload):
        if not gate.accept(art.title, art.themes):
            continue
        yield RawDocument(
            doc_id=RawDocument.make_id("gdelt", art.url),
            source="gdelt",
            kind=SourceKind.NEWS,
            published_at=art.published_at,
            text=art.title,
            url=art.url,
            publisher=art.domain or None,
            meta={"gdelt_tone": art.tone,
                  "themes": sorted(t for t in art.themes if t.startswith(("ECON_", "SANCTIONS", "ARMEDCONFLICT")))[:8]},
        )


def iter_range(start: datetime, end: datetime, gate: Gate) -> Iterator[tuple[datetime, list[RawDocument]]]:
    """Historical fetch: every slot in ``[start, end)``, oldest first. Used to build replays."""
    slot = slot_floor(start)
    while slot < end:
        payload = fetch_slot(slot)
        yield slot, (list(parse_gkg(payload, gate)) if payload else [])
        slot += SLOT


class GdeltSource:
    """Live feed: each poll returns the articles of any 15-minute slot published since the last poll."""

    name = "gdelt"

    def __init__(self, gate: Gate, interval_seconds: float = 120.0, lookback_slots: int = 2):
        self.interval_seconds = interval_seconds
        self._gate = gate
        self._next_slot: datetime | None = None
        self._lookback = lookback_slots

    def poll(self) -> list[RawDocument]:
        newest = slot_floor(datetime.now(timezone.utc))
        if self._next_slot is None:
            self._next_slot = newest - self._lookback * SLOT
        docs: list[RawDocument] = []
        while self._next_slot <= newest:
            payload = fetch_slot(self._next_slot)
            if payload is None:
                # The newest slot is often published a few minutes late: retry it next poll,
                # but do not stall forever on a slot that never appears.
                if newest - self._next_slot < 4 * SLOT:
                    break
            else:
                docs.extend(parse_gkg(payload, self._gate))
            self._next_slot += SLOT
        return docs
