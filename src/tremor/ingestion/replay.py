"""Replay ("time machine") source: feeds a recorded historical window through the live pipeline.

Documents are released when a *virtual clock* passes their timestamp; the clock runs ``speed``
times faster than real time and can be paused, resumed and re-speeded from the dashboard.
Nothing downstream knows it is a replay - which is the point: the demo exercises exactly the
code that runs on live feeds, on a real crisis, deterministically and offline.
"""

from __future__ import annotations

import gzip
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

from tremor.paths import DATA_DIR
from tremor.schemas import RawDocument

REPLAY_DIR = DATA_DIR / "replay"


def list_packs() -> list[dict]:
    packs = []
    for manifest in sorted(REPLAY_DIR.glob("*/manifest.json")):
        info = json.loads(manifest.read_text(encoding="utf-8"))
        packs.append({"name": manifest.parent.name, "title": info.get("title", manifest.parent.name),
                      "start": info.get("start"), "end": info.get("end"),
                      "news": (info.get("news") or {}).get("kept_articles"), "social": (info.get("social") or {}).get("posts")})
    return packs


def _read_jsonl(path: Path) -> list[RawDocument]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [RawDocument.model_validate_json(line) for line in fh if line.strip()]


def load_pack(name: str) -> list[RawDocument]:
    folder = REPLAY_DIR / name
    docs: list[RawDocument] = []
    for stem in ("news", "social"):
        for path in (folder / f"{stem}.jsonl", folder / f"{stem}.jsonl.gz"):
            if path.exists():
                docs.extend(_read_jsonl(path))
    if not docs:
        raise FileNotFoundError(f"replay pack {name!r} has no news/social files in {folder}")
    return sorted(docs, key=lambda d: d.published_at)


class ReplaySource:
    name = "replay"
    interval_seconds = 0.5
    max_batch = 1500  # documents released per poll at most

    def __init__(self, pack: str, speed: float = 900.0, start_offset_hours: float = 0.0):
        self.pack = pack
        self._docs = load_pack(pack)
        self.start = self._docs[0].published_at + timedelta(hours=start_offset_hours)
        self.end = self._docs[-1].published_at
        self.speed = speed
        self._cursor = 0
        while self._cursor < len(self._docs) and self._docs[self._cursor].published_at < self.start:
            self._cursor += 1
        self._virtual = self.start  # virtual time reached at the last anchor
        self._anchor: float | None = None  # wall-clock time of the last anchor
        self.paused = False

    # ------------------------------------------------------------------ clock
    @property
    def virtual_now(self) -> datetime:
        if self._anchor is None or self.paused:
            return self._virtual
        return min(self.end, self._virtual + timedelta(seconds=(time.monotonic() - self._anchor) * self.speed))

    def _reanchor(self) -> None:
        self._virtual = self.virtual_now
        self._anchor = None if self.paused else time.monotonic()

    def pause(self) -> None:
        self._reanchor()
        self.paused = True
        self._anchor = None

    def resume(self) -> None:
        self.paused = False
        self._anchor = time.monotonic()

    def set_speed(self, speed: float) -> None:
        self._reanchor()
        self.speed = max(1.0, float(speed))

    @property
    def finished(self) -> bool:
        return self._cursor >= len(self._docs)

    def status(self) -> dict:
        span = (self.end - self.start).total_seconds() or 1.0
        return {"pack": self.pack, "virtual_now": self.virtual_now.isoformat(), "start": self.start.isoformat(), "end": self.end.isoformat(),
                "progress": round(min(1.0, (self.virtual_now - self.start).total_seconds() / span), 4), "speed": self.speed,
                "paused": self.paused, "finished": self.finished, "released": self._cursor, "total": len(self._docs)}

    # ------------------------------------------------------------------ source protocol
    def poll(self) -> list[RawDocument]:
        if self._anchor is None and not self.paused:
            self._anchor = time.monotonic()
        now = self.virtual_now
        out = []
        while self._cursor < len(self._docs) and self._docs[self._cursor].published_at <= now and len(out) < self.max_batch:
            out.append(self._docs[self._cursor])
            self._cursor += 1
        if len(out) == self.max_batch and not self.finished and self._docs[self._cursor].published_at <= now:
            # Backpressure: the engine is the bottleneck, so let the virtual clock follow what was
            # actually released instead of racing ahead of the analysis.
            self._virtual = out[-1].published_at
            self._anchor = None if self.paused else time.monotonic()
        return out
