"""Disk cache for embeddings, so re-training heads takes seconds instead of minutes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from tremor.nlp.encoder import EncoderSpec, OnnxEncoder
from tremor.paths import RAW_DIR


def _key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Maps text -> embedding for one specific encoder; persisted under ``data/raw/emb_cache``."""

    def __init__(self, encoder: OnnxEncoder, cache_dir: Path = RAW_DIR / "emb_cache"):
        self.encoder = encoder
        spec: EncoderSpec = encoder.spec
        tag = _key(f"{spec.repo}|{spec.onnx_file}|{spec.pooling}|{spec.max_length}")[:10]
        self._vec_path = cache_dir / f"{tag}.npy"
        self._idx_path = cache_dir / f"{tag}.json"
        cache_dir.mkdir(parents=True, exist_ok=True)
        if self._vec_path.exists() and self._idx_path.exists():
            self._vectors = np.load(self._vec_path)
            self._index: dict[str, int] = json.loads(self._idx_path.read_text(encoding="utf-8"))
        else:
            self._vectors = np.zeros((0, encoder.dim), dtype=np.float32)
            self._index = {}

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        keys = [_key(t) for t in texts]
        missing = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in self._index))
        if missing:
            fresh = self.encoder.encode(missing, batch_size=batch_size)
            base = len(self._vectors)
            self._vectors = np.vstack([self._vectors, fresh])
            for offset, text in enumerate(missing):
                self._index[_key(text)] = base + offset
            np.save(self._vec_path, self._vectors)
            self._idx_path.write_text(json.dumps(self._index), encoding="utf-8")
        return self._vectors[[self._index[k] for k in keys]]
