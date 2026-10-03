"""Task heads: small classifiers that sit on top of the shared sentence embedding.

Heads are stored as plain NumPy arrays in one ``.npz`` file, so inference needs neither
scikit-learn nor PyTorch and the whole trained model is a few hundred kilobytes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass
class Head:
    """Multinomial classifier: an optional hidden ReLU layer, then a softmax layer."""

    name: str
    labels: tuple[str, ...]
    weights: list[np.ndarray]  # each (in, out)
    biases: list[np.ndarray]  # each (out,)
    temperature: float = 1.0  # >1 softens over-confident probabilities (set by calibration)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        h = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if i < len(self.weights) - 1:
                h = np.maximum(h, 0.0)
        return softmax(h / self.temperature)


class HeadBundle:
    """All heads plus the metadata needed to audit them (training data, metrics, encoder)."""

    def __init__(self, heads: dict[str, Head], meta: dict):
        self.heads = heads
        self.meta = meta

    def __getitem__(self, name: str) -> Head:
        return self.heads[name]

    def __contains__(self, name: str) -> bool:
        return name in self.heads

    def save(self, path: Path) -> None:
        arrays: dict[str, np.ndarray] = {}
        index = {}
        for name, head in self.heads.items():
            index[name] = {"labels": list(head.labels), "layers": len(head.weights), "temperature": head.temperature}
            for i, (w, b) in enumerate(zip(head.weights, head.biases)):
                arrays[f"{name}__w{i}"] = w.astype(np.float32)
                arrays[f"{name}__b{i}"] = b.astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        path.with_suffix(".json").write_text(json.dumps({"heads": index, "meta": self.meta}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> HeadBundle:
        info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(path) as arrays:
            heads = {}
            for name, spec in info["heads"].items():
                heads[name] = Head(
                    name=name,
                    labels=tuple(spec["labels"]),
                    weights=[arrays[f"{name}__w{i}"] for i in range(spec["layers"])],
                    biases=[arrays[f"{name}__b{i}"] for i in range(spec["layers"])],
                    temperature=float(spec.get("temperature", 1.0)),
                )
        return cls(heads, info.get("meta", {}))
