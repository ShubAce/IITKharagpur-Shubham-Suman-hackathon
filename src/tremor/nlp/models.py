"""The engine's text model. Three interchangeable implementations of one interface:

``MultiTaskOnnxModel``  the fine-tuned encoder (``python main.py finetune``): one forward pass
                        returns the embedding, document sentiment and event type; a second,
                        entity-conditioned pass gives sentiment *towards a named company*.
``ProbeTextModel``      a frozen off-the-shelf encoder with small trained heads
                        (``python main.py train``): the first baseline the fine-tuned model beats.
``RuleTextModel``       keyword counting: the naive baseline, and the fallback when no model file
                        can be loaded, so the system always runs.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from tremor.config import Settings, Taxonomy
from tremor.nlp.cues import CueMatcher
from tremor.nlp.heads import softmax
from tremor.nlp.lexicon import hashing_embedding, lexicon_sentiment
from tremor.paths import MODELS_DIR

log = logging.getLogger(__name__)
HEADS_PATH = MODELS_DIR / "heads.npz"
ENCODER_DIR = MODELS_DIR / "tremor-encoder"


@dataclass
class ModelOutput:
    embeddings: np.ndarray  # (n, d) unit-norm
    sentiment: np.ndarray  # (n, 3) probabilities: negative, neutral, positive
    event: np.ndarray  # (n, k) probabilities over taxonomy.ids

    @property
    def sentiment_score(self) -> np.ndarray:
        """Signed score in [-1, 1]: P(positive) - P(negative)."""
        return self.sentiment[:, 2] - self.sentiment[:, 0]


class TextModel(Protocol):
    name: str
    supports_targets: bool

    def predict(self, texts: Sequence[str]) -> ModelOutput: ...

    def target_sentiment(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """Sentiment in [-1, 1] towards ``entity`` for each ``(entity, text)`` pair."""
        ...


class MultiTaskOnnxModel:
    name = "tremor-encoder"
    supports_targets = True

    def __init__(self, taxonomy: Taxonomy, model_dir: Path = ENCODER_DIR, threads: int | None = None, batch_size: int = 64):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.meta = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
        if list(self.meta["event_labels"]) != list(taxonomy.ids):
            raise RuntimeError("model event labels do not match configs/taxonomy.yaml; run `python main.py finetune`")
        cfg = self.meta["config"]
        self._max_length, self._pair_max_length = cfg["max_length"], cfg["pair_max_length"]
        self._temperature = self.meta.get("temperature", {})
        self._batch_size = batch_size
        self._tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tok.enable_padding(pad_id=self._tok.token_to_id("[PAD]") or 0, pad_token="[PAD]")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or min(8, os.cpu_count() or 4)
        self._sess = ort.InferenceSession(str(model_dir / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])

    def _run(self, items: list, max_length: int) -> tuple[np.ndarray, ...]:
        """Tokenise (single texts or pairs), run in length-sorted batches, return outputs in input order."""
        self._tok.enable_truncation(max_length=max_length)
        order = sorted(range(len(items)), key=lambda i: len(items[i]) if isinstance(items[i], str) else len(items[i][1]))
        outputs: list[list[np.ndarray]] = [[], [], [], []]
        for start in range(0, len(order), self._batch_size):
            enc = self._tok.encode_batch([items[i] for i in order[start:start + self._batch_size]])
            feeds = {
                "input_ids": np.asarray([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.asarray([e.attention_mask for e in enc], dtype=np.int64),
                "token_type_ids": np.asarray([e.type_ids for e in enc], dtype=np.int64),
            }
            for sink, value in zip(outputs, self._sess.run(None, feeds)):
                sink.append(value)
        inverse = np.argsort(order)
        return tuple(np.concatenate(chunks)[inverse] for chunks in outputs)

    def predict(self, texts: Sequence[str]) -> ModelOutput:
        embedding, sentiment, event, _ = self._run(list(texts), self._max_length)
        return ModelOutput(embedding, softmax(sentiment / self._temperature.get("sentiment", 1.0)),
                           softmax(event / self._temperature.get("event", 1.0)))

    def target_sentiment(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        *_, target = self._run([tuple(p) for p in pairs], self._pair_max_length)
        probs = softmax(target / self._temperature.get("target", 1.0))
        return probs[:, 2] - probs[:, 0]


class ProbeTextModel:
    name = "frozen-probe"
    supports_targets = False

    def __init__(self, settings: Settings, taxonomy: Taxonomy, heads_path: Path = HEADS_PATH):
        from tremor.nlp.encoder import EncoderSpec, OnnxEncoder
        from tremor.nlp.heads import HeadBundle

        cfg = settings.encoder
        self._encoder = OnnxEncoder(EncoderSpec(cfg.repo, cfg.onnx_file, cfg.pooling, cfg.max_length), threads=cfg.threads)
        self._heads = HeadBundle.load(heads_path)
        self._batch_size = cfg.batch_size
        trained_on = self._heads.meta.get("encoder", {})
        if trained_on and (trained_on.get("repo"), trained_on.get("onnx_file")) != (cfg.repo, cfg.onnx_file):
            raise RuntimeError("heads.npz was trained on a different encoder than settings.yaml selects; run `python main.py train`")
        if tuple(self._heads["event"].labels) != tuple(taxonomy.ids):
            raise RuntimeError("event head labels do not match configs/taxonomy.yaml; run `python main.py train`")

    def predict(self, texts: Sequence[str]) -> ModelOutput:
        emb = self._encoder.encode(list(texts), batch_size=self._batch_size)
        return ModelOutput(emb, self._heads["sentiment"].predict_proba(emb), self._heads["event"].predict_proba(emb))

    def target_sentiment(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        raise NotImplementedError


class RuleTextModel:
    """Keyword counting: lexicon sentiment, cue-based event type, hashed bag-of-words embedding."""

    name = "rules"
    supports_targets = False

    def __init__(self, taxonomy: Taxonomy):
        self._cues = CueMatcher(taxonomy)
        self._ids = taxonomy.ids
        self._fallback = self._ids.index("OTHER")

    def predict(self, texts: Sequence[str]) -> ModelOutput:
        texts = list(texts)
        sentiment = np.asarray([lexicon_sentiment(t) for t in texts], dtype=np.float32).reshape(-1, 3)
        event = np.full((len(texts), len(self._ids)), 0.02, dtype=np.float32)
        for row, text in enumerate(texts):
            hits = self._cues.match(text).by_type
            if hits:
                for type_id, n in hits.items():
                    event[row, self._ids.index(type_id)] += n
            else:
                event[row, self._fallback] += 1.0
        event /= event.sum(axis=1, keepdims=True)
        return ModelOutput(hashing_embedding(texts), sentiment, event)

    def target_sentiment(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        raise NotImplementedError


def load_text_model(settings: Settings, taxonomy: Taxonomy, prefer: str = "auto") -> TextModel:
    """Best available model: fine-tuned encoder, else frozen probe, else rules (never raises)."""
    if prefer == "rules":
        return RuleTextModel(taxonomy)
    candidates = []
    if prefer in ("auto", "tremor-encoder"):
        candidates.append(lambda: MultiTaskOnnxModel(taxonomy, threads=settings.encoder.threads, batch_size=settings.encoder.batch_size))
    if prefer in ("auto", "frozen-probe"):
        candidates.append(lambda: ProbeTextModel(settings, taxonomy))
    for build in candidates:
        try:
            return build()
        except Exception as exc:  # missing model file, no network on first run, label mismatch ...
            log.warning("model unavailable (%s: %s)", type(exc).__name__, exc)
    log.warning("falling back to the rule baseline")
    return RuleTextModel(taxonomy)
