"""Sentence encoder running on ONNX Runtime - no PyTorch needed at inference time.

One forward pass per text produces an L2-normalised embedding that every downstream head
reuses: sentiment, event type, relevance, story clustering and historical-analog retrieval.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tremor.paths import MODEL_CACHE_DIR


@dataclass(frozen=True)
class EncoderSpec:
    """Where to find an ONNX sentence encoder on the Hugging Face Hub and how to pool it."""

    repo: str
    onnx_file: str
    pooling: str = "mean"  # "mean" | "cls"
    max_length: int = 96
    tokenizer_file: str = "tokenizer.json"


class OnnxEncoder:
    """Tokenise -> ONNX transformer -> pool -> L2-normalise."""

    def __init__(self, spec: EncoderSpec, threads: int | None = None, cache_dir: Path = MODEL_CACHE_DIR):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        if spec.pooling not in ("mean", "cls"):
            raise ValueError(f"unknown pooling {spec.pooling!r}")
        self.spec = spec
        cache_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = hf_hub_download(spec.repo, spec.onnx_file, cache_dir=cache_dir)
        tok_path = hf_hub_download(spec.repo, spec.tokenizer_file, cache_dir=cache_dir)

        self._tok = Tokenizer.from_file(tok_path)
        self._tok.enable_truncation(max_length=spec.max_length)
        pad_id = self._tok.token_to_id("[PAD]") or 0
        self._tok.enable_padding(pad_id=pad_id, pad_token="[PAD]")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or min(8, os.cpu_count() or 4)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self._sess.get_inputs()}
        self.dim = int(self._sess.get_outputs()[0].shape[-1])

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """Return an ``(n, dim)`` float32 matrix of unit-norm embeddings, in input order."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        # Sorting by length keeps padding (and therefore wasted compute) minimal per batch.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            enc = self._tok.encode_batch([texts[i] for i in idx])
            ids = np.asarray([e.ids for e in enc], dtype=np.int64)
            mask = np.asarray([e.attention_mask for e in enc], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self._sess.run(None, feeds)[0]  # (batch, seq, dim)
            if self.spec.pooling == "cls":
                pooled = hidden[:, 0]
            else:
                m = mask[..., None].astype(np.float32)
                pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            out[idx] = pooled
        return out
