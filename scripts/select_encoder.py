"""Model selection: which compact ONNX sentence encoder should TREMOR's heads sit on?

For each candidate we freeze the encoder, fit a logistic-regression probe on its embeddings
and measure accuracy on three held-out tasks plus CPU throughput. The winner is the best
accuracy-per-millisecond trade-off, not simply the largest model.

    python scripts/select_encoder.py            # writes docs/results/encoder_selection.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from tremor.nlp.encoder import EncoderSpec, OnnxEncoder  # noqa: E402
from tremor.paths import DOCS_DIR  # noqa: E402
from tremor.training.datasets import load_fpb, load_tfn_topic, load_tfns  # noqa: E402

CANDIDATES = {
    "MiniLM-L6 (int8, 23 MB)": EncoderSpec("sentence-transformers/all-MiniLM-L6-v2", "onnx/model_quint8_avx2.onnx", "mean"),
    "MiniLM-L6 (fp32, 90 MB)": EncoderSpec("sentence-transformers/all-MiniLM-L6-v2", "onnx/model.onnx", "mean"),
    "MiniLM-L12 (int8, 34 MB)": EncoderSpec("sentence-transformers/all-MiniLM-L12-v2", "onnx/model_quint8_avx2.onnx", "mean"),
    "bge-small (int8, 34 MB)": EncoderSpec("Xenova/bge-small-en-v1.5", "onnx/model_quantized.onnx", "cls"),
    "bge-small (fp32, 133 MB)": EncoderSpec("BAAI/bge-small-en-v1.5", "onnx/model.onnx", "cls"),
    "gte-small (int8, 34 MB)": EncoderSpec("Xenova/gte-small", "onnx/model_quantized.onnx", "mean"),
    "mpnet-base (int8, 110 MB)": EncoderSpec("sentence-transformers/all-mpnet-base-v2", "onnx/model_quint8_avx2.onnx", "mean"),
    "bge-base (int8, 110 MB)": EncoderSpec("Xenova/bge-base-en-v1.5", "onnx/model_quantized.onnx", "cls"),
}


def probe(x_train, y_train, x_test, y_test) -> dict[str, float]:
    clf = LogisticRegression(C=10.0, max_iter=2000)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return {"accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4)}


def main() -> None:
    tasks = {"news_sentiment (PhraseBank)": load_fpb(), "tweet_sentiment (TFNS)": load_tfns()}
    topic = load_tfn_topic().rename(columns={"label": "topic_id"})
    topic["label"] = topic["topic_id"]
    tasks["topic (20 classes)"] = topic

    results = {}
    for name, spec in CANDIDATES.items():
        enc = OnnxEncoder(spec)
        enc.encode(["warm-up"] * 8)
        row: dict[str, object] = {"repo": spec.repo, "file": spec.onnx_file, "dim": enc.dim}
        n_texts, seconds = 0, 0.0
        for task, df in tasks.items():
            t0 = time.perf_counter()
            emb = enc.encode(df["text"].tolist())
            seconds += time.perf_counter() - t0
            n_texts += len(df)
            tr, te = (df["split"] == "train").to_numpy(), (df["split"] == "test").to_numpy()
            row[task] = probe(emb[tr], df["label"].to_numpy()[tr], emb[te], df["label"].to_numpy()[te])
        row["docs_per_second"] = round(n_texts / seconds, 1)
        results[name] = row
        print(f"{name:28s} " + " | ".join(f"{t.split(' ')[0]} acc={row[t]['accuracy']:.3f} f1={row[t]['macro_f1']:.3f}" for t in tasks)
              + f" | {row['docs_per_second']:.0f} docs/s", flush=True)

    out = DOCS_DIR / "results" / "encoder_selection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
