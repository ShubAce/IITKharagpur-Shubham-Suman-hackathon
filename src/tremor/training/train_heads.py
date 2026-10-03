"""Train the sentiment and event heads on top of the frozen sentence encoder.

The encoder is never fine-tuned: training fits two small classifiers on cached embeddings,
which takes well under a minute on a laptop CPU and makes the whole model reproducible from
the public data by anyone (``python main.py train``).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np

from tremor.config import load_settings, load_taxonomy
from tremor.nlp.encoder import EncoderSpec, OnnxEncoder
from tremor.nlp.heads import Head, HeadBundle
from tremor.paths import DOCS_DIR, MODELS_DIR
from tremor.training.corpora import SENTIMENT_LABELS, event_corpus, sentiment_corpus
from tremor.training.embed import EmbeddingCache

HEADS_PATH = MODELS_DIR / "heads.npz"


def fit_head(name: str, x: np.ndarray, y: np.ndarray, weight: np.ndarray, labels: tuple[str, ...],
             hidden: int | None = 256, seed: int = 0) -> Head:
    """Fit a softmax classifier (one hidden layer by default) and export it to plain arrays."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    if hidden:
        # MLPClassifier has no sample_weight, so weights are applied by resampling.
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x), size=len(x), replace=True, p=weight / weight.sum())
        clf = MLPClassifier(hidden_layer_sizes=(hidden,), alpha=1e-3, batch_size=256, learning_rate_init=1e-3,
                            max_iter=60, early_stopping=True, validation_fraction=0.1, n_iter_no_change=6, random_state=seed)
        clf.fit(x[idx], y[idx])
        weights, biases = list(clf.coefs_), list(clf.intercepts_)
    else:
        clf = LogisticRegression(C=10.0, max_iter=1000)
        clf.fit(x, y, sample_weight=weight)
        weights, biases = [clf.coef_.T], [clf.intercept_]
    assert list(clf.classes_) == list(range(len(labels))), "every class needs at least one training example"
    return Head(name=name, labels=labels, weights=[w.astype(np.float32) for w in weights],
                biases=[b.astype(np.float32) for b in biases])


def calibrate_temperature(head: Head, x: np.ndarray, y: np.ndarray) -> float:
    """Pick the softmax temperature that minimises negative log-likelihood on held-out data."""
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.5, 3.0, 26):
        head.temperature = float(t)
        p = head.predict_proba(x)
        nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, None)).mean()
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    head.temperature = best_t
    return best_t


def classification_report(y: np.ndarray, pred: np.ndarray, labels: tuple[str, ...]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    present = sorted(set(y.tolist()))
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=present, zero_division=0)
    return {
        "n": int(len(y)),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, labels=present, average="macro", zero_division=0)), 4),
        "per_class": {labels[c]: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                                  "f1": round(float(f[i]), 3), "n": int(n[i])} for i, c in enumerate(present)},
    }


def balance_weights(y: np.ndarray, base: np.ndarray, power: float = 0.5) -> np.ndarray:
    """Soften class imbalance: weight classes by (1 / frequency) ** power."""
    counts = np.bincount(y)
    factor = (counts.max() / np.clip(counts, 1, None)) ** power
    return base * factor[y]


def train(hidden: int | None = 256, verbose: bool = True) -> dict:
    settings, taxonomy = load_settings(), load_taxonomy()
    enc_cfg = settings.encoder
    encoder = OnnxEncoder(EncoderSpec(enc_cfg.repo, enc_cfg.onnx_file, enc_cfg.pooling, enc_cfg.max_length), threads=enc_cfg.threads)
    cache = EmbeddingCache(encoder)
    report: dict = {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "encoder": enc_cfg.model_dump(), "head": f"mlp-{hidden}" if hidden else "linear"}
    heads: dict[str, Head] = {}

    # ------------------------------------------------------------------ sentiment
    sent = sentiment_corpus()
    t0 = time.perf_counter()
    x = cache.encode(sent["text"].tolist())
    if verbose:
        print(f"sentiment corpus: {len(sent):,} texts embedded in {time.perf_counter() - t0:.1f}s")
    y = sent["label"].to_numpy()
    tr = (sent["split"] == "train").to_numpy()
    head = fit_head("sentiment", x[tr], y[tr], balance_weights(y[tr], sent["weight"].to_numpy()[tr]), SENTIMENT_LABELS, hidden)
    report["sentiment"] = {"train_rows": sent[tr].groupby("dataset").size().to_dict(), "test": {}}
    # Calibrate on the pooled test rows of the 3-class sets (StockTwits has no neutral class).
    cal = (~tr) & (sent["dataset"] != "stocktwits").to_numpy()
    report["sentiment"]["temperature"] = calibrate_temperature(head, x[cal], y[cal])
    for name, part in sent[~tr].groupby("dataset"):
        idx = part.index.to_numpy()
        proba = head.predict_proba(x[idx])
        if name == "stocktwits":  # two-class ground truth: judge the sign of the score
            score = proba[:, 2] - proba[:, 0]
            truth = y[idx] == 2
            report["sentiment"]["test"][name] = {"n": int(len(idx)), "directional_accuracy": round(float(((score > 0) == truth).mean()), 4)}
        else:
            report["sentiment"]["test"][name] = classification_report(y[idx], proba.argmax(1), SENTIMENT_LABELS)
    heads["sentiment"] = head

    # ------------------------------------------------------------------ event type
    events = event_corpus(taxonomy)
    labels = tuple(taxonomy.ids)
    label_id = {name: i for i, name in enumerate(labels)}
    t0 = time.perf_counter()
    xe = cache.encode(events["text"].tolist())
    if verbose:
        print(f"event corpus: {len(events):,} texts embedded in {time.perf_counter() - t0:.1f}s")
    ye = events["label"].map(label_id).to_numpy()
    tre = (events["split"] == "train").to_numpy()
    ehead = fit_head("event", xe[tre], ye[tre], balance_weights(ye[tre], events["weight"].to_numpy()[tre]), labels, hidden)
    report["event"] = {
        "train_rows": events[tre].groupby("origin").size().to_dict(),
        "train_by_class": events[tre].groupby("label").size().to_dict(),
        "temperature": calibrate_temperature(ehead, xe[~tre], ye[~tre]),
        "test_human_labels": classification_report(ye[~tre], ehead.predict_proba(xe[~tre]).argmax(1), labels),
    }
    heads["event"] = ehead

    bundle = HeadBundle(heads, meta={k: report[k] for k in ("trained_at", "encoder", "head")})
    bundle.save(HEADS_PATH)
    out = DOCS_DIR / "results" / "head_training.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if verbose:
        _print_summary(report)
        print(f"saved {HEADS_PATH.name} ({HEADS_PATH.stat().st_size / 1024:.0f} KB) and {out.relative_to(DOCS_DIR.parent)}")
    return report


def _print_summary(report: dict) -> None:
    print("\nSENTIMENT (held-out)")
    for name, m in report["sentiment"]["test"].items():
        if "accuracy" in m:
            print(f"  {name:11s} n={m['n']:5d}  accuracy={m['accuracy']:.3f}  macro-F1={m['macro_f1']:.3f}")
        else:
            print(f"  {name:11s} n={m['n']:5d}  directional accuracy={m['directional_accuracy']:.3f}")
    ev = report["event"]["test_human_labels"]
    print(f"\nEVENT TYPE (human-labelled test set)  n={ev['n']}  accuracy={ev['accuracy']:.3f}  macro-F1={ev['macro_f1']:.3f}")
    for name, m in ev["per_class"].items():
        print(f"  {name:22s} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}  n={m['n']}")


if __name__ == "__main__":
    train()
