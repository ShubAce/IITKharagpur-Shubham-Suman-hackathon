"""Evaluation harness: every model, the same held-out data, through the real inference path.

Compares the naive keyword baseline, the frozen-encoder probe and the fine-tuned TREMOR encoder
(int8 ONNX, exactly as the engine runs it) on:

* sentiment     - news (PhraseBank, SEntFiN, FiQA headlines) and social (TFNS tweets, StockTwits)
* event type    - human-labelled finance tweets mapped onto the taxonomy
* entity-level  - headlines naming several companies, often with *opposite* sentiment
                  ("Apple gains as Samsung stumbles"): does each company get its own score?
* efficiency    - documents per second on this machine's CPU, model size on disk

Only test splits are used, and test splits are never used for training or model selection.

    python main.py evaluate            # also calibrates the fine-tuned model's probabilities first
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from tremor.config import load_settings, load_taxonomy
from tremor.nlp.heads import softmax
from tremor.nlp.models import ENCODER_DIR, HEADS_PATH, MultiTaskOnnxModel, ProbeTextModel, RuleTextModel
from tremor.paths import DOCS_DIR
from tremor.training.corpora import event_corpus, sentiment_corpus
from tremor.training.finetune import _dev_split, target_corpus

RESULTS = DOCS_DIR / "results" / "evaluation.json"


def _fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.5, 4.0, 36):
        p = softmax(logits / t)
        nll = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-9, None)).mean())
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return round(best_t, 3)


def calibrate() -> dict:
    """Temperature-scale the fine-tuned model's three heads on the dev split; store in model.json."""
    taxonomy = load_taxonomy()
    model = MultiTaskOnnxModel(taxonomy)
    event_id = {name: i for i, name in enumerate(taxonomy.ids)}
    sent = _dev_split(sentiment_corpus())
    sent = sent[(sent["split"] == "dev") & (sent["dataset"] != "stocktwits")]
    ev = _dev_split(event_corpus(taxonomy))
    ev = ev[ev["split"] == "dev"]
    tg = _dev_split(target_corpus())
    tg = tg[tg["split"] == "dev"]
    _, s_logits, _, _ = model._run(sent["text"].tolist(), model._max_length)
    _, _, e_logits, _ = model._run(ev["text"].tolist(), model._max_length)
    *_, t_logits = model._run(list(zip(tg["entity"], tg["text"])), model._pair_max_length)
    temps = {"sentiment": _fit_temperature(s_logits, sent["label"].to_numpy()),
             "event": _fit_temperature(e_logits, ev["label"].map(event_id).to_numpy()),
             "target": _fit_temperature(t_logits, tg["label"].to_numpy())}
    path = ENCODER_DIR / "model.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["temperature"] = temps
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return temps


def _sentiment_metrics(model, df: pd.DataFrame) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    out = {}
    for name, part in df.groupby("dataset"):
        probs = model.predict(part["text"].tolist()).sentiment
        y = part["label"].to_numpy()
        if name == "stocktwits":
            out[name] = {"n": len(part), "directional_accuracy": round(float((((probs[:, 2] - probs[:, 0]) > 0) == (y == 2)).mean()), 4)}
        else:
            pred = probs.argmax(1)
            out[name] = {"n": len(part), "accuracy": round(float(accuracy_score(y, pred)), 4),
                         "macro_f1": round(float(f1_score(y, pred, average="macro")), 4)}
    return out


def _event_metrics(model, df: pd.DataFrame, labels: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    probs = model.predict(df["text"].tolist()).event
    y = df["label"].map({n: i for i, n in enumerate(labels)}).to_numpy()
    pred = probs.argmax(1)
    present = sorted(set(y.tolist()))
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=present, zero_division=0)
    return {"n": len(df), "accuracy": round(float(accuracy_score(y, pred)), 4),
            "macro_f1": round(float(f1_score(y, pred, labels=present, average="macro", zero_division=0)), 4),
            "per_class": {labels[c]: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3), "f1": round(float(f[i]), 3),
                                      "n": int(n[i])} for i, c in enumerate(present)}}


def _entity_metrics(model, df: pd.DataFrame) -> dict:
    """Accuracy of the sentiment *towards each named entity*; models without an entity-level head
    fall back to giving every entity the whole text's sentiment (the naive approach)."""
    if getattr(model, "supports_targets", False):
        # Classify through the entity-conditioned head's probabilities (one pass per entity-text pair).
        *_, logits = model._run(list(zip(df["entity"], df["text"])), model._pair_max_length)
        pred = softmax(logits / model._temperature.get("target", 1.0)).argmax(1)
        method = "entity-conditioned head"
    else:
        pred = model.predict(df["text"].tolist()).sentiment.argmax(1)
        method = "document sentiment copied to every entity"
    y = df["label"].to_numpy()
    out = {"method": method, "n": len(df), "accuracy": round(float((pred == y).mean()), 4)}
    multi = df["n_entities"].to_numpy() > 1
    out["multi_entity_accuracy"] = round(float((pred[multi] == y[multi]).mean()), 4)
    # Headlines where the named entities have *different* labels: the hardest, most decision-relevant case.
    conflicting = df.groupby("text")["label"].transform("nunique").to_numpy() > 1
    out["conflicting_n"] = int(conflicting.sum())
    out["conflicting_accuracy"] = round(float((pred[conflicting] == y[conflicting]).mean()), 4)
    return out


def _throughput(model, texts: list[str]) -> dict:
    model.predict(texts[:16])  # warm-up
    t0 = time.perf_counter()
    model.predict(texts)
    batch_rate = len(texts) / (time.perf_counter() - t0)
    single = []
    for text in texts[:50]:
        t0 = time.perf_counter()
        model.predict([text])
        single.append((time.perf_counter() - t0) * 1000)
    return {"docs_per_second_batched": round(batch_rate, 1), "latency_ms_single_p50": round(float(np.median(single)), 2),
            "cpu_threads": min(8, os.cpu_count() or 4)}


def evaluate() -> dict:
    settings, taxonomy = load_settings(), load_taxonomy()
    labels = taxonomy.ids
    sent = sentiment_corpus()
    sent_test = sent[sent["split"] == "test"]
    ev = event_corpus(taxonomy)
    ev_test = ev[(ev["split"] == "test") & (ev["origin"] == "human")]
    tg = target_corpus()
    tg_test = tg[(tg["split"] == "test") & (tg["dataset"] == "sentfin")]
    speed_texts = ev_test["text"].sample(2000, random_state=0, replace=len(ev_test) < 2000).tolist()

    models = {"keyword baseline (naive)": RuleTextModel(taxonomy)}
    if HEADS_PATH.exists():
        try:
            models["frozen encoder + MLP head"] = ProbeTextModel(settings, taxonomy)
        except Exception as exc:  # heads trained for another encoder
            print("skipping frozen probe:", exc)
    models["TREMOR fine-tuned (int8 ONNX)"] = MultiTaskOnnxModel(taxonomy)

    report = {"evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "models": {}}
    for name, model in models.items():
        print(f"evaluating {name} ...", flush=True)
        size = (ENCODER_DIR / "model.onnx").stat().st_size if name.startswith("TREMOR") else None
        report["models"][name] = {
            "sentiment": _sentiment_metrics(model, sent_test),
            "event": _event_metrics(model, ev_test, labels),
            "entity_sentiment": _entity_metrics(model, tg_test),
            "efficiency": {**_throughput(model, speed_texts), **({"model_mb": round(size / 1e6, 1)} if size else {})},
        }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(report)
    return report


def print_report(report: dict) -> None:
    names = list(report["models"])
    first = report["models"][names[0]]
    print(f"\n{'':34s}" + "".join(f"{n[:26]:>28s}" for n in names))
    for ds in first["sentiment"]:
        key = "directional_accuracy" if ds == "stocktwits" else "accuracy"
        print(f"{'sentiment ' + ds + ' (' + key[:3] + ')':34s}" + "".join(f"{report['models'][n]['sentiment'][ds][key]:>28.3f}" for n in names))
    print(f"{'event type macro-F1':34s}" + "".join(f"{report['models'][n]['event']['macro_f1']:>28.3f}" for n in names))
    print(f"{'entity sentiment (conflicting)':34s}" + "".join(f"{report['models'][n]['entity_sentiment']['conflicting_accuracy']:>28.3f}" for n in names))
    print(f"{'docs / second (batched, CPU)':34s}" + "".join(f"{report['models'][n]['efficiency']['docs_per_second_batched']:>28.0f}" for n in names))
    print(f"{'latency ms (single doc, p50)':34s}" + "".join(f"{report['models'][n]['efficiency']['latency_ms_single_p50']:>28.1f}" for n in names))


if __name__ == "__main__":
    print("temperatures:", calibrate())
    evaluate()
