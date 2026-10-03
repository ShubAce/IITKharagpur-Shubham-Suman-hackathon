"""Industry reference point: FinBERT (ProsusAI/finbert) on the same held-out sentiment sets.

FinBERT is the most widely used open financial sentiment model. Note it was itself trained on
Financial PhraseBank, so its PhraseBank score is partly in-sample; the tweet and StockTwits
scores are the fair comparison. Runs on CPU (as the engine does) to compare throughput too.

Needs PyTorch + transformers (requirements-dev.txt):
    python scripts/benchmark_finbert.py      # -> docs/results/finbert_reference.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from tremor.training.corpora import sentiment_corpus  # noqa: E402


def main() -> int:
    torch.set_num_threads(8)
    name = "ProsusAI/finbert"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).eval()
    # FinBERT's label order -> ours (0 negative, 1 neutral, 2 positive)
    order = [model.config.label2id[k] for k in ("negative", "neutral", "positive")]

    @torch.no_grad()
    def predict(texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], truncation=True, max_length=64, padding=True, return_tensors="pt")
            out.append(model(**enc).logits.softmax(-1)[:, order].numpy())
        return np.concatenate(out)

    test = sentiment_corpus()
    test = test[test["split"] == "test"]
    report = {"model": name, "parameters_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1), "sentiment": {}}
    for ds, part in test.groupby("dataset"):
        probs = predict(part["text"].tolist())
        y = part["label"].to_numpy()
        if ds == "stocktwits":
            report["sentiment"][ds] = {"n": len(part), "directional_accuracy": round(float((((probs[:, 2] - probs[:, 0]) > 0) == (y == 2)).mean()), 4)}
        else:
            report["sentiment"][ds] = {"n": len(part), "accuracy": round(float(accuracy_score(y, probs.argmax(1))), 4),
                                       "macro_f1": round(float(f1_score(y, probs.argmax(1), average="macro")), 4)}
        print(ds, report["sentiment"][ds], flush=True)
    sample = test["text"].sample(1000, random_state=0).tolist()
    t0 = time.perf_counter()
    predict(sample)
    report["docs_per_second_batched_cpu"] = round(len(sample) / (time.perf_counter() - t0), 1)
    print("throughput:", report["docs_per_second_batched_cpu"], "docs/s")
    out = ROOT / "docs" / "results" / "finbert_reference.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
