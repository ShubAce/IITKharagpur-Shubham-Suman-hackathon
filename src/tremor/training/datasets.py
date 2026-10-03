"""Loaders for the public labelled datasets (fetched by ``scripts/fetch_data.py``).

Every loader returns a tidy DataFrame with a ``text`` column and a fixed, reproducible
``split`` column, so training and evaluation never leak into each other.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from tremor.paths import RAW_DIR

SENTIMENT_LABELS = ("negative", "neutral", "positive")
_SENT_ID = {name: i for i, name in enumerate(SENTIMENT_LABELS)}

# Label ids of zeroshot/twitter-financial-news-topic, in dataset order.
TFN_TOPICS = (
    "Analyst Update", "Fed | Central Banks", "Company | Product News", "Treasuries | Corporate Debt",
    "Dividend", "Earnings", "Energy | Oil", "Financials", "Currencies", "General News | Opinion",
    "Gold | Metals | Materials", "IPO", "Legal | Regulation", "M&A | Investments", "Macro", "Markets",
    "Politics", "Personnel Change", "Stock Commentary", "Stock Movement",
)


def stable_split(text: str, test_fraction: float = 0.2) -> str:
    """Deterministic train/test assignment from a hash of the text (stable across runs and machines)."""
    bucket = int(hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if bucket < test_fraction else "train"


def load_fpb() -> pd.DataFrame:
    """Financial PhraseBank: news sentences labelled by 5-8 annotators.

    ``agreement`` is the strictest agreement tier a sentence belongs to (50/66/75/100 %).
    """
    base = RAW_DIR / "fpb" / "FinancialPhraseBank"
    tiers = {"Sentences_50Agree.txt": 50, "Sentences_66Agree.txt": 66, "Sentences_75Agree.txt": 75, "Sentences_AllAgree.txt": 100}
    best: dict[str, tuple[str, int]] = {}
    for fname, tier in tiers.items():
        for line in (base / fname).read_text(encoding="latin-1").splitlines():
            if "@" not in line:
                continue
            text, label = line.rsplit("@", 1)
            best[text.strip()] = (label.strip(), tier)  # later (stricter) tiers overwrite
    df = pd.DataFrame([{"text": t, "label": _SENT_ID[lab], "agreement": tier} for t, (lab, tier) in best.items()])
    df["split"] = df["text"].map(stable_split)
    df["source"] = "news"
    return df


def load_tfns() -> pd.DataFrame:
    """Twitter Financial News Sentiment: finance tweets, official train/validation split."""
    remap = {0: _SENT_ID["negative"], 1: _SENT_ID["positive"], 2: _SENT_ID["neutral"]}  # bearish, bullish, neutral
    parts = []
    for fname, split in (("sent_train.csv", "train"), ("sent_valid.csv", "test")):
        part = pd.read_csv(RAW_DIR / "tfns" / fname)
        part["label"] = part["label"].map(remap)
        part["split"] = split
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    df["source"] = "social"
    return df


def load_tfn_topic() -> pd.DataFrame:
    """Twitter Financial News Topic: 20 topic labels, official train/validation split."""
    parts = []
    for fname, split in (("topic_train.csv", "train"), ("topic_valid.csv", "test")):
        part = pd.read_csv(RAW_DIR / "tfn_topic" / fname)
        part["topic"] = part["label"].map(lambda i: TFN_TOPICS[i])
        part["split"] = split
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def load_fiqa() -> pd.DataFrame:
    """FiQA-2018 task 1: sentiment towards a named target on a continuous -1..1 scale."""
    base = RAW_DIR / "fiqa" / "data"
    parts = []
    for prefix, split in (("train", "train"), ("valid", "train"), ("test", "test")):
        part = pd.read_parquet(next(base.glob(f"{prefix}-*.parquet")))
        part["split"] = split
        parts.append(part)
    df = pd.concat(parts, ignore_index=True).rename(columns={"sentence": "text"})
    df["source"] = df["type"].map({"headline": "news", "post": "social"})
    return df[["text", "target", "aspect", "score", "source", "split"]]


def load_sentfin() -> pd.DataFrame:
    """SEntFiN 1.1: one row per (headline, entity) with that entity's own sentiment label."""
    raw = pd.read_csv(RAW_DIR / "sentfin" / "SEntFiN-v1.1.csv")
    rows = []
    for title, decisions in zip(raw["Title"], raw["Decisions"]):
        try:
            entities = json.loads(decisions)
        except (TypeError, json.JSONDecodeError):
            continue
        split = stable_split(title)
        for entity, label in entities.items():
            if label in _SENT_ID:
                rows.append({"text": title, "entity": entity, "label": _SENT_ID[label],
                             "n_entities": len(entities), "split": split})
    df = pd.DataFrame(rows)
    df["source"] = "news"
    return df
