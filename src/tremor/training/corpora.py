"""Assemble the training / evaluation corpora for the two heads from public datasets.

Two rules keep the evaluation honest:

* every dataset has a fixed split, and any training text that also appears in *any* test
  split is dropped (the public datasets overlap with each other);
* weak (rule-derived) labels are only ever used for training, never reported as accuracy.
"""

from __future__ import annotations

import json

import pandas as pd

from tremor.config import Taxonomy
from tremor.nlp.cues import CueMatcher
from tremor.nlp.preprocess import clean_text, fingerprint
from tremor.paths import RAW_DIR
from tremor.training.datasets import (
    SENTIMENT_LABELS,
    load_fiqa,
    load_fpb,
    load_sentfin,
    load_tfn_topic,
    load_tfns,
    stable_split,
)

FIQA_NEUTRAL_BAND = 0.15  # |score| below this counts as neutral when FiQA is used as a 3-class set

# Topics of the human-labelled TFN-topic set that map cleanly onto one event type.
TOPIC_TO_EVENT = {
    "Fed | Central Banks": "MACROECONOMIC",
    "Macro": "MACROECONOMIC",
    "Currencies": "MACROECONOMIC",
    "Energy | Oil": "MACROECONOMIC",
    "Gold | Metals | Materials": "MACROECONOMIC",
    "Politics": "GEOPOLITICAL",
    "M&A | Investments": "MERGER_ACQUISITION",
    "IPO": "MERGER_ACQUISITION",
    "Earnings": "EARNINGS_GUIDANCE",
    "Dividend": "EARNINGS_GUIDANCE",
    "Legal | Regulation": "REGULATORY_LEGAL",
    "Personnel Change": "MANAGEMENT_GOVERNANCE",
    "Analyst Update": "MARKET_COMMENTARY",
    "Stock Commentary": "MARKET_COMMENTARY",
    "Stock Movement": "MARKET_COMMENTARY",
    "Markets": "MARKET_COMMENTARY",
    "General News | Opinion": "OTHER",
}
# Topics that are really *sector* or mixed tags; a rule cue decides the event type, and the row
# is dropped from training when no cue fires.
MIXED_TOPICS = {"Company | Product News": "PRODUCT_LAUNCH", "Treasuries | Corporate Debt": None, "Financials": None}


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() >= 8]
    df["fp"] = df["text"].map(fingerprint)
    df = df.drop_duplicates(subset=["fp", "split"])
    test_fps = set(df.loc[df["split"] == "test", "fp"])
    leak = (df["split"] == "train") & df["fp"].isin(test_fps)
    return df[~leak].drop(columns="fp").reset_index(drop=True)


def load_stocktwits() -> pd.DataFrame:
    """Self-labelled StockTwits posts collected by ``scripts/collect_stocktwits.py`` (may be absent)."""
    path = RAW_DIR / "stocktwits" / "messages.jsonl"
    if not path.exists():
        return pd.DataFrame(columns=["text", "label", "split", "source"])
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            msg = json.loads(line)
            tag = msg.get("user_sentiment")
            if tag in ("Bullish", "Bearish"):
                rows.append({"text": msg["body"], "label": 2 if tag == "Bullish" else 0})
    df = pd.DataFrame(rows, columns=["text", "label"])
    df["split"] = df["text"].map(stable_split)
    df["source"] = "social"
    return df


def sentiment_corpus() -> pd.DataFrame:
    """Columns: text, label (0 neg / 1 neu / 2 pos), split, dataset, source, weight."""
    fpb = load_fpb()
    fpb["dataset"] = "fpb"
    fpb["weight"] = fpb["agreement"].map({50: 0.6, 66: 0.8, 75: 0.9, 100: 1.0})  # trust unanimous labels more

    tfns = load_tfns()
    tfns["dataset"], tfns["weight"] = "tfns", 1.0

    fiqa = load_fiqa()
    fiqa["label"] = fiqa["score"].map(lambda s: 0 if s <= -FIQA_NEUTRAL_BAND else (2 if s >= FIQA_NEUTRAL_BAND else 1))
    fiqa["dataset"], fiqa["weight"] = "fiqa", 1.0

    sentfin = load_sentfin()
    sentfin = sentfin[sentfin["n_entities"] == 1].copy()  # headline-level label is only unambiguous with one entity
    sentfin["dataset"], sentfin["weight"] = "sentfin", 1.0

    twits = load_stocktwits()
    twits["dataset"], twits["weight"] = "stocktwits", 1.0

    cols = ["text", "label", "split", "dataset", "source", "weight"]
    return _finish(pd.concat([d[cols] for d in (fpb, tfns, fiqa, sentfin, twits) if len(d)], ignore_index=True))


def _weak_pool(texts: pd.Series, cues: CueMatcher, origin: str, cap_per_class: int, seed: int = 0) -> pd.DataFrame:
    """Label a pool of unlabelled text with the rule cues, keeping only unambiguous hits."""
    rows = []
    for text in texts.dropna().unique():
        hits = cues.match(text).by_type
        if not hits:
            continue
        ranked = sorted(hits.items(), key=lambda kv: -kv[1])
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            continue  # two event types tie: too ambiguous to trust as a label
        rows.append({"text": text, "label": ranked[0][0]})
    df = pd.DataFrame(rows, columns=["text", "label"])
    # Shuffle, then keep up to ``cap_per_class`` rows per label (groupby.apply drops the key column in pandas 3).
    df = df.sample(frac=1.0, random_state=seed).groupby("label").head(cap_per_class).reset_index(drop=True)
    df["split"], df["origin"], df["weight"] = "train", origin, 0.5
    return df


def event_corpus(taxonomy: Taxonomy, weak_cap: int = 2500) -> pd.DataFrame:
    """Columns: text, label (event type id), split, origin, weight.

    ``origin == "human"`` rows carry labels made by people (TFN-topic); they form the test set.
    """
    cues = CueMatcher(taxonomy)
    topic = load_tfn_topic()
    rows = []
    for text, name, split in zip(topic["text"], topic["topic"], topic["split"]):
        if name in TOPIC_TO_EVENT:
            rows.append({"text": text, "label": TOPIC_TO_EVENT[name], "split": split, "origin": "human", "weight": 1.0})
        elif split == "train":
            best = cues.match(clean_text(text)).best_type or MIXED_TOPICS[name]
            if best:
                rows.append({"text": text, "label": best, "split": "train", "origin": "human+cue", "weight": 0.7})
    frames = [pd.DataFrame(rows)]

    # Rule-labelled pools add the classes the human set barely covers (credit events,
    # operational incidents, armed conflict) and real-world noise.
    bz_path = next((RAW_DIR / "benzinga" / "data").glob("*.parquet"), None)
    if bz_path is not None:
        headlines = pd.read_parquet(bz_path, columns=["headline"])["headline"].sample(250_000, random_state=0)
        frames.append(_weak_pool(headlines, cues, "weak:benzinga", weak_cap))
    search_path = RAW_DIR / "event_headlines" / "headlines.jsonl"
    if search_path.exists():
        # Headlines found by class-specific news searches: the query class is the weak label,
        # unless the rule cues point clearly at a different class.
        found = pd.read_json(search_path, lines=True)
        keep = []
        for title, query_class in zip(found["title"], found["query_class"]):
            best = cues.match(title).best_type
            if best is None or best == query_class:
                keep.append({"text": title, "label": query_class})
        searched = pd.DataFrame(keep)
        searched["split"], searched["origin"], searched["weight"] = "train", "weak:search", 0.6
        frames.append(searched)
    gdelt_path = RAW_DIR / "gdelt_sample" / "titles.jsonl"
    if gdelt_path.exists():
        titles = pd.read_json(gdelt_path, lines=True)["title"]
        frames.append(_weak_pool(titles, cues, "weak:gdelt", weak_cap, seed=1))
        # General news with no risk cue at all is the engine's main source of "noise" examples.
        quiet = titles[titles.map(lambda t: not cues.match(t).by_type)].drop_duplicates()
        noise = pd.DataFrame({"text": quiet.sample(min(len(quiet), 6000), random_state=2)})
        noise["label"], noise["split"], noise["origin"], noise["weight"] = "OTHER", "train", "weak:gdelt-noise", 0.5
        frames.append(noise)

    df = _finish(pd.concat([f for f in frames if len(f)], ignore_index=True))
    return df[df["label"].isin(taxonomy.ids)].reset_index(drop=True)


__all__ = ["SENTIMENT_LABELS", "sentiment_corpus", "event_corpus", "load_stocktwits"]
