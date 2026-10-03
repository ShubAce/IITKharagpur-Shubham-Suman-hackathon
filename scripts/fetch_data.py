"""Download the public datasets used to train, calibrate and evaluate TREMOR.

Everything lands in ``data/raw/`` which is git-ignored: the repository only ships the
small, curated files the prototype needs to run (see ``data/README.md``).

Usage:
    python scripts/fetch_data.py                 # everything
    python scripts/fetch_data.py --only fpb tfns # a subset
    python scripts/fetch_data.py --list

Kaggle datasets need a Kaggle API token (``~/.kaggle/kaggle.json``); Hugging Face
datasets are public and need no token.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


@dataclass(frozen=True)
class Dataset:
    name: str
    kind: str  # "kaggle" | "hf"
    repo: str
    purpose: str
    files: tuple[str, ...] = field(default_factory=tuple)  # hf only; empty = whole kaggle dataset


DATASETS: tuple[Dataset, ...] = (
    # --- named in the hackathon brief -------------------------------------------------
    Dataset("fpb", "kaggle", "ankurzing/sentiment-analysis-for-financial-news",
            "Financial PhraseBank: 4.8k news sentences with analyst-agreed sentiment"),
    Dataset("tweet_returns", "kaggle", "thedevastator/tweet-sentiment-s-impact-on-stock-returns",
            "Stock tweets with forward returns: impact calibration + Module A backtest"),
    Dataset("transactions", "kaggle", "computingvictor/transactions-fraud-datasets",
            "Financial Transactions Dataset: seeds the Module B synthetic portfolio",
            files=("mcc_codes.json", "users_data.csv", "cards_data.csv", "transactions_data.csv")),
    # --- augmentations ---------------------------------------------------------------
    Dataset("sentfin", "kaggle", "ankurzing/aspect-based-sentiment-analysis-for-financial-news",
            "SEntFiN: 10.7k headlines with per-entity sentiment"),
    Dataset("stock_tweets", "kaggle", "equinxx/stock-tweets-for-sentiment-analysis-and-prediction",
            "80k ticker-tagged tweets, Sep 2021 - Sep 2022: social leg of crisis replays"),
    Dataset("tfns", "hf", "zeroshot/twitter-financial-news-sentiment",
            "Finance tweets with bearish/bullish/neutral labels",
            files=("sent_train.csv", "sent_valid.csv")),
    Dataset("tfn_topic", "hf", "zeroshot/twitter-financial-news-topic",
            "Finance tweets with 20 topic labels: seeds the event classifier",
            files=("topic_train.csv", "topic_valid.csv")),
    Dataset("fiqa", "hf", "TheFinAI/fiqa-sentiment-classification",
            "FiQA 2018 task 1: target-level sentiment on a continuous -1..1 scale",
            files=("data/train-00000-of-00001-aeefa1eadf5be10b.parquet",
                   "data/valid-00000-of-00001-51867fe1ac59af78.parquet",
                   "data/test-00000-of-00001-0fb9f3a47c7d0fce.parquet")),
    Dataset("benzinga", "hf", "ashraq/financial-news",
            "1.8M ticker-tagged headlines 2009-2020: weak labels for the event head",
            files=("data/train-00000-of-00001-8ec327f23bbe0948.parquet",)),
    Dataset("gdelt_sample", "gdelt", "gdeltproject.org (raw GKG 2.0)",
            "Headlines from 40 random 15-minute slots, 2019-2023: general-news negatives and weak labels"),
)

# Windows used for crisis replays. Training data never samples from them, so the demo is
# always run on text the models have not seen.
REPLAY_BLACKOUTS = (("2022-02-17", "2022-03-03"), ("2023-03-06", "2023-03-17"))


def fetch_kaggle(ds: Dataset, dest: Path) -> None:
    exe = shutil.which("kaggle")
    if exe is None:
        raise RuntimeError("kaggle CLI not found: pip install kaggle and add ~/.kaggle/kaggle.json")
    if ds.files:
        for f in ds.files:
            subprocess.run([exe, "datasets", "download", "-d", ds.repo, "-f", f, "-p", str(dest)], check=True)
        for z in dest.glob("*.zip"):
            shutil.unpack_archive(z, dest)
            z.unlink()
    else:
        subprocess.run([exe, "datasets", "download", "-d", ds.repo, "-p", str(dest), "--unzip"], check=True)


def fetch_hf(ds: Dataset, dest: Path) -> None:
    from huggingface_hub import hf_hub_download

    for f in ds.files:
        hf_hub_download(repo_id=ds.repo, filename=f, repo_type="dataset", local_dir=dest)


def fetch_gdelt_sample(ds: Dataset, dest: Path, n_slots: int = 40, seed: int = 7) -> None:
    """Every headline from ``n_slots`` randomly chosen 15-minute GDELT slots (reproducible via the seed)."""
    import json
    import random
    from datetime import datetime, timedelta, timezone

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tremor.ingestion.gdelt import fetch_slot, iter_gkg

    rng = random.Random(seed)
    first, last = datetime(2019, 1, 1, tzinfo=timezone.utc), datetime(2023, 12, 31, tzinfo=timezone.utc)
    blackouts = [(datetime.fromisoformat(a).replace(tzinfo=timezone.utc), datetime.fromisoformat(b).replace(tzinfo=timezone.utc))
                 for a, b in REPLAY_BLACKOUTS]
    done = 0
    with (dest / "titles.jsonl").open("w", encoding="utf-8") as fh:
        while done < n_slots:
            slot = first + timedelta(minutes=15 * rng.randrange(int((last - first).total_seconds() // 900)))
            if any(a <= slot <= b for a, b in blackouts):
                continue
            payload = fetch_slot(slot)
            if payload is None:
                continue
            n = 0
            for art in iter_gkg(payload):
                fh.write(json.dumps({"published_at": art.published_at.isoformat(), "domain": art.domain, "title": art.title,
                                     "themes": sorted(art.themes)[:40]}, ensure_ascii=False) + "\n")
                n += 1
            done += 1
            print(f"  slot {done:2d}/{n_slots}  {slot:%Y-%m-%d %H:%M}  {n} headlines", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", help="dataset names to fetch (default: all)")
    parser.add_argument("--skip", nargs="*", default=[], help="dataset names to skip")
    parser.add_argument("--list", action="store_true", help="list datasets and exit")
    parser.add_argument("--force", action="store_true", help="re-download even if the folder is not empty")
    args = parser.parse_args()

    if args.list:
        for ds in DATASETS:
            print(f"{ds.name:14s} {ds.kind:6s} {ds.repo}\n{'':14s} {ds.purpose}")
        return 0

    known = {ds.name for ds in DATASETS}
    unknown = set(args.only or []) | set(args.skip)
    if unknown - known:
        parser.error(f"unknown dataset(s): {sorted(unknown - known)}; choose from {sorted(known)}")

    failed: list[str] = []
    for ds in DATASETS:
        if (args.only and ds.name not in args.only) or ds.name in args.skip:
            continue
        dest = RAW_DIR / ds.name
        if dest.exists() and any(dest.iterdir()) and not args.force:
            print(f"[skip] {ds.name}: already in {dest}")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[fetch] {ds.name} <- {ds.kind}:{ds.repo}")
        try:
            {"kaggle": fetch_kaggle, "hf": fetch_hf, "gdelt": fetch_gdelt_sample}[ds.kind](ds, dest)
        except Exception as exc:  # keep going: one unavailable source must not block the rest
            failed.append(ds.name)
            print(f"[FAIL] {ds.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if failed:
        print(f"\n{len(failed)} dataset(s) failed: {failed}", file=sys.stderr)
        return 1
    print("\nAll requested datasets are in", RAW_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
