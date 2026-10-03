# TREMOR: Text-driven Risk Engine for Market Observation & Response - S&P Global & Crisil Campus Hackathon

**Candidate Name:** Shubham Suman
**College Email ID:** shubhamsuman@kgpian.iitkgp.ac.in
**College / Campus:** IIT Kharagpur
**Demo Video Link:** _to be added (YouTube, unlisted)_
**Slide Deck Link (if hosted externally):** [`docs/presentation.pdf`](docs/presentation.pdf) (in this repository)

---

## 1. Project Overview / Problem Statement & Approach

**The problem.** Markets move on news before they move on numbers, but news arrives as an unstructured,
massively duplicated stream: the same wire story is reprinted by hundreds of outlets, most of the
global news flow is irrelevant to any portfolio, and social media is fast but noisy. A risk team needs
to know, within minutes, *which* developments matter, *to whom*, *how much* - and what they mean for its
positions. The brief asks for an AI/NLP engine that turns this text into machine-readable risk signals
(sentiment, event type, impact) and for applications that act on them.

**The approach.** TREMOR is built around three ideas:

1. **Events, not articles.** Documents are clustered into *stories* (one report and its reprints) and
   stories into *events* (one situation). Fifty outlets repeating a headline become one event with fifty
   corroborations, which fixes double counting and turns repetition into evidence. One fine-tuned encoder,
   exported to ONNX and run on CPU, produces four outputs per text in one pass - sentiment, event type,
   entity-level sentiment and the embedding used for clustering.
2. **An impact score that explains itself.** Impact (1-10) is a points-based scorecard, the way credit
   analysts build ratings: a base severity for the event type plus points for severe language,
   *independent* corroboration, velocity, breadth and market linkage, minus points for single-source or
   social-only stories. Every score comes with its line items, and it rises as an event develops, so one
   alarming tweet cannot trigger a stress test.
3. **Applications that behave like the real thing.** Module A is written as an index methodology
   (parent index, tilt, single-stock and sector caps, turnover control, costs, a benchmark); Module B is
   a credit-risk stress test (historical-analog scenarios, Vasicek-shifted PDs, IFRS 9 expected credit
   loss, Basel IRB risk-weighted assets, CET1) on a synthetic wholesale book whose loan sleeve is derived
   from the 13.3M card transactions of the dataset named in the brief.

Both modules are built and they *subscribe* to the engine's signals, as the brief specifies: Module A to
entity sentiment, Module B to event type and impact. Everything can be replayed on real history - the
demo replays the Russian invasion of Ukraine (21-24 Feb 2022, 39k documents) through exactly the code that
runs on live feeds.

## 2. Architecture & Tech Stack

![Architecture](docs/architecture.png)

**Data flow.**

1. **Sources** - GDELT 2.0 raw 15-minute files (global news, no key), RSS (Yahoo Finance per ticker, Google
   News topics, Federal Reserve), StockTwits and Bluesky (social; the X API is paid), replay packs of real
   history, Yahoo Finance prices.
2. **Gate** - a document must name a tracked entity (or carry risk content) to be analysed; this cuts a
   global news firehose by roughly an order of magnitude at zero model cost.
3. **Engine** - de-duplication by text fingerprint; entity linking over 80 entities (companies,
   countries, central banks, commodities) with an ambiguity guard ("Ford" only counts with a finance cue);
   the multi-task encoder; rule cues as a transparent prior; online story and event clustering; the impact
   scorecard; a per-entity Bayesian sentiment state that decays to neutral and reports its uncertainty.
4. **Signals** - `DocSignal`, `EventSignal`, `EntitySignal` (Pydantic schemas in `src/tremor/schemas.py`),
   published on an in-process bus, served by a REST API with a live server-sent-event stream, and appended
   to `data/output/signals.jsonl`.
5. **Modules** - A rebalances the TREMOR-20 index; B stress-tests the book when an event crosses the
   threshold. A dashboard shows all of it live.

**Example signal** (abridged; the top event of the replay, end of 24 Feb 2022):

```json
{"scope": "event", "event_type": "GEOPOLITICAL", "impact_score": 9.9, "sentiment_score": -0.09,
 "headline": "Russian troops try to seize Chernobyl nuclear plant amid…", "regions": ["RU", "UA", "GB", "US"],
 "n_docs": 11344, "n_stories": 2522, "n_publishers": 1937,
 "impact_factors": [
   {"name": "Event type",     "points": 5.0,  "detail": "Geopolitical: base severity"},
   {"name": "Intensity",      "points": 1.15, "detail": "extreme-severity language in 3917/9825 reports; amounts in the billions"},
   {"name": "Corroboration",  "points": 2.0,  "detail": "2522 independent reports across 1937 publishers"},
   {"name": "Velocity",       "points": 1.0,  "detail": "46 new reports in the last hour"},
   {"name": "Breadth",        "points": 0.25, "detail": "9 countries/regions named"},
   {"name": "Market linkage", "points": 0.5,  "detail": "520 reports (5%) discuss markets, prices or listed companies"}]}
```

**Tech stack and why.**

| Layer      | Choice                                                                                                    | Why                                                                                                                                                             |
| ---------- | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Text model | `BAAI/bge-small-en-v1.5` fine-tuned multi-task (PyTorch, training only) -> ONNX int8                    | Frozen encoders plateau around 0.80 on financial sentiment (8 candidates tested); fine-tuning one small model beats FinBERT on 4 of 5 test sets at 5x its speed |
| Inference  | ONNX Runtime + Hugging Face`tokenizers`                                                                 | 34 MB model, no PyTorch or GPU needed to run; ~4 ms per headline                                                                                                |
| Service    | FastAPI + Uvicorn, server-sent events                                                                     | typed schemas, auto-generated API docs at`/docs`, live push to the dashboard                                                                                  |
| Numerics   | NumPy, pandas, SciPy                                                                                      | vectorised revaluation of the whole book in milliseconds                                                                                                        |
| Data       | GDELT, feedparser + requests, yfinance, Kaggle / Hugging Face datasets                                    | free, keyless where possible, reproducible scripts                                                                                                              |
| Dashboard  | plain HTML + ES modules + hand-built SVG charts                                                           | no build step, no CDN: works offline during a live pitch                                                                                                        |
| Quality    | pytest (76 tests incl. the full replay through the API and a reproducibility check), ruff, GitHub Actions |                                                                                                                                                                 |

**API** (interactive docs at `http://127.0.0.1:8000/docs`): `GET /api/events`, `/api/events/{id}`,
`/api/entities`, `/api/entities/{id}`, `/api/documents`, `/api/stream` (SSE), `POST /api/analyze`,
`GET /api/index`, `/api/index/history`, `/api/index/rebalances`, `/api/backtest`, `GET /api/portfolio`,
`/api/stress/runs`, `/api/stress/runs/{id}`, `POST /api/stress/run`, `/api/replay/*`, `/api/evaluation`.

## 3. Dataset Used

All data is public or synthetic; no proprietary or client data is used. Full data card, sources,
licences and leakage controls: [`data/README.md`](data/README.md).

| Purpose               | Data                                                                                                                                                                                                                                            |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Live demo (replay)    | 38,366 GDELT news headlines + 876 timestamped tweets, 21-24 Feb 2022 (`data/replay/`)                                                                                                                                                         |
| Module B portfolio    | Kaggle*Financial Transactions Dataset* (13.3M card transactions, named in the brief) -> cash-flow profiles of 120 merchants -> mid-market loan book; plus large-corporate loans, bonds and derivatives (synthetic, `data/portfolio/`)       |
| Module B scenarios    | 26 historical stress episodes (2001-2025) x 26 risk factors measured from Yahoo Finance (`data/scenarios/`)                                                                                                                                   |
| Module A backtest     | ~49k dated company headlines (Google News archive) + ~63k timestamped tweets, Oct 2021 - Sep 2022; daily prices (`data/prices/`, `data/backtest/`)                                                                                          |
| Training / evaluation | Financial PhraseBank (named in the brief), Twitter Financial News sentiment and topic, FiQA-2018, SEntFiN 1.1, 4k self-labelled StockTwits posts, weak labels from news searches and GDELT (fetched by`scripts/fetch_data.py`, not committed) |

**Key assumptions.** Merchant facility size = 40x annual card turnover (card sales are only part of a
company's revenue); ratings of real companies and all exposure sizes are illustrative; credit-spread moves
of historical episodes are proxied from bond ETFs; backtest headlines are stamped at the end of their day
so nothing is traded before it could have been read.

## 4. Quickstart & Installation

Runtime: **Python 3.12** (3.10+ should work) on **Windows 11** (tested); CPU only, no API keys.

```bash
git clone <your-repo-url>
cd IITKharagpur-Shubham-Suman-hackathon
python -m venv .venv
.venv\Scripts\activate            # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py                    # dashboard on the Feb-2022 crisis replay -> http://127.0.0.1:8000
```

| Command                                                  | What it does                                                                                          |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `python main.py`                                       | API + dashboard, replaying 21-24 Feb 2022 at 30 min of history per second (speed and pause in the UI) |
| `python main.py serve --mode live`                     | the same on live feeds (GDELT, RSS, StockTwits, Bluesky, live prices)                                 |
| `python main.py replay`                                | run the whole replay headless and print the events, stress tests and index moves                      |
| `python main.py analyze "Moody's cuts Boeing to junk"` | analyse one text from the command line                                                                |
| `pytest`                                               | 76 tests: entity linking, index methodology, credit math, stress engine, pipeline, API                |
| `python main.py evaluate`                              | accuracy and speed of every model on held-out data                                                    |
| `python main.py backtest --rebuild`                    | recompute the one-year Module A backtest                                                              |

Rebuilding everything from public data (optional): `pip install -r requirements-dev.txt`, then
`python scripts/fetch_data.py`, `python main.py finetune` (8 minutes on a laptop GPU),
`python main.py evaluate`, `python scripts/build_portfolio.py`, `python scripts/build_analog_library.py`.

**Suggested 5-minute demo.** Risk radar: watch the Russia-Ukraine event climb from 7.7 to 9.9 and open its
scorecard -> Stress lab: the escalation ladder of stress tests, the scenario's historical analogs, the
before/after value and CET1 -> edit a shock and re-run -> Index rebalancer: JPMorgan underweighted on
sanctions news, the weight heatmap, the rebalance log with the headline behind each trade -> Analyze text:
paste a headline naming two companies with opposite news -> Model & results.

## 5. Key Results & Domain Impact

Full tables, generated from the result files: [`docs/RESULTS.md`](docs/RESULTS.md).

**NLP engine - held-out test sets, real inference path (int8 ONNX on CPU).**

| Task                                        | Keyword baseline (naive) | Frozen encoder + head | FinBERT | **TREMOR** |
| ------------------------------------------- | -----------------------: | --------------------: | ------: | ---------------: |
| News sentiment, PhraseBank (acc.)           |                    0.665 |                 0.765 |  0.879* |  **0.830** |
| Tweet sentiment, TFNS (acc.)                |                    0.703 |                 0.763 |   0.714 |  **0.867** |
| Indian financial news, SEntFiN (acc.)       |                    0.673 |                 0.745 |   0.726 |  **0.870** |
| Target-level sentiment, FiQA (acc.)         |                    0.485 |                 0.665 |   0.533 |  **0.709** |
| StockTwits self-labels, n = 792 (direction) |                    0.420 |                 0.770 |   0.646 |  **0.814** |
| Event type, human labels (macro-F1)         |                    0.377 |                 0.778 |       - |  **0.880** |
| Opposite-sentiment entities in one headline |                    0.453 |                 0.458 |       - |  **0.850** |
| Throughput, docs/s (laptop CPU)             |                    5,218 |                   492 |      61 |    **307** |

\* FinBERT was trained on PhraseBank, so its PhraseBank score is partly in-sample.

**Efficiency vs. the naive and the standard approach.** On the four-day crisis replay the engine processed
39,242 documents in 90-100 s on a laptop CPU (end to end, including clustering and both modules), folded 6,692 syndicated
copies into corroboration and discarded 7,834 as noise. Rolling articles up into events turned 39k
documents into scored events of which only **14** warranted a stress test, on a 34 MB model that needs
neither a GPU nor an API key. (Counts are from the headless replay, `python main.py replay`; in the dashboard
the replay is fed in time-sliced batches whose boundaries depend on the replay speed, so the number of stress
tests can differ by one or two.)

**Module B on the crisis.** The Russia-Ukraine situation was stress-tested as it escalated: impact 7.7 on
21 Feb, 8.8 when the US moved to cut Russian banks off, re-runs as coverage doubled around Putin's
recognition of the separatist regions (9.2), on the invasion itself - *"Putin Announces Military Assault
Against Ukraine in Surprise Speech"*, 24 Feb 05:45 UTC, impact 9.4 - and as coverage reached 539 reports
an hour (9.9). Each run shows value before/after, P&L by risk channel, ECL, downgrades (Sberbank and Gazprom
to default at impact 9+, while the bank's CDS protection on Gazprom pays out) and CET1 (13.5% -> 11.6-11.9%).
Built strictly from analogs that ended before the invasion, the scenario got the direction of **7 of 9**
risk factors right (HY spreads +59 bp vs +55 bp realised); it missed oil and the euro, because no earlier
episode involved a war-driven supply shock from a major commodity exporter - stated, not hidden.

**Module A backtest (Oct 2021 - Sep 2022, bear market, 20 stocks, 5 bp costs).** TREMOR-20 returned
-19.6% vs -19.9% for the equal-weight benchmark (+0.32% a year, information ratio 0.33); the naive
keyword-sentiment tilt lost 0.23% a year. Neither signal's daily information coefficient is statistically
distinguishable from zero over one year (TREMOR +0.006, t = 0.3; naive +0.012, t = 0.7), and an earlier
training run of the same model gave +0.63% - so the honest reading is that one year of 20 stocks cannot
establish alpha either way. The backtest demonstrates an investable, cost-aware, look-ahead-free
methodology; proving a durable signal needs a longer, broader sample.

**Domain impact.** For an index provider, the methodology shows how a sentiment overlay can be run as a
rules-based, capped, low-turnover index with every weight change traceable to its source - the governance
an index committee needs. For a credit-risk and ratings business, the engine is an early-warning system:
on the replay it flagged credit events (the Credit Suisse "Suisse Secrets" leak, HSBC's China-property loss,
Chinese developer default warnings) and tracked the geopolitical escalation hour by hour, and it converts them into scenario-consistent, explainable
estimates of losses, provisions and capital - the questions a CRO asks the morning a crisis breaks. The
entity-level sentiment, trained partly on Indian financial news (SEntFiN), and the INR / Indian
sovereign / Indian corporate exposures make it directly relevant to Indian markets.

**Limitations and next steps.** Ratings and exposures are illustrative; analog scenarios inherit the gaps of
history (the oil miss above) - next: condition the analog search on the commodities an event names and add
an LLM "scenario reviewer" for narrative and plausibility checks; the backtest covers one year; RSS and
search archives give dates but not exact times; the in-process bus would become Kafka or Redis Streams in
production; GDELT coverage is English-language here, while GDELT itself covers 100+ languages (a
multilingual encoder is the next step).

---

**AI usage.** This project was built with AI assistance (Anthropic Claude) for code and documentation, in
line with the hackathon's AI-usage guideline. Design decisions, data, model training and every reported
number are reproducible from the scripts in this repository.

**Repository layout.**

```
main.py                    entry point (serve | replay | analyze | evaluate | backtest | train | finetune)
configs/                   universe (entities), taxonomy (events, cues), scenarios (factors, episodes), settings
src/tremor/
  ingestion/               GDELT, RSS, StockTwits, Bluesky, replay, prices
  nlp/                     entity linking, cues, encoder, models, impact scorecard
  engine/                  clustering, sentiment state, pipeline, runtime (bus, store)
  modules/rebalancer/      Module A: methodology, live index, backtest
  modules/stress/          Module B: portfolio, credit math, scenarios, stress engine
  training/                corpora, fine-tuning, evaluation
  api/                     FastAPI app + dashboard (static)
models/tremor-encoder/     the fine-tuned model (int8 ONNX, 34 MB) and its metadata
data/                      everything the prototype runs on (see data/README.md)
scripts/                   data collection, replay / portfolio / analog builders, benchmarks, diagrams
docs/                      architecture.png, RESULTS.md, results/*.json, screenshots/, presentation.pdf
tests/                     pytest suite
```

Licence: MIT.
