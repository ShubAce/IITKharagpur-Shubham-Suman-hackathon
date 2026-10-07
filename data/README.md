# Data card

Everything the prototype needs to run is in this folder. Everything here is either **public**
(GDELT, Yahoo Finance prices, public news headlines, a public Kaggle tweet archive, public Hacker
News posts, published rating actions as reported in the press) or **synthetic** (the bank
portfolio). No proprietary, confidential or client data is used.

Bulk raw downloads used for *training and calibration* are not committed (they are large and
some are licensed for redistribution only from their original source). They are reproducible
with `python scripts/fetch_data.py` into `data/raw/` (git-ignored).

## Files used at runtime

| Path | What it is | Source | How it was made |
|---|---|---|---|
| `replay/ukraine_2022/news.jsonl.gz` | 38,366 news headlines, 21-24 Feb 2022 (the Russian invasion of Ukraine) | GDELT 2.0 Global Knowledge Graph, raw 15-minute files | `scripts/build_replay.py`: every 15-minute slot downloaded, filtered by the entity/risk gate, at most 2 syndicated copies kept per headline |
| `replay/ukraine_2022/social.jsonl.gz` | 876 timestamped stock tweets from the same window | Kaggle `equinxx/stock-tweets-for-sentiment-analysis-and-prediction` | same script |
| `replay/svb_2023/news.jsonl.gz` | 39,240 news headlines, 8-15 Mar 2023 (Silicon Valley Bank, Signature Bank, Credit Suisse) | GDELT 2.0, raw 15-minute files | `scripts/build_replay.py`, same gate, at most 2 copies per headline |
| `replay/svb_2023/social.jsonl.gz` | 4,489 timestamped Hacker News stories and comment excerpts naming the banks of the run (the tweet archive ends in Sep 2022, the X API is paid, and StockTwits removed delisted tickers such as $SIVB) | Algolia Hacker News Search API (keyless) | `scripts/collect_hn.py`: typo-tolerance off, each comment cut to the sentence naming a crisis bank |
| `reference/rating_actions.csv` | 11 public rating actions on Russia (Feb-Mar 2022) and on SVB, Signature, First Republic, Western Alliance (Mar 2023), each with the URL of the report it comes from | press reports and S&P research pages | hand-entered; used only to measure the watchlist's lead time |
| `prices/constituents_daily.csv` | Daily adjusted open/close of the 20 index stocks and the S&P 500, Aug 2021 - Mar 2023 | Yahoo Finance via `yfinance` | `scripts/fetch_prices.py` |
| `portfolio/merchant_profiles.csv` | Cash-flow credit profiles of 120 merchants (turnover, volatility, growth, refund and decline rates, customer count, derived rating) | Kaggle *Financial Transactions Dataset* (`computingvictor/transactions-fraud-datasets`), 13.3M card transactions - the transaction dataset named in the brief | `scripts/build_portfolio.py` (step 1) |
| `portfolio/positions.csv` | The synthetic wholesale book: 388 positions - loans, revolving credit facilities, bonds, interest-rate swaps, FX forwards, CDS, equity total-return swaps, an oil swap | synthetic (fixed seed) | `scripts/build_portfolio.py` (step 2) |
| `scenarios/analog_shocks.csv` | Measured moves of 26 risk factors during 26 historical stress episodes (2001-2025); with each episode's description and day-one headline (`configs/scenarios.yaml`) it is also the answer key of the point-in-time scenario backtest (`python main.py validate`) | Yahoo Finance via `yfinance` | `scripts/build_analog_library.py` |
| `backtest/signals_daily.csv` | The engine's sentiment state for each index stock at every market open, Oct 2021 - Sep 2022 | derived | `python main.py backtest --rebuild` |
| `backtest/signals_daily_naive.csv` | The same, from the naive keyword baseline | derived | same |
| `backtest/impact_daily.csv` | Per company and day, Oct 2021 - Sep 2022: the highest impact of an event about the company (fine-tuned and keyword engines), tone intensity, headline and post counts - the input of the impact event study | derived | `python main.py impact --rebuild` |

## Data used to train, calibrate and evaluate (fetched, not committed)

| Name | Records | Used for | Source |
|---|---|---|---|
| Financial PhraseBank | 4,846 news sentences, 5-8 annotators each | sentiment (train / test) | Kaggle `ankurzing/sentiment-analysis-for-financial-news` (named in the brief) |
| Twitter Financial News Sentiment | 11,931 tweets | sentiment (train / test) | Hugging Face `zeroshot/twitter-financial-news-sentiment` |
| Twitter Financial News Topic | 21,107 tweets, 20 topics | event type (human-labelled test set) | Hugging Face `zeroshot/twitter-financial-news-topic` |
| FiQA-2018 task 1 | 1,173 headlines / posts with a target and a -1..1 score | entity-level sentiment | Hugging Face `TheFinAI/fiqa-sentiment-classification` |
| SEntFiN 1.1 | 10,753 headlines, 14,409 (headline, entity, sentiment) triples | entity-level sentiment | Kaggle `ankurzing/aspect-based-sentiment-analysis-for-financial-news` |
| StockTwits | ~10k public posts, ~35% self-tagged Bullish/Bearish by their author | social sentiment (train / test) | public StockTwits API, `scripts/collect_stocktwits.py` |
| Tweet Sentiment's Impact on Stock Returns | 862k brand tweets with forward returns | named in the brief; used to study noise (e.g. "Ford" = Christine Blasey Ford, Sept 2018) | Kaggle `thedevastator/tweet-sentiment-s-impact-on-stock-returns` |
| Benzinga partner headlines | 1.8M headlines 2009-2020 | weak (rule-derived) event-type labels | Hugging Face `ashraq/financial-news` |
| Class-specific news searches | 12,747 headlines | weak event-type labels for rare classes (credit events, incidents, conflict) | Google News RSS, `scripts/collect_event_headlines.py` |
| GDELT random sample | 42,657 headlines from 40 random 15-minute slots, 2019-2023 | general-news "noise" examples | GDELT, `scripts/fetch_data.py --only gdelt_sample` |
| Backtest news | 49k dated company headlines, Oct 2021 - Sep 2022 | Module A backtest | Google News RSS archive search, `scripts/collect_backtest_news.py` |

**Leakage controls.** Every dataset has a fixed, hash-based split. Training rows whose text also
appears in *any* test split are dropped. Weak labels are used for training only, never reported as
accuracy. The training sample of GDELT excludes both replay windows (17 Feb - 3 Mar 2022 and
6 - 17 Mar 2023, `REPLAY_BLACKOUTS` in `scripts/fetch_data.py`), so the replays run on text the model
has never seen. Checked for the 2023 pack: no training text falls inside its window; 12 of the 42,657
general-news training headlines, all from later dates (25 Mar - 11 May 2023), mention SVB, Signature
Bank or First Republic.

**Day-one headlines for the scenario backtest** (`headline` in `configs/scenarios.yaml`) state only
the trigger of each past crisis ("Britain votes to leave the European Union"), never its market
outcome, so the generator cannot read the answer from its input.

## Assumptions

- **Portfolio sizing.** Card turnover in the transaction dataset is a fraction of a company's
  revenue, so each merchant's facility = 40x its annual card turnover (capped at $150m). Ratings come
  from quintiles of a cash-flow stability score (mostly BB/B, as mid-market borrowers are).
- **Large-corporate ratings and exposure sizes are illustrative**, not any agency's or bank's.
  The rating-to-PD master scale is a generic long-run average.
- **Credit spreads for analog episodes** are proxied from bond ETFs (LQD, HYG, EMB) relative to
  duration-matched Treasury ETFs, because index OAS histories are licensed.
- **Backtest headlines have reliable dates but not times**; they are stamped at 23:59 UTC so they are
  only tradable from the next day's open (no look-ahead).
- **Rating actions have dates, not times**, so the watchlist's lead time is counted in calendar days.
- **Hacker News stands in for social media in the 2023 replay.** It is a technology community rather
  than a finance one, which suits this crisis (Silicon Valley Bank was the startup world's bank) but
  would not suit every crisis.
