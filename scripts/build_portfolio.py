"""Build Module B's synthetic wholesale portfolio.

Step 1 (needs the raw Kaggle *Financial Transactions* dataset, ~1.3 GB): aggregate 13M card
transactions into cash-flow credit profiles of the top merchants -> data/portfolio/merchant_profiles.csv
Step 2: generate the full book -> data/portfolio/positions.csv

The repository ships both outputs, so step 1 is only needed to rebuild from scratch.

    python scripts/build_portfolio.py              # both steps (raw data required)
    python scripts/build_portfolio.py --reuse      # step 2 only, from the committed profiles
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from tremor.config import load_universe  # noqa: E402
from tremor.modules.stress.portfolio import (  # noqa: E402
    POSITIONS_PATH,
    PROFILES_PATH,
    build_merchant_profiles,
    generate_portfolio,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reuse", action="store_true", help="skip step 1 and reuse the committed merchant profiles")
    args = parser.parse_args()

    PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if args.reuse:
        profiles = pd.read_csv(PROFILES_PATH)
    else:
        t0 = time.time()
        profiles = build_merchant_profiles()
        profiles.to_csv(PROFILES_PATH, index=False)
        print(f"merchant profiles: {len(profiles)} borrowers in {time.time() - t0:.0f}s -> {PROFILES_PATH}")
        print(profiles.groupby(["sector", "rating"]).size().unstack(fill_value=0))
    book = generate_portfolio(profiles, load_universe())
    book.to_csv(POSITIONS_PATH, index=False)
    print(f"\nportfolio: {len(book)} positions -> {POSITIONS_PATH}")
    print(book.groupby("asset_class")["notional"].agg(["count", "sum"]).assign(sum=lambda d: (d["sum"] / 1e9).round(2))
          .rename(columns={"sum": "notional_usd_bn"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
