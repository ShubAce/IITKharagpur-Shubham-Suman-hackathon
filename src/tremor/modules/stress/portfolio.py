"""The synthetic wholesale banking book that Module B stresses.

Four sleeves, ~USD 25bn in total, generated deterministically (fixed seed):

1. Mid-market lending - derived from the brief's *Financial Transactions* dataset. Each of the
   top merchants by card turnover becomes a corporate borrower: facility size is scaled from its
   turnover, and its rating from the *stability* of that turnover (volatility, trend, customer
   concentration, refund and decline rates). This is transaction-based credit assessment, the way
   banks use cash-flow data to underwrite small and mid-sized companies.
2. Large-corporate lending - term loans and revolving credit facilities to the companies the
   engine tracks, so a news event about a company maps straight onto the book's exposure to it.
3. Securities - sovereign bonds (liquidity buffer) and investment-grade / high-yield corporates.
4. Derivatives - interest-rate swaps hedging the bond book, FX forwards hedging non-USD assets,
   CDS (protection bought on large exposures, some sold), equity total-return swaps, an oil swap.

All names, sizes and ratings are fictitious and for demonstration only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tremor.config import Universe
from tremor.paths import DATA_DIR, RAW_DIR

PORTFOLIO_DIR = DATA_DIR / "portfolio"
POSITIONS_PATH = PORTFOLIO_DIR / "positions.csv"
PROFILES_PATH = PORTFOLIO_DIR / "merchant_profiles.csv"

# Merchant category code (MCC) ranges -> GICS-style sector.
MCC_SECTORS = [
    ((4000, 4799), "Industrials"), ((4800, 4899), "Communication Services"), ((4900, 4999), "Utilities"),
    ((5000, 5199), "Industrials"), ((5200, 5299), "Consumer Discretionary"), ((5300, 5399), "Consumer Staples"),
    ((5400, 5499), "Consumer Staples"), ((5500, 5539), "Consumer Discretionary"), ((5540, 5549), "Energy"),
    ((5550, 5999), "Consumer Discretionary"), ((6000, 6999), "Financials"), ((7000, 7999), "Consumer Discretionary"),
    ((8000, 8099), "Health Care"), ((8100, 8999), "Industrials"), ((3000, 3999), "Industrials"),
]
REGION_OF_COUNTRY = {"US": "US", "GB": "UK", "DE": "EU", "FR": "EU", "CH": "EU", "EU": "EU", "IN": "IN",
                     "CN": "EM", "TW": "EM", "RU": "EM", "SA": "EM", "TR": "EM", "JP": "JP"}
CURRENCY_OF_REGION = {"US": "USD", "UK": "GBP", "EU": "EUR", "IN": "INR", "EM": "USD", "JP": "USD"}


def mcc_sector(mcc: int) -> str:
    for (lo, hi), sector in MCC_SECTORS:
        if lo <= mcc <= hi:
            return sector
    return "Consumer Discretionary"


# --------------------------------------------------------------------------- sleeve 1: from transactions
def build_merchant_profiles(raw_dir: Path = RAW_DIR / "transactions", top_n: int = 120) -> pd.DataFrame:
    """Aggregate 13M card transactions into a cash-flow credit profile per merchant."""
    cols = ["date", "client_id", "amount", "merchant_id", "merchant_city", "merchant_state", "mcc", "errors"]
    parts = []
    for chunk in pd.read_csv(raw_dir / "transactions_data.csv", usecols=cols, chunksize=2_000_000):
        chunk["amount"] = chunk["amount"].str.replace("$", "", regex=False).astype(float)
        chunk["month"] = chunk["date"].str[:7]
        chunk["refund"] = chunk["amount"] < 0
        chunk["error"] = chunk["errors"].notna()
        parts.append(chunk.groupby(["merchant_id", "month"]).agg(
            turnover=("amount", lambda a: a[a > 0].sum()), refunds=("refund", "sum"), errors=("error", "sum"),
            n=("amount", "size"), mcc=("mcc", "first"), city=("merchant_city", "first"), state=("merchant_state", "first")))
    monthly = pd.concat(parts).groupby(level=[0, 1]).agg(
        turnover=("turnover", "sum"), refunds=("refunds", "sum"), errors=("errors", "sum"), n=("n", "sum"),
        mcc=("mcc", "first"), city=("city", "first"), state=("state", "first")).reset_index()
    clients = pd.concat(
        pd.read_csv(raw_dir / "transactions_data.csv", usecols=["merchant_id", "client_id"], chunksize=4_000_000)
    ).groupby("merchant_id")["client_id"].nunique()

    rows = []
    for merchant_id, g in monthly.groupby("merchant_id"):
        g = g.sort_values("month")
        if len(g) < 24:
            continue  # need two years of history to judge stability
        last12, prev12 = g["turnover"].iloc[-12:].sum(), g["turnover"].iloc[-24:-12].sum()
        rows.append({
            "merchant_id": int(merchant_id), "mcc": int(g["mcc"].iloc[0]), "city": g["city"].iloc[0], "state": g["state"].iloc[0],
            "months": len(g), "annual_turnover": last12,
            "volatility": float(g["turnover"].iloc[-36:].std() / max(g["turnover"].iloc[-36:].mean(), 1e-9)),
            "growth": float(last12 / prev12 - 1.0) if prev12 > 0 else 0.0,
            "refund_rate": float(g["refunds"].sum() / g["n"].sum()), "decline_rate": float(g["errors"].sum() / g["n"].sum()),
            "customers": int(clients.get(merchant_id, 0)),
        })
    prof = pd.DataFrame(rows).sort_values("annual_turnover", ascending=False).head(top_n).reset_index(drop=True)
    prof["sector"] = prof["mcc"].map(mcc_sector)
    mcc_names = json.loads((raw_dir / "mcc_codes.json").read_text(encoding="utf-8"))
    prof["category"] = prof["mcc"].astype(str).map(mcc_names).fillna("Other")

    # Cash-flow credit score: stable, growing, diversified, low-dispute merchants score higher.
    z = lambda s: (s - s.mean()) / (s.std() or 1.0)  # noqa: E731
    prof["credit_score"] = (-z(prof["volatility"]) + 0.5 * z(prof["growth"].clip(-0.5, 0.5))
                            + 0.5 * z(np.log1p(prof["customers"])) - 0.5 * z(prof["refund_rate"]) - 0.5 * z(prof["decline_rate"]))
    # Mid-market borrowers are mostly sub-investment-grade: map score quintiles to BBB .. B / CCC.
    grades = ["CCC", "B", "BB", "BB", "BBB"]
    quintile = pd.qcut(prof["credit_score"].rank(method="first"), 5, labels=False)
    prof["rating"] = [grades[q] for q in quintile]
    return prof


# --------------------------------------------------------------------------- generator
def _bond_analytics(coupon: float, years: float, yld: float) -> tuple[float, float]:
    """Modified duration and convexity of an annual-pay bullet bond."""
    t = np.arange(1, int(np.ceil(years)) + 1, dtype=float)
    t[-1] = years
    cf = np.full(len(t), coupon)
    cf[-1] += 1.0
    disc = (1 + yld) ** -t
    price = float((cf * disc).sum())
    macaulay = float((t * cf * disc).sum() / price)
    convexity = float((t * (t + 1) * cf * disc).sum() / (price * (1 + yld) ** 2))
    return macaulay / (1 + yld), convexity


def generate_portfolio(profiles: pd.DataFrame, universe: Universe, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ent = universe.by_id()
    rows: list[dict] = []

    def add(**kw) -> None:
        kw.setdefault("position_id", f"P{len(rows) + 1:04d}")
        rows.append(kw)

    # 1. Mid-market loans from transaction profiles. Card turnover covers only part of a company's
    #    revenue, so facility size scales turnover by an assumed multiple (documented in data/README.md).
    TURNOVER_TO_FACILITY = 40.0
    for p in profiles.itertuples():
        facility = float(np.clip(p.annual_turnover * TURNOVER_TO_FACILITY, 5e6, 150e6))
        name = f"Merchant {p.merchant_id} ({p.category[:34]}, {p.state if isinstance(p.state, str) else 'online'})"
        common = dict(obligor_id=f"M{p.merchant_id}", obligor_name=name, sector=p.sector, country="US", region="US",
                      currency="USD", rating=p.rating, lgd=0.40, source="transactions")
        add(asset_class="loan", notional=round(facility * 0.7, -3), drawn=round(facility * 0.7, -3), undrawn=0.0, ccf=0.0,
            maturity=float(rng.integers(3, 7)), fixed_float="float", coupon=0.0, direction=1, **common)
        add(asset_class="revolver", notional=round(facility * 0.3, -3), drawn=round(facility * 0.3 * rng.uniform(0.2, 0.5), -3),
            undrawn=0.0, ccf=0.5, maturity=3.0, fixed_float="float", coupon=0.0, direction=1, **common)
        rows[-1]["undrawn"] = rows[-1]["notional"] - rows[-1]["drawn"]

    # 2. Large-corporate lending to tracked companies (sizes and ratings illustrative).
    corporate_ratings = {"AAPL": "AA", "MSFT": "AAA", "NVDA": "A", "AMZN": "AA", "GOOGL": "AA", "META": "A", "TSLA": "BBB",
                         "NFLX": "BBB", "DIS": "A", "INTC": "BBB", "AMD": "A", "PG": "AA", "KO": "A", "COST": "A", "BA": "BBB",
                         "LMT": "A", "VZ": "BBB", "JPM": "A", "XOM": "AA", "JNJ": "AAA", "GS": "A", "BAC": "A", "C": "BBB",
                         "MS": "A", "WFC": "A", "DB": "BBB", "CS": "BBB", "HSBC": "A", "CVX": "AA", "SHEL": "A", "BP": "A",
                         "GAZP": "BB", "SBER": "BB", "PFE": "A", "UNH": "A", "WMT": "AA", "CAT": "A", "GE": "BBB", "RTX": "BBB",
                         "AIR": "A", "F": "BB", "GM": "BBB", "VOW": "BBB", "NKE": "AA", "PYPL": "A", "CRM": "A", "TSM": "AA",
                         "BABA": "A", "EVERG": "CCC", "RELIANCE": "BBB", "HDFCB": "BBB", "TATAMOT": "BB", "ADANI": "BB"}
    for entity_id, rating in corporate_ratings.items():
        e = ent[entity_id]
        region = REGION_OF_COUNTRY.get(e.country or "US", "EM")
        size = {"AAA": 250e6, "AA": 350e6, "A": 450e6, "BBB": 400e6, "BB": 250e6, "B": 120e6, "CCC": 80e6}[rating] * rng.uniform(0.6, 1.4)
        common = dict(obligor_id=entity_id, obligor_name=e.name, sector=e.sector, country=e.country, region=region,
                      currency=CURRENCY_OF_REGION[region], rating=rating, lgd=0.45, source="large_corporate")
        add(asset_class="loan", notional=round(size * 0.5, -5), drawn=round(size * 0.5, -5), undrawn=0.0, ccf=0.0,
            maturity=float(rng.integers(3, 8)), fixed_float=rng.choice(["float", "fixed"], p=[0.7, 0.3]),
            coupon=0.0, direction=1, **common)
        drawn_share = rng.uniform(0.05, 0.25)
        add(asset_class="revolver", notional=round(size * 0.5, -5), drawn=round(size * 0.5 * drawn_share, -5),
            undrawn=round(size * 0.5 * (1 - drawn_share), -5), ccf=0.4, maturity=5.0, fixed_float="float", coupon=0.0, direction=1, **common)

    # 3. Securities. Sovereigns are the liquidity buffer; corporates are a credit-spread book.
    sovereigns = [("UST 2Y", "US", "USD", 1.5e9, 2), ("UST 5Y", "US", "USD", 1.5e9, 5), ("UST 10Y", "US", "USD", 1.2e9, 10),
                  ("Bund 10Y", "DE", "EUR", 0.8e9, 10), ("Gilt 10Y", "GB", "GBP", 0.6e9, 10), ("India G-sec 10Y", "IN", "INR", 0.5e9, 10),
                  ("EM sovereign basket", "EM", "USD", 0.5e9, 8)]
    for name, country, currency, size, years in sovereigns:
        rating = {"US": "AA", "DE": "AAA", "GB": "AA", "IN": "BBB", "EM": "BB"}[country]
        coupon = {"USD": 0.04, "EUR": 0.025, "GBP": 0.04, "INR": 0.07}[currency] + (0.025 if country == "EM" else 0.0)
        duration, convexity = _bond_analytics(coupon, years, coupon)
        add(asset_class="bond", bond_type="sovereign", obligor_id=f"SOV_{country}", obligor_name=name, sector="Sovereign",
            country=country, region=REGION_OF_COUNTRY.get(country, "EM"), currency=currency, rating=rating, lgd=0.6,
            notional=size, drawn=size, maturity=float(years), coupon=coupon, duration=round(duration, 3),
            convexity=round(convexity, 3), direction=1, source="securities")
    corporate_bonds = ["JPM", "BAC", "GS", "C", "DB", "CS", "XOM", "SHEL", "AAPL", "VZ", "BA", "INTC", "F", "TSLA", "GAZP", "EVERG"]
    for entity_id in corporate_bonds:
        e, rating = ent[entity_id], corporate_ratings[entity_id]
        years = float(rng.choice([3, 5, 7, 10]))
        coupon = {"AAA": 0.04, "AA": 0.042, "A": 0.045, "BBB": 0.05, "BB": 0.065, "B": 0.08, "CCC": 0.11}[rating]
        duration, convexity = _bond_analytics(coupon, years, coupon)
        size = (200e6 if rating in ("AAA", "AA", "A", "BBB") else 80e6) * rng.uniform(0.6, 1.4)
        region = REGION_OF_COUNTRY.get(e.country or "US", "EM")
        add(asset_class="bond", bond_type="corporate", obligor_id=entity_id, obligor_name=f"{e.name} {years:.0f}Y senior",
            sector=e.sector, country=e.country, region=region, currency="USD", rating=rating, lgd=0.6, notional=round(size, -5),
            drawn=round(size, -5), maturity=years, coupon=coupon, duration=round(duration, 3), convexity=round(convexity, 3),
            direction=1, source="securities")

    # 4. Derivatives (MtM starts at zero: the stress result is their P&L).
    for currency, notional, years, direction in [("USD", 2.5e9, 7, -1), ("USD", 0.8e9, 5, 1), ("EUR", 0.6e9, 10, -1), ("GBP", 0.4e9, 10, -1)]:
        duration, _ = _bond_analytics(0.035, years, 0.035)
        # direction +1 = receive fixed (gains when rates fall), -1 = pay fixed (hedges fixed-rate bonds)
        add(asset_class="irs", obligor_id="CCP", obligor_name=f"{'Receive' if direction > 0 else 'Pay'}-fixed IRS {currency} {years}Y",
            sector="Derivatives", country="XX", region="US", currency=currency, rating="AA", notional=notional, maturity=float(years),
            duration=round(duration, 3), direction=direction, source="derivatives")
    for currency, notional in [("EUR", 1.2e9), ("GBP", 0.8e9), ("INR", 0.6e9)]:
        add(asset_class="fx_forward", obligor_id="CCP", obligor_name=f"FX forward: sell {currency} / buy USD", sector="Derivatives",
            country="XX", region="US", currency=currency, rating="AA", notional=notional, maturity=0.5, direction=-1, source="derivatives")
    for entity_id, notional, direction in [("BA", 150e6, 1), ("F", 100e6, 1), ("GAZP", 120e6, 1), ("EVERG", 40e6, 1),
                                           ("DB", 100e6, -1), ("INTC", 100e6, -1), ("JPM", 150e6, -1)]:
        e = ent[entity_id]
        add(asset_class="cds", obligor_id=entity_id, obligor_name=f"CDS 5Y {'protection bought' if direction > 0 else 'protection sold'}: {e.name}",
            sector=e.sector, country=e.country, region=REGION_OF_COUNTRY.get(e.country or "US", "EM"), currency="USD",
            rating=corporate_ratings[entity_id], notional=notional, maturity=5.0, duration=4.5, direction=direction, source="derivatives")
    for entity_id, notional in [("AAPL", 120e6), ("JPM", 100e6), ("XOM", 80e6), ("LMT", 60e6)]:
        e = ent[entity_id]
        add(asset_class="equity_trs", obligor_id=entity_id, obligor_name=f"Equity TRS (long) {e.name}", sector=e.sector,
            country=e.country, region="US", currency="USD", rating=corporate_ratings[entity_id], notional=notional,
            maturity=1.0, direction=1, source="derivatives")
    add(asset_class="commodity_swap", obligor_id="CCP", obligor_name="Oil swap: client airline hedge (bank short oil)",
        sector="Derivatives", country="XX", region="US", currency="USD", rating="A", notional=200e6, maturity=1.0, direction=-1,
        source="derivatives")

    df = pd.DataFrame(rows)
    defaults = {"bond_type": "", "drawn": 0.0, "undrawn": 0.0, "ccf": 0.0, "coupon": 0.0, "duration": 0.0, "convexity": 0.0,
                "fixed_float": "", "lgd": 0.45}
    for col, value in defaults.items():
        df[col] = df[col].fillna(value) if col in df else value
    ordered = ["position_id", "asset_class", "bond_type", "obligor_id", "obligor_name", "sector", "country", "region", "currency",
               "rating", "lgd", "notional", "drawn", "undrawn", "ccf", "maturity", "fixed_float", "coupon", "duration",
               "convexity", "direction", "source"]
    return df[ordered]


def load_portfolio(path: Path = POSITIONS_PATH) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""])
