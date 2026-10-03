"""Measure how every risk factor actually moved during each historical episode in configs/scenarios.yaml.

The output, ``data/scenarios/analog_shocks.csv``, is what lets Module B say "this stress scenario
is 60% Crimea 2014 and 40% Brexit 2016, scaled to the event's impact" instead of inventing numbers.

Measurement conventions (start = last close on/before the episode start; end = last close on/before its end):
    pct      percentage change of the proxy (FX quoted so that a positive number = currency strengthens vs USD)
    bp       change of a yield index, in basis points
    pts      change in index points (VIX)
    spread   credit-spread change implied by a bond ETF's return in excess of a Treasury ETF:
             -(r_credit - r_treasury) / duration, in basis points (a standard proxy where index OAS data is licensed)
    relative sector ETF return minus the market's return

    python scripts/build_analog_library.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

OUT = ROOT / "data" / "scenarios" / "analog_shocks.csv"


def main() -> int:
    import yfinance as yf

    cfg = yaml.safe_load((ROOT / "configs" / "scenarios.yaml").read_text(encoding="utf-8"))
    factors: dict[str, dict] = cfg["risk_factors"]
    tickers = sorted({f["proxy"] for f in factors.values()} | {f[k] for f in factors.values() for k in ("treasury", "relative_to") if k in f})
    print(f"downloading {len(tickers)} proxies from Yahoo Finance ...", flush=True)
    prices = yf.download(tickers, start="2000-01-01", auto_adjust=True, progress=False)["Close"].sort_index()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    def close_on(ticker: str, day: str) -> float:
        series = prices[ticker].dropna()
        series = series[series.index <= pd.Timestamp(day)]
        return float(series.iloc[-1]) if len(series) and (pd.Timestamp(day) - series.index[-1]).days <= 5 else np.nan

    rows = []
    for ep in cfg["episodes"]:
        start, end = str(ep["start"]), str(ep["end"])
        row = {"episode_id": ep["id"], "event_type": ep["event_type"], "title": ep["title"], "start": start, "end": end}
        for fid, f in factors.items():
            p0, p1 = close_on(f["proxy"], start), close_on(f["proxy"], end)
            if f["unit"] == "bp" and "treasury" not in f:
                value = (p1 - p0) * 100.0  # yield indices are quoted in percent
            elif f["unit"] == "pts":
                value = p1 - p0
            elif "treasury" in f:
                t0, t1 = close_on(f["treasury"], start), close_on(f["treasury"], end)
                value = -((p1 / p0 - 1.0) - (t1 / t0 - 1.0)) / f["duration"] * 1e4
            else:
                value = (p0 / p1 - 1.0) * 100.0 if f.get("invert") else (p1 / p0 - 1.0) * 100.0
                if "relative_to" in f:
                    m0, m1 = close_on(f["relative_to"], start), close_on(f["relative_to"], end)
                    value -= (m1 / m0 - 1.0) * 100.0
            row[fid] = round(float(value), 2) if np.isfinite(value) else np.nan
        rows.append(row)
        shown = ", ".join(f"{k} {row[k]:+.1f}" for k in ("EQ_US", "IR_USD_10Y", "CS_HY", "CMD_OIL", "FX_EUR") if np.isfinite(row[k]))
        print(f"  {ep['id']:22s} {shown}", flush=True)

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    meta = {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": "Yahoo Finance daily closes via yfinance",
            "proxies": {fid: {k: v for k, v in f.items() if k != "label"} for fid, f in factors.items()},
            "coverage": {fid: int(df[fid].notna().sum()) for fid in factors}}
    OUT.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {OUT} ({len(df)} episodes x {len(factors)} factors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
