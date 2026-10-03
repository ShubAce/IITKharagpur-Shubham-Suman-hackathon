"""Revalue the wholesale book under a scenario, and decide when an event deserves a stress test.

Each instrument is revalued through the channel that actually drives it:

    loans / revolvers   credit losses: Vasicek-stressed PD, downturn LGD, revolver drawdown (higher
                        EAD), IFRS 9 stage migration -> change in expected-credit-loss allowance
    bonds               rates + credit spread: -duration x dy + 1/2 convexity x dy^2
    interest-rate swaps DV01 x rate move
    CDS                 CS01 x spread move, plus jump-to-default if the reference entity defaults
    FX forwards / non-USD assets   FX translation
    equity TRS, oil swap           delta x price move

Then capital: Basel IRB risk-weighted assets before and after (stressed PDs raise RWA at the same
time as losses eat capital - the pro-cyclical squeeze), and the CET1 ratio against requirements.
Everything is vectorised over the ~400 positions, so a full revaluation takes milliseconds and the
analyst can move any shock in the dashboard and see the result immediately.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from tremor.modules.stress.credit import (
    RATING_ORDER,
    SICR_MULTIPLE,
    expected_credit_loss,
    irb_risk_weight,
    rating_to_pd,
    standardised_bond_weight,
    stressed_pd,
)
from tremor.modules.stress.scenarios import Scenario, ScenarioLibrary, epicentre_of
from tremor.schemas import EventSignal

EQUITY_SIGMA_PCT = 20.0  # a 20% equity fall = a one-standard-deviation systematic credit shock
REGION_EQUITY = {"US": "EQ_US", "EU": "EQ_EU", "UK": "EQ_EU", "IN": "EQ_IN", "EM": "EQ_EM", "JP": "EQ_EM"}
CURRENCY_FX = {"EUR": "FX_EUR", "GBP": "FX_GBP", "INR": "FX_INR", "JPY": "FX_JPY"}
RATE_BETA = {"USD": 1.0, "EUR": 0.6, "GBP": 0.8, "INR": 0.3}  # local rate move per USD rate move (assumption)
DEVELOPED_SOVEREIGNS = {"US", "DE", "GB", "JP"}
IG = {"AAA", "AA", "A", "BBB"}
NOTCH_SPREAD_BP = {"IG": 25.0, "HY": 75.0}  # idiosyncratic spread per notch of downgrade
SECTOR_SPREAD_BETA = {"IG": 3.0, "HY": 8.0}  # bp of extra spread per 1% sector underperformance
ESCALATION_MIN_VELOCITY = 60  # reports per hour before a coverage surge alone can re-run a stress test
ESCALATION_VELOCITY_MULTIPLE = 1.8  # ... and the surge must be this many times the coverage at the last run
CET1_START = 0.135  # assumed starting CET1 ratio of the synthetic bank
CET1_REQUIREMENT = 0.095  # 4.5% minimum + 2.5% conservation buffer + ~2.5% Pillar 2 / systemic buffers
DERIVATIVE_RWA_FACTOR = 0.05  # simplified counterparty-credit RWA per unit of derivative notional
CDS_RECOVERY = 0.4
# Basel IRB PDs are through-the-cycle: regulatory capital reacts only partly to a point-in-time
# stress, whereas IFRS 9 expected losses must use the full point-in-time PD. Share passed through:
RWA_PIT_SHARE = 0.3

CHANNELS = ("Credit losses (ECL)", "Interest rates", "Credit spreads", "Default (CDS)", "FX", "Equity", "Commodities")


def _notch(rating: str, notches: int) -> str:
    if notches <= 0 or rating == "D":
        return rating
    return RATING_ORDER[min(RATING_ORDER.index(rating) + notches, len(RATING_ORDER) - 1)]


def _rate_shock_bp(shocks: dict[str, float], maturity: np.ndarray, currency: pd.Series) -> np.ndarray:
    """USD rate move interpolated between the 5Y and 10Y points, scaled to the local currency."""
    r5, r10 = shocks.get("IR_USD_5Y", 0.0), shocks.get("IR_USD_10Y", 0.0)
    weight = np.clip((maturity - 5.0) / 5.0, 0.0, 1.0)
    return (r5 + (r10 - r5) * weight) * currency.map(RATE_BETA).fillna(1.0).to_numpy()


@dataclass
class StressResult:
    run_id: str
    created_at: datetime
    scenario: Scenario
    trigger: dict | None
    totals: dict
    capital: dict
    credit: dict
    by_channel: list[dict]
    by_asset_class: list[dict]
    by_sector: list[dict]
    by_region: list[dict]
    top_losses: list[dict]
    positions: pd.DataFrame = field(repr=False)

    def to_dict(self, include_positions: bool = False) -> dict:
        out = {k: v for k, v in self.__dict__.items() if k not in ("positions", "scenario")}
        out["created_at"] = self.created_at.isoformat()
        out["scenario"] = self.scenario.to_dict()
        if include_positions:
            out["positions"] = self.positions.to_dict(orient="records")
        return out


def run_stress(book: pd.DataFrame, scenario: Scenario, library: ScenarioLibrary, trigger: dict | None = None) -> StressResult:
    s = scenario.shocks
    df = book.copy()
    n = len(df)
    eq_region = df["region"].map(REGION_EQUITY).map(lambda f: s.get(f, 0.0) if isinstance(f, str) else 0.0).fillna(0.0)
    eq_sector = df["sector"].map(library.sector_factor).map(lambda f: s.get(f, 0.0) if isinstance(f, str) else 0.0).fillna(0.0)
    equity_move = (eq_region + eq_sector).to_numpy()  # percent
    z = equity_move / EQUITY_SIGMA_PCT  # systematic credit factor (negative = downturn)

    epicentre = scenario.epicentre
    notches = np.array([max(epicentre.get(o, 0), epicentre.get(c, 0)) for o, c in zip(df["obligor_id"], df["country"].astype(str))])
    rating_after = [_notch(r, k) for r, k in zip(df["rating"], notches)]
    pnl = {c: np.zeros(n) for c in CHANNELS}

    # ------------------------------------------------------------------ loans and revolvers
    lending = df["asset_class"].isin(["loan", "revolver"]).to_numpy()
    pd0 = rating_to_pd(df["rating"])
    pd_notched = rating_to_pd(rating_after)
    pd1 = np.where(np.array(rating_after) == "D", 1.0, stressed_pd(pd_notched, np.minimum(z, 2.0)))
    downturn = np.clip(-z, 0.0, None)
    lgd0 = df["lgd"].to_numpy(float)
    lgd1 = np.minimum(lgd0 * (1.0 + 0.15 * downturn), 0.9)
    drawn, undrawn, ccf0 = (df[c].to_numpy(float) for c in ("drawn", "undrawn", "ccf"))
    ccf1 = np.minimum(1.0, ccf0 + 0.15 * downturn)  # firms draw on their credit lines in a crisis
    ead0, ead1 = drawn + ccf0 * undrawn, drawn + ccf1 * undrawn
    maturity = df["maturity"].to_numpy(float)
    stage2_0 = df["rating"].isin(["CCC"]).to_numpy()
    stage2_1 = stage2_0 | (pd1 >= SICR_MULTIPLE * pd0) | np.isin(rating_after, ["CCC", "D"])
    ecl0 = np.where(lending, expected_credit_loss(pd0, lgd0, ead0, maturity, stage2_0), 0.0)
    ecl1 = np.where(lending, expected_credit_loss(pd1, lgd1, ead1, maturity, stage2_1), 0.0)
    pnl["Credit losses (ECL)"] -= ecl1 - ecl0

    # ------------------------------------------------------------------ bonds
    bonds = (df["asset_class"] == "bond").to_numpy()
    rate_bp = _rate_shock_bp(s, maturity, df["currency"])
    grade = np.where(np.isin(rating_after, list(IG)), "IG", "HY")
    sovereign = (df["bond_type"] == "sovereign").to_numpy()
    developed = df["country"].isin(DEVELOPED_SOVEREIGNS).to_numpy()
    base_spread = np.where(sovereign, np.where(developed, 0.0, s.get("CS_EM", 0.0)),
                           np.where(grade == "IG", s.get("CS_IG", 0.0), s.get("CS_HY", 0.0)))
    sector_addon = np.where(sovereign, 0.0, np.clip(-eq_sector.to_numpy(), 0.0, None) * np.vectorize(SECTOR_SPREAD_BETA.get)(grade))
    idio = notches * np.vectorize(NOTCH_SPREAD_BP.get)(grade)
    spread_bp = base_spread + sector_addon + idio
    dur, conv = df["duration"].to_numpy(float), df["convexity"].to_numpy(float)
    notional = df["notional"].to_numpy(float)
    dy_rate, dy_total = rate_bp / 1e4, (rate_bp + spread_bp) / 1e4
    rate_part = -dur * dy_rate + 0.5 * conv * dy_rate ** 2
    total_part = -dur * dy_total + 0.5 * conv * dy_total ** 2
    pnl["Interest rates"] += np.where(bonds, notional * rate_part, 0.0)
    pnl["Credit spreads"] += np.where(bonds, notional * (total_part - rate_part), 0.0)

    # ------------------------------------------------------------------ derivatives
    direction = df["direction"].to_numpy(float)
    irs = (df["asset_class"] == "irs").to_numpy()
    pnl["Interest rates"] += np.where(irs, -direction * notional * dur * 1e-4 * rate_bp, 0.0)
    cds = (df["asset_class"] == "cds").to_numpy()
    cds_spread = np.where(grade == "IG", s.get("CS_IG", 0.0), s.get("CS_HY", 0.0)) + sector_addon + idio
    defaulted = np.array(rating_after) == "D"
    pnl["Credit spreads"] += np.where(cds & ~defaulted, direction * notional * dur * 1e-4 * cds_spread, 0.0)
    pnl["Default (CDS)"] += np.where(cds & defaulted, direction * notional * (1.0 - CDS_RECOVERY), 0.0)
    fx_move = df["currency"].map(CURRENCY_FX).map(lambda f: s.get(f, 0.0) if isinstance(f, str) else 0.0).fillna(0.0).to_numpy() / 100
    fwd = (df["asset_class"] == "fx_forward").to_numpy()
    pnl["FX"] += np.where(fwd, direction * notional * fx_move, 0.0)
    trs = (df["asset_class"] == "equity_trs").to_numpy()
    pnl["Equity"] += np.where(trs, direction * notional * (equity_move / 100 - 0.05 * notches), 0.0)
    oil = (df["asset_class"] == "commodity_swap").to_numpy()
    pnl["Commodities"] += np.where(oil, direction * notional * s.get("CMD_OIL", 0.0) / 100, 0.0)

    # FX translation of non-USD loans and bonds (the forwards above hedge most of it)
    carrying0 = np.where(lending, drawn - ecl0, np.where(bonds, notional, 0.0))
    carrying_pre_fx = np.where(lending, drawn - ecl1, np.where(bonds, notional * (1 + total_part), 0.0))
    pnl["FX"] += (lending | bonds) * carrying_pre_fx * fx_move

    df["pnl"] = sum(pnl.values())
    for channel, values in pnl.items():
        df[f"pnl_{channel}"] = values
    df["rating_after"] = rating_after
    df["pd_before"], df["pd_after"] = pd0, np.where(lending | bonds | cds, pd1, np.nan)
    df["stage2_after"] = stage2_1 & lending

    # ------------------------------------------------------------------ capital
    rw_loan0 = irb_risk_weight(pd0, lgd0, maturity) * ead0
    pd_capital = pd0 ** (1 - RWA_PIT_SHARE) * pd1 ** RWA_PIT_SHARE
    lgd_capital = lgd0 + RWA_PIT_SHARE * (lgd1 - lgd0)
    # Defaulted exposures: Basel charges capital for unexpected loss on top of the provision
    # (LGD in-default minus best-estimate loss); floored here at a 100% risk weight to stay conservative.
    rw_default = np.maximum(12.5 * np.maximum(lgd1 - lgd0, 0.0), 1.0)
    rw_loan1 = np.where(defaulted, rw_default, irb_risk_weight(pd_capital, lgd_capital, maturity)) * ead1
    rw_bond0 = standardised_bond_weight(df["rating"]) * notional
    rw_bond1 = standardised_bond_weight(rating_after) * notional * (1 + total_part)
    deriv = (irs | cds | fwd | trs | oil)
    rwa0 = float(np.where(lending, rw_loan0, 0).sum() + np.where(bonds, rw_bond0, 0).sum() + (deriv * notional).sum() * DERIVATIVE_RWA_FACTOR)
    rwa1 = float(np.where(lending, rw_loan1, 0).sum() + np.where(bonds, rw_bond1, 0).sum() + (deriv * notional).sum() * DERIVATIVE_RWA_FACTOR)
    total_pnl = float(df["pnl"].sum())
    cet1_0 = CET1_START * rwa0
    cet1_1 = cet1_0 + total_pnl
    value0 = float(carrying0.sum())

    def group(col: str) -> list[dict]:
        g = df.groupby(col)["pnl"].sum().sort_values()
        return [{"name": k, "pnl": round(float(v), 0)} for k, v in g.items()]

    worst = df.nsmallest(10, "pnl")
    created = datetime.now(timezone.utc)
    run_id = "run_" + hashlib.sha1(f"{scenario.scenario_id}|{created.isoformat()}".encode()).hexdigest()[:10]
    return StressResult(
        run_id=run_id, created_at=created, scenario=scenario, trigger=trigger,
        totals={"value_before": round(value0, 0), "value_after": round(value0 + total_pnl, 0), "pnl": round(total_pnl, 0),
                "pnl_pct": round(total_pnl / value0 * 100, 3), "positions": n},
        capital={"rwa_before": round(rwa0, 0), "rwa_after": round(rwa1, 0), "cet1_before": round(cet1_0, 0), "cet1_after": round(cet1_1, 0),
                 "cet1_ratio_before": round(cet1_0 / rwa0 * 100, 2), "cet1_ratio_after": round(cet1_1 / rwa1 * 100, 2),
                 "capital_depletion_bp": round(-total_pnl / rwa0 * 1e4, 1),  # loss in bp of starting RWA
                 "requirement_pct": CET1_REQUIREMENT * 100, "breach": bool(cet1_1 / rwa1 < CET1_REQUIREMENT)},
        credit={"ecl_before": round(float(ecl0.sum()), 0), "ecl_after": round(float(ecl1.sum()), 0),
                "stage2_before": int((stage2_0 & lending).sum()), "stage2_after": int((stage2_1 & lending).sum()),
                "downgraded_obligors": int(df.loc[lending & (np.array(rating_after) != df["rating"].to_numpy()), "obligor_id"].nunique()),
                "defaults": int(df.loc[defaulted & (lending | bonds), "obligor_id"].nunique()),
                "avg_pd_before_bp": round(float(np.average(pd0[lending], weights=ead0[lending])) * 1e4, 1),
                "avg_pd_after_bp": round(float(np.average(pd1[lending], weights=ead1[lending])) * 1e4, 1)},
        by_channel=[{"name": c, "pnl": round(float(v.sum()), 0)} for c, v in pnl.items()],
        by_asset_class=group("asset_class"), by_sector=group("sector"), by_region=group("region"),
        top_losses=[{"position_id": r.position_id, "obligor": r.obligor_name, "asset_class": r.asset_class, "rating": r.rating,
                     "rating_after": r.rating_after, "pnl": round(float(r.pnl), 0)} for r in worst.itertuples()],
        positions=df,
    )


# --------------------------------------------------------------------------- trigger logic
class StressMonitor:
    """Subscribes to event signals and runs a stress test when one crosses the threshold.

    One *situation* (say, the Russia-Ukraine war) produces many related high-impact events in
    a day; the monitor groups them by event type and the countries / companies involved, and runs
    once per situation - again only if the situation's impact rises by ``retrigger_delta``.
    """

    def __init__(self, book: pd.DataFrame, library: ScenarioLibrary, threshold: float = 7.0, retrigger_delta: float = 1.0,
                 trigger_types: tuple[str, ...] = ("GEOPOLITICAL", "MACROECONOMIC", "CREDIT_EVENT"), window_hours: float = 48.0,
                 type_thresholds: dict[str, float] | None = None):
        self.book, self.library = book, library
        self.threshold, self.retrigger_delta, self.trigger_types = threshold, retrigger_delta, trigger_types
        # Macro news is abundant every day; only a more severe macro event warrants a full stress test.
        self.type_thresholds = type_thresholds if type_thresholds is not None else {"MACROECONOMIC": max(threshold, 8.0)}
        self.window = timedelta(hours=window_hours)
        self.runs: list[StressResult] = []
        self._situations: list[dict] = []  # {type, keys, impact, at, run_id, events}
        self._event_situation: dict[str, dict] = {}

    @staticmethod
    def _keys(event: EventSignal) -> set[str]:
        """Identity of the situation: who is directly hit (RU+UA for the Ukraine war).

        Macroeconomic shocks are global, so all macro events in the window are one situation."""
        return {"GLOBAL"} if event.event_type == "MACROECONOMIC" else set(epicentre_of(event))

    def threshold_for(self, event_type: str) -> float:
        return self.type_thresholds.get(event_type, self.threshold)

    def on_event(self, event: EventSignal) -> StressResult | None:
        # Strictly above the threshold, as in the brief's rule ("Geopolitical with an Impact Score > 7").
        if event.event_type not in self.trigger_types or event.impact_score <= self.threshold_for(event.event_type):
            return None
        keys, now = self._keys(event), event.last_updated
        sit = self._event_situation.get(event.event_id)
        if sit is None:
            live = [s for s in self._situations if s["type"] == event.event_type and now - s["seen"] <= self.window]
            overlapping = [s for s in live if keys & s["keys"]]
            if overlapping:  # hits the same countries / companies as a situation already being tracked
                sit = max(overlapping, key=lambda s: len(keys & s["keys"]))
            elif not keys and live:  # names no epicentre: commentary on the crisis already stressed
                sit = live[-1]
        if sit is None:
            sit = {"type": event.event_type, "keys": set(keys), "impact": event.impact_score, "at": now, "seen": now,
                   "velocity": event.reports_last_hour, "events": set()}
            self._situations.append(sit)
            self._bind(event, sit)
            return self._run(event, sit, "new situation")
        self._bind(event, sit)
        # Re-run only on material escalation: impact up by retrigger_delta, or - because the 1-10
        # scale saturates for the biggest stories - coverage velocity nearly doubling.
        if event.impact_score >= sit["impact"] + self.retrigger_delta:
            reason = f"escalation: impact {sit['impact']:.1f} -> {event.impact_score:.1f}"
        elif event.reports_last_hour >= max(ESCALATION_VELOCITY_MULTIPLE * sit["velocity"], ESCALATION_MIN_VELOCITY):
            reason = f"escalation: coverage {sit['velocity']} -> {event.reports_last_hour} reports/hour"
        else:
            return None  # this situation was already stressed at a similar severity
        sit.update(impact=max(sit["impact"], event.impact_score), at=now, velocity=event.reports_last_hour)
        return self._run(event, sit, reason)

    def _bind(self, event: EventSignal, sit: dict) -> None:
        sit["events"].add(event.event_id)
        sit["seen"] = max(sit["seen"], event.last_updated)  # still in the news: keep the situation alive
        self._event_situation[event.event_id] = sit

    def _run(self, event: EventSignal, situation: dict, reason: str) -> StressResult:
        scenario = self.library.for_event(event)
        trigger = {"event_id": event.event_id, "headline": event.headline, "event_type": event.event_type,
                   "event_type_label": event.event_type_label, "impact_score": event.impact_score,
                   "detected_at": event.last_updated.isoformat(), "situation": sorted(situation["keys"]), "reason": reason,
                   "rule": f"{event.event_type_label} event with impact {event.impact_score:.1f} > "
                           f"{self.threshold_for(event.event_type):.1f} ({reason})"}
        result = run_stress(self.book, scenario, self.library, trigger)
        situation["run_id"] = result.run_id
        self.runs.append(result)
        return result
