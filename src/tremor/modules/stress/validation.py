"""Is the scenario generator any good? A point-in-time backtest on real crises.

Every historical episode in the library is treated as breaking news. The generator gets only the
episode's day-one headline (the trigger, with no market outcome in it) and only the episodes that had
*ended* before it began, and the scenario it builds is compared with what markets then actually did.
Simpler ways of choosing the shocks go through exactly the same test, so the generator has to beat
something:

    naive         the brief's own example: equities -10%, interest rates +200 bp, nothing else
    template      the expert-defined shock for the event type (configs/scenarios.yaml)
    all crises    the average move of every earlier episode ("the average crisis")
    same type     the average of earlier episodes of the same event type
    nearest       the closest analogs alone (event type + headline similarity, similarity-weighted)
    TREMOR        the production method, unchanged (ScenarioLibrary.historical_shocks): the closest
                  analogs credibility-weighted 50/50 against the average crisis

A past crisis has no impact score from the engine and names no obligor, so every scenario is compared
unscaled (severity 1.0, i.e. impact 8.5) and without epicentre notching: the test is about the *shape*
of the shocks. Two scores:

    direction     share of the materially moving headline factors whose sign the scenario got right;
                  a factor the scenario leaves at zero counts as a miss (it captures nothing)
    P&L error     |P&L of the synthetic book under the scenario - P&L under the realised moves| in
                  USD m, and the same for the post-stress CET1 ratio in percentage points: how wrong
                  the number a risk committee would act on was
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from tremor.modules.stress.engine import run_stress
from tremor.modules.stress.scenarios import REFERENCE_IMPACT, ScenarioLibrary

# The factors a risk committee reads first: equities, rates, credit, FX, commodities, volatility.
HEADLINE_FACTORS = ("EQ_US", "EQ_EU", "EQ_EM", "EQ_IN", "IR_USD_10Y", "CS_IG", "CS_HY", "CS_EM", "FX_EUR", "FX_JPY",
                    "CMD_OIL", "CMD_GOLD", "VOL_VIX")
MATERIAL = {"pct": 1.0, "bp": 5.0, "pts": 1.0}  # a smaller realised move is noise: its sign is not scored
# The example shock in the hackathon brief ("a 10% drop in all equity prices, a 2% increase in interest rates").
NAIVE_SHOCK = {"EQ_US": -10.0, "EQ_EU": -10.0, "EQ_EM": -10.0, "EQ_IN": -10.0, "IR_USD_5Y": 200.0, "IR_USD_10Y": 200.0}
METHODS = {
    "naive": "Naive fixed shock (the brief's example)",
    "template": "Expert template per event type",
    "all_mean": "Average of all earlier crises",
    "same_type_mean": "Average of earlier crises of the same type",
    "nearest": "Closest analogs alone",
    "tremor": "TREMOR (analogs, credibility-weighted)",
}
MIN_PRIOR = 4  # an episode is a test case once at least this many earlier episodes can serve as analogs


def _as_of(start) -> datetime:
    return pd.Timestamp(start).to_pydatetime().replace(tzinfo=timezone.utc)


def eligible_pool(library: ScenarioLibrary, start) -> pd.DataFrame:
    """Episodes a scenario built on ``start`` may learn from (the production rule, at impact 8.5)."""
    return library.episodes[library.eligible(_as_of(start), REFERENCE_IMPACT)]


def candidate_scenarios(library: ScenarioLibrary, episode: pd.Series, pool: pd.DataFrame) -> dict[str, dict]:
    """Every method's shocks (and the analogs the retrieval methods chose) for one test episode."""
    def mean_of(ids: list[str]) -> dict[str, float]:
        return library.blend([{"episode_id": i, "weight": 1.0 / len(ids)} for i in ids]) if ids else {}

    as_of = _as_of(episode["start"])
    same_type = pool.loc[pool["event_type"] == episode["event_type"], "id"].tolist()
    shocks, analogs, _ = library.historical_shocks(episode["event_type"], episode["headline"], as_of, REFERENCE_IMPACT)
    chosen = [{"episode_id": a["episode_id"], "title": a["title"], "weight": a["weight"], "similarity": a["similarity"]} for a in analogs]
    template = library.templates.get(episode["event_type"]) or library.templates["GEOPOLITICAL"]
    return {
        "naive": {"shocks": dict(NAIVE_SHOCK)},
        "template": {"shocks": dict(template["shocks"])},
        "all_mean": {"shocks": mean_of(pool["id"].tolist())},
        "same_type_mean": {"shocks": mean_of(same_type) or mean_of(pool["id"].tolist())},
        "nearest": {"shocks": library.blend(analogs), "analogs": chosen},
        "tremor": {"shocks": shocks, "analogs": chosen},
    }


def direction_hits(shocks: dict[str, float], realised: dict[str, float], units: dict[str, str]) -> dict[str, bool]:
    """For each materially moving headline factor: did the scenario move it the same way?"""
    return {f: shocks.get(f, 0.0) * realised[f] > 0 for f in HEADLINE_FACTORS
            if realised.get(f) is not None and abs(realised[f]) >= MATERIAL[units[f]]}


def backtest(library: ScenarioLibrary, book: pd.DataFrame, min_prior: int = MIN_PRIOR) -> dict:
    units = {f: spec["unit"] for f, spec in library.factors.items()}
    episodes = library.episodes.sort_values("start")
    rows = []
    for _, ep in episodes.iterrows():
        pool = eligible_pool(library, ep["start"])
        if len(pool) < min_prior or ep["id"] not in library.analogs.index:
            continue
        realised = {f: float(v) for f, v in library.analogs.loc[ep["id"], list(library.factors)].items() if pd.notna(v)}
        base = run_stress(book, library.custom("realised", realised), library)
        row = {"episode_id": ep["id"], "title": ep["title"], "event_type": ep["event_type"], "start": str(pd.Timestamp(ep["start"]).date()),
               "headline": ep["headline"], "n_prior": len(pool), "realised": {f: realised.get(f) for f in HEADLINE_FACTORS},
               "realised_pnl_musd": round(base.totals["pnl"] / 1e6, 1), "realised_cet1_after": base.capital["cet1_ratio_after"],
               "methods": {}}
        for method, cand in candidate_scenarios(library, ep, pool).items():
            # Compare like with like: a factor with no realised measurement is not shocked either.
            shocks = {f: round(v, 2) for f, v in cand["shocks"].items() if f in realised}
            result = run_stress(book, library.custom(method, shocks), library)
            hits = direction_hits(shocks, realised, units)
            row["methods"][method] = {
                "hits": sum(hits.values()), "scored": len(hits), "factor_hits": hits,
                "pnl_musd": round(result.totals["pnl"] / 1e6, 1),
                "pnl_error_musd": round(abs(result.totals["pnl"] - base.totals["pnl"]) / 1e6, 1),
                "cet1_after": result.capital["cet1_ratio_after"],
                "cet1_error_pp": round(abs(result.capital["cet1_ratio_after"] - base.capital["cet1_ratio_after"]), 2),
                "shocks": {f: shocks.get(f) for f in HEADLINE_FACTORS},
                **({"analogs": cand["analogs"]} if "analogs" in cand else {}),
            }
        rows.append(row)
    return {"description": __doc__.strip().split("\n\n")[0].replace("\n", " "), "methods": METHODS,
            "headline_factors": list(HEADLINE_FACTORS), "materiality": MATERIAL, "naive_shock": NAIVE_SHOCK,
            "min_prior": min_prior, "episodes": rows, "summary": summarise(rows)}


def summarise(rows: list[dict]) -> dict:
    out = {}
    for method in METHODS:
        m = [r["methods"][method] for r in rows]
        errors = np.array([x["pnl_error_musd"] for x in m])
        per_factor = {}
        for f in HEADLINE_FACTORS:
            calls = [x["factor_hits"][f] for x in m if f in x["factor_hits"]]
            if calls:
                per_factor[f] = round(sum(calls) / len(calls), 2)
        out[method] = {
            "label": METHODS[method],
            "direction_hit_rate": round(sum(x["hits"] for x in m) / max(1, sum(x["scored"] for x in m)), 3),
            "pnl_error_mean_musd": round(float(errors.mean()), 1),
            "pnl_error_median_musd": round(float(np.median(errors)), 1),
            "cet1_error_mean_pp": round(float(np.mean([x["cet1_error_pp"] for x in m])), 2),
            "closer_than_naive": sum(1 for r in rows if r["methods"][method]["pnl_error_musd"] < r["methods"]["naive"]["pnl_error_musd"]),
            "closer_than_average": sum(1 for r in rows if r["methods"][method]["pnl_error_musd"] < r["methods"]["all_mean"]["pnl_error_musd"]),
            "per_factor_hit_rate": per_factor,
        }
    rates = [r["realised"]["IR_USD_10Y"] for r in rows if r["realised"].get("IR_USD_10Y") is not None]
    out["context"] = {"episodes": len(rows), "first": rows[0]["start"] if rows else None, "last": rows[-1]["start"] if rows else None,
                      "rates_fell": sum(1 for v in rates if v <= -MATERIAL["bp"]), "rates_rose": sum(1 for v in rates if v >= MATERIAL["bp"]),
                      "mean_abs_realised_pnl_musd": round(float(np.mean([abs(r["realised_pnl_musd"]) for r in rows])), 1) if rows else None}
    return out
