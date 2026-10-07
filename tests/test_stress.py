import numpy as np
import pytest

from tremor.modules.stress.credit import irb_risk_weight, pd_to_rating, rating_to_pd, stressed_pd
from tremor.modules.stress.engine import StressMonitor, run_stress
from tremor.modules.stress.portfolio import load_portfolio
from tremor.modules.stress.scenarios import ScenarioLibrary, severity_multiplier
from tremor.schemas import EventSignal

BOOK = load_portfolio()
LIB = ScenarioLibrary()  # no encoder: analog retrieval falls back to event-type matching


def scenario(**shocks):
    return LIB.custom("test", shocks)


def test_basel_irb_risk_weight_matches_hand_calculation():
    # PD 0.2%, LGD 45%, M 2.5y: correlation 0.229, K = 3.51%, RW = 43.9% (worked by hand from BCBS formula).
    assert irb_risk_weight(np.array([0.002]), np.array([0.45]), np.array([2.5]))[0] == pytest.approx(0.439, abs=0.003)


def test_stressed_pd_is_consistent_and_monotone():
    pd = rating_to_pd(["AA", "BBB", "BB", "B"])
    assert np.allclose(stressed_pd(pd, 0.0), pd)  # no shock, no change
    worse, better = stressed_pd(pd, -1.5), stressed_pd(pd, 1.0)
    assert (worse > pd).all() and (better < pd).all()
    assert pd_to_rating([0.0021])[0] == "BBB"


def test_a_zero_scenario_changes_nothing():
    result = run_stress(BOOK, scenario(), LIB)
    assert result.totals["pnl"] == pytest.approx(0.0, abs=1.0)
    assert result.capital["rwa_after"] == pytest.approx(result.capital["rwa_before"], rel=1e-9)


def test_rates_up_hurts_bonds_and_helps_pay_fixed_swaps():
    result = run_stress(BOOK, scenario(IR_USD_5Y=100, IR_USD_10Y=100), LIB)
    pos = result.positions.set_index("position_id")
    ust = pos[pos["obligor_name"] == "UST 10Y"]["pnl"].iloc[0]
    payer = pos[pos["obligor_name"].str.startswith("Pay-fixed IRS USD")]["pnl"].iloc[0]
    assert ust < 0 < payer


def test_wider_spreads_pay_protection_buyers():
    pos = run_stress(BOOK, scenario(CS_IG=100, CS_HY=300), LIB).positions
    cds = pos[pos["asset_class"] == "cds"]
    assert (cds.loc[cds["direction"] > 0, "pnl"] > 0).all()
    assert (cds.loc[cds["direction"] < 0, "pnl"] < 0).all()


def test_bigger_equity_crash_means_bigger_credit_losses():
    mild = run_stress(BOOK, scenario(EQ_US=-10, EQ_EU=-10, EQ_EM=-10), LIB)
    severe = run_stress(BOOK, scenario(EQ_US=-30, EQ_EU=-30, EQ_EM=-30), LIB)
    assert severe.credit["ecl_after"] > mild.credit["ecl_after"] > mild.credit["ecl_before"]
    assert severe.totals["pnl"] < mild.totals["pnl"] < 0


def test_epicentre_notching_hits_named_obligors():
    result = run_stress(BOOK, LIB.custom("sanctions", {}, epicentre={"RU": 3}), LIB)
    russian = result.positions[(result.positions["country"] == "RU") & result.positions["asset_class"].isin(["loan", "bond"])]
    assert (russian["rating_after"] != russian["rating"]).all()
    assert russian["pnl"].sum() < 0


def test_severity_scales_with_impact():
    assert severity_multiplier(8.5) == pytest.approx(1.0)
    assert severity_multiplier(7.0) < 1.0 < severity_multiplier(10.0)


def _event(event_id, impact, regions=("RU", "UA"), event_type="GEOPOLITICAL"):
    from datetime import datetime, timezone

    now = datetime(2022, 2, 24, 6, tzinfo=timezone.utc)
    return EventSignal(event_id=event_id, first_seen=now, last_updated=now, headline="Russia invades Ukraine", sentiment_score=-0.4,
                       event_type=event_type, event_type_label="Geopolitical", impact_score=impact, market_wide=True,
                       entities=list(regions), regions=list(regions))


def test_monitor_runs_once_per_situation_and_re_runs_on_escalation():
    monitor = StressMonitor(BOOK, LIB, threshold=7.0, retrigger_delta=1.0)
    assert monitor.on_event(_event("evt_a", 6.5)) is None  # below threshold
    assert monitor.on_event(_event("evt_a", 7.0)) is None  # the rule is strictly "> 7"
    assert monitor.on_event(_event("evt_a", 7.4)) is not None  # crosses: first stress test
    assert monitor.on_event(_event("evt_b", 7.9)) is None  # same situation (RU/UA), not materially worse
    assert monitor.on_event(_event("evt_b", 8.6)) is not None  # escalated by >= 1 point: re-run
    assert monitor.on_event(_event("evt_c", 7.2, regions=("CN", "TW"))) is not None  # a different situation
    assert len(monitor.runs) == 3


# --------------------------------------------------------------------------- scenario generator
def test_scenario_blends_closest_analogs_with_the_average_crisis():
    from datetime import datetime, timezone

    from tremor.modules.stress.scenarios import ANALOG_CREDIBILITY

    as_of = datetime(2022, 2, 24, tzinfo=timezone.utc)
    shocks, analogs, n_prior = LIB.historical_shocks("GEOPOLITICAL", "Russia invades Ukraine", as_of, impact=8.5)
    nearest, (prior, n) = LIB.blend(analogs), LIB.average_crisis(as_of, 8.5)
    assert analogs and n_prior == n > len(analogs)
    for f in ("EQ_US", "IR_USD_10Y", "CS_HY", "CMD_OIL"):
        assert shocks[f] == pytest.approx(ANALOG_CREDIBILITY * nearest[f] + (1 - ANALOG_CREDIBILITY) * prior[f])
    assert all(a["end"] < "2022-02-24" for a in analogs)  # point-in-time: nothing from the future


def test_news_on_a_price_overrules_history_by_direction_not_tone():
    # Reports say oil is rising (31 up, 4 down) in a negative tone ("oil soars on war fears").
    event = _event("evt_oil", 9.0).model_copy(update={"price_moves": {"OIL": {"up": 31, "down": 4}},
                                                       "entity_sentiment": {"OIL": -0.3}, "entity_mentions": {"OIL": 40}})
    scenario = LIB.for_event(event)
    assert scenario.shocks["CMD_OIL"] > 15
    assert "31 reports say oil is rising" in scenario.narrative
    assert "Crude oil" in scenario.narrative and "CMD_OIL" not in scenario.narrative  # written for a reader, not in factor ids
    split = event.model_copy(update={"price_moves": {"OIL": {"up": 10, "down": 9}}})  # the reports disagree: history stands
    assert LIB.for_event(split).shocks["CMD_OIL"] == pytest.approx(LIB.for_event(_event("evt_plain", 9.0)).shocks["CMD_OIL"])


def test_scenario_backtest_is_point_in_time_and_beats_the_naive_shock():
    import pandas as pd

    from tremor.modules.stress.validation import backtest

    result = backtest(LIB, BOOK)
    assert result["summary"]["context"]["episodes"] >= 15
    ends = LIB.episodes.set_index("id")["end"]
    for row in result["episodes"]:
        for analog in row["methods"]["tremor"].get("analogs", []):
            assert ends[analog["episode_id"]] < pd.Timestamp(row["start"])
    s = result["summary"]
    assert s["tremor"]["direction_hit_rate"] > s["naive"]["direction_hit_rate"] + 0.3
    assert s["tremor"]["pnl_error_mean_musd"] < s["naive"]["pnl_error_mean_musd"]


def test_epicentre_is_what_a_firm_specific_event_is_about():
    from tremor.modules.stress.scenarios import epicentre_of

    run = _event("evt_run", 8.0, regions=("US",), event_type="CREDIT_EVENT").model_copy(update={
        "market_wide": False, "entities": ["SIVB", "JPM", "WFC"], "entity_mentions": {"SIVB": 40, "JPM": 9, "WFC": 6}})
    assert epicentre_of(run) == ["SIVB"]  # banks named in coverage of SVB's run are not notched as if they failed
    deal = run.model_copy(update={"entities": ["CS", "UBS"], "entity_mentions": {"CS": 30, "UBS": 22}})
    assert epicentre_of(deal) == ["CS", "UBS"]


def test_risk_memo_records_the_evidence_as_it_stood_and_escapes_text():
    from tremor.modules.stress.memo import render_memo

    monitor = StressMonitor(BOOK, LIB, threshold=7.0)
    event = _event("evt_memo", 9.2).model_copy(update={"headline": "Russia invades <script>alert(1)</script> Ukraine"})
    run = monitor.on_event(event)
    html = render_memo(run, None, [], LIB.factors)  # no live event: the memo uses the run's own snapshot
    assert "Impact scorecard" in html and "Suggested actions" in html and "CET1" in html
    assert "<script>" not in html and "&lt;script&gt;" in html
    concentration = next(line for line in html.split("<li>") if line.startswith("Concentration"))
    names = concentration.split("(", 1)[1].split(")")[0].split(", ")
    assert len(names) == len(set(names))  # aggregated by obligor, no name twice


def test_the_memo_reviews_the_names_this_scenario_hits_first():
    from tremor.modules.stress.memo import names_to_review, render_memo

    run = StressMonitor(BOOK, LIB, threshold=7.0).on_event(_event("evt_review", 9.2))
    hit = run.positions[run.positions["obligor_id"] != "CCP"].groupby("obligor_id")["pnl"].sum().idxmin()
    watch = [  # the watchlist is bank-wide and ranked by its own score: the bystander comes first there
        {"entity_id": "ELSEWHERE", "name": "Bystander Corp", "status": "Watch Negative", "exposure_musd": 100.0},
        {"entity_id": hit, "name": "Hit Corp", "status": "Watch Negative", "exposure_musd": 50.0},
        {"entity_id": "MONITORED", "name": "Monitored Corp", "status": "Monitor", "exposure_musd": 80.0},
    ]
    assert [w["name"] for w in names_to_review(run, watch)] == ["Hit Corp", "Bystander Corp"]
    html = render_memo(run, None, watch, LIB.factors)
    assert "hardest hit by this scenario first): Hit Corp" in html
    assert "(no language model)" in html  # no summary drafted: the footer says so
    drafted = render_memo(run, None, watch, LIB.factors, summary={"source": "llm", "text": "Prose.", "note": "drafted"})
    assert "(no language model)" not in drafted and "worded by a local language model" in drafted
