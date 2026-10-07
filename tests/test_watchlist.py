from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from tremor.config import load_settings, load_taxonomy, load_universe
from tremor.engine.pipeline import RiskEngine
from tremor.modules.stress.portfolio import load_portfolio
from tremor.modules.watchlist import CreditWatch, _status, exposures
from tremor.nlp.models import load_text_model
from tremor.schemas import EventSignal

SETTINGS, UNIVERSE, TAXONOMY = load_settings(), load_universe(), load_taxonomy()
ENGINE = RiskEngine(SETTINGS, UNIVERSE, TAXONOMY, load_text_model(SETTINGS, TAXONOMY))  # no evidence: neutral sentiment
BOOK = load_portfolio()
NOW = datetime(2023, 3, 15, 14, tzinfo=timezone.utc)
NO_RUNS = SimpleNamespace(stress_runs=[])


def credit_event(event_id, impact, entities, n_docs=40, mentions=None, minutes_ago=30):
    at = NOW - timedelta(minutes=minutes_ago)
    return EventSignal(event_id=event_id, first_seen=at, last_updated=at, headline="Credit Suisse shares plunge as top investor rules out more help",
                       sentiment_score=-0.6, event_type="CREDIT_EVENT", event_type_label="Credit Event", impact_score=impact,
                       entities=entities, n_docs=n_docs, entity_mentions=mentions or {e: n_docs for e in entities})


def test_a_severe_credit_event_puts_its_subject_on_watch_negative():
    watch = CreditWatch(UNIVERSE, BOOK)
    watch.on_event(credit_event("evt_cs", 9.0, ["CS", "UBS"]))
    changes = watch.update(ENGINE, NO_RUNS, NOW)
    cs = next(e for e in watch.entries if e["entity_id"] == "CS")
    assert cs["status"] == "Watch Negative" and cs["factors"][0]["name"] == "Credit event"
    assert cs["exposure_musd"] > 0 and cs["rating"] == "BBB"  # the book's exposure is attached to the flag
    assert watch.first_flagged["CS"]["Watch Negative"] == NOW.isoformat()  # lead time is recorded
    assert any(c["entity_id"] == "CS" and c["to"] == "Watch Negative" for c in changes)


def test_a_bystander_named_in_passing_gets_at_most_one_point():
    watch = CreditWatch(UNIVERSE, BOOK)
    # Credit Suisse is what the event is about; JPMorgan is quoted in 2 of 40 reports.
    watch.on_event(credit_event("evt_cs", 9.0, ["CS", "JPM"], mentions={"CS": 38, "JPM": 2}))
    watch.update(ENGINE, NO_RUNS, NOW)
    flagged = {e["entity_id"]: e for e in watch.entries}
    assert flagged["CS"]["factors"][0]["points"] == 3.5
    assert "JPM" not in flagged  # one point from someone else's credit event is not enough to be flagged


def test_stale_events_drop_out():
    watch = CreditWatch(UNIVERSE, BOOK, window_hours=48)
    watch.on_event(credit_event("evt_old", 9.0, ["CS"], minutes_ago=60 * 50))
    watch.update(ENGINE, NO_RUNS, NOW)
    assert not any(e["entity_id"] == "CS" for e in watch.entries)


def test_hysteresis_stops_a_name_flapping_at_the_threshold():
    assert _status(1.5, None) == "Monitor" and _status(1.4, None) is None
    assert _status(1.4, "Monitor") == "Monitor"  # a flagged name needs a clearly lower score to leave
    assert _status(0.9, "Monitor") is None
    assert _status(2.7, "Watch Negative") == "Watch Negative" and _status(2.4, "Watch Negative") == "Monitor"


def test_exposure_reports_credit_protection_bought_separately():
    ex = exposures(BOOK)
    assert ex.loc["GAZP", "protection"] > 0  # the bank bought CDS protection on Gazprom
    assert ex.loc["GAZP", "exposure"] > ex.loc["GAZP", "protection"]


def test_a_company_much_of_the_coverage_speaks_of_negatively_is_a_subject_not_a_bystander():
    # Signature Bank's closure reported inside the larger SVB story; JPMorgan quoted in a few reports.
    event = credit_event("evt_svb", 9.0, ["SIVB", "SBNY", "JPM"], n_docs=1000, mentions={"SIVB": 800, "SBNY": 120, "JPM": 25})
    event = event.model_copy(update={"entity_sentiment": {"SIVB": -0.5, "SBNY": -0.5, "JPM": -0.5}})
    watch = CreditWatch(UNIVERSE, BOOK)
    watch.on_event(event)
    watch.update(ENGINE, NO_RUNS, NOW)
    flagged = {e["entity_id"]: e for e in watch.entries}
    assert flagged["SBNY"]["status"] == "Watch Negative"  # 120 negative reports: a subject of the event
    assert "JPM" not in flagged  # 25 reports, 2.5% of the story: named in passing


def test_a_situation_still_in_the_news_keeps_its_obligors_flagged():
    # The last stress test of the Russia-Ukraine situation is 50 hours old, but reports keep coming.
    from tremor.modules.stress.engine import StressMonitor
    from tremor.modules.stress.scenarios import ScenarioLibrary

    def war(event_id, impact, hours):
        at = NOW + timedelta(hours=hours)
        return EventSignal(event_id=event_id, first_seen=at, last_updated=at, headline="Russia presses offensive in Ukraine",
                           sentiment_score=-0.5, event_type="GEOPOLITICAL", event_type_label="Geopolitical", impact_score=impact,
                           market_wide=True, entities=["RU", "UA"], regions=["RU", "UA"])

    monitor = StressMonitor(BOOK, ScenarioLibrary(), threshold=7.0)
    assert monitor.on_event(war("evt_1", 8.0, 0)) is not None  # the stress test
    for i, hours in enumerate((24, 48)):  # still in the news, not materially worse: no new run
        assert monitor.on_event(war(f"evt_{i + 2}", 8.2, hours)) is None
    store = SimpleNamespace(stress_runs=monitor.runs)
    later = NOW + timedelta(hours=50)
    with_monitor, without = CreditWatch(UNIVERSE, BOOK), CreditWatch(UNIVERSE, BOOK)
    with_monitor.update(ENGINE, store, later, monitor)
    without.update(ENGINE, store, later)
    reasons = {e["entity_id"]: {f["name"] for f in e["factors"]} for e in with_monitor.entries}
    assert "Epicentre" in reasons["GAZP"] and "Epicentre" in reasons["SBER"]
    assert not any(e["entity_id"] == "GAZP" and any(f["name"] == "Epicentre" for f in e["factors"]) for e in without.entries)
