import re
import time
from datetime import datetime, timezone

import pytest
import requests

from tremor.config import LlmSettings
from tremor.modules.stress import narrative
from tremor.modules.stress.engine import StressMonitor
from tremor.modules.stress.memo import render_memo
from tremor.modules.stress.narrative import (
    _PLACEHOLDER,
    _TERMS,
    SummaryDrafts,
    check_draft,
    executive_summary,
    fill,
    slot_facts,
    slots,
    tidy,
)
from tremor.modules.stress.portfolio import load_portfolio
from tremor.modules.stress.scenarios import ScenarioLibrary
from tremor.schemas import EventSignal

LIB = ScenarioLibrary()
NOW = datetime(2022, 2, 24, 5, 45, tzinfo=timezone.utc)
WATCH = [{"status": "Watch Negative", "exposure_musd": 258.0, "name": "Sberbank of Russia"}]
GOOD = ("A [EVENT_TYPE], [HEADLINE], reached impact [IMPACT]. Under the scenario the book would lose [LOSS], and the "
        "CET1 ratio would fall from [CET1_BEFORE] to [CET1_AFTER], still above the [REQUIREMENT] requirement. Review [NAMES] first.")


@pytest.fixture(scope="module")
def run():
    event = EventSignal(event_id="evt_inv", first_seen=NOW, last_updated=NOW, headline="Putin announces military assault on Ukraine",
                        sentiment_score=-0.4, event_type="GEOPOLITICAL", event_type_label="Geopolitical", impact_score=9.4,
                        market_wide=True, entities=["RU", "UA"], regions=["RU", "UA"], n_docs=9294, n_stories=1741, n_publishers=1645)
    result = StressMonitor(load_portfolio(), LIB, threshold=7.0).on_event(event)
    assert not result.capital["breach"]  # the checks below assume capital stays above the requirement
    return result


def test_the_model_writes_prose_and_the_engine_writes_every_figure(run):
    out = executive_summary(run, WATCH, generate=lambda prompt: GOOD)
    values = slots(run, WATCH)
    assert out["source"] == "llm" and "[" not in out["text"]
    assert values["LOSS"] in out["text"] and values["CET1_AFTER"] in out["text"]  # exact, as computed


@pytest.mark.parametrize("draft, problem", [
    (GOOD.replace("[LOSS]", "about $270m"), "numbers of its own"),  # it computed a figure itself
    (GOOD.replace("still above", "below"), "falls below the requirement"),  # wrong capital claim (it stays above)
    (GOOD.replace("would lose", "has incurred a loss of"), "has happened"),  # a scenario told as a realised loss
    (GOOD.replace("[CET1_AFTER]", "a lower level"), "left out CET1_AFTER"),
    (GOOD + " Spreads widen by [SPREAD].", "not in the facts: SPREAD"),
    (GOOD.replace("still above the [REQUIREMENT] requirement", "a shortfall of [SHORTFALL]"), "not in the facts: SHORTFALL"),
    (GOOD.replace("would fall", "would increase"), "CET1 ratio rises"),  # the direction flipped (seen in a repaired draft)
    (GOOD.replace("Review [NAMES] first.", "The bank has been hit by the event."), "already been hit"),
    (GOOD + " This revised draft fixes the capital statement.", "talks about the draft"),
    (GOOD.removesuffix("."), "stops mid-sentence"),
    (GOOD + " According to the reports, volatility could rise.", "only count the reports"),  # seen in six drafts
    (GOOD.replace("would fall from [CET1_BEFORE] to", "would remain above [CET1_BEFORE] to"), "with its own level"),
])
def test_a_draft_that_breaks_a_rule_is_rejected_for_the_template(run, draft, problem):
    out = executive_summary(run, WATCH, generate=lambda prompt: draft)
    assert out["source"] == "template" and problem in out["note"]
    assert out["text"] == executive_summary(run, WATCH)["text"]  # the deterministic summary instead


def test_what_a_chat_model_wraps_around_the_paragraph_is_dropped_and_figures_read_cleanly(run):
    assert tidy(f"Here is the revised opening paragraph: {GOOD} Note: I fixed the capital statement.") == GOOD
    text = fill("The book would lose $[LOSS], [LOSS_PCT]% of its value. [EVENT_TYPE] news.", slots(run, WATCH))
    assert "$$" not in text and "%%" not in text and "Geopolitical event news." in text  # capitalised to start a sentence
    assert '""' not in fill('A "[HEADLINE]" story.', slots(run, WATCH))  # the headline brings its own quotes
    assert fill("A [EVENT_TYPE] and an [IMPACT].", {"EVENT_TYPE": "operational incident event", "IMPACT": "7.9"}) == (
        "An operational incident event and a 7.9.")
    assert fill("An [HEADLINE] story.", {"HEADLINE": '"European stocks at risk"'}) == 'A "European stocks at risk" story.'


def test_names_with_digits_are_not_figures():
    assert check_draft(GOOD.replace("CET1 ratio", "Common Equity Tier 1 (CET1) ratio, as defined under IFRS 9,"), breach=False) == []
    assert check_draft(GOOD.replace("[IMPACT]", "[IMPACT] out of 10"), breach=False) == []  # the scale, not a figure
    assert "numbers of its own (8)" in check_draft(GOOD.replace("[IMPACT]", "8 out of 10"), breach=False)[0]  # a made-up score
    assert check_draft(GOOD.replace("[IMPACT]", "[IMPACT] on a one-to-ten scale"), breach=False) == []
    assert "numbers of its own (five)" in check_draft(GOOD + " It blends five earlier crises.", breach=False)[0]  # spelled out


def test_the_model_sees_the_facts_in_words_and_never_a_digit(run):
    facts = "\n".join(slot_facts(run))
    assert not re.search(r"\d", _TERMS.sub("", _PLACEHOLDER.sub("", facts)))  # so any digit in a draft is one it made up
    assert "equities fall" in facts and "credit spreads widen" in facts  # which way the scenario moves markets
    assert "the largest losses come through credit losses (ECL)" in facts


def test_a_rejected_draft_goes_back_with_the_reasons_and_can_be_repaired(run):
    prompts = []

    def model(prompt):
        prompts.append(prompt)
        return GOOD.replace("[IMPACT]", "8 out of 10") if len(prompts) == 1 else GOOD

    out = executive_summary(run, WATCH, generate=model)
    assert out["source"] == "llm" and out["attempts"] == 2 and "attempt 2" in out["note"]
    assert "numbers of its own (8)" in prompts[1] and "8 out of 10" in prompts[1]  # the reasons and the rejected draft


def test_a_failing_model_falls_back_and_the_memo_shows_only_accepted_drafts(run):
    def broken(prompt):
        raise requests.Timeout("too slow")

    out = executive_summary(run, WATCH, generate=broken)
    assert out["source"] == "template" and "Timeout" in out["note"]
    html = render_memo(run, None, WATCH, LIB.factors, summary=out)
    assert "Executive summary:" in html and "<h2>Executive summary</h2>" not in html
    accepted = executive_summary(run, WATCH, generate=lambda prompt: GOOD)
    assert "<h2>Executive summary</h2>" in render_memo(run, None, WATCH, LIB.factors, summary=accepted)


def test_without_a_server_the_memo_never_waits(run):
    drafts = SummaryDrafts(LlmSettings(enabled=True, url="http://127.0.0.1:9"))  # nothing listens on port 9
    out = drafts.get(run, WATCH)
    assert out["source"] == "template" and "no local language model" in out["note"]


def test_a_drafted_summary_names_who_is_on_the_watchlist_now(run, monkeypatch):
    monkeypatch.setattr(narrative, "_ollama", lambda settings: lambda prompt: GOOD)  # a model that answers at once
    drafts = SummaryDrafts(LlmSettings(enabled=True))
    assert "drafting" in drafts.get(run, WATCH)["note"]  # the first request queues the draft and does not wait
    deadline = time.monotonic() + 10
    while drafts.get(run, WATCH)["source"] != "llm" and time.monotonic() < deadline:
        time.sleep(0.01)
    later = drafts.get(run, WATCH + [{"status": "Watch Negative", "exposure_musd": 120.0, "name": "Gazprom"}])
    assert later["source"] == "llm" and "Gazprom" in later["text"]  # filled again with the watchlist of this request


def test_the_language_model_is_off_unless_an_analyst_switches_it_on(run):
    out = SummaryDrafts(LlmSettings()).get(run, WATCH)
    assert out["source"] == "template" and "note" not in out  # no model, no note: the memo is deterministic
    html = render_memo(run, None, WATCH, LIB.factors, summary=out)
    assert "Executive summary" not in html and "(no language model)" in html
