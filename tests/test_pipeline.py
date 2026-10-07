import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tremor.config import load_settings, load_taxonomy, load_universe
from tremor.engine.pipeline import RiskEngine
from tremor.nlp.models import load_text_model
from tremor.schemas import RawDocument, SourceKind

SETTINGS, UNIVERSE, TAXONOMY = load_settings(), load_universe(), load_taxonomy()
MODEL = load_text_model(SETTINGS, TAXONOMY)
T0 = datetime(2022, 2, 24, 3, 0, tzinfo=timezone.utc)


def doc(text, minutes=0, publisher="example.com", kind=SourceKind.NEWS):
    return RawDocument(doc_id=RawDocument.make_id(publisher, f"{text}|{minutes}"), source="test", kind=kind,
                       published_at=T0 + timedelta(minutes=minutes), text=text, publisher=publisher)


def engine():
    return RiskEngine(SETTINGS, UNIVERSE, TAXONOMY, MODEL)


def test_model_is_the_fine_tuned_encoder():
    assert MODEL.name == "tremor-encoder", "models/tremor-encoder is missing: the engine fell back to keywords"


def test_signals_carry_the_three_required_fields():
    result = engine().process([doc("Russia launches full-scale military invasion of Ukraine; oil prices surge")])
    event = result.events[0]
    assert -1.0 <= event.sentiment_score <= 1.0
    assert event.event_type in TAXONOMY.ids
    assert 1.0 <= event.impact_score <= 10.0


def test_syndicated_copies_are_folded_into_corroboration():
    eng = engine()
    text = "Boeing cuts 737 MAX delivery forecast after new production flaw"
    eng.process([doc(text, publisher="a.com")])
    result = eng.process([doc(text, 5, publisher="b.com"), doc(text, 6, publisher="c.com")])
    assert result.duplicates == 2
    assert result.events[0].n_docs == 3 and result.events[0].n_publishers == 3


def test_independent_reports_raise_impact():
    eng = engine()
    first = eng.process([doc("Russian troops cross into Ukraine as Putin declares military operation", 0, "a.com")]).events[0]
    reports = [
        "Russia invades Ukraine, explosions heard in Kyiv as war begins",
        "Putin orders attack on Ukraine; markets tumble and oil spikes above $100",
        "Western leaders condemn Russian invasion of Ukraine and vow severe sanctions",
        "Ukraine declares martial law after Russian missile strikes on military targets",
        "Stocks plunge worldwide as Russia launches assault on Ukraine",
        "Russian forces attack Ukraine from the north, east and south",
    ]
    later = eng.process([doc(t, 10 + i, f"outlet{i}.com") for i, t in enumerate(reports)]).events
    biggest = max(later, key=lambda e: e.n_stories)
    assert biggest.event_type == "GEOPOLITICAL"
    assert biggest.n_stories >= 3
    assert biggest.impact_score > first.impact_score


def test_noise_moves_nothing():
    result = engine().process([doc("Retweet and follow us for a chance to win a £100 Amazon voucher #giveaway", kind=SourceKind.SOCIAL)])
    assert result.noise == 1 and not result.events


def test_entity_level_sentiment_separates_winners_and_losers():
    out = engine().analyse_text("Apple shares surge to a record high on blowout iPhone sales while Intel plunges after slashing its forecast")
    scores = {e["id"]: e["sentiment"] for e in out["entities"]}
    assert out["entity_level_sentiment"]
    assert scores["AAPL"] > 0 > scores["INTC"]


def test_state_decays_back_to_neutral():
    eng = engine()
    eng.process([doc("JPMorgan shares plunge after a surprise multibillion-dollar trading loss", 0, "a.com"),
                 doc("JPMorgan stock tumbles as regulators open a probe into the trading loss", 3, "b.com")])
    now = eng.entity_signal("JPM").sentiment_score
    later = eng.entity_signal("JPM", now=T0 + timedelta(days=10)).sentiment_score
    assert now < -0.05
    assert abs(later) < abs(now) / 10
    assert eng.entity_signal("JPM", now=T0 + timedelta(days=10)).sentiment_uncertainty > eng.entity_signal("JPM").sentiment_uncertainty


@pytest.mark.parametrize("headline,expected", [
    ("Evergrande misses bond coupon payment, edges towards default", "CREDIT_EVENT"),
    ("Microsoft agrees to buy Activision Blizzard for $69 billion", "MERGER_ACQUISITION"),
    ("Fed raises interest rates by 75 basis points to fight inflation", "MACROECONOMIC"),
    ("Apple unveils the iPhone 14 and a new Apple Watch at its September event", "PRODUCT_LAUNCH"),
])
def test_event_classification(headline, expected):
    assert engine().analyse_text(headline)["event_type"] == expected


@pytest.mark.parametrize("headline", [
    "Amazon and Walmart are locked in a price war over delivery fees",
    "Two of Ecommerce's Biggest Names Just Launched a Silent War",
    "First Look! God of War Ragnarök Funko POPs! Going live at the links below",
    "Microsoft builds a $100 billion war chest for acquisitions",
])
def test_war_as_a_figure_of_speech_is_not_geopolitical(headline):
    result = engine().analyse_text(headline)
    assert result["event_type"] != "GEOPOLITICAL"
    assert not any("extreme" in f["detail"] for f in result["impact_factors"])  # an idiom is not severe language


@pytest.mark.parametrize("headline", [
    "Russia launches full-scale war on Ukraine",
    "War fears send stocks lower as troops mass on the border",
    "Russia's silent war on Ukraine's power grid escalates",
    "Price war erupts in Russian oil as sanctions bite",
])
def test_literal_war_stays_geopolitical(headline):
    assert engine().analyse_text(headline)["event_type"] == "GEOPOLITICAL"


@pytest.mark.parametrize("headline, geopolitical", [
    ("SVB's Balance-Sheet Time Bomb Was 'Sitting in Plain Sight' (SIVB)", False),  # the model alone says Geopolitical, 0.95
    ("Why the commercial property time bomb could hit banks next", False),
    ("Russia's seizure of Chernobyl is a ticking time bomb", True),  # a country is named: literal
    ("Bomb attack kills dozens in Pakistan market", True),
])
def test_time_bomb_as_a_figure_of_speech_is_not_geopolitical(headline, geopolitical):
    assert (engine().analyse_text(headline)["event_type"] == "GEOPOLITICAL") is geopolitical


REPRODUCIBILITY_PROBE = """
import json
from tremor.config import load_settings, load_taxonomy, load_universe
from tremor.engine.pipeline import RiskEngine
from tremor.ingestion.replay import load_pack
from tremor.nlp.models import load_text_model
settings, taxonomy = load_settings(), load_taxonomy()
eng = RiskEngine(settings, load_universe(), taxonomy, load_text_model(settings, taxonomy))
docs, out = load_pack("ukraine_2022")[:2500], {}
for i in range(0, len(docs), 250):
    result = eng.process(docs[i:i + 250])
    for e in result.events:
        out[e.event_id] = [e.headline, e.impact_score, [s.story_id for s in e.stories]]
    for s in result.entities:
        out[s.entity_id] = [s.top_event_id, s.event_type]
print(json.dumps(out, sort_keys=True))
"""


def test_results_do_not_depend_on_python_hash_seed():
    # Sets iterate in a per-process random order. Ties in the story ranking once made two runs of the
    # same replay pick different headlines, and so different stress-test scenarios.
    root = Path(__file__).resolve().parents[1]
    runs = [subprocess.Popen([sys.executable, "-W", "ignore", "-c", REPRODUCIBILITY_PROBE], cwd=root, stdout=subprocess.PIPE,
                             text=True, env={**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONHASHSEED": seed})
            for seed in ("1", "2")]
    outputs = [p.communicate(timeout=600)[0] for p in runs]
    assert [p.returncode for p in runs] == [0, 0]
    assert outputs[0] and outputs[0] == outputs[1]


@pytest.mark.parametrize("headline, geopolitical", [
    ("Warren, Porter Take First Step to Repeal Trump-Era Law Blamed for SVB Collapse", False),  # the model alone: Geopolitical
    ("CEO of collapsed Silicon Valley Bank successfully lobbied Congress to loosen rules", False),
    ("Congress weighs new sanctions on Russia", True),  # a foreign actor and a conflict cue
    ("Lawmakers call for hearing on Taiwan security as China masses ships", True),
])
def test_domestic_politics_is_not_geopolitics(headline, geopolitical):
    assert (engine().analyse_text(headline)["event_type"] == "GEOPOLITICAL") is geopolitical
