from tremor.config import load_universe
from tremor.nlp.entities import EntityLinker
from tremor.nlp.preprocess import clean_text, fingerprint, tidy_headline

linker = EntityLinker(load_universe())


def test_plain_alias_and_cashtag():
    assert linker.entity_ids("Boeing wins $3bn order while $TSLA slides") == ["BA", "TSLA"]


def test_longest_alias_wins():
    mentions = linker.link("JPMorgan Chase raises its dividend")
    assert [(m.entity_id, m.surface) for m in mentions] == [("JPM", "JPMorgan Chase")]


def test_ambiguous_name_needs_a_finance_cue():
    # The September 2018 problem: "Ford" tweets about Christine Blasey Ford, not Ford Motor.
    assert linker.entity_ids("Christine Ford testifies before the Senate committee") == []
    assert linker.entity_ids("Ford shares slide after it cuts guidance") == ["F"]
    assert linker.entity_ids("The Amazon rainforest is burning at a record rate") == []
    assert linker.entity_ids("Amazon earnings beat estimates on AWS strength") == ["AMZN"]


def test_finance_feed_unlocks_ambiguous_names():
    assert linker.entity_ids("Apple looking strong into the close", finance_context=True) == ["AAPL"]


def test_exact_aliases_are_case_sensitive():
    assert linker.entity_ids("Any intel on the meeting?") == []
    assert linker.entity_ids("Intel unveils its new processor") == ["INTC"]


def test_macro_actors_are_linked():
    ids = linker.entity_ids("Russia invades Ukraine; oil prices surge as the Fed weighs its response")
    assert ids == ["RU", "UA", "OIL", "FED"]


def test_word_boundaries():
    assert linker.entity_ids("The metadata was stored in a costly intelligent system") == []


def test_clean_text_and_fingerprint_collapse_retweets():
    original = "Tesla recalls 500,000 vehicles https://t.co/abc123"
    retweet = "RT @newsbot: Tesla recalls 500,000 vehicles https://t.co/xyz789"
    assert clean_text(retweet) == "Tesla recalls 500,000 vehicles"
    assert fingerprint(original) == fingerprint(retweet)
    assert clean_text("Stocks sink on war fears - Reuters", strip_publisher_suffix=True) == "Stocks sink on war fears"


def test_tidy_headline_drops_site_names_and_marks_truncation():
    assert tidy_headline("NATO chief: peace shattered in Europe | iNFOnews | Local News") == (
        "NATO chief: peace shattered in Europe")
    assert tidy_headline("Russian troops try to seize Chernobyl nuclear plant amid") == (
        "Russian troops try to seize Chernobyl nuclear plant amid…")
    assert tidy_headline("Oil jumps as Russia attacks, Biden says") == "Oil jumps as Russia attacks, Biden says"
