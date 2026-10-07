import pytest

from tremor.nlp.price_moves import price_directions, share_direction


@pytest.mark.parametrize("headline, expected", [
    # Real headlines from the Feb-2022 replay.
    ("Oil, gold cede gains on prospects for Biden-Putin Ukraine summit", {"OIL": -1, "GOLD": -1}),
    ("Brent breaches $94 mark as oil rises on supply concerns", {"OIL": 1}),
    ("Oil prices break $100 on Russian 'military operation' in Ukraine", {"OIL": 1}),
    ("Russia-Ukraine tensions send crude oil, gold prices spiralling high", {"OIL": 1, "GOLD": 1}),
    ("GLOBAL MARKETS-Stock futures rally, oil turns tail on Ukraine hopes", {"OIL": -1}),
    ("Ukraine, crude price surge seen as risks to financial stability", {"OIL": 1}),
    ("With gas prices rising, U.S. eyes emergency oil release", {"GAS": 1}),
    ("Gold slips as dollar firms; oil higher", {"GOLD": -1, "OIL": 1}),
    ("A jump in crude lifts energy shares", {"OIL": 1}),
])
def test_direction_of_named_prices(headline, expected):
    assert price_directions(headline, {"OIL", "GAS", "GOLD"}) == expected


@pytest.mark.parametrize("headline", [
    "US stocks fall amid Ukraine crisis, oil flirts with $100/barrel",  # no verb attached to oil
    "OPEC oil output falls to a two-year low",  # a quantity, not the price
    "Saudi Aramco Sees Good Signs Oil Demand's Rising as Shares Hit Record",
    "China's oil dependence on imports sees drop",
    "U.S. eyes oil reserves release as prices rise on Ukraine",
    "Greenhouse gas emissions fall 5%",
])
def test_quantities_and_unattached_verbs_give_no_direction(headline):
    assert price_directions(headline, {"OIL", "GAS"}) == {}


def test_direction_is_not_sentiment():
    # A negative sentence about a rising price: tone says down, the price is going up.
    assert price_directions("Oil soars as Russia invades Ukraine and stocks crash", {"OIL"}) == {"OIL": 1}


def test_only_linked_entities_are_considered():
    assert price_directions("Wall Street stocks tumble as oil surges", {"OIL"}) == {"OIL": 1}
    assert price_directions("Wall Street stocks tumble as oil surges", {"OIL", "SPX"}) == {"OIL": 1, "SPX": -1}


@pytest.mark.parametrize("headline, name, expected", [
    ("Apple shares jump as iPhone sales beat estimates; Intel slides on weak guidance", "Apple", 1),
    ("Amazon stock plunges after company issues disappointing revenue forecast", "Amazon", -1),
    ("AMD Stock Jumps 10% on Monday, Propelled by Meta (Facebook) Deal", "AMD", 1),
    ("Tesla jumps 9% after record deliveries", "Tesla", 1),  # bare name, but the move is quantified
    ("Netflix Share Price Crash: 4 Stocks to Shake Off the Trouble in Tech", "Netflix", -1),
    # A company name alone is not a share price.
    ("Boeing loses out to Airbus on a $37 billion order", "Boeing", 0),
    ("ExxonMobil plunges into plastics recycling", "ExxonMobil", 0),
    ("Apple sales jump 20% in China", "Apple", 0),
])
def test_share_price_direction_needs_a_price_context(headline, name, expected):
    assert share_direction(headline, name) == expected
