import numpy as np
import pytest

from tremor.modules.rebalancer.methodology import (
    TiltConfig,
    apply_bounds,
    limit_turnover,
    rebalance,
    target_weights,
    tilt,
)

N = 20
PARENT = np.full(N, 1.0 / N)
SECTORS = np.array(["Tech"] * 5 + ["Comm"] * 5 + ["Staples"] * 3 + ["Disc"] * 2 + ["Ind"] * 2 + ["Fin", "Energy", "Health"])
CFG = TiltConfig()


def random_sentiment(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1, 1, N)


def test_neutral_sentiment_leaves_the_parent_index_unchanged():
    assert np.allclose(target_weights(PARENT, np.zeros(N), SECTORS, CFG), PARENT)


def test_positive_sentiment_increases_weight_and_negative_decreases_it():
    sentiment = np.zeros(N)
    sentiment[0], sentiment[1] = 0.5, -0.5
    w = target_weights(PARENT, sentiment, SECTORS, CFG)
    assert w[0] > PARENT[0] > w[1]


@pytest.mark.parametrize("seed", range(25))
def test_weights_are_fully_invested_and_respect_every_constraint(seed):
    w = target_weights(PARENT, random_sentiment(seed), SECTORS, CFG)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= CFG.min_multiple * PARENT - 1e-9).all()
    assert (w <= np.minimum(CFG.max_multiple * PARENT, CFG.max_weight) + 1e-9).all()
    for sector in np.unique(SECTORS):
        inside = SECTORS == sector
        assert abs(w[inside].sum() - PARENT[inside].sum()) <= CFG.sector_band + 1e-6


@pytest.mark.parametrize("seed", range(10))
def test_raising_one_sentiment_never_lowers_that_weight(seed):
    sentiment = random_sentiment(seed)
    before = target_weights(PARENT, sentiment, SECTORS, CFG)
    bumped = sentiment.copy()
    bumped[3] = min(1.0, bumped[3] + 0.3)
    after = target_weights(PARENT, bumped, SECTORS, CFG)
    assert after[3] >= before[3] - 1e-9


def test_tilt_is_invariant_to_a_market_wide_shift_in_sentiment():
    # If every stock's sentiment falls by the same amount, relative weights should not change.
    sentiment = random_sentiment(1) * 0.4
    assert np.allclose(tilt(PARENT, sentiment, 2.0), tilt(PARENT, sentiment - 0.3, 2.0))


def test_apply_bounds_rejects_infeasible_bounds():
    with pytest.raises(ValueError):
        apply_bounds(PARENT, np.full(N, 0.06), np.full(N, 0.10))


def test_turnover_is_capped_and_small_trades_are_skipped():
    target = target_weights(PARENT, random_sentiment(4), SECTORS, CFG)
    weights, turnover = limit_turnover(PARENT.copy(), target, max_turnover=0.05, no_trade_band=0.0025)
    assert turnover <= 0.05 + 1e-12
    assert weights.sum() == pytest.approx(1.0)
    assert 0.5 * np.abs(weights - PARENT).sum() == pytest.approx(turnover)

    nearly_same = PARENT + np.where(np.arange(N) % 2 == 0, 0.001, -0.001)
    weights, turnover = limit_turnover(PARENT.copy(), nearly_same, max_turnover=0.10, no_trade_band=0.0025)
    assert turnover == 0.0 and np.allclose(weights, PARENT)


def test_rebalance_reports_cost_from_turnover():
    result = rebalance(PARENT, PARENT.copy(), random_sentiment(2), SECTORS, CFG)
    assert result.cost == pytest.approx(result.turnover * CFG.cost_bps / 1e4)
    assert result.weights.sum() == pytest.approx(1.0)
