"""Index methodology: how sentiment becomes weights.

Written the way an index provider writes a tilted-index rulebook - a parent index, a tilt, and
explicit constraints - because an unconstrained "more sentiment, more weight" rule produces an
index nobody could run: it concentrates in a few names, drifts away from its sector mix and
trades itself to death on noise.

    1. tilt        w_i  proportional to  parent_i * exp(tilt_strength * sentiment_i)
    2. stock caps  each weight stays within [min_multiple, max_multiple] x parent and below max_weight
    3. sector band each sector stays within +/- sector_band of its parent weight
    4. turnover    trades smaller than no_trade_band are skipped; total one-way turnover is capped

Step 1 guarantees the brief's requirement: holding everything else fixed, a higher sentiment
score never gives a lower weight. All functions are pure and work on NumPy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel


class TiltConfig(BaseModel):
    tilt_strength: float = 2.0  # sentiment of +0.35 doubles a weight before constraints (exp(2 * 0.35) = 2.0)
    min_multiple: float = 0.4  # floor, as a multiple of the parent weight: nothing is ever sold out completely
    max_multiple: float = 2.0  # cap, as a multiple of the parent weight
    max_weight: float = 0.12  # absolute single-stock cap
    sector_band: float = 0.08  # maximum deviation of any sector from its parent weight
    max_turnover: float = 0.10  # one-way turnover allowed per rebalance
    no_trade_band: float = 0.0025  # weight changes smaller than this are not worth the trading cost
    cost_bps: float = 5.0  # transaction cost charged per unit of one-way turnover


@dataclass(frozen=True)
class Rebalance:
    target: np.ndarray  # where the methodology wants to be
    weights: np.ndarray  # where it actually moves to after the turnover rules
    turnover: float  # one-way: half the sum of absolute weight changes
    cost: float  # as a fraction of index value


def tilt(parent: np.ndarray, sentiment: np.ndarray, strength: float) -> np.ndarray:
    raw = parent * np.exp(strength * np.clip(sentiment, -1.0, 1.0))
    return raw / raw.sum()


def apply_bounds(weights: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Clamp to [lower, upper] and hand the excess to the unclamped names in proportion to their weight."""
    if lower.sum() > 1.0 + 1e-9 or upper.sum() < 1.0 - 1e-9:
        raise ValueError("bounds are infeasible: they cannot sum to 100%")
    w = weights.astype(float).copy()
    fixed = np.zeros(len(w), dtype=bool)
    for _ in range(len(w) + 1):
        free = ~fixed
        if not free.any():
            break
        w[free] *= (1.0 - w[fixed].sum()) / w[free].sum()
        over, under = free & (w > upper + 1e-12), free & (w < lower - 1e-12)
        if not over.any() and not under.any():
            break
        w[over], w[under] = upper[over], lower[under]
        fixed |= over | under
    return w


def apply_sector_bands(weights: np.ndarray, parent: np.ndarray, sectors: np.ndarray, band: float,
                       lower: np.ndarray, upper: np.ndarray, max_passes: int = 25) -> np.ndarray:
    """Scale any sector outside parent +/- band back to the edge, re-applying the stock bounds each pass."""
    w = weights.copy()
    for _ in range(max_passes):
        moved = False
        for sector in np.unique(sectors):
            inside = sectors == sector
            if inside.all():
                continue
            parent_weight, current = parent[inside].sum(), w[inside].sum()
            limit = min(max(current, parent_weight - band), parent_weight + band)
            if abs(limit - current) > 1e-9:
                w[inside] *= limit / current
                w[~inside] *= (1.0 - limit) / w[~inside].sum()
                moved = True
        w = apply_bounds(w, lower, upper)
        if not moved:
            break
    return w


def limit_turnover(current: np.ndarray, target: np.ndarray, max_turnover: float, no_trade_band: float) -> tuple[np.ndarray, float]:
    """Move from ``current`` towards ``target``, skipping tiny trades and capping total turnover."""
    delta = target - current
    traded = np.abs(delta) >= no_trade_band
    delta[~traded] = 0.0
    if traded.any():
        # Skipping small trades leaves the book slightly unbalanced; spread that residual over the real trades.
        delta[traded] -= delta.sum() * np.abs(delta[traded]) / np.abs(delta[traded]).sum()
    turnover = 0.5 * float(np.abs(delta).sum())
    if turnover > max_turnover:
        delta *= max_turnover / turnover
        turnover = max_turnover
    return current + delta, turnover


def target_weights(parent: np.ndarray, sentiment: np.ndarray, sectors: np.ndarray, cfg: TiltConfig) -> np.ndarray:
    """Steps 1-3: the constrained weights the methodology aims for, ignoring trading frictions."""
    lower = cfg.min_multiple * parent
    upper = np.minimum(cfg.max_multiple * parent, cfg.max_weight)
    upper = np.maximum(upper, lower)  # a parent weight above max_weight keeps its floor
    w = apply_bounds(tilt(parent, sentiment, cfg.tilt_strength), lower, upper)
    return apply_sector_bands(w, parent, sectors, cfg.sector_band, lower, upper)


def rebalance(parent: np.ndarray, current: np.ndarray, sentiment: np.ndarray, sectors: np.ndarray, cfg: TiltConfig) -> Rebalance:
    target = target_weights(parent, sentiment, sectors, cfg)
    weights, turnover = limit_turnover(current, target, cfg.max_turnover, cfg.no_trade_band)
    return Rebalance(target=target, weights=weights, turnover=turnover, cost=turnover * cfg.cost_bps / 1e4)
