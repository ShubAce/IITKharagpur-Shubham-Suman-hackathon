"""Dictionary sentiment and a hashing embedder: the *naive baseline*, and the offline fallback.

This is what a keyword-counting pipeline does. The engine uses it only when the neural model
cannot be loaded (no network on first run), and the evaluation report uses it as the
reference point that every learned component has to beat.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_POSITIVE = frozenset("""
beat beats beating surge surges surged soar soars soared jump jumps jumped gain gains gained rally rallies rallied
rise rises rose rising climb climbs climbed up upgrade upgrades upgraded outperform outperforms strong stronger
strength record growth grow grows grew profit profits profitable win wins won boost boosts boosted raise raises
raised bullish bull buy buying breakout rebound rebounds recover recovers recovery approve approves approved
approval success successful exceed exceeds exceeded top tops topped expand expands expansion positive optimistic
optimism upbeat robust solid momentum breakthrough moon rocket winner outpace higher best improve improves improved
""".split())
_NEGATIVE = frozenset("""
miss misses missed plunge plunges plunged plummet plummets slump slumps slumped fall falls fell falling drop drops
dropped decline declines declined slide slides slid sink sinks sank tumble tumbles tumbled crash crashes crashed
down downgrade downgrades downgraded underperform weak weaker weakness loss losses lose loses lost cut cuts
cutting slash slashes slashed warn warns warned warning bearish bear sell selling selloff lawsuit sued fraud probe
investigation recall recalls default defaults bankruptcy bankrupt layoff layoffs fired fine fined penalty risk
risks fear fears concern concerns crisis collapse collapses collapsed war sanctions negative pessimistic worst
lower hurt hurts delay delays delayed halt halts halted shortage breach scandal dump bagholder puts
""".split())
_NEGATORS = frozenset({"not", "no", "never", "without", "less", "fewer", "n't", "cannot", "neither", "nor"})
_TOKEN = re.compile(r"[a-z']+")


def lexicon_sentiment(text: str) -> tuple[float, float, float]:
    """Return ``(p_negative, p_neutral, p_positive)`` from word counts with simple negation flipping."""
    tokens = _TOKEN.findall(text.lower())
    pos = neg = 0
    for i, token in enumerate(tokens):
        polarity = 1 if token in _POSITIVE else (-1 if token in _NEGATIVE else 0)
        if polarity == 0:
            continue
        if any(t in _NEGATORS or t.endswith("n't") for t in tokens[max(0, i - 3): i]):
            polarity = -polarity
        if polarity > 0:
            pos += 1
        else:
            neg += 1
    if pos == neg == 0:
        return 0.1, 0.8, 0.1
    score = (pos - neg) / (pos + neg)  # -1 .. 1
    strength = min(1.0, (pos + neg) / 3.0) * 0.8
    p_pos = max(score, 0.0) * strength + (1 - strength) / 3
    p_neg = max(-score, 0.0) * strength + (1 - strength) / 3
    return p_neg, max(0.0, 1.0 - p_pos - p_neg), p_pos


def hashing_embedding(texts: list[str], dim: int = 256) -> np.ndarray:
    """Bag-of-words hashing trick: good enough to group near-identical headlines, nothing more."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in _TOKEN.findall(text.lower()):
            if len(token) > 2:
                out[row, int(hashlib.md5(token.encode()).hexdigest()[:8], 16) % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(norms, 1e-12, None)
