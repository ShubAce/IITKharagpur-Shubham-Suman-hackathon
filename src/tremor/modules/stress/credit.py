"""Credit-risk building blocks: rating scale, stressed default probabilities, expected loss, capital.

Standard, published methodology only:

* PD under stress - the Vasicek single-factor model behind the Basel IRB formula. A borrower
  defaults when its asset value falls below its debt; a systematic shock ``z`` (in standard
  deviations; negative = bad economy) shifts every borrower's asset value at once, which moves
  the probit of each PD by a correlation-dependent amount:

      PD(z) = N( N^-1(PD) - z * sqrt(rho / (1 - rho)) )

* Expected credit loss (IFRS 9 style) - 12-month ECL for performing loans (stage 1), lifetime ECL
  once credit risk has increased significantly (stage 2: PD up 3x or more since origination).
* Risk-weighted assets - the Basel IRB corporate risk-weight function with the regulatory asset
  correlation and maturity adjustment.

The rating -> PD master scale is illustrative (long-run average corporate default rates), not
any agency's or bank's proprietary scale.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri  # standard normal CDF and its inverse

RATING_PD = {  # one-year probability of default per rating grade (illustrative master scale)
    "AAA": 0.0001, "AA": 0.0002, "A": 0.0006, "BBB": 0.0020, "BB": 0.0090, "B": 0.0350, "CCC": 0.2000, "D": 1.0,
}
RATING_ORDER = list(RATING_PD)
PD_FLOOR = 0.0003  # Basel floor for corporate PD
SICR_MULTIPLE = 3.0  # significant increase in credit risk: stressed PD at least 3x origination PD


def rating_to_pd(ratings) -> np.ndarray:
    return np.asarray([RATING_PD[r] for r in ratings], dtype=float)


def pd_to_rating(pd_values) -> list[str]:
    """Nearest grade on a log scale (used to show stressed ratings)."""
    grades = np.log(np.asarray([RATING_PD[g] for g in RATING_ORDER[:-1]]))
    out = []
    for p in np.atleast_1d(pd_values):
        out.append("D" if p >= 0.999 else RATING_ORDER[int(np.argmin(np.abs(grades - np.log(max(p, 1e-6)))))])
    return out


def asset_correlation(pd: np.ndarray) -> np.ndarray:
    """Basel corporate asset correlation: 12% for risky borrowers rising to 24% for the safest."""
    w = (1.0 - np.exp(-50.0 * pd)) / (1.0 - np.exp(-50.0))
    return 0.12 * w + 0.24 * (1.0 - w)


def stressed_pd(pd: np.ndarray, z: np.ndarray | float) -> np.ndarray:
    """PD after a systematic shock ``z`` (standard deviations; negative = downturn).

    Vasicek's conditional default threshold, expressed relative to the borrower's *current* PD:
    the probit of the PD moves by ``-z * sqrt(rho / (1 - rho))``. Unlike the raw conditional formula
    evaluated at z = 0 (which returns the median, below the average PD), a zero shock leaves every
    PD exactly where it is - so a no-change scenario produces no change in losses.
    """
    pd = np.clip(pd, 1e-6, 1.0 - 1e-9)
    rho = asset_correlation(pd)
    return ndtr(ndtri(pd) - np.asarray(z) * np.sqrt(rho / (1.0 - rho)))


def lifetime_pd(pd_1y: np.ndarray, years: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - np.clip(pd_1y, 0, 1)) ** np.clip(years, 1.0, None)


def expected_credit_loss(pd_1y: np.ndarray, lgd: np.ndarray, ead: np.ndarray, years: np.ndarray, stage2: np.ndarray) -> np.ndarray:
    horizon_pd = np.where(stage2, lifetime_pd(pd_1y, years), pd_1y)
    return horizon_pd * lgd * ead


def irb_risk_weight(pd: np.ndarray, lgd: np.ndarray, maturity: np.ndarray) -> np.ndarray:
    """Basel II/III IRB risk weight for corporate exposures (as a fraction of EAD)."""
    pd = np.clip(pd, PD_FLOOR, 0.9999)
    rho = asset_correlation(pd)
    b = (0.11852 - 0.05478 * np.log(pd)) ** 2
    m = np.clip(maturity, 1.0, 5.0)
    k = lgd * ndtr((ndtri(pd) + np.sqrt(rho) * ndtri(0.999)) / np.sqrt(1.0 - rho)) - pd * lgd
    k = k * (1.0 + (m - 2.5) * b) / (1.0 - 1.5 * b)
    return np.clip(k, 0.0, None) * 12.5


def standardised_bond_weight(ratings) -> np.ndarray:
    """Basel standardised risk weights for rated corporate and sovereign bonds (simplified)."""
    table = {"AAA": 0.2, "AA": 0.2, "A": 0.5, "BBB": 1.0, "BB": 1.0, "B": 1.5, "CCC": 1.5, "D": 1.5}
    return np.asarray([table[r] for r in ratings], dtype=float)
