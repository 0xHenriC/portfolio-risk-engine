"""Backtest of the 99 % one-day VaR: exceptions, Kupiec, Christoffersen, Basel zones.

Each day's VaR is estimated on the 250 previous returns only, then compared with that
day's return: an exception is a day whose loss is worse than the VaR forecast the day
before. The returns are those of today's book revalued on past days, so the test says
whether the method would have held on this composition, not how the book was managed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2, norm

BASEL_WINDOW = 250
BACKTEST_LEVEL = 0.99
BASEL_YELLOW_FROM = 5
BASEL_RED_FROM = 10


def rolling_var(returns_pct: pd.Series, method: str, level: float = BACKTEST_LEVEL,
                window: int = BASEL_WINDOW) -> pd.Series:
    """VaR forecast for each day, estimated on the preceding `window` returns only."""
    rolling = returns_pct.rolling(window)
    if method == "historical":
        forecast = rolling.quantile(1 - level, interpolation="linear")
    elif method == "normal":
        forecast = rolling.mean() - rolling.std(ddof=1) * norm.ppf(level)
    else:
        raise ValueError(f"unknown VaR method '{method}'")
    return forecast.shift(1).dropna()


def kupiec_pof(n: int, exceptions: int, p: float) -> tuple[float, float]:
    """Proportion-of-failures likelihood ratio and its chi-square(1) p-value.

    H0: the true exception probability is p. A low p-value means too many, or too few,
    exceptions for the stated confidence level.
    """
    x, phat = exceptions, exceptions / n
    lr = -2 * (xlogy(n - x, 1 - p) + xlogy(x, p) - xlogy(n - x, 1 - phat) - xlogy(x, phat))
    return float(lr), float(chi2.sf(lr, 1))


def christoffersen_independence(hits: np.ndarray) -> tuple[float, float]:
    """Independence likelihood ratio of the exception sequence, chi-square(1) p-value.

    H0: an exception today is as likely after an exception as after a quiet day. A low
    p-value means exceptions come in clusters, so the VaR reacts too slowly.
    """
    prev, curr = hits[:-1], hits[1:]
    n00 = int(np.sum(~prev & ~curr))
    n01 = int(np.sum(~prev & curr))
    n10 = int(np.sum(prev & ~curr))
    n11 = int(np.sum(prev & curr))
    pi0 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi1 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / len(curr)
    log_h0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    log_h1 = xlogy(n00, 1 - pi0) + xlogy(n01, pi0) + xlogy(n10, 1 - pi1) + xlogy(n11, pi1)
    lr = max(-2 * (log_h0 - log_h1), 0.0)
    return float(lr), float(chi2.sf(lr, 1))


def basel_zone(exceptions_in_250: int) -> str:
    """Traffic light for 250 observations at 99 %: green 0-4, yellow 5-9, red 10 or more."""
    if exceptions_in_250 >= BASEL_RED_FROM:
        return "red"
    return "yellow" if exceptions_in_250 >= BASEL_YELLOW_FROM else "green"


def backtest_var(returns_pct: pd.Series, method: str, level: float = BACKTEST_LEVEL,
                 window: int = BASEL_WINDOW) -> dict:
    """Run the full backtest of one VaR method over every day with a forecast."""
    forecast = rolling_var(returns_pct, method, level, window)
    realised = returns_pct.loc[forecast.index]
    hits = (realised < forecast).to_numpy()
    n, x = len(hits), int(hits.sum())
    last = hits[-BASEL_WINDOW:]
    lr_pof, p_pof = kupiec_pof(n, x, 1 - level)
    lr_ind, p_ind = christoffersen_independence(hits)
    return {
        "method": method, "level": level, "days": n, "exceptions": x,
        "expected": n * (1 - level), "rate": x / n,
        "kupiec_lr": lr_pof, "kupiec_p": p_pof,
        "christoffersen_lr": lr_ind, "christoffersen_p": p_ind,
        "conditional_coverage_p": float(chi2.sf(lr_pof + lr_ind, 2)),
        "exceptions_last_250": int(last.sum()), "zone_last_250": basel_zone(int(last.sum())),
        "worst_250_exceptions": int(pd.Series(hits).rolling(BASEL_WINDOW).sum().max()),
        "forecast": forecast, "realised": realised,
    }
