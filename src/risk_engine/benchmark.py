"""Composite benchmark and alpha / beta of the book against it.

The benchmark is built from regional index trackers ("bricks"), weighted like the
book's currency exposure (cash excluded), and rebalanced daily at today's weights.
Beta is the OLS slope of daily portfolio returns on benchmark returns; alpha is the
daily intercept annualised simply (x 252). Both sides are total return: a distributing
tracker has its distributions added back. The book's dividends are gross while index
trackers follow net-of-tax indices, so the resulting bias is measured and shown as a
band, not corrected.
"""

from __future__ import annotations

import math

import pandas as pd

CURRENCY_TO_BRICK = {
    **dict.fromkeys(["EUR", "GBP", "CHF", "SEK", "DKK", "NOK"], "Europe"),
    "JPY": "Japan", "USD": "US", "INR": "India", "Gold": "Gold",
    **dict.fromkeys(["HKD", "SGD", "KRW", "TWD"], "Asia ex-Japan"),
}
BRICK_FX_HEDGED = {"Europe": False, "Japan": True, "Asia ex-Japan": False, "US": True,
                   "India": False, "Gold": None}
MIN_OBSERVATIONS = 60
MAX_FILL_SESSIONS = 3
TRADING_DAYS = 252
WITHHOLDING_BAND = (0.15, 0.25)


def composite_weights(currency_weights: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Brick weights from currency weights, and the currencies no brick covers.

    Unmapped currencies are excluded and reported; they are never pushed into a brick
    by default, which would claim a coverage the benchmark does not have.
    """
    brick = currency_weights.index.map(lambda c: CURRENCY_TO_BRICK.get(c))
    mapped = currency_weights[brick.notna()]
    weights = mapped.groupby(brick[brick.notna()]).sum()
    return weights / weights.sum(), currency_weights[brick.isna()]


def total_return(prices: pd.Series, distributions: pd.Series | None = None) -> pd.Series:
    """Total-return index: TR_t = TR_{t-1} x (P_t + D_t) / P_{t-1}."""
    d = (distributions if distributions is not None else pd.Series(dtype=float))
    d = d.groupby(level=0).sum().reindex(prices.index).fillna(0.0)
    step = (prices + d) / prices.shift(1)
    return 100 * step.fillna(1.0).cumprod()


def align(series: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Series on a Monday-Friday calendar, gaps filled over at most three sessions.

    Unbounded filling would create runs of zero returns, which inflate the benchmark's
    apparent stability and drag beta down.
    """
    grid = series.index.union(calendar)
    return series.reindex(grid).ffill(limit=MAX_FILL_SESSIONS).reindex(calendar)


def compute_alpha_beta(portfolio_total: pd.Series, portfolio_price: pd.Series,
                       bricks: pd.DataFrame, distributions: pd.DataFrame,
                       weights: pd.Series, all_held_from: pd.Timestamp) -> dict:
    """Beta, annualised alpha and the diagnostics that say how far to trust them."""
    used = list(weights.index)
    tr = {b: total_return(bricks[b], distributions.loc[distributions["brick"] == b]
                          .set_index("ex_date")["amount"]) for b in used}
    series = [portfolio_total, *tr.values()]
    start = max(s.first_valid_index() for s in series)
    end = min(s.last_valid_index() for s in series)
    calendar = pd.bdate_range(start, end)
    frame = pd.DataFrame({"pf_total": align(portfolio_total, calendar),
                          "pf_price": align(portfolio_price, calendar),
                          **{b: align(s, calendar) for b, s in tr.items()}})
    dropped = int(frame.isna().any(axis=1).sum())
    rets = frame.dropna().pct_change(fill_method=None).dropna()
    r_b = (rets[used] * weights).sum(axis=1)
    r_p = rets["pf_total"]
    n = len(rets)
    base = {"n": n, "dropped_dates": dropped, "weights": weights,
            "composition_coverage_pct": float((calendar >= all_held_from).mean() * 100)}
    if n < MIN_OBSERVATIONS or r_b.var(ddof=1) == 0:
        return {**base, "ok": False,
                "message": f"{n} observations, {MIN_OBSERVATIONS} required, or flat benchmark"}
    beta = float(r_p.cov(r_b) / r_b.var(ddof=1))
    alpha_daily = float(r_p.mean() - beta * r_b.mean())
    dividend_pct = float((r_p.mean() - rets["pf_price"].mean()) * TRADING_DAYS * 100)
    return {
        **base, "ok": True, "beta": beta,
        "alpha_annual_pct": alpha_daily * TRADING_DAYS * 100,
        "r_squared": float(r_p.corr(r_b) ** 2),
        "tracking_error_pct": float((r_p - r_b).std(ddof=1) * math.sqrt(TRADING_DAYS) * 100),
        "dividend_contribution_pct": dividend_pct,
        "withholding_bias_band_pct": tuple(dividend_pct * x for x in WITHHOLDING_BAND),
        "benchmark_index": 100 * (1 + r_b).cumprod(),
        "portfolio_index": 100 * (1 + r_p).cumprod(),
    }
