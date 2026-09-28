"""Gross dividends: EUR conversion, entitlement, yield on cost and a plausibility cap.

Dividends are dated by their ex-date and kept gross (no withholding tax). The EUR rate is
the one implied by the same security's closes (EUR close / local close) on the ex-date
or the last session before it, so the dividend and the price it detaches from are
converted consistently. Without any known rate the dividend is dropped, never converted
at 1.
"""

from __future__ import annotations

import pandas as pd

MAX_PLAUSIBLE_YIELD = 0.30


def dividends_eur(dividends: pd.DataFrame, prices_local: pd.DataFrame,
                  prices_eur: pd.DataFrame) -> pd.DataFrame:
    """Add the EUR amount per share of each dividend, at the implied ex-date rate."""
    implied = (prices_eur / prices_local[prices_eur.columns]).ffill()
    rows = []
    for d in dividends.itertuples():
        if d.ticker not in implied:
            continue
        rate = implied[d.ticker].loc[:d.ex_date].dropna()
        if rate.empty:
            continue
        rows.append({"ticker": d.ticker, "ex_date": d.ex_date, "amount": d.amount,
                     "amount_eur": d.amount * float(rate.iloc[-1])})
    return pd.DataFrame(rows, columns=["ticker", "ex_date", "amount", "amount_eur"])


def entitled_quantity(lots: pd.DataFrame, ticker: str, ex_date: pd.Timestamp) -> float:
    """Quantity entitled to a dividend: held at the close of the session before the ex-date.

    Market rule: a share bought on the ex-date trades without the dividend, so it gives no
    right to it; a share sold on the ex-date was still held the evening before and keeps
    it. Only trades dated strictly before the ex-date count.
    """
    mask = (lots["ticker"] == ticker) & (lots["date"] < ex_date)
    return float(lots.loc[mask, "quantity"].sum())


def dividends_received(lots: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """Cash received per dividend: entitled quantity times the amount per share.

    Nothing is received before the first purchase or between an exit and a rebuy, since
    the entitled quantity is then zero.
    """
    rows = []
    for d in dividends.itertuples():
        qty = entitled_quantity(lots, d.ticker, d.ex_date)
        if qty > 0:
            rows.append({"ticker": d.ticker, "ex_date": d.ex_date, "quantity": qty,
                         "received": qty * d.amount, "received_eur": qty * d.amount_eur})
    return pd.DataFrame(rows, columns=["ticker", "ex_date", "quantity", "received",
                                       "received_eur"])


def yield_on_cost(received: pd.DataFrame, open_cost_eur: pd.Series, since: pd.Timestamp
                  ) -> pd.Series:
    """Dividends received since a date divided by the open cost of the current cycle, in %."""
    recent = received[received["ex_date"] > since].groupby("ticker")["received_eur"].sum()
    cost = open_cost_eur[open_cost_eur > 0]
    return (recent.reindex(cost.index).fillna(0.0) / cost * 100).rename("yield_on_cost_pct")


def is_plausible_yield(dividend_yield: float | None) -> bool:
    """A trailing yield outside [0, 30 %] is treated as a data error, not as information."""
    return dividend_yield is not None and pd.notna(dividend_yield) and (
        0.0 <= dividend_yield <= MAX_PLAUSIBLE_YIELD)


def weighted_portfolio_yield(values_eur: pd.Series, trailing_yield: pd.Series) -> float | None:
    """Value-weighted trailing yield over the lines whose yield is plausible."""
    keep = [t for t in values_eur.index if is_plausible_yield(trailing_yield.get(t))]
    if not keep:
        return None
    v = values_eur[keep]
    return float((v * trailing_yield[keep]).sum() / v.sum())
