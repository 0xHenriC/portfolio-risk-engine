"""A portfolio defined by quantities, valued every day in EUR.

Two readings of the same book: the valuation at one date (each line is quantity x last
close x rate, with a bounded carry-forward of stale prices), and the daily history of
value together with a chained price index and a chained total-return index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_CARRY_FORWARD_SESSIONS = 3


def carry_forward_bounded(matrix: pd.DataFrame, max_sessions: int = MAX_CARRY_FORWARD_SESSIONS
                          ) -> pd.DataFrame:
    """Carry a close forward over at most N Monday-Friday sessions.

    The count runs on a calendar Monday-Friday grid, not on the rows present, so a gap
    of four sessions is caught even when the matrix has no row for those days. Weekend
    rows keep their own values and do not use up the allowance.
    """
    weekdays = matrix.index.weekday < 5
    grid = pd.bdate_range(matrix.index.min(), matrix.index.max())
    filled = matrix[weekdays].reindex(grid).ffill(limit=max_sessions)
    out = matrix.copy()
    out.loc[weekdays] = filled.reindex(matrix.index[weekdays]).to_numpy()
    return out


def current_quantities(lots: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.Series:
    """Net quantity per ticker, from the lots dated on or before the valuation date."""
    if as_of is not None:
        lots = lots[lots["date"] <= as_of]
    qty = lots.groupby("ticker")["quantity"].sum()
    return qty[qty.abs() > 1e-9]


def value_positions(quantities: pd.Series, eur_prices: pd.DataFrame, as_of: pd.Timestamp,
                    cash_eur: dict[str, float] | None = None) -> dict:
    """Line-by-line valuation at one date, with the lines left out of the total named.

    A line whose last close is older than the allowance has no price: it is excluded
    from the total rather than valued at an old close, and the total says it is
    incomplete. A line valued at an earlier close within the allowance is flagged.
    """
    bounded = carry_forward_bounded(eur_prices.loc[:as_of])
    rows, missing = [], []
    for ticker, qty in quantities.items():
        raw = eur_prices[ticker].loc[:as_of].dropna()
        price = bounded[ticker].iloc[-1] if ticker in bounded else np.nan
        if raw.empty or pd.isna(price):
            missing.append(ticker)
            continue
        rows.append({"ticker": ticker, "quantity": float(qty), "price_eur": float(price),
                     "price_date": raw.index[-1], "value_eur": float(qty * price),
                     "stale": raw.index[-1] < bounded.index[-1]})
    lines = pd.DataFrame(rows).set_index("ticker")
    securities_total = float(lines["value_eur"].sum())
    cash_total = float(sum((cash_eur or {}).values()))
    return {"lines": lines, "securities_eur": securities_total, "cash_eur": cash_total,
            "total_eur": securities_total + cash_total, "missing_price": missing,
            "complete": not missing}


def _quantity_matrix(lots: pd.DataFrame, index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    """Quantity held at each date: cumulative sum of lots dated on or before it."""
    flows = lots.pivot_table(index="date", columns="ticker", values="quantity", aggfunc="sum")
    flows = flows.reindex(index.union(flows.index)).fillna(0.0).cumsum()
    return flows.reindex(index).reindex(columns=columns, fill_value=0.0)


def portfolio_value_history(lots: pd.DataFrame, eur_prices: pd.DataFrame,
                            dividends_eur: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily value, invested capital and two chained indices (price, total return).

    Each day compares only positions valued both yesterday and today, at yesterday's
    quantities: a purchase joins the index the next session, and shares sold on a day
    still carry that day's move. The index therefore measures market performance and
    never mistakes a deposit or a sale for a gain or a loss. Gross dividends are added
    on their ex-date for the total-return index.
    """
    first = lots["date"].min()
    prices = eur_prices.loc[first:].ffill()
    tickers = prices.columns.intersection(lots["ticker"].unique())
    prices = prices[tickers]
    qty = _quantity_matrix(lots, prices.index, tickers)
    prev_qty, prev_px = qty.shift(1).fillna(0.0), prices.shift(1)
    both = prices.notna() & prev_px.notna()
    v_today = (prev_qty * prices).where(both).sum(axis=1)
    v_yest = (prev_qty * prev_px).where(both).sum(axis=1)
    div = pd.DataFrame(0.0, index=prices.index, columns=tickers)
    if dividends_eur is not None and not dividends_eur.empty:
        per_share = dividends_eur.pivot_table(index="ex_date", columns="ticker",
                                              values="amount_eur", aggfunc="sum")
        div = per_share.reindex(index=prices.index, columns=tickers).fillna(0.0)
    d_today = (prev_qty * div).sum(axis=1)
    ok = v_yest > 0
    price_step = (v_today / v_yest).where(ok, 1.0)
    total_step = ((v_today + d_today) / v_yest).where(ok, 1.0)
    invested = lots.groupby("date")["acquisition_value_eur"].sum()
    invested = invested.reindex(prices.index.union(invested.index)).fillna(0.0).cumsum()
    value = (qty * prices).sum(axis=1)
    history = pd.DataFrame({
        "value_eur": value,
        "invested_eur": invested.reindex(prices.index),
        "price_index": 100 * price_step.cumprod(),
        "total_return_index": 100 * total_step.cumprod(),
        "n_positions": (qty.abs() > 1e-9).sum(axis=1),
    })
    history["unrealized_pct"] = (history["value_eur"] / history["invested_eur"] - 1) * 100
    return history
