"""From a trade journal to lots, cost basis and realised P&L.

Cost basis is the weighted average cost per holding cycle: a cycle ends when the
quantity held returns to zero, and a later purchase starts a fresh average. The journal
has no time of day, so trades of one session follow a fixed rule: the day's sales first
draw on what was held before the session, at the prior average cost, then on the day's
purchases at their own average price. All amounts are in EUR at the trade-date rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

QTY_TOLERANCE = 1e-6
REQUIRED_COLUMNS = ("date", "ticker", "side", "quantity", "price", "currency", "fx_to_eur")


class JournalError(ValueError):
    """Raised with every problem found at once, so that one run lists them all."""

    def __init__(self, problems: list[str]):
        super().__init__("\n".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class LotBook:
    """Lots (signed quantities) plus the open cost of the current cycle of each ticker."""

    lots: pd.DataFrame
    open_cost_eur: pd.Series
    open_quantity: pd.Series


def validate_journal(journal: pd.DataFrame, known_currencies: set[str],
                     as_of: pd.Timestamp) -> None:
    """Check sides, signs, dates and currencies before any lot is built."""
    problems = []
    for col in REQUIRED_COLUMNS:
        if col not in journal or journal[col].isna().all():
            problems.append(f"column '{col}' is missing or empty")
    if problems:
        raise JournalError(problems)
    for i, row in journal.iterrows():
        side = str(row["side"]).strip().upper()
        if side not in ("BUY", "SELL"):
            problems.append(f"row {i}: side '{row['side']}' is neither BUY nor SELL")
        for col in ("quantity", "price", "fx_to_eur"):
            if not pd.notna(row[col]) or float(row[col]) <= 0:
                problems.append(f"row {i}: {col} must be strictly positive")
        if not isinstance(row["date"], pd.Timestamp) or row["date"] > as_of:
            problems.append(f"row {i}: date is not a valid date on or before {as_of.date()}")
        if row["currency"] not in known_currencies:
            problems.append(f"row {i}: unknown currency '{row['currency']}'")
    ccy_count = journal.groupby("ticker")["currency"].nunique()
    problems += [f"{t}: several currencies in the journal" for t in ccy_count[ccy_count > 1].index]
    if problems:
        raise JournalError(problems)


def split_same_day_sale(prior_qty: float, prior_avg: float, day_buy_qty: float,
                        day_buy_avg: float, sold_qty: float) -> tuple[float, float]:
    """Blended unit cost of a session's sales, and the part drawn on prior holdings."""
    from_prior = min(sold_qty, prior_qty)
    from_day = sold_qty - from_prior
    if from_day > day_buy_qty + QTY_TOLERANCE:
        raise JournalError([f"sales of {sold_qty} exceed holdings plus the day's purchases"])
    return (from_prior * prior_avg + from_day * day_buy_avg) / sold_qty, from_prior


def _ticker_lots(trades: pd.DataFrame) -> tuple[list[dict], float, float]:
    """Walk one ticker's sessions in date order and emit its lots."""
    lots, qty, cost = [], 0.0, 0.0
    for date, day in trades.groupby("date", sort=True):
        buys = day[day["side"] == "BUY"]
        sells = day[day["side"] == "SELL"]
        buy_qty = float(buys["quantity"].sum())
        buy_cost = float((buys["quantity"] * buys["price"] * buys["fx_to_eur"]).sum())
        buy_avg = buy_cost / buy_qty if buy_qty else 0.0
        joining_qty, joining_cost = buy_qty, buy_cost
        if not sells.empty:
            sold = float(sells["quantity"].sum())
            prior_avg = cost / qty if qty > QTY_TOLERANCE else 0.0
            unit_cost, from_prior = split_same_day_sale(qty, prior_avg, buy_qty, buy_avg, sold)
            for s in sells.itertuples():
                lots.append({"date": date, "quantity": -s.quantity,
                             "acquisition_value_eur": -s.quantity * unit_cost,
                             "proceeds_eur": s.quantity * s.price * s.fx_to_eur})
            qty, cost = qty - from_prior, cost - from_prior * prior_avg
            if qty <= QTY_TOLERANCE:
                qty, cost = 0.0, 0.0
            joining_qty = buy_qty - (sold - from_prior)
            joining_cost = joining_qty * buy_avg
        for b in buys.itertuples():
            lots.append({"date": date, "quantity": b.quantity,
                         "acquisition_value_eur": b.quantity * b.price * b.fx_to_eur,
                         "proceeds_eur": None})
        qty, cost = qty + joining_qty, cost + joining_cost
        if qty <= QTY_TOLERANCE:
            qty, cost = 0.0, 0.0
    return lots, qty, cost


def build_lots(journal: pd.DataFrame) -> LotBook:
    """Lots for every ticker: buys as positive lots, sales as negative lots with cost relief."""
    journal = journal.assign(side=journal["side"].str.strip().str.upper())
    rows, open_cost, open_qty = [], {}, {}
    for ticker, trades in journal.groupby("ticker", sort=True):
        lots, qty, cost = _ticker_lots(trades)
        rows += [{"ticker": ticker, **lot} for lot in lots]
        open_cost[ticker], open_qty[ticker] = cost, qty
    lots = pd.DataFrame(rows).sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)
    return LotBook(lots, pd.Series(open_cost, dtype=float), pd.Series(open_qty, dtype=float))


def reconcile_with_holdings(book: LotBook, declared: pd.Series) -> None:
    """The journal's running quantity must equal the independently declared holding."""
    held = book.lots.groupby("ticker")["quantity"].sum()
    problems = []
    for ticker in held.index.union(declared.index):
        j, d = float(held.get(ticker, 0.0)), float(declared.get(ticker, 0.0))
        if abs(j - d) > QTY_TOLERANCE:
            problems.append(f"{ticker}: journal holds {j:g}, declared holding is {d:g}")
    if problems:
        raise JournalError(problems)


def quantity_held_at(lots: pd.DataFrame, ticker: str, date: pd.Timestamp) -> float:
    """Signed sum of the ticker's lots dated on or before the given date."""
    mask = (lots["ticker"] == ticker) & (lots["date"] <= date)
    return float(lots.loc[mask, "quantity"].sum())


def realized_pnl_by_sale(lots: pd.DataFrame) -> pd.DataFrame:
    """One row per sale: proceeds plus the (negative) cost relief, in EUR, FX included."""
    sales = lots[lots["quantity"] < 0].copy()
    sales["pnl_eur"] = sales["proceeds_eur"].astype(float) + sales["acquisition_value_eur"]
    return sales[["date", "ticker", "quantity", "pnl_eur"]].reset_index(drop=True)


def realized_gains_losses(lots: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
                          ) -> dict[str, float]:
    """Realised gains and losses booked in (start, end], summed separately by sign."""
    pnl = realized_pnl_by_sale(lots)
    window = pnl[(pnl["date"] > start) & (pnl["date"] <= end)]["pnl_eur"]
    return {"gains_eur": float(window[window > 0].sum()),
            "losses_eur": float(window[window < 0].sum())}
