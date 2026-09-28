"""Position register: one row per security ever held, open or closed.

For each line: holding periods, capital invested, proceeds, remaining value, dividends,
the return in local currency, the total return in EUR, and the currency effect deduced
as the residual of the two, so that (1 + total) = (1 + local) x (1 + currency) holds
exactly. Returns are multiples of capital over the whole life of the line, never
annualised.
"""

from __future__ import annotations

import pandas as pd

from risk_engine.journal import LotBook, realized_pnl_by_sale


def holding_periods(lots: pd.DataFrame, last_date: pd.Timestamp) -> list[tuple]:
    """Intervals over which the running quantity is strictly positive."""
    running = lots.groupby("date")["quantity"].sum().sort_index().cumsum()
    periods, start = [], None
    for date, qty in running.items():
        if qty > 1e-9 and start is None:
            start = date
        elif qty <= 1e-9 and start is not None:
            periods.append((start, date))
            start = None
    if start is not None:
        periods.append((start, last_date))
    return periods


def _line(ticker: str, trades: pd.DataFrame, lots: pd.DataFrame, open_cost: float,
          local_close: pd.Series, eur_close: pd.Series, received: pd.DataFrame,
          unit_factor: float) -> dict:
    """Register row of one security."""
    buys, sells = trades[trades["side"] == "BUY"], trades[trades["side"] == "SELL"]
    qty_now = float(lots["quantity"].sum())
    last_date = local_close.dropna().index[-1]
    periods = holding_periods(lots, last_date)
    in_periods = pd.concat([local_close.loc[a:b] for a, b in periods]).dropna()
    invested = float((buys["quantity"] * buys["price"] * buys["fx_to_eur"]).sum())
    proceeds = float((sells["quantity"] * sells["price"] * sells["fx_to_eur"]).sum())
    remaining = qty_now * float(eur_close.dropna().iloc[-1]) if qty_now > 1e-9 else 0.0
    div = received[received["ticker"] == ticker]
    local_cost = float((buys["quantity"] * buys["price"]).sum())
    local_exit = (float((sells["quantity"] * sells["price"]).sum())
                  + qty_now * float(local_close.dropna().iloc[-1]) * unit_factor
                  + float(div["received"].sum()))
    local_ret = local_exit / local_cost - 1
    total_ret = (proceeds + remaining + float(div["received_eur"].sum())) / invested - 1
    realized = float(realized_pnl_by_sale(lots)["pnl_eur"].sum())
    unrealized = remaining - open_cost if qty_now > 1e-9 else None
    state = "exited" if qty_now <= 1e-9 else ("rebought" if len(periods) > 1 else "held")
    return {
        "ticker": ticker, "state": state, "quantity": qty_now, "periods": len(periods),
        "days_held": sum((b - a).days for a, b in periods),
        "invested_eur": invested, "proceeds_eur": proceeds, "remaining_eur": remaining,
        "dividends_eur": float(div["received_eur"].sum()),
        "local_return_pct": local_ret * 100, "total_return_pct": total_ret * 100,
        "fx_effect_pct": ((1 + total_ret) / (1 + local_ret) - 1) * 100,
        "realized_eur": realized, "unrealized_eur": unrealized,
        "identity_gap_eur": (unrealized or 0.0) + realized - (proceeds + remaining - invested),
        "min_local": float(in_periods.min()), "max_local": float(in_periods.max()),
    }


def build_register(journal: pd.DataFrame, book: LotBook, prices_local: pd.DataFrame,
                   eur_prices: pd.DataFrame, received: pd.DataFrame,
                   securities: pd.DataFrame) -> pd.DataFrame:
    """Register of every ticker in the journal, sorted by capital invested."""
    rows = []
    for ticker, trades in journal.groupby("ticker"):
        lots = book.lots[book.lots["ticker"] == ticker]
        rows.append(_line(ticker, trades.assign(side=trades["side"].str.upper()), lots,
                          float(book.open_cost_eur.get(ticker, 0.0)), prices_local[ticker],
                          eur_prices[ticker], received,
                          float(securities.at[ticker, "unit_factor"])))
    return pd.DataFrame(rows).set_index("ticker").sort_values("invested_eur", ascending=False)


def register_summary(register: pd.DataFrame) -> dict:
    """Capital committed, exits, mean holding time and capital-weighted exit return."""
    exited = register[register["state"] == "exited"]
    weighted = (float((exited["total_return_pct"] * exited["invested_eur"]).sum()
                      / exited["invested_eur"].sum()) if len(exited) else None)
    return {"lines": len(register), "exited_lines": len(exited),
            "invested_eur": float(register["invested_eur"].sum()),
            "mean_days_held": float(register["days_held"].mean()),
            "exited_return_weighted_pct": weighted,
            "best": register["total_return_pct"].idxmax(),
            "worst": register["total_return_pct"].idxmin()}
