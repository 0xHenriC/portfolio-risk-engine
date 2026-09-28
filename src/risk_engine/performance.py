"""Performance by period: a chained headline return and a market-versus-flow decomposition.

The headline is the time-weighted return read on the total-return index (gross
dividends included), never annualised. The decomposition answers a different question,
in EUR: of the change in value over the window, how much is market (price and FX moves
on what was held) and how much is flow (money put in by purchases, taken out by sales).
The two parts must add up to the change in value to the cent, or the block is refused.
"""

from __future__ import annotations

import pandas as pd

from risk_engine.journal import realized_gains_losses
from risk_engine.portfolio import carry_forward_bounded

PERIODS: dict[str, int | str] = {"1W": 7, "1M": 30, "YTD": "ytd", "1Y": 365, "ITD": "itd"}
BALANCE_TOLERANCE_EUR = 0.01


class PeriodError(ValueError):
    """The window cannot be measured honestly (no session, or the parts do not balance)."""


def window_bounds(sessions: pd.DatetimeIndex, end_target: pd.Timestamp, period: str,
                  first_purchase: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Start and end sessions of a window, both on weekdays actually present."""
    sessions = sessions[sessions.weekday < 5]
    ends = sessions[sessions <= end_target]
    if ends.empty:
        raise PeriodError(f"no session on or before {end_target.date()}")
    end = ends[-1]
    spec = PERIODS[period]
    if spec == "itd":
        starts = sessions[sessions >= first_purchase]
        start = starts[0] if len(starts) else end
    else:
        target = (pd.Timestamp(end.year - 1, 12, 31) if spec == "ytd"
                  else end - pd.Timedelta(days=int(spec)))
        before = sessions[sessions <= target]
        start = before[-1] if len(before) else sessions[0]
    if start >= end:
        raise PeriodError(f"{period}: window start {start.date()} is not before its end")
    return start, end


def _match_window(start_qty: float, p_start: float, p_end: float, trades: pd.DataFrame
                  ) -> dict[str, float]:
    """Market gain and flow of one line over the window.

    Sold units are matched first against the holding at the window start, then against
    the window's purchases in date order; a session's sales use the day's own purchases
    only after the prior holding. Units still held are valued at the end price.
    """
    queue = [[start_qty, p_start]] if start_qty > 0 else []
    gain = flow = buy_cost = 0.0
    for _, day in trades.groupby("date", sort=True):
        buys = day[day["quantity"] > 0]
        sells = day[day["quantity"] < 0]
        pending = [[q, c / q] for q, c in zip(buys["quantity"], buys["acquisition_value_eur"],
                                              strict=True)]
        buy_cost += float(buys["acquisition_value_eur"].sum())
        flow += float(buys["acquisition_value_eur"].sum() - sells["proceeds_eur"].sum())
        for qty, proceeds in zip(-sells["quantity"], sells["proceeds_eur"], strict=True):
            unit_sale = proceeds / qty
            while qty > 1e-12:
                if not queue:
                    if not pending:
                        raise PeriodError("a sale exceeds the units available in the window")
                    queue, pending = pending, []
                take = min(qty, queue[0][0])
                gain += take * (unit_sale - queue[0][1])
                queue[0][0] -= take
                qty -= take
                if queue[0][0] <= 1e-12:
                    queue.pop(0)
        queue += pending
    gain += sum(q * (p_end - unit) for q, unit in queue)
    return {"market_eur": gain, "flow_eur": flow, "base_eur": start_qty * p_start + buy_cost}


def period_block(lots: pd.DataFrame, eur_prices: pd.DataFrame, history: pd.DataFrame,
                 period: str, end_target: pd.Timestamp) -> dict:
    """Headline return, EUR decomposition and top/bottom contributors for one window."""
    start, end = window_bounds(eur_prices.index, end_target, period, lots["date"].min())
    px = carry_forward_bounded(eur_prices.loc[:end])
    rows = []
    for ticker, tl in lots.groupby("ticker"):
        q_start = float(tl.loc[tl["date"] <= start, "quantity"].sum())
        window = tl[(tl["date"] > start) & (tl["date"] <= end)]
        q_end = q_start + float(window["quantity"].sum())
        if abs(q_start) < 1e-9 and window.empty:
            continue
        p_start, p_end = px.at[start, ticker], px.at[end, ticker]
        if (q_start > 0 and pd.isna(p_start)) or (q_end > 1e-9 and pd.isna(p_end)):
            raise PeriodError(f"{period}: {ticker} has no usable price at a window bound")
        p_start = 0.0 if pd.isna(p_start) else float(p_start)
        p_end = 0.0 if pd.isna(p_end) else float(p_end)
        parts = _match_window(q_start, p_start, p_end, window)
        rows.append({"ticker": ticker, "value_start_eur": q_start * p_start,
                     "value_end_eur": q_end * p_end, **parts})
    table = pd.DataFrame(rows).set_index("ticker")
    change = table["value_end_eur"].sum() - table["value_start_eur"].sum()
    explained = table["market_eur"].sum() + table["flow_eur"].sum()
    if abs(change - explained) > BALANCE_TOLERANCE_EUR:
        raise PeriodError(f"{period}: market + flow = {explained:.2f}, change = {change:.2f}")
    table["contribution_pct"] = table["market_eur"] / table["base_eur"].where(
        table["base_eur"] > 0) * 100
    tr = history["total_return_index"]
    ranked = table.sort_values("market_eur", ascending=False)
    return {
        "period": period, "start": start, "end": end,
        "return_pct": (float(tr.asof(end)) / float(tr.asof(start)) - 1) * 100,
        "market_eur": float(table["market_eur"].sum()),
        "flow_eur": float(table["flow_eur"].sum()),
        "value_start_eur": float(table["value_start_eur"].sum()),
        "value_end_eur": float(table["value_end_eur"].sum()),
        "realized": realized_gains_losses(lots, start, end),
        "top": ranked.head(5), "bottom": ranked.tail(5).iloc[::-1], "lines": table,
    }
