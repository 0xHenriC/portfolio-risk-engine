"""Data controls run before any figure is trusted.

A visible stop beats a silent wrong number. The price-jump guard stops on moves that
look like an unadjusted split (a close divided or multiplied by a round factor); the
other controls only report, with two levels (info, alert) so that a signal firing every
day does not stop being read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ROUND_FACTORS = (2, 3, 4, 5, 10, 20, 50, 100)
FACTOR_TOLERANCE = 0.05
GAP_THRESHOLD = 0.15
MIN_PEERS = 5
QUANTITY_CHANGE_ALERT = 0.50
CONCORDANCE_THRESHOLD = 0.02


def split_like_factor(ratio: float) -> float | None:
    """The round factor (or its inverse) a price ratio sits within 5 % of, if any.

    3:2 is left out on purpose: a genuine crash of about a third would be taken for it.
    """
    if not np.isfinite(ratio) or ratio <= 0:
        return None
    for f in ROUND_FACTORS:
        for target in (f, 1 / f):
            if abs(ratio / target - 1) <= FACTOR_TOLERANCE:
                return target
    return None


def scan_price_jumps(prices: pd.DataFrame) -> pd.DataFrame:
    """Every close whose ratio to the previous close of the same series looks like a split."""
    rows = []
    for ticker in prices.columns:
        s = prices[ticker].dropna()
        ratios = (s / s.shift(1)).iloc[1:]
        for date, r in ratios.items():
            factor = split_like_factor(float(r))
            if factor is not None:
                rows.append({"ticker": ticker, "date": date, "ratio": float(r), "factor": factor})
    return pd.DataFrame(rows, columns=["ticker", "date", "ratio", "factor"])


def raw_unit_trades(journal: pd.DataFrame, prices: pd.DataFrame, securities: pd.DataFrame
                    ) -> pd.DataFrame:
    """Journal lines whose price looks like it predates a split (non-blocking warning)."""
    flagged = []
    for i, row in journal.iterrows():
        close = prices[row["ticker"]].loc[:row["date"]].dropna()
        if close.empty:
            continue
        unit = float(securities.at[row["ticker"], "unit_factor"])
        factor = split_like_factor(row["price"] / (float(close.iloc[-1]) * unit))
        if factor is not None:
            flagged.append({"row": i, "ticker": row["ticker"], "date": row["date"],
                            "factor": factor})
    return pd.DataFrame(flagged, columns=["row", "ticker", "date", "factor"])


def gap_shortfall(prices: pd.DataFrame, venue: pd.Series) -> pd.DataFrame:
    """Series with markedly fewer sessions than their exchange peers over the same span.

    Peers are the tickers of the same exchange already quoted when the series starts;
    the reference is the upper median of their session counts. Exchanges with fewer than
    five tickers are reported as not comparable rather than judged.
    """
    rows = []
    for place, tickers in venue.groupby(venue).groups.items():
        tickers = [t for t in tickers if t in prices.columns]
        for t in tickers:
            s = prices[t].dropna()
            if len(tickers) < MIN_PEERS:
                rows.append({"ticker": t, "venue": place, "status": "not comparable"})
                continue
            d0 = s.index[0]
            peers = [p for p in tickers if p != t and prices[p].first_valid_index() <= d0]
            counts = sorted(int(prices[p].loc[d0:].notna().sum()) for p in peers)
            m = counts[len(counts) // 2] if counts else len(s)
            shortfall = (m - len(s)) / m if m else 0.0
            rows.append({"ticker": t, "venue": place, "sessions": len(s), "peer_median": m,
                         "shortfall": shortfall,
                         "status": "alert" if shortfall > GAP_THRESHOLD else "ok"})
    return pd.DataFrame(rows)


def freshness(prices: pd.DataFrame, venue: pd.Series, as_of: pd.Timestamp) -> pd.DataFrame:
    """Lag of each series in observed sessions, and whether it is a holiday or a gap.

    A session is observed when any listed equity has a close. If an exchange peer has
    the missing session, the gap is real (alert). If no peer has it, the exchange was
    probably closed: one session is information, two or more is an alert.
    """
    listed = [t for t in prices.columns if t in venue.index]
    reference = prices[listed].loc[:as_of].dropna(how="all").index
    reference = reference[reference.weekday < 5]
    rows = []
    for t in listed:
        last = prices[t].loc[:as_of].last_valid_index()
        missing = reference[reference > last]
        peers = [p for p in listed if p != t and venue[p] == venue[t]]
        peer_has = bool(len(missing)) and any(pd.notna(prices.at[missing[0], p]) for p in peers)
        lag = len(missing)
        level = "ok" if lag == 0 else ("alert" if peer_has or lag >= 2 else "info")
        rows.append({"ticker": t, "last_close": last, "lag_sessions": lag, "level": level})
    return pd.DataFrame(rows).set_index("ticker")


def quantity_plausibility(previous: pd.Series, current: pd.Series) -> list[str]:
    """Warn when a holding moves by more than half, or disappears, between two runs."""
    out = []
    for t, q0 in previous.items():
        q1 = float(current.get(t, 0.0))
        if q1 == 0:
            out.append(f"{t}: no longer in the holdings")
        elif abs(q1 / q0 - 1) > QUANTITY_CHANGE_ALERT:
            out.append(f"{t}: quantity moved from {q0:g} to {q1:g}")
    return out


def valuation_concordance(reported: pd.Series, rebuilt: pd.Series) -> pd.Series:
    """Lines where a reported value and the rebuilt value differ by more than 2 %."""
    gap = (rebuilt.reindex(reported.index) / reported - 1).abs()
    return gap[gap > CONCORDANCE_THRESHOLD]
