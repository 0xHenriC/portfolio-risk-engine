"""Conversion of quote-currency closes into EUR.

The reference currency is the euro and currency moves are part of the risk: a USD line
loses value for a EUR investor when the dollar weakens, even if its USD price is flat.
Rates are expressed as EUR per one unit of foreign currency, so EUR = local x rate.
"""

from __future__ import annotations

import pandas as pd

REFERENCE_CURRENCY = "EUR"


def rate_matrix(securities: pd.DataFrame, index: pd.Index, fx_to_eur: pd.DataFrame
                ) -> pd.DataFrame:
    """EUR per unit of each security's quote currency, on the exact date of each row.

    No carry-forward: a close without a same-day rate stays unconverted, which is safer
    than borrowing a rate from another day without saying so.
    """
    rates = fx_to_eur.reindex(index)
    columns = {}
    for ticker, ccy in securities["currency"].items():
        columns[ticker] = 1.0 if ccy == REFERENCE_CURRENCY else rates[ccy]
    return pd.DataFrame(columns, index=index)


def to_eur(prices: pd.DataFrame, securities: pd.DataFrame, fx_to_eur: pd.DataFrame,
           apply_unit_factor: bool = True) -> pd.DataFrame:
    """EUR price per held unit: close x same-day rate x unit factor.

    The unit factor bridges a quote unit and a held unit (a metal quoted per ounce but
    held in bars). It is constant, so it cancels in returns but not in weights.
    """
    secs = securities.loc[[t for t in prices.columns if t in securities.index]]
    eur = prices[secs.index] * rate_matrix(secs, prices.index, fx_to_eur)
    if apply_unit_factor:
        eur = eur * secs["unit_factor"]
    return eur


def cash_to_eur(cash: dict[str, float], fx_to_eur: pd.DataFrame, as_of: pd.Timestamp
                ) -> dict[str, float]:
    """EUR value of each cash pocket at the last rate on or before the valuation date."""
    out = {}
    for ccy, amount in cash.items():
        known = fx_to_eur[ccy].loc[:as_of].dropna() if ccy != REFERENCE_CURRENCY else None
        out[ccy] = amount * (1.0 if known is None else float(known.iloc[-1]))
    return out
