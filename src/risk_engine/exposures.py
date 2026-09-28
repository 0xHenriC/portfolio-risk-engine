"""Exposures and concentration, each measured on the base that makes it meaningful.

Concentration is read on securities (cash excluded), sectors on equities only, and
currency and geography on total wealth (cash included, since a USD cash pocket is a
dollar exposure). Metal gets its own pseudo-currency: a bar is not a euro exposure
just because its value is reported in euros.
"""

from __future__ import annotations

import pandas as pd

ZONES = ["Greater China", "Other Asia", "India", "Eurozone", "Europe ex-euro",
         "North America", "Oceania", "Rest of world"]
COUNTRY_TO_ZONE = {
    **dict.fromkeys(["china", "hong kong", "macau"], "Greater China"),
    **dict.fromkeys(["japan", "south korea", "taiwan", "singapore", "indonesia", "malaysia",
                     "thailand", "philippines", "vietnam"], "Other Asia"),
    "india": "India",
    **dict.fromkeys(["austria", "belgium", "croatia", "cyprus", "estonia", "finland", "france",
                     "germany", "greece", "ireland", "italy", "latvia", "lithuania",
                     "luxembourg", "malta", "netherlands", "portugal", "slovakia", "slovenia",
                     "spain", "bulgaria", "monaco"], "Eurozone"),
    **dict.fromkeys(["united kingdom", "switzerland", "sweden", "norway", "denmark",
                     "iceland", "poland", "czech republic", "hungary", "romania"],
                    "Europe ex-euro"),
    **dict.fromkeys(["united states", "canada", "mexico", "bermuda"], "North America"),
    **dict.fromkeys(["australia", "new zealand"], "Oceania"),
}
CURRENCY_TO_ZONE = {"EUR": "Eurozone", "USD": "North America", "CAD": "North America",
                    "GBP": "Europe ex-euro", "CHF": "Europe ex-euro", "SEK": "Europe ex-euro",
                    "NOK": "Europe ex-euro", "DKK": "Europe ex-euro", "JPY": "Other Asia",
                    "HKD": "Greater China", "CNY": "Greater China", "INR": "India",
                    "AUD": "Oceania"}
METAL_PSEUDO_CURRENCY = "Gold"
CONCENTRATION_ALERT = 0.20
FRAGILE_POSITION_COUNT = 5


def zone_for_country(country: str | None) -> str:
    """Zone of a headquarters country; unknown countries fall into 'Rest of world'."""
    if country is None or pd.isna(country):
        return "Not provided"
    return COUNTRY_TO_ZONE.get(" ".join(str(country).lower().split()), "Rest of world")


def _weights(values: pd.Series, keys: pd.Series) -> pd.Series:
    """Share of the base held in each group, largest first."""
    grouped = values.groupby(keys).sum()
    return (grouped / values.sum()).sort_values(ascending=False)


def compute_exposures(values_eur: pd.Series, securities: pd.DataFrame,
                      cash_eur: dict[str, float], betas: pd.Series | None = None,
                      metal_zone: str = "Eurozone") -> dict:
    """Concentration, sector, currency, zone and weighted-beta figures of one valuation.

    `metal_zone` is where the bars are held in custody, which is what a geographic map
    of assets should show for physical metal.
    """
    secs = securities.loc[values_eur.index]
    equity = secs["asset_class"] == "equity"
    currency = secs["currency"].where(equity, METAL_PSEUDO_CURRENCY)
    cash = pd.Series(cash_eur, dtype=float)
    cash_ids = [f"CASH-{c}" for c in cash.index]
    wealth = pd.concat([values_eur, pd.Series(cash.to_numpy(), index=cash_ids)])
    wealth_ccy = pd.concat([currency, pd.Series(cash.index, index=cash_ids)])
    zones = pd.concat([
        secs["country"].map(zone_for_country).where(equity, metal_zone),
        pd.Series([CURRENCY_TO_ZONE.get(c, "Rest of world") for c in cash.index], index=cash_ids),
    ])
    line_weights = (values_eur / values_eur.sum()).sort_values(ascending=False)
    out = {
        "securities_eur": float(values_eur.sum()),
        "wealth_eur": float(wealth.sum()),
        "cash_share": float(cash.sum() / wealth.sum()),
        "positions": int(len(values_eur)),
        "fragile": len(values_eur) < FRAGILE_POSITION_COUNT,
        "top_lines": line_weights.head(10),
        "largest_weight": float(line_weights.iloc[0]),
        "sector": _weights(values_eur[equity], secs.loc[equity, "sector"].fillna("Not available")),
        "currency": _weights(wealth, wealth_ccy),
        "currency_ex_cash": _weights(values_eur, currency),
        "zone": _weights(wealth, zones).reindex(ZONES).fillna(0.0),
    }
    if betas is not None:
        b = betas.reindex(values_eur.index)[equity].dropna()
        out["weighted_beta"] = float((values_eur[b.index] * b).sum() / values_eur[b.index].sum())
        out["beta_coverage"] = float(values_eur[b.index].sum() / values_eur.sum())
    return out


def interpretation_sentences(exp: dict) -> list[str]:
    """Plain descriptive sentences; they describe the book and never advise."""
    sector, ccy = exp["sector"], exp["currency"]
    out = [f"Largest sector: {sector.index[0]}, {sector.iloc[0]:.1%} of equities.",
           f"Main currency: {ccy.index[0]}, {ccy.iloc[0]:.1%} of wealth, "
           f"alongside {len(ccy) - 1} other exposure(s)."]
    top = exp["top_lines"]
    out.append(f"Largest line: {top.index[0]}, {top.iloc[0]:.1%} of securities.")
    if exp["largest_weight"] >= CONCENTRATION_ALERT:
        out.append("One line alone weighs 20 % or more and strongly drives the result.")
    if exp["cash_share"] > 0:
        out.append(f"Cash pockets: {exp['cash_share']:.1%} of wealth.")
    if exp["fragile"]:
        out.append("Fewer than five positions: statistics on this book are fragile.")
    return out
