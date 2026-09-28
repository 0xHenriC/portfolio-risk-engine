"""Market stress scenarios: instant shocks from scenarios.toml, and historical windows.

Each scenario revalues today's positions in full (price shock and currency shock
compounded per line) and reports the loss in EUR, in percent of wealth, and line by line.
Historical windows replay the worst run of consecutive sessions inside a past episode.
"""

from __future__ import annotations

import tomllib
from importlib import resources

import pandas as pd

from risk_engine.exposures import zone_for_country


def load_scenarios() -> dict:
    """Read the packaged scenarios.toml."""
    text = resources.files("risk_engine").joinpath("scenarios.toml").read_text(encoding="utf-8")
    return tomllib.loads(text)


def _line_shock(scenario: dict, asset_class: str, zone: str, currency: str) -> float:
    """Compounded relative change of one line's EUR value under a scenario."""
    price = 1 + scenario.get(asset_class, 0.0)
    price *= 1 + scenario.get("zone", {}).get(zone, 0.0)
    return price * (1 + scenario.get("currency", {}).get(currency, 0.0)) - 1


def apply_scenario(scenario: dict, values_eur: pd.Series, securities: pd.DataFrame,
                   cash_eur: dict[str, float]) -> dict:
    """Loss of the whole wealth under one scenario, with each line's contribution."""
    rows = {}
    for ticker, value in values_eur.items():
        sec = securities.loc[ticker]
        zone = zone_for_country(sec["country"]) if sec["asset_class"] == "equity" else "Metal"
        rows[ticker] = value * _line_shock(scenario, sec["asset_class"], zone, sec["currency"])
    for ccy, value in cash_eur.items():
        rows[f"CASH-{ccy}"] = value * scenario.get("currency", {}).get(ccy, 0.0)
    contrib = pd.Series(rows, dtype=float)
    wealth = float(values_eur.sum() + sum(cash_eur.values()))
    return {"name": scenario["name"], "pnl_eur": float(contrib.sum()),
            "pnl_pct": float(contrib.sum()) / wealth * 100,
            "contributions": contrib.sort_values()}


def run_scenarios(values_eur: pd.Series, securities: pd.DataFrame,
                  cash_eur: dict[str, float]) -> list[dict]:
    """Every hypothetical scenario of the packaged file."""
    return [apply_scenario(s, values_eur, securities, cash_eur)
            for s in load_scenarios()["scenario"]]


def worst_window(eur_prices: pd.DataFrame, values_eur: pd.Series, start: str | None = None,
                 end: str | None = None, sessions: int = 5) -> dict | None:
    """Worst run of `sessions` consecutive weekdays for today's book inside a date range.

    Lines without prices over the window are left out and named, so a partial history
    is visible rather than hidden in a smaller loss.
    """
    px = eur_prices.loc[start:end, values_eur.index]
    px = px[px.index.weekday < 5].ffill()
    if len(px) <= sessions:
        return None
    move = px / px.shift(sessions) - 1
    available = move.notna()
    pnl = (move.fillna(0.0) * values_eur).sum(axis=1).where(available.any(axis=1))
    worst_end = pnl.idxmin()
    if pd.isna(worst_end):
        return None
    worst_start = px.index[px.index.get_loc(worst_end) - sessions]
    contrib = (move.loc[worst_end] * values_eur).dropna()
    missing = [t for t in values_eur.index if t not in contrib.index]
    return {"start": worst_start, "end": worst_end, "pnl_eur": float(pnl[worst_end]),
            "pnl_pct": float(pnl[worst_end]) / float(values_eur[contrib.index].sum()) * 100,
            "contributions": contrib.sort_values(), "missing": missing}
