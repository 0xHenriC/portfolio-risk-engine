"""Independent witness of the risk figures. It imports nothing from risk_engine.

A second implementation, written separately, that starts again from raw closes in
quote currency, FX rates and quantities, and must land on the same numbers as the
engine. It checks that the engine implements the method correctly; it does not test
another method. A witness that imported the code it checks would share its bugs and
prove nothing, so every rule it needs is restated here on purpose: EUR conversion at the
same-day rate, the unit factor of metal, Monday-Friday sessions, the risk window, the
minimum number of positions per day, and the quantile interpolation. The cost is that a
rule changed on one side must be changed on the other.

Where the engine revalues positions with a matrix product, the witness loops over days
and averages weights; where the engine calls numpy and scipy for quantiles and the
normal law, the witness interpolates by hand and uses the standard library.
"""

from __future__ import annotations

import math
from datetime import datetime
from statistics import NormalDist

import numpy as np
import pandas as pd

WINDOW_YEARS = 10
MIN_POSITIONS_PER_DAY = 2
MIN_DAYS = 5
ANNUALISATION = 252
LEVELS = (0.95, 0.975, 0.99)


def is_business_day(day: datetime) -> bool:
    """Monday to Friday; no exchange calendar, a holiday is a weekday without a close."""
    return day.weekday() < 5


def eur_matrix(prices: pd.DataFrame, fx_to_eur: pd.DataFrame, securities: pd.DataFrame,
               tickers: list[str]) -> pd.DataFrame:
    """EUR value of one held unit per day: close x rate of the same date x unit factor."""
    out = {}
    for t in tickers:
        ccy = securities.at[t, "currency"]
        rate = 1.0 if ccy == "EUR" else fx_to_eur[ccy].reindex(prices.index)
        out[t] = prices[t] * rate * float(securities.at[t, "unit_factor"])
    return pd.DataFrame(out, index=prices.index)


def current_weights(quantities: pd.Series, eur: pd.DataFrame, as_of: datetime
                    ) -> dict[str, float]:
    """Weight of each position from its quantity and last known EUR close on or before as_of."""
    values = {}
    for t, q in quantities.items():
        known = eur[t][eur.index <= as_of].dropna()
        if len(known):
            values[t] = float(q) * float(known.iloc[-1])
    total = sum(values.values())
    return {t: v / total for t, v in values.items()}


def interpolated_quantile(sorted_values: np.ndarray, p: float) -> float:
    """Quantile with linear interpolation between order statistics, position p(n-1)."""
    h = p * (len(sorted_values) - 1)
    lo = math.floor(h)
    hi = min(lo + 1, len(sorted_values) - 1)
    return float(sorted_values[lo] + (h - lo) * (sorted_values[hi] - sorted_values[lo]))


def daily_returns(weights: dict[str, float], eur: pd.DataFrame, as_of: datetime
                  ) -> pd.Series:
    """Portfolio return per session in percent, by day-by-day reweighting."""
    start = as_of - pd.Timedelta(days=int(365.25 * WINDOW_YEARS))
    window = eur[(eur.index >= start) & (eur.index <= as_of)]
    window = window.loc[[d for d in window.index if is_business_day(d)]]
    covered = [t for t in weights if window[t].notna().sum() >= 2]
    w_sum = sum(weights[t] for t in covered)
    w = {t: weights[t] / w_sum for t in covered}
    rets = window[covered].pct_change(fill_method=None) * 100
    floor = max(MIN_POSITIONS_PER_DAY, len(covered) // 2)
    kept = {}
    for day, row in rets.iterrows():
        available = row.dropna()
        if len(available) < floor:
            continue
        weight_today = sum(w[t] for t in available.index)
        if weight_today > 0:
            kept[day] = sum(w[t] * available[t] for t in available.index) / weight_today
    return pd.Series(kept, dtype=float)


def independent_calculation(prices: pd.DataFrame, fx_to_eur: pd.DataFrame,
                            securities: pd.DataFrame, quantities: pd.Series,
                            as_of: datetime) -> dict[str, float]:
    """Every risk figure of the book, rebuilt from closes, rates and quantities."""
    eur = eur_matrix(prices, fx_to_eur, securities, list(quantities.index))
    r = daily_returns(current_weights(quantities, eur, as_of), eur, as_of)
    if len(r) < MIN_DAYS:
        raise ValueError(f"only {len(r)} usable days, {MIN_DAYS} required")
    x = r.to_numpy()
    ordered = np.sort(x)
    mu, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    level_index = 100 * np.cumprod(1 + x / 100)
    peaks = np.maximum.accumulate(level_index)
    out = {"n_days": float(len(x)), "volatility_daily_pct": sd,
           "volatility_annualized_pct": sd * math.sqrt(ANNUALISATION),
           "max_drawdown_pct": float(np.min((level_index - peaks) / peaks)) * 100}
    for c in LEVELS:
        tag = f"{c * 100:g}"
        var = interpolated_quantile(ordered, 1 - c)
        out[f"historical_var_{tag}"] = var
        out[f"historical_es_{tag}"] = float(np.mean(x[x <= var]))
        z = NormalDist().inv_cdf(c)
        out[f"normal_var_{tag}"] = mu - sd * z
        out[f"normal_es_{tag}"] = mu - sd * NormalDist().pdf(z) / (1 - c)
        k = max(1, round(len(ordered) * (1 - c)))
        out[f"rank_var_{tag}"] = float(ordered[k - 1])
        out[f"rank_es_{tag}"] = float(np.mean(ordered[: k - 1])) if k > 1 else float(ordered[0])
    return out


def max_relative_difference(engine: dict[str, float], witness: dict[str, float]
                            ) -> tuple[float, str]:
    """Largest relative gap between two sets of figures, and the figure where it occurs."""
    worst, where = 0.0, ""
    for key, w in witness.items():
        e = engine[key]
        gap = abs(e - w) / max(abs(w), 1e-12)
        if gap > worst:
            worst, where = gap, key
    return worst, where
