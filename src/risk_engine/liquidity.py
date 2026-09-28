"""Liquidity and the liquidity stress test: how long would it take to sell the book?

Days to liquidate a line = shares held / (participation rate x 10-day average volume).
Selling no more than 20 % of a stock's daily volume is the usual bound to avoid moving
the price. The stress test replays the same calculation with a market that absorbs
less: half (10 %), then a quarter (5 %) of that rate. Since the time is inversely
proportional to the rate, the stressed scores are exactly twice and four times the
normal one; the test is a sensitivity on a stated assumption, not a calibrated
liquidity-adjusted VaR (no bid/ask history is used).
"""

from __future__ import annotations

import pandas as pd

PARTICIPATION_RATE = 0.20
STRESS_RATES = {"moderate": 0.10, "severe": 0.05}
DRYING_UP_THRESHOLD = 0.7


def average_volumes(volumes: pd.DataFrame) -> pd.DataFrame:
    """10-session and 90-session average traded shares per ticker."""
    return pd.DataFrame({"adv10": volumes.tail(10).mean(), "adv90": volumes.tail(90).mean()})


def _named_surplus(trend: float) -> bool:
    """Named only if the excess over normal volume shows as at least 1 % once rounded."""
    return int(format((trend - 1.0) * 100.0, ".0f")) >= 1


def compute_liquidity_metrics(quantities: pd.Series, values_eur: pd.Series,
                              adv: pd.DataFrame) -> dict:
    """Days to liquidate per line under normal and stressed participation, and volume trends."""
    rates = {"normal": PARTICIPATION_RATE, **STRESS_RATES}
    lines = pd.DataFrame({"quantity": quantities, "value_eur": values_eur})
    lines = lines.join(adv, how="left")
    usable = lines["adv10"] > 0
    for name, rate in rates.items():
        lines[f"days_{name}"] = (lines["quantity"] / (lines["adv10"] * rate)).where(usable)
    lines["position_to_volume"] = (lines["quantity"] / lines["adv10"]).where(usable)
    lines["volume_trend"] = (lines["adv10"] / lines["adv90"]).where(lines["adv90"] > 0)
    covered = lines[lines["days_normal"].notna()]
    covered_value = float(covered["value_eur"].sum())
    scores = {name: float((covered["value_eur"] * covered[f"days_{name}"]).sum() / covered_value)
              for name in rates} if covered_value else dict.fromkeys(rates)
    trend = lines["volume_trend"].dropna()
    surplus = trend[trend > 1.0].sort_values(ascending=False)
    return {
        "lines": lines,
        "coverage_pct": covered_value / float(values_eur.sum()) * 100,
        "covered_lines": len(covered), "total_lines": len(lines),
        "score_days": scores,
        "severity_ratio": {n: scores[n] / scores["normal"] for n in STRESS_RATES}
        if scores["normal"] else {},
        "least_liquid": covered["days_normal"].sort_values(ascending=False).head(10),
        "drying_up": trend[trend < DRYING_UP_THRESHOLD].sort_values(),
        "surplus": surplus[surplus.map(_named_surplus)].head(5),
        "surplus_count": len(surplus),
    }
