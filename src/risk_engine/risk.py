"""Market risk of the current book: volatility, VaR, Expected Shortfall and drawdown.

Method: today's positions are revalued on each past day of the window with that day's
EUR returns (historical simulation at current composition). This is the risk of the
portfolio as it stands, not a record of past performance. Conventions:

- returns and risk figures are daily, in percent of the covered value;
- VaR and ES are reported as returns, so a loss is a negative number;
- a day is kept only if at least max(2, n/2) positions have a valid return; on that
  day the positions without a return are left out and the others reweighted;
- a missing close is an unknown return, never a zero return;
- weekends are excluded before any return is computed;
- volatility is annualised with a fixed sqrt(252).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

RISK_WINDOW_YEARS = 10
TRADING_DAYS = 252
MIN_DAILY_OBSERVATIONS = 5
CONFIDENCE_LEVELS = (0.95, 0.975, 0.99)
HEADLINE_LEVEL = 0.95


class RiskConsistencyError(ArithmeticError):
    """Raised when a result breaks an identity that must always hold (ES beyond VaR)."""


def load_price_history_matrix(eur_prices: pd.DataFrame, tickers: list[str], as_of: pd.Timestamp,
                              window_years: int = RISK_WINDOW_YEARS) -> pd.DataFrame:
    """EUR closes of the held tickers over the risk window ending at the valuation date.

    The window is fixed on purpose: a deeper history must not silently move the VaR,
    and it can only worsen the maximum drawdown.
    """
    start = as_of - pd.Timedelta(days=int(365.25 * window_years))
    cols = [t for t in tickers if t in eur_prices.columns]
    return eur_prices.loc[start:as_of, cols]


def business_days_only(matrix: pd.DataFrame) -> pd.DataFrame:
    """Monday to Friday rows only, from the calendar and not from the data."""
    return matrix[matrix.index.weekday < 5]


def portfolio_returns(position_values: pd.Series, price_matrix: pd.DataFrame
                      ) -> tuple[pd.Series, list[str]]:
    """Daily return of the current book, in percent, by full revaluation.

    Each day's P&L is the sum over positions of value x that day's return; positions
    without a return that day are left out and the P&L is scaled back to the whole
    covered value. Dividing by the covered value gives the portfolio return.
    """
    covered = [t for t in position_values.index
               if t in price_matrix.columns and price_matrix[t].notna().sum() >= 2]
    rets = price_matrix[covered].pct_change(fill_method=None)
    valid = rets.notna().to_numpy()
    values = position_values[covered].to_numpy(dtype=float)
    pnl = np.where(valid, rets.fillna(0.0).to_numpy(), 0.0) @ values
    available_value = valid @ values
    threshold = max(2, len(covered) // 2)
    keep = (valid.sum(axis=1) >= threshold) & (available_value > 0)
    pct = pnl[keep] / available_value[keep] * 100
    return pd.Series(pct, index=rets.index[keep], name="return_pct"), covered


def historical_var_es(returns_pct: pd.Series, level: float) -> tuple[float, float]:
    """Historical VaR (interpolated quantile) and ES (mean of returns at or below it)."""
    r = returns_pct.to_numpy()
    var = float(np.quantile(r, 1 - level))
    tail = r[r <= var]
    return var, float(tail.mean()) if tail.size else var


def parametric_var_es(returns_pct: pd.Series, level: float) -> tuple[float, float]:
    """Normal VaR and ES from the sample mean and standard deviation (ddof=1)."""
    mu, sigma = float(returns_pct.mean()), float(returns_pct.std(ddof=1))
    z = norm.ppf(level)
    return mu - sigma * z, mu - sigma * norm.pdf(z) / (1 - level)


def rank_var_es(returns_pct: pd.Series, level: float) -> tuple[float, float]:
    """Textbook rank method: VaR is the k-th worst return, k = round(n(1-c)), no interpolation;
    ES is the mean of the k-1 returns strictly worse than it."""
    worst_first = np.sort(returns_pct.to_numpy())
    k = max(1, round(len(worst_first) * (1 - level)))
    var = float(worst_first[k - 1])
    return var, float(worst_first[: k - 1].mean()) if k > 1 else var


def check_es_beyond_var(var_pct: float, es_pct: float, label: str) -> None:
    """ES averages the tail beyond the VaR, so it can never be a smaller loss."""
    if es_pct > var_pct + 1e-12:
        raise RiskConsistencyError(f"{label}: ES {es_pct:.6f} is above VaR {var_pct:.6f}")


def max_drawdown(returns_pct: pd.Series) -> dict:
    """Largest fall of the chained index from a previous peak, with its dates and length."""
    index = 100.0 * (1 + returns_pct / 100).cumprod()
    running_max = index.cummax()
    dd = (index - running_max) / running_max * 100
    trough = dd.idxmin()
    peak = index.loc[:trough].idxmax()
    after = index.loc[trough:]
    recovered = after[after >= index[peak]]
    recovery = recovered.index[0] if len(recovered) else None
    pos = index.index.get_loc
    return {"max_drawdown_pct": float(dd.min()), "peak": peak, "trough": trough,
            "recovery": recovery, "sessions_to_trough": pos(trough) - pos(peak),
            "sessions_to_recovery": (pos(recovery) - pos(peak)) if recovery is not None else None,
            "drawdown_pct": dd}


def scale_to_horizon(one_day_pct: float, days: int = 10) -> float:
    """Square-root-of-time scaling of a 1-day figure.

    Exact only for independent, identically distributed returns with zero mean. With fat
    tails, volatility clustering or autocorrelation it misstates the h-day loss, and it
    ignores that positions would be traded over h days. Given as an approximation.
    """
    return one_day_pct * math.sqrt(days)


def _measures(returns_pct: pd.Series, covered_value: float, levels: tuple[float, ...]) -> dict:
    """VaR and ES for every method and level, in percent and in EUR."""
    methods = {"historical": historical_var_es, "normal": parametric_var_es,
               "rank": rank_var_es}
    out = {}
    for name, fn in methods.items():
        for level in levels:
            var, es = fn(returns_pct, level)
            check_es_beyond_var(var, es, f"{name} {level:.1%}")
            out[(name, level)] = {"var_pct": var, "es_pct": es,
                                  "var_eur": var / 100 * covered_value,
                                  "es_eur": es / 100 * covered_value}
    return out


def compute_advanced_risk_from_price_history(position_values: pd.Series,
                                             price_matrix: pd.DataFrame,
                                             levels: tuple[float, ...] = CONFIDENCE_LEVELS
                                             ) -> dict:
    """All risk figures of the current book over the given EUR price matrix."""
    total = float(position_values.sum())
    returns, covered = portfolio_returns(position_values, price_matrix)
    if not covered:
        return {"ok": False, "message": "no held position has a usable price history"}
    if len(returns) < MIN_DAILY_OBSERVATIONS:
        return {"ok": False, "message": f"{len(returns)} usable days, "
                                        f"{MIN_DAILY_OBSERVATIONS} required"}
    covered_value = float(position_values[covered].sum())
    vol = float(returns.std(ddof=1))
    top = position_values[covered].sort_values(ascending=False).index[:10]
    corr = price_matrix[top].pct_change(fill_method=None).corr(min_periods=MIN_DAILY_OBSERVATIONS)
    return {
        "ok": True, "returns_pct": returns, "n_days": len(returns),
        "first_day": returns.index[0], "last_day": returns.index[-1],
        "covered": covered, "coverage_pct": covered_value / total * 100,
        "covered_value_eur": covered_value,
        "volatility_daily_pct": vol,
        "volatility_annualized_pct": vol * math.sqrt(TRADING_DAYS),
        **max_drawdown(returns),
        "measures": _measures(returns, covered_value, levels),
        "pnl_eur": returns / 100 * covered_value,
        "correlation": corr,
    }


def compute_advanced_risk(position_values: pd.Series, eur_prices: pd.DataFrame,
                          as_of: pd.Timestamp, window_years: int = RISK_WINDOW_YEARS,
                          levels: tuple[float, ...] = CONFIDENCE_LEVELS) -> dict:
    """Load the window, drop weekends, then measure. Cash pockets must not be passed in:
    a constant series has no market risk and would only dilute the figures."""
    matrix = load_price_history_matrix(eur_prices, list(position_values.index), as_of,
                                       window_years)
    return compute_advanced_risk_from_price_history(position_values,
                                                    business_days_only(matrix), levels)
