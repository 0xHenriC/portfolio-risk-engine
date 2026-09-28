"""Seeded synthetic market: prices, FX, volumes, dividends, benchmark bricks and a trade journal.

Everything the demo and the tests need is produced here from one integer seed, so that
every figure in the README can be reproduced exactly. The securities are real listed
names used only as labels; every price, volume, dividend and beta below is simulated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_SEED = 20240613
N_DAYS = 1500
END_DATE = "2025-10-31"
STUDENT_DF = 4
TROY_OUNCES_PER_KG = 1000 / 31.1034768
BOOK_SCALE = 13

SECURITIES = pd.DataFrame(
    [
        # ticker, name, currency, asset class, sector, industry, country, venue, unit factor
        ("PG", "Procter & Gamble", "USD", "equity", "Consumer Staples",
         "Household Products", "United States", "NYSE", 1.0),
        ("ADBE", "Adobe", "USD", "equity", "Information Technology",
         "Software", "United States", "NASDAQ", 1.0),
        ("ADS.DE", "adidas", "EUR", "equity", "Consumer Discretionary",
         "Footwear", "Germany", "XETRA", 1.0),
        ("PHIA.AS", "Philips", "EUR", "equity", "Health Care",
         "Medical Equipment", "Netherlands", "Euronext Amsterdam", 1.0),
        ("KNEBV.HE", "Kone", "EUR", "equity", "Industrials",
         "Machinery", "Finland", "Nasdaq Helsinki", 1.0),
        ("6758.T", "Sony Group", "JPY", "equity", "Consumer Discretionary",
         "Consumer Electronics", "Japan", "Tokyo", 1.0),
        ("1299.HK", "AIA Group", "HKD", "equity", "Financials",
         "Life Insurance", "Hong Kong", "Hong Kong", 1.0),
        ("GIVN.SW", "Givaudan", "CHF", "equity", "Materials",
         "Specialty Chemicals", "Switzerland", "SIX Swiss", 1.0),
        ("GOLD-KG", "Gold, 1 kg bar", "USD", "metal", None,
         None, None, "Metal", TROY_OUNCES_PER_KG),
    ],
    columns=["ticker", "name", "currency", "asset_class", "sector", "industry",
             "country", "venue", "unit_factor"],
).set_index("ticker")

# Annualised volatility, annual drift, starting price in quote currency.
_ASSET_PARAMS = {
    "PG": (0.18, 0.06, 150.0),
    "ADBE": (0.32, 0.08, 420.0),
    "ADS.DE": (0.30, 0.05, 190.0),
    "PHIA.AS": (0.30, 0.02, 30.0),
    "KNEBV.HE": (0.24, 0.04, 65.0),
    "6758.T": (0.30, 0.07, 2600.0),
    "1299.HK": (0.28, 0.03, 80.0),
    "GIVN.SW": (0.21, 0.05, 3600.0),
    "GOLD-KG": (0.15, 0.05, 1750.0),
}
_REGION = {"PG": "US", "ADBE": "US", "ADS.DE": "EU", "PHIA.AS": "EU", "KNEBV.HE": "EU",
           "6758.T": "JP", "1299.HK": "HK", "GIVN.SW": "CH", "GOLD-KG": "GOLD"}
_FX_PARAMS = {"USD": (0.075, 1.10), "JPY": (0.095, 130.0), "CHF": (0.055, 1.02)}
_HKD_PEG = 7.80

# (ticker, month, day, amount per share): gross dividends, ex-date in each year.
_DIVIDENDS = [
    ("PG", 1, 20, 0.95), ("PG", 4, 20, 0.95), ("PG", 7, 20, 0.95), ("PG", 10, 20, 0.95),
    ("ADS.DE", 5, 15, 3.00), ("PHIA.AS", 5, 10, 0.85), ("KNEBV.HE", 3, 5, 1.75),
    ("6758.T", 3, 28, 40.0), ("6758.T", 9, 27, 40.0),
    ("1299.HK", 5, 20, 1.20), ("1299.HK", 8, 25, 0.50), ("GIVN.SW", 3, 25, 68.0),
]


@dataclass(frozen=True)
class MarketData:
    """All market inputs of one run. Prices are closes in quote currency per quote unit."""

    securities: pd.DataFrame
    prices: pd.DataFrame
    fx_to_eur: pd.DataFrame
    volumes: pd.DataFrame
    dividends: pd.DataFrame
    betas: pd.Series
    bricks: pd.DataFrame
    brick_distributions: pd.DataFrame
    journal: pd.DataFrame
    declared_holdings: pd.Series
    cash: dict[str, float]


def _correlation(names: list[str]) -> np.ndarray:
    """Block correlation: regions, a weakly linked gold, and risk-off FX behaviour."""
    fx_equity = {"USD": 0.10, "JPY": 0.30, "CHF": 0.20}
    corr = np.eye(len(names))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i >= j:
                continue
            ra, rb = _REGION.get(a, a), _REGION.get(b, b)
            if a in _FX_PARAMS and b in _FX_PARAMS:
                c = 0.45
            elif a in _FX_PARAMS or b in _FX_PARAMS:
                fx, other = (a, b) if a in _FX_PARAMS else (b, a)
                c = 0.30 if other == "GOLD-KG" and fx == "USD" else (
                    0.0 if other == "GOLD-KG" else fx_equity[fx])
            elif "GOLD" in (ra, rb):
                c = 0.05
            else:
                c = 0.55 if ra == rb else 0.35
            corr[i, j] = corr[j, i] = c
    return corr


def _student_t_shocks(rng: np.random.Generator, corr: np.ndarray, n: int) -> np.ndarray:
    """Multivariate Student-t draws with unit variance and the given correlation."""
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((n, corr.shape[0])) @ chol.T
    w = rng.chisquare(STUDENT_DF, size=(n, 1)) / STUDENT_DF
    return z / np.sqrt(w) * np.sqrt((STUDENT_DF - 2) / STUDENT_DF)


def _volatility_regime(n: int) -> np.ndarray:
    """Calm baseline with two stressed stretches, so that losses cluster in time."""
    regime = np.full(n, 0.85)
    regime[140:215] = 2.2
    regime[900:950] = 1.8
    return regime


def _holiday_mask(rng: np.random.Generator, dates: pd.DatetimeIndex, tickers: list[str]
                  ) -> pd.DataFrame:
    """About nine closed sessions a year per exchange; FX and metal never close."""
    mask = pd.DataFrame(False, index=dates, columns=tickers)
    for region in ("US", "EU", "JP", "HK", "CH"):
        closed = rng.random(len(dates)) < 9 / 252
        for t in tickers:
            if _REGION[t] == region:
                mask[t] = closed
    return mask


def _simulate_paths(rng: np.random.Generator, dates: pd.DatetimeIndex
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Log-return paths for securities (quote currency) and FX (units per EUR)."""
    tickers = list(SECURITIES.index)
    names = tickers + list(_FX_PARAMS)
    n = len(dates)
    shocks = _student_t_shocks(rng, _correlation(names), n) * _volatility_regime(n)[:, None]
    vols = np.array([_ASSET_PARAMS[t][0] for t in tickers] + [v for v, _ in _FX_PARAMS.values()])
    drift = np.array([_ASSET_PARAMS[t][1] for t in tickers] + [0.0] * len(_FX_PARAMS))
    log_ret = shocks * vols / np.sqrt(252) + drift / 252 - 0.5 * (vols / np.sqrt(252)) ** 2
    for day, size in ((160, -0.07), (175, -0.05), (182, 0.04), (905, -0.06), (1320, -0.045)):
        for k, t in enumerate(tickers):
            log_ret[day, k] += 0.4 * abs(size) if t == "GOLD-KG" else size
        log_ret[day, len(tickers) + 1] += 0.5 * size  # yen strengthens: fewer JPY per EUR
    log_ret[0] = 0.0
    start = np.array([_ASSET_PARAMS[t][2] for t in tickers] + [s for _, s in _FX_PARAMS.values()])
    levels = start * np.exp(np.cumsum(log_ret, axis=0))
    prices = pd.DataFrame(levels[:, : len(tickers)], index=dates, columns=tickers)
    fx_per_eur = pd.DataFrame(levels[:, len(tickers):], index=dates, columns=list(_FX_PARAMS))
    fx_per_eur["HKD"] = fx_per_eur["USD"] * _HKD_PEG * np.exp(rng.normal(0, 4e-4, n))
    return prices, fx_per_eur


def _dividend_table(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Gross dividend per share on each ex-date inside the simulated range."""
    rows = []
    for year in range(dates[0].year, dates[-1].year + 1):
        for ticker, month, day, amount in _DIVIDENDS:
            ex = pd.Timestamp(year, month, day)
            ex = ex + pd.offsets.BDay(0) if ex.weekday() >= 5 else ex
            if dates[0] < ex <= dates[-1]:
                rows.append((ticker, ex, amount))
    return pd.DataFrame(rows, columns=["ticker", "ex_date", "amount"]).sort_values("ex_date")


def _apply_ex_dividend_drops(prices: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """Lower every later price by the dividend yield on the ex-date, as a real quote would."""
    adjusted = prices.copy()
    for row in dividends.itertuples():
        if row.ex_date in adjusted.index:
            level = adjusted.at[row.ex_date, row.ticker]
            adjusted.loc[row.ex_date:, row.ticker] *= 1 - row.amount / level
    return adjusted


def _add_metal_weekends(prices: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """The metal vendor also quotes on weekends: add those rows over the last two years."""
    last = prices.index[-1]
    weekends = pd.date_range(last - pd.DateOffset(years=2), last, freq="D")
    weekends = weekends[weekends.weekday >= 5]
    extended = prices.reindex(prices.index.union(weekends))
    friday = extended["GOLD-KG"].ffill()
    noise = np.exp(rng.normal(0, 0.002, len(extended)))
    is_weekend = extended.index.weekday >= 5
    extended.loc[is_weekend, "GOLD-KG"] = (friday * noise)[is_weekend]
    return extended


def _volumes(rng: np.random.Generator, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily traded shares over the last 120 sessions, with one drying-up and one busy name."""
    adv = {"PG": 7e6, "ADBE": 3e6, "ADS.DE": 6e5, "PHIA.AS": 3e6, "KNEBV.HE": 5e5,
           "6758.T": 5e6, "1299.HK": 2.5e7, "GIVN.SW": 2.5e4, "GOLD-KG": np.nan}
    recent = dates[-120:]
    data = {}
    for t, base in adv.items():
        v = base * rng.lognormal(0, 0.35, len(recent))
        if t == "GIVN.SW":
            v[-10:] *= 0.55
        if t == "ADS.DE":
            v[-10:] *= 1.45
        data[t] = v
    return pd.DataFrame(data, index=recent)


def _betas() -> pd.Series:
    """Illustrative vendor-style betas, used for the value-weighted beta of the book."""
    return pd.Series({"PG": 0.45, "ADBE": 1.20, "ADS.DE": 1.10, "PHIA.AS": 0.90,
                      "KNEBV.HE": 0.70, "6758.T": 0.95, "1299.HK": 0.85, "GIVN.SW": 0.60},
                     name="beta")

def _bricks(rng: np.random.Generator, prices: pd.DataFrame, fx_to_eur: pd.DataFrame
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """EUR-listed index trackers per currency brick, each a noisy proxy of its region."""
    sessions = prices.index[prices.index.weekday < 5]
    local = prices.loc[sessions].ffill().pct_change(fill_method=None)
    eur = (prices.loc[sessions] * _eur_rate_matrix(prices.loc[sessions], fx_to_eur)).ffill()
    eur = eur.pct_change(fill_method=None)
    specs = {
        "US": (local[["PG", "ADBE"]].mean(axis=1), 0.009),
        "Europe": (eur[["ADS.DE", "PHIA.AS", "KNEBV.HE", "GIVN.SW"]].mean(axis=1), 0.007),
        "Japan": (local["6758.T"] * 0.7, 0.008),
        "Asia ex-Japan": (eur["1299.HK"] * 0.7, 0.009),
        "Gold": (eur["GOLD-KG"], 0.001),
    }
    out = {}
    for brick, (proxy, noise) in specs.items():
        r = proxy.fillna(0.0) + rng.normal(0, noise, len(proxy))
        r.iloc[0] = 0.0
        out[brick] = 100 * np.cumprod(1 + r)
    bricks = pd.DataFrame(out, index=sessions)
    ex_dates = [d for d in sessions if d.month in (6, 12) and d.day >= 15 and d.weekday() == 2]
    ex_dates = [d for i, d in enumerate(ex_dates) if i == 0 or ex_dates[i - 1].month != d.month]
    dist = pd.DataFrame({"brick": "Asia ex-Japan", "ex_date": ex_dates,
                         "amount": [0.012 * bricks.at[d, "Asia ex-Japan"] for d in ex_dates]})
    for row in dist.itertuples():
        bricks.loc[row.ex_date:, "Asia ex-Japan"] -= row.amount
    return bricks, dist


def _eur_rate_matrix(prices: pd.DataFrame, fx_to_eur: pd.DataFrame) -> pd.DataFrame:
    """EUR per unit of each security's currency, aligned on the price grid."""
    rates = fx_to_eur.reindex(prices.index).ffill()
    rates["EUR"] = 1.0
    cols = {t: rates[SECURITIES.at[t, "currency"]] for t in prices.columns}
    return pd.DataFrame(cols, index=prices.index)


def build_journal(prices: pd.DataFrame, fx_to_eur: pd.DataFrame, securities: pd.DataFrame,
                  quantity_scale: dict[str, float] | None = None) -> pd.DataFrame:
    """A trade history exercising cycles, a same-day sale and rebuy, and a closed line.

    Trades are placed on the last N_DAYS sessions of the price grid and priced at the
    close of their session (plus or minus 0.1 % of fees), so the same script works on
    synthetic and downloaded closes.
    """
    sessions = prices.index[prices.index.weekday < 5]
    offset = max(0, len(sessions) - N_DAYS)
    scale = quantity_scale or {}

    def day(i: int) -> pd.Timestamp:
        return sessions[offset + i]

    trades = [
        (5, "PG", "BUY", 600), (5, "ADBE", "BUY", 150), (6, "ADS.DE", "BUY", 350),
        (6, "PHIA.AS", "BUY", 2500), (7, "KNEBV.HE", "BUY", 900), (8, "6758.T", "BUY", 2800),
        (8, "1299.HK", "BUY", 8000), (9, "GIVN.SW", "BUY", 18), (10, "GOLD-KG", "BUY", 1),
        (260, "PG", "BUY", 200), (300, "ADBE", "SELL", 50), (410, "6758.T", "SELL", 2800),
        (520, "6758.T", "BUY", 2500), (610, "KNEBV.HE", "SELL", 400), (700, "GIVN.SW", "BUY", 6),
        (820, "KNEBV.HE", "SELL", 500), (880, "PHIA.AS", "SELL", 1000),
        (880, "PHIA.AS", "BUY", 1500), (1000, "1299.HK", "BUY", 4000), (1100, "PG", "BUY", 150),
        (1250, "ADS.DE", "SELL", 100), (1400, "ADBE", "BUY", 40),
    ]
    rows = []
    for i, name, side, qty in trades:
        ticker = name if name in securities.index else _replacement(name, securities)
        d = day(i)
        close = prices[ticker].loc[:d].dropna().iloc[-1]
        unit = securities.at[ticker, "unit_factor"]
        fee = 1.001 if side == "BUY" else 0.999
        ccy = securities.at[ticker, "currency"]
        rate = 1.0 if ccy == "EUR" else float(fx_to_eur[ccy].loc[:d].dropna().iloc[-1])
        rows.append((d, ticker, side, float(qty * BOOK_SCALE * scale.get(ticker, 1.0)),
                     float(close * unit * fee), ccy, rate))
    cols = ["date", "ticker", "side", "quantity", "price", "currency", "fx_to_eur"]
    return pd.DataFrame(rows, columns=cols)


def _replacement(name: str, securities: pd.DataFrame) -> str:
    """The security standing in for a synthetic one (the metal line in live mode)."""
    same_class = securities[securities["asset_class"] == SECURITIES.at[name, "asset_class"]]
    return same_class.index[0]


def declared_holdings(journal: pd.DataFrame) -> pd.Series:
    """Holdings as a custodian would report them, derived here from the journal."""
    signed = journal["quantity"].where(journal["side"] == "BUY", -journal["quantity"])
    held = signed.groupby(journal["ticker"]).sum()
    return held[held.abs() > 1e-9]


def generate_market(seed: int = DEFAULT_SEED) -> MarketData:
    """Build the full synthetic data set. Same seed, same numbers, on any machine."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=END_DATE, periods=N_DAYS)
    prices, fx_per_eur = _simulate_paths(rng, dates)
    dividends = _dividend_table(dates)
    prices = _apply_ex_dividend_drops(prices, dividends)
    prices = prices.mask(_holiday_mask(rng, dates, list(prices.columns)))
    fx_to_eur = 1.0 / fx_per_eur
    prices = _add_metal_weekends(prices, rng)
    fx_to_eur = fx_to_eur.reindex(prices.index).ffill()
    bricks, dist = _bricks(rng, prices, fx_to_eur)
    journal = build_journal(prices, fx_to_eur, SECURITIES)
    return MarketData(
        securities=SECURITIES.copy(),
        prices=prices,
        fx_to_eur=fx_to_eur,
        volumes=_volumes(rng, dates),
        dividends=dividends.reset_index(drop=True),
        betas=_betas(),
        bricks=bricks,
        brick_distributions=dist,
        journal=journal,
        declared_holdings=declared_holdings(journal),
        cash={"EUR": 600_000.0, "USD": 500_000.0},
    )
