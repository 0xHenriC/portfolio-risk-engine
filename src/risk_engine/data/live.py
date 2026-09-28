"""Optional live mode: daily closes from Yahoo Finance through yfinance, cached locally.

Downloads go to .cache/ (ignored by git): Yahoo's terms do not allow redistributing the
data, so none of it is committed. Only completed sessions are used: a row dated today may
be an intraday quote, not a close. Closes are split-adjusted but not dividend-adjusted;
dividends are downloaded separately and added on their ex-date, as in synthetic mode,
so they are never counted twice. The metal line is replaced by a listed gold tracker
(held in shares, unit factor 1) and the two fictitious illiquid lines are left out, since
they have no market data. Betas keep the illustrative synthetic values, and the benchmark
bricks are not downloaded.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from risk_engine.data import synthetic

CACHE_DIR = Path(".cache")
START = "2007-01-02"
FX_SYMBOLS = {"USD": "EURUSD=X", "JPY": "EURJPY=X", "HKD": "EURHKD=X", "CHF": "EURCHF=X"}
GOLD_TRACKER = "GLD"
GOLD_SHARES_PER_BAR = 350


def _download(symbol: str, refresh: bool) -> pd.DataFrame:
    """Daily history of one symbol, read from the cache when it is there."""
    path = CACHE_DIR / f"{symbol.replace('=', '_').replace('^', '_')}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf

    raw = yf.Ticker(symbol).history(start=START, auto_adjust=False, actions=True)
    if raw.empty:
        raise RuntimeError(f"no data returned for {symbol}")
    raw.index = pd.DatetimeIndex(raw.index.tz_localize(None).normalize())
    frame = raw[["Close", "Volume", "Dividends"]]
    CACHE_DIR.mkdir(exist_ok=True)
    frame.to_csv(path)
    return frame


def live_securities() -> pd.DataFrame:
    """The synthetic universe with the metal bar swapped for a gold tracker in shares."""
    secs = synthetic.SECURITIES.drop(index=["GOLD-KG", *synthetic.FICTITIOUS])
    gold = pd.DataFrame([{"name": "Gold tracker", "currency": "USD", "asset_class": "metal",
                          "sector": None, "industry": None, "country": None,
                          "venue": "Metal", "unit_factor": 1.0}], index=[GOLD_TRACKER])
    return pd.concat([secs, gold])


def load_market(refresh: bool = False) -> synthetic.MarketData:
    """Market data of the same universe from Yahoo, in the synthetic layout."""
    secs = live_securities()
    frames = {t: _download(t, refresh) for t in secs.index}
    prices = pd.DataFrame({t: f["Close"] for t, f in frames.items()}).sort_index()
    prices = prices[prices.index < pd.Timestamp.today().normalize()]  # today is not a close yet
    volumes = pd.DataFrame({t: f["Volume"] for t, f in frames.items()}).sort_index()
    divs = [(t, d, float(a)) for t, f in frames.items()
            for d, a in f["Dividends"].items() if a > 0]
    fx = pd.DataFrame({c: 1.0 / _download(s, refresh)["Close"] for c, s in FX_SYMBOLS.items()})
    fx = fx.sort_index().reindex(prices.index.union(fx.index)).ffill()
    journal = synthetic.build_journal(prices, fx, secs, {GOLD_TRACKER: GOLD_SHARES_PER_BAR})
    base = synthetic.generate_market()
    betas = base.betas.reindex(secs.index)
    return synthetic.MarketData(
        securities=secs, prices=prices, fx_to_eur=fx,
        volumes=volumes[volumes.index.weekday < 5].tail(120),
        dividends=pd.DataFrame(divs, columns=["ticker", "ex_date", "amount"]),
        betas=betas, bricks=pd.DataFrame(),
        brick_distributions=pd.DataFrame(columns=["brick", "ex_date", "amount"]),
        journal=journal, declared_holdings=synthetic.declared_holdings(journal),
        cash=base.cash,
    )
