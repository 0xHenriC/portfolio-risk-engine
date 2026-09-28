import pandas as pd
import pytest

from risk_engine import demo
from risk_engine.data.synthetic import generate_market


@pytest.fixture(scope="session")
def market():
    return generate_market()


@pytest.fixture(scope="session")
def result(market):
    return demo.run(market)


def make_journal(rows: list[tuple]) -> pd.DataFrame:
    """Journal from (date, ticker, side, quantity, price) rows, EUR, rate 1."""
    df = pd.DataFrame(rows, columns=["date", "ticker", "side", "quantity", "price"])
    df["date"] = pd.to_datetime(df["date"])
    df["currency"] = "EUR"
    df["fx_to_eur"] = 1.0
    return df
