import numpy as np
import pandas as pd
import pytest
from conftest import make_journal

from risk_engine import dividends, journal


def test_dividend_without_any_known_rate_is_dropped():
    idx = pd.bdate_range("2024-01-01", periods=3)
    local = pd.DataFrame({"A": [10.0, 10.0, 10.0]}, index=idx)
    eur = pd.DataFrame({"A": [np.nan, 9.0, 9.0]}, index=idx)
    divs = pd.DataFrame({"ticker": ["A", "A"], "ex_date": [idx[0], idx[2]], "amount": [1.0, 1.0]})
    out = dividends.dividends_eur(divs, local, eur)
    assert len(out) == 1 and out["amount_eur"].iloc[0] == pytest.approx(0.9)


def test_no_dividend_between_an_exit_and_a_rebuy():
    j = make_journal([("2024-01-02", "A", "BUY", 10, 10.0), ("2024-03-01", "A", "SELL", 10, 10.0),
                      ("2024-06-03", "A", "BUY", 5, 10.0)])
    lots = journal.build_lots(j).lots
    divs = pd.DataFrame({"ticker": "A", "ex_date": pd.to_datetime(["2023-12-01", "2024-02-01",
                                                                   "2024-04-01", "2024-07-01"]),
                         "amount": 1.0, "amount_eur": 1.0})
    got = dividends.dividends_received(lots, divs)
    assert got["ex_date"].dt.strftime("%m").tolist() == ["02", "07"]
    assert got["received_eur"].tolist() == [10.0, 5.0]


def test_ex_date_trades_follow_the_market_rule():
    # Bought on the ex-date: no right to the dividend. Sold on the ex-date: the right stays.
    j = make_journal([("2024-03-01", "A", "BUY", 10, 10.0), ("2024-05-02", "B", "BUY", 20, 10.0),
                      ("2024-06-03", "B", "SELL", 20, 10.0)])
    lots = journal.build_lots(j).lots
    divs = pd.DataFrame({"ticker": ["A", "B"],
                         "ex_date": pd.to_datetime(["2024-03-01", "2024-06-03"]),
                         "amount": 1.0, "amount_eur": 1.0})
    got = dividends.dividends_received(lots, divs)
    assert got["ticker"].tolist() == ["B"]
    assert got["received_eur"].tolist() == [20.0]
    assert dividends.entitled_quantity(lots, "A", pd.Timestamp("2024-03-04")) == 10.0


@pytest.mark.parametrize(("y", "ok"), [(0.0, True), (0.30, True), (0.31, False), (-0.01, False),
                                       (None, False), (np.nan, False)])
def test_yield_plausibility_cap(y, ok):
    assert dividends.is_plausible_yield(y) is ok


def test_portfolio_yield_ignores_implausible_yields():
    v = pd.Series({"A": 100.0, "B": 100.0, "C": 800.0})
    y = pd.Series({"A": 0.02, "B": 0.04, "C": 4.0})
    assert dividends.weighted_portfolio_yield(v, y) == pytest.approx(0.03)


def test_register_identities(result):
    reg = result["register"]
    assert reg["identity_gap_eur"].abs().max() < 1e-6
    lhs = 1 + reg["total_return_pct"] / 100
    rhs = (1 + reg["local_return_pct"] / 100) * (1 + reg["fx_effect_pct"] / 100)
    assert np.allclose(lhs, rhs)
    assert set(reg["state"]) == {"held", "rebought", "exited"}
    assert pd.isna(reg.loc["KNEBV.HE", "unrealized_eur"])  # a closed line has no latent P&L
