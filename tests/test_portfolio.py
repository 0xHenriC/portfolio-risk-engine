import numpy as np
import pandas as pd
import pytest
from conftest import make_journal

from risk_engine import journal, portfolio


def test_carry_forward_counts_calendar_sessions_not_rows():
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-04", "2024-01-05"])  # Mon, Thu, Fri
    m = pd.DataFrame({"A": [10.0, np.nan, np.nan]}, index=idx)
    out = portfolio.carry_forward_bounded(m, 3)
    assert out.at[pd.Timestamp("2024-01-04"), "A"] == 10.0  # 3 sessions later: allowed
    assert np.isnan(out.at[pd.Timestamp("2024-01-05"), "A"])  # 4 sessions later: too old


def test_weekend_rows_keep_their_values():
    idx = pd.DatetimeIndex(["2024-01-05", "2024-01-06", "2024-01-08"])  # Fri, Sat, Mon
    m = pd.DataFrame({"G": [1.0, 1.5, np.nan], "E": [2.0, np.nan, np.nan]}, index=idx)
    out = portfolio.carry_forward_bounded(m, 3)
    assert out.at[pd.Timestamp("2024-01-06"), "G"] == 1.5
    assert np.isnan(out.at[pd.Timestamp("2024-01-06"), "E"])
    assert out.at[pd.Timestamp("2024-01-08"), "E"] == 2.0


def test_a_purchase_does_not_move_the_index():
    idx = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"A": [10.0, 10.0, 10.0, 11.0, 11.0]}, index=idx)
    j = make_journal([("2024-01-01", "A", "BUY", 10, 10.0), ("2024-01-03", "A", "BUY", 1000, 10.0)])
    h = portfolio.portfolio_value_history(journal.build_lots(j).lots, prices)
    assert h["price_index"].iloc[2] == pytest.approx(100.0)
    assert h["price_index"].iloc[3] == pytest.approx(110.0)
    assert h["value_eur"].iloc[-1] == pytest.approx(1010 * 11.0)


def test_dividends_enter_the_total_return_index_only():
    idx = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=idx)
    j = make_journal([("2024-01-01", "A", "BUY", 10, 100.0)])
    div = pd.DataFrame({"ticker": ["A"], "ex_date": [idx[2]], "amount": [1.0], "amount_eur": [1.0]})
    h = portfolio.portfolio_value_history(journal.build_lots(j).lots, prices, div)
    assert h["price_index"].iloc[-1] == pytest.approx(100.0)
    assert h["total_return_index"].iloc[-1] == pytest.approx(101.0)


def test_valuation_excludes_a_line_without_a_recent_price():
    idx = pd.bdate_range("2024-01-01", periods=8)
    prices = pd.DataFrame({"A": [10.0] * 8, "B": [5.0, 5.0, 5.0] + [np.nan] * 5}, index=idx)
    v = portfolio.value_positions(pd.Series({"A": 1.0, "B": 1.0}), prices, idx[-1])
    assert v["missing_price"] == ["B"] and not v["complete"]
    assert v["total_eur"] == pytest.approx(10.0)


def test_header_total_equals_sum_of_lines(result):
    v = result["valuation"]
    assert v["securities_eur"] == pytest.approx(v["lines"]["value_eur"].sum(), abs=1e-6)
    assert v["total_eur"] == pytest.approx(v["securities_eur"] + v["cash_eur"], abs=1e-6)
