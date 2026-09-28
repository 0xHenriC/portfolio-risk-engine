import numpy as np
import pandas as pd
import pytest
from conftest import make_journal

from risk_engine import journal, performance, portfolio


def test_every_period_balances_to_the_cent(result):
    for p in result["periods"]:
        change = p["value_end_eur"] - p["value_start_eur"]
        assert change == pytest.approx(p["market_eur"] + p["flow_eur"], abs=0.01)


def test_headline_is_the_ratio_of_the_total_return_index(result):
    tr = result["history"]["total_return_index"]
    for p in result["periods"]:
        assert p["return_pct"] == pytest.approx((tr.asof(p["end"]) / tr.asof(p["start"]) - 1) * 100)


def test_window_start_is_the_last_session_on_or_before_the_target():
    sessions = pd.bdate_range("2023-12-01", "2024-03-04")
    start, end = performance.window_bounds(sessions, pd.Timestamp("2024-03-04"), "1M",
                                           sessions[0])
    assert end == pd.Timestamp("2024-03-04")
    assert start == pd.Timestamp("2024-02-02")  # the target 3 February is a Saturday
    start, _ = performance.window_bounds(sessions, pd.Timestamp("2024-03-04"), "YTD", sessions[0])
    assert start == pd.Timestamp("2023-12-29")


def test_market_and_flow_on_a_hand_case():
    idx = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.DataFrame({"A": [10.0] * 5 + [12.0] * 5}, index=idx)
    j = make_journal([("2024-01-01", "A", "BUY", 100, 10.0), ("2024-01-09", "A", "BUY", 50, 11.0)])
    lots = journal.build_lots(j).lots
    history = portfolio.portfolio_value_history(lots, prices)
    block = performance.period_block(lots, prices, history, "1W", idx[-1])
    # Held 100 at 10 at the start: +200; bought 50 at 11, worth 12: +50; paid 550 in.
    assert block["market_eur"] == pytest.approx(250.0)
    assert block["flow_eur"] == pytest.approx(550.0)


def test_sale_in_window_is_matched_on_the_start_holding_first():
    idx = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.DataFrame({"A": np.linspace(10, 19, 10)}, index=idx)
    j = make_journal([("2024-01-01", "A", "BUY", 100, 10.0), ("2024-01-10", "A", "SELL", 40, 18.0)])
    lots = journal.build_lots(j).lots
    history = portfolio.portfolio_value_history(lots, prices)
    block = performance.period_block(lots, prices, history, "1W", idx[-1])
    p_start = prices["A"].asof(block["start"])
    assert block["market_eur"] == pytest.approx(40 * (18 - p_start) + 60 * (19 - p_start))
    assert block["flow_eur"] == pytest.approx(-40 * 18.0)
