import pandas as pd
import pytest
from conftest import make_journal

from risk_engine import journal

AS_OF = pd.Timestamp("2024-12-31")


def test_weighted_average_cost_and_cycle_reset():
    j = make_journal([("2024-01-02", "A", "BUY", 100, 10.0), ("2024-01-03", "A", "BUY", 100, 20.0),
                      ("2024-01-04", "A", "SELL", 50, 30.0), ("2024-01-05", "A", "SELL", 150, 12.0),
                      ("2024-01-08", "A", "BUY", 10, 50.0)])
    book = journal.build_lots(j)
    pnl = journal.realized_pnl_by_sale(book.lots)["pnl_eur"].tolist()
    assert pnl == pytest.approx([50 * (30 - 15), 150 * (12 - 15)])
    # The position went back to zero: the rebuy starts a new average, not 15.
    assert book.open_cost_eur["A"] == pytest.approx(500.0)
    assert book.open_quantity["A"] == pytest.approx(10.0)


def test_same_day_sale_draws_on_prior_holding_first():
    j = make_journal([("2024-01-02", "B", "BUY", 100, 10.0), ("2024-01-03", "B", "SELL", 150, 12.0),
                      ("2024-01-03", "B", "BUY", 100, 16.0)])
    book = journal.build_lots(j)
    sale = book.lots[book.lots["quantity"] < 0].iloc[0]
    # 100 units at the prior average 10, then 50 of the day's units at 16.
    assert sale["acquisition_value_eur"] == pytest.approx(-(100 * 10 + 50 * 16))
    assert book.open_cost_eur["B"] == pytest.approx(50 * 16)
    assert journal.split_same_day_sale(100, 10, 100, 16, 150) == pytest.approx((12.0, 100))


def test_overselling_is_refused():
    j = make_journal([("2024-01-02", "C", "BUY", 10, 10.0), ("2024-01-03", "C", "SELL", 11, 10.0)])
    with pytest.raises(journal.JournalError):
        journal.build_lots(j)


def test_validation_reports_every_problem_at_once():
    j = make_journal([("2024-01-02", "A", "HOLD", 10, 10.0), ("2024-01-03", "A", "BUY", -1, 10.0),
                      ("2025-06-01", "A", "BUY", 1, 10.0)])
    j.loc[0, "currency"] = "XYZ"
    with pytest.raises(journal.JournalError) as err:
        journal.validate_journal(j, {"EUR"}, AS_OF)
    assert len(err.value.problems) >= 4


def test_reconciliation_with_declared_holdings():
    j = make_journal([("2024-01-02", "A", "BUY", 10, 10.0)])
    book = journal.build_lots(j)
    journal.reconcile_with_holdings(book, pd.Series({"A": 10.0}))
    with pytest.raises(journal.JournalError):
        journal.reconcile_with_holdings(book, pd.Series({"A": 9.0}))
    with pytest.raises(journal.JournalError):
        journal.reconcile_with_holdings(book, pd.Series({"A": 10.0, "B": 5.0}))


def test_realized_gains_and_losses_by_window():
    j = make_journal([("2024-01-02", "A", "BUY", 100, 10.0), ("2024-02-01", "A", "SELL", 10, 12.0),
                      ("2024-03-01", "A", "SELL", 10, 8.0)])
    lots = journal.build_lots(j).lots
    start, end = pd.Timestamp("2024-01-31"), pd.Timestamp("2024-03-01")
    out = journal.realized_gains_losses(lots, start, end)
    assert out == pytest.approx({"gains_eur": 20.0, "losses_eur": -20.0})
    assert journal.quantity_held_at(lots, "A", pd.Timestamp("2024-02-15")) == 90


def test_synthetic_journal_reconciles(market):
    book = journal.build_lots(market.journal)
    journal.reconcile_with_holdings(book, market.declared_holdings)
    assert book.open_quantity["KNEBV.HE"] == 0.0
