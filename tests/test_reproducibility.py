import pandas as pd

from risk_engine.data.synthetic import generate_market


def test_same_seed_same_market():
    a, b = generate_market(7), generate_market(7)
    pd.testing.assert_frame_equal(a.prices, b.prices)
    pd.testing.assert_frame_equal(a.fx_to_eur, b.fx_to_eur)
    pd.testing.assert_frame_equal(a.journal, b.journal)


def test_other_seed_other_market():
    a, b = generate_market(7), generate_market(8)
    assert not a.prices.equals(b.prices)


def test_same_seed_same_risk(market, result):
    from risk_engine import demo

    again = demo.run(generate_market())
    assert again["risk"]["measures"] == result["risk"]["measures"]


def test_market_shape(market):
    weekdays = market.prices[market.prices.index.weekday < 5]
    assert len(weekdays) == 1500
    assert (market.prices.index.weekday >= 5).any()  # metal weekend quotes exist
    assert weekdays.iloc[:, :-1].isna().any().all()  # every exchange has holidays
    assert weekdays.iloc[:, -1].notna().all()  # the metal never closes on a weekday
