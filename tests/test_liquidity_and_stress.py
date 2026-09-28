import pandas as pd
import pytest

from risk_engine import liquidity, stress
from risk_engine.data.synthetic import SECURITIES


def test_stressed_scores_are_exactly_two_and_four_times_normal(result):
    liq = result["liquidity"]
    assert liq["severity_ratio"]["moderate"] == pytest.approx(2.0)
    assert liq["severity_ratio"]["severe"] == pytest.approx(4.0)
    assert liq["coverage_pct"] < 100  # the metal line has no traded volume


def test_illiquid_lines_take_several_days_to_sell(result):
    liq = result["liquidity"]
    lines = liq["lines"]
    assert list(liq["least_liquid"].index[:2]) == ["FIC-LV", "FIC-SC"]
    assert (lines.loc[["FIC-LV", "FIC-SC"], "days_normal"] > 3).all()
    share = lines.loc[["FIC-LV", "FIC-SC"], "value_eur"].sum() / result["valuation"]["total_eur"]
    assert share == pytest.approx(0.10, abs=0.02)


def test_days_to_liquidate_and_volume_trend():
    adv = pd.DataFrame({"adv10": [1000.0, 500.0], "adv90": [1000.0, 1000.0]}, index=["A", "B"])
    out = liquidity.compute_liquidity_metrics(pd.Series({"A": 400.0, "B": 400.0}),
                                              pd.Series({"A": 1.0, "B": 3.0}), adv)
    assert out["lines"].at["A", "days_normal"] == pytest.approx(400 / (0.2 * 1000))
    assert out["lines"].at["B", "days_severe"] == pytest.approx(400 / (0.05 * 500))
    assert out["score_days"]["normal"] == pytest.approx((1 * 2 + 3 * 4) / 4)
    assert list(out["drying_up"].index) == ["B"]


@pytest.mark.parametrize(("trend", "named"), [(1.004, False), (1.006, True), (1.3, True)])
def test_surplus_is_named_from_one_percent_once_rounded(trend, named):
    assert liquidity._named_surplus(trend) is named


def _book() -> pd.Series:
    return pd.Series({"PG": 100.0, "ADS.DE": 100.0, "1299.HK": 100.0, "GOLD-KG": 100.0})


def test_equity_shock_hits_equities_only():
    out = stress.apply_scenario({"name": "x", "equity": -0.2}, _book(), SECURITIES, {})
    assert out["pnl_eur"] == pytest.approx(-60.0)
    assert out["contributions"]["GOLD-KG"] == 0.0


def test_currency_shock_hits_lines_and_cash_in_that_currency():
    out = stress.apply_scenario({"name": "x", "currency": {"USD": -0.1}}, _book(), SECURITIES,
                                {"USD": 50.0, "EUR": 50.0})
    # PG and the metal (USD-quoted) and the USD cash lose 10 %.
    assert out["pnl_eur"] == pytest.approx(-25.0)
    assert out["pnl_pct"] == pytest.approx(-25.0 / 500 * 100)


def test_shocks_compound_on_one_line():
    out = stress.apply_scenario({"name": "x", "equity": -0.2, "zone": {"Greater China": -0.25},
                                 "currency": {"HKD": -0.1}}, _book(), SECURITIES, {})
    assert out["contributions"]["1299.HK"] == pytest.approx(100 * (0.8 * 0.75 * 0.9 - 1))


def test_packaged_scenarios_load():
    names = [s["name"] for s in stress.load_scenarios()["scenario"]]
    assert "Equities -20%" in names and "USD -10% vs EUR" in names
