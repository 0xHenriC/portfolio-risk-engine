import numpy as np
import pandas as pd
import pytest

from risk_engine import benchmark, exposures, quality, report
from risk_engine.data.synthetic import SECURITIES


@pytest.mark.parametrize(("ratio", "factor"), [(0.5, 0.5), (0.49, 0.5), (1.97, 2), (10.3, 10),
                                               (0.667, None), (0.66, None), (1.3, None)])
def test_split_like_ratios(ratio, factor):
    assert quality.split_like_factor(ratio) == factor


def test_price_jump_scan_finds_an_unadjusted_split():
    idx = pd.bdate_range("2024-01-01", periods=4)
    s = pd.DataFrame({"A": [100.0, 101.0, 50.4, 51.0]}, index=idx)
    jumps = quality.scan_price_jumps(s)
    assert jumps["date"].tolist() == [s.index[2]]


def test_freshness_tells_a_holiday_from_a_gap():
    idx = pd.bdate_range("2024-01-01", periods=4)
    p = pd.DataFrame({"A": [1.0, 1.0, 1.0, np.nan], "B": [1.0, 1.0, 1.0, 1.0],
                      "C": [1.0, 1.0, 1.0, np.nan]}, index=idx)
    venue = pd.Series({"A": "X", "B": "X", "C": "Y"})
    out = quality.freshness(p, venue, idx[-1])
    assert out.at["A", "level"] == "alert"  # a peer on the same exchange has the session
    assert out.at["C", "level"] == "info"  # alone on its exchange: probably a holiday
    assert out.at["B", "level"] == "ok"


def test_synthetic_prices_carry_no_split_like_jump(result):
    assert result["price_jumps"].empty
    assert result["raw_unit_trades"].empty


def test_quantity_plausibility():
    warns = quality.quantity_plausibility(pd.Series({"A": 100.0, "B": 10.0, "C": 5.0}),
                                          pd.Series({"A": 160.0, "B": 12.0}))
    assert len(warns) == 2


def test_exposure_bases(result):
    exp = result["exposures"]
    assert exp["sector"].sum() == pytest.approx(1.0)
    assert exp["currency"].sum() == pytest.approx(1.0)
    assert exp["zone"].sum() == pytest.approx(1.0)
    assert "Gold" in exp["currency"].index and "Gold" not in exp["sector"].index
    assert exposures.zone_for_country("  Hong   Kong ") == "Greater China"
    assert exposures.zone_for_country("Atlantis") == "Rest of world"


def test_beta_is_recovered_on_constructed_data():
    idx = pd.bdate_range("2022-01-03", periods=600)
    rng = np.random.default_rng(9)
    rb = rng.normal(0, 0.01, 600)
    rp = 0.0002 + 1.5 * rb + rng.normal(0, 0.002, 600)
    bench = pd.DataFrame({"US": 100 * np.cumprod(1 + rb)}, index=idx)
    pf = pd.Series(100 * np.cumprod(1 + rp), index=idx)
    out = benchmark.compute_alpha_beta(pf, pf, bench, pd.DataFrame(columns=["brick", "ex_date",
                                                                            "amount"]),
                                       pd.Series({"US": 1.0}), idx[0])
    assert out["ok"] and out["beta"] == pytest.approx(1.5, abs=0.03)
    assert out["alpha_annual_pct"] == pytest.approx(0.0002 * 252 * 100, abs=2.0)
    short = benchmark.compute_alpha_beta(pf.iloc[:50], pf.iloc[:50], bench, pd.DataFrame(
        columns=["brick", "ex_date", "amount"]), pd.Series({"US": 1.0}), idx[0])
    assert not short["ok"]


def test_composite_weights_exclude_unmapped_currencies():
    w, unmapped = benchmark.composite_weights(pd.Series({"EUR": 0.3, "CHF": 0.1, "USD": 0.4,
                                                         "BRL": 0.2}))
    assert w.to_dict() == pytest.approx({"Europe": 0.5, "US": 0.5})
    assert list(unmapped.index) == ["BRL"]


def test_total_return_adds_distributions():
    p = pd.Series([100.0, 98.0], index=pd.bdate_range("2024-01-01", periods=2))
    tr = benchmark.total_return(p, pd.Series({p.index[1]: 2.0}))
    assert tr.iloc[-1] == pytest.approx(100.0)


def test_demo_report_and_charts(result, tmp_path):
    text = report.format_report(result)
    assert "Witness check: OK" in text
    paths = report.save_charts(result, tmp_path)
    assert all(p.stat().st_size > 10_000 for p in paths)
    assert SECURITIES.index.isin(result["register"].index).all()
