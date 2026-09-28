import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from risk_engine import risk


def test_parametric_var_is_mean_plus_z_sigma():
    r = pd.Series(np.random.default_rng(1).normal(0.05, 1.2, 5000))
    var, es = risk.parametric_var_es(r, 0.99)
    mu, sd = r.mean(), r.std(ddof=1)
    assert var == pytest.approx(mu - norm.ppf(0.99) * sd, rel=1e-12)
    assert es == pytest.approx(mu - sd * norm.pdf(norm.ppf(0.99)) / 0.01, rel=1e-12)


def test_parametric_var_on_zero_mean_is_z_times_sigma():
    x = np.random.default_rng(2).normal(0, 1.0, 10_000)
    r = pd.Series(x - x.mean())
    var, _ = risk.parametric_var_es(r, 0.975)
    assert var == pytest.approx(-1.959964 * r.std(ddof=1), rel=1e-6)


def test_historical_var_converges_to_normal_on_normal_returns():
    r = pd.Series(np.random.default_rng(3).normal(0, 1.0, 400_000))
    for level in (0.95, 0.975, 0.99):
        hist, hist_es = risk.historical_var_es(r, level)
        para, para_es = risk.parametric_var_es(r, level)
        assert hist == pytest.approx(para, rel=0.01)
        assert hist_es == pytest.approx(para_es, rel=0.01)


def test_historical_var_on_a_hand_computed_array():
    r = pd.Series([-5.0, -3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # 10 % quantile, linear: position 0.1 * 9 = 0.9, between -5 and -3.
    var, es = risk.historical_var_es(r, 0.90)
    assert var == pytest.approx(-5 + 0.9 * 2)
    assert es == pytest.approx(-5.0)
    # 20 % quantile: position 1.8, between -3 and -1; ES averages -5 and -3.
    var, es = risk.historical_var_es(r, 0.80)
    assert var == pytest.approx(-3 + 0.8 * 2)
    assert es == pytest.approx(-4.0)


def test_rank_method_takes_the_kth_worst_without_interpolation():
    r = pd.Series(np.arange(-10.0, 90.0))  # 100 returns, worst is -10
    var, es = risk.rank_var_es(r, 0.95)  # k = round(100 * 0.05) = 5
    assert var == -6.0
    assert es == pytest.approx(np.mean([-10, -9, -8, -7]))


def test_es_is_never_a_smaller_loss_than_var(result):
    for (method, level), m in result["risk"]["measures"].items():
        assert m["es_pct"] <= m["var_pct"], (method, level)


def test_consistency_check_raises():
    with pytest.raises(risk.RiskConsistencyError):
        risk.check_es_beyond_var(-2.0, -1.5, "test")


def test_full_revaluation_reweights_days_with_missing_prices():
    idx = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [50.0, 45.0, np.nan],
                           "C": [10.0, 10.0, 11.0], "D": [20.0, 20.0, 20.0]}, index=idx)
    values = pd.Series({"A": 100.0, "B": 300.0, "C": 100.0, "D": 500.0})
    r, covered = risk.portfolio_returns(values, prices)
    assert covered == ["A", "B", "C", "D"]
    assert r.iloc[0] == pytest.approx((100 * 0.10 + 300 * -0.10) / 1000 * 100)
    # Day 2: B has no close, so its value is left out and the others reweighted.
    assert r.iloc[1] == pytest.approx((100 * 0.10 + 100 * 0.10) / 700 * 100)


def test_days_with_too_few_positions_are_dropped():
    idx = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({t: [1.0, 1.1, np.nan] for t in "ABCDEF"}, index=idx)
    prices.loc[idx[2], ["A", "B"]] = 1.2  # 2 of 6 positions: below max(2, 6 // 2) = 3
    r, _ = risk.portfolio_returns(pd.Series(1.0, index=list("ABCDEF")), prices)
    assert list(r.index) == [idx[1]]


def test_weekends_are_removed_from_the_calendar():
    idx = pd.date_range("2024-01-05", "2024-01-08")  # Fri to Mon
    m = pd.DataFrame({"G": [1.0, 1.1, 1.2, 1.3]}, index=idx)
    assert list(risk.business_days_only(m).index.weekday) == [4, 0]


def test_max_drawdown_on_a_hand_path():
    r = pd.Series([10.0, -50.0, 20.0, 100.0], index=pd.bdate_range("2024-01-01", periods=4))
    dd = risk.max_drawdown(r)
    assert dd["max_drawdown_pct"] == pytest.approx(-50.0)
    assert dd["peak"] == r.index[0] and dd["trough"] == r.index[1]
    assert dd["recovery"] == r.index[3] and dd["sessions_to_recovery"] == 3


def test_square_root_of_time():
    assert risk.scale_to_horizon(-2.0, 10) == pytest.approx(-2.0 * math.sqrt(10))


def test_volatility_is_annualised_with_sqrt_252(result):
    rk = result["risk"]
    assert rk["volatility_annualized_pct"] == pytest.approx(
        rk["returns_pct"].std(ddof=1) * math.sqrt(252))
