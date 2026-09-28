import math

import numpy as np
import pandas as pd
import pytest

from risk_engine import backtest


def test_kupiec_on_a_known_case():
    # 10 exceptions in 250 days at 99 %: LR = 12.955, p = 0.00032 (textbook red zone case).
    lr, p = backtest.kupiec_pof(250, 10, 0.01)
    expected = -2 * ((240 * math.log(0.99) + 10 * math.log(0.01))
                     - (240 * math.log(0.96) + 10 * math.log(0.04)))
    assert lr == pytest.approx(expected, rel=1e-12)
    assert lr == pytest.approx(12.955, abs=1e-3)
    assert p == pytest.approx(3.19e-4, rel=0.01)


def test_kupiec_with_no_exception():
    lr, _ = backtest.kupiec_pof(250, 0, 0.01)
    assert lr == pytest.approx(-2 * 250 * math.log(0.99))


def test_kupiec_accepts_the_expected_count():
    _, p = backtest.kupiec_pof(1000, 10, 0.01)
    assert p == pytest.approx(1.0)


def test_christoffersen_flags_clusters_and_not_isolated_hits():
    clustered = np.zeros(500, dtype=bool)
    clustered[100:105] = True
    isolated = np.zeros(500, dtype=bool)
    isolated[[50, 150, 250, 350, 450]] = True
    assert backtest.christoffersen_independence(clustered)[1] < 0.01
    assert backtest.christoffersen_independence(isolated)[1] > 0.5


def test_christoffersen_hand_computed():
    hits = np.array([0, 1, 1, 0, 0, 0, 1, 0], dtype=bool)
    # Seven transitions: n00=2, n01=2, n10=2, n11=1.
    lr, _ = backtest.christoffersen_independence(hits)
    pi0, pi1, pi = 2 / 4, 1 / 3, 3 / 7
    log_h0 = 4 * math.log(1 - pi) + 3 * math.log(pi)
    log_h1 = (2 * math.log(1 - pi0) + 2 * math.log(pi0)
              + 2 * math.log(1 - pi1) + math.log(pi1))
    assert lr == pytest.approx(-2 * (log_h0 - log_h1))


@pytest.mark.parametrize(("n", "zone"), [(0, "green"), (4, "green"), (5, "yellow"),
                                         (9, "yellow"), (10, "red"), (15, "red")])
def test_basel_zones(n, zone):
    assert backtest.basel_zone(n) == zone


def test_forecast_uses_only_past_returns():
    r = pd.Series(np.random.default_rng(4).normal(size=400),
                  index=pd.bdate_range("2020-01-01", periods=400))
    shocked = r.copy()
    shocked.iloc[300] = -50.0
    a = backtest.rolling_var(r, "historical")
    b = backtest.rolling_var(shocked, "historical")
    assert a.loc[:r.index[300]].equals(b.loc[:r.index[300]])
    assert a.iloc[-1] != b.iloc[-1]


def test_normal_var_under_covers_fat_tails():
    # Student-t with 3 degrees of freedom, scaled to unit variance.
    x = np.random.default_rng(5).standard_t(3, size=6000) / math.sqrt(3)
    r = pd.Series(x, index=pd.bdate_range("2000-01-03", periods=6000))
    normal = backtest.backtest_var(r, "normal")
    hist = backtest.backtest_var(r, "historical")
    assert normal["exceptions"] > hist["exceptions"]
    assert normal["kupiec_p"] < 0.05
